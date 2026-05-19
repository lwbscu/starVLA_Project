# Copyright 2025 starVLA community. All rights reserved.
"""Phase 2: AWAC actor with Q(s, a_pi) - Q(s, a_data) weighting (no V network)."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import get_scheduler

from starVLA.dataloader.awac_transition_dataset import build_awac_dataloader
from starVLA.model.framework.VLM4A.QwenPI_awac import Qwen_PI_AWAC
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.model.modules.critic import AWACQCritic
from starVLA.model.lora_utils import apply_lora_if_enabled
from starVLA.training.awac_train_utils import assert_module_frozen, freeze_module
from starVLA.training.trainer_utils.config_tracker import wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, normalize_dotlist_args

deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
logger = get_logger(__name__)


def setup_directories(cfg) -> Path:
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)
    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)
    return output_dir


def load_awac_critic_checkpoint(actor, cfg, device):
    """Load frozen Q critic for advantage weighting only."""
    ckpt_path = cfg.trainer.awac_critic_checkpoint
    payload = torch.load(ckpt_path, map_location="cpu")
    if "value_net" in payload:
        logger.warning(
            "Checkpoint contains legacy value_net weights; ignoring them (AWAC uses Q-only advantage)."
        )
    awac_cfg = cfg.awac
    action_cfg = cfg.framework.action_model
    critic = AWACQCritic(
        qwen_vl_interface=actor.qwen_vl_interface,
        action_dim=int(action_cfg.action_dim),
        state_dim=int(action_cfg.state_dim),
        action_horizon=int(action_cfg.action_horizon),
        hidden_dim=int(awac_cfg.hidden_dim),
        num_q_heads=int(awac_cfg.num_q_heads),
        num_layers=int(awac_cfg.transformer_layers),
        nhead=int(awac_cfg.get("nhead", 8)),
        dim_feedforward=int(awac_cfg.get("dim_feedforward", 2048)),
        freeze_visual=True,
        max_vision_tokens=int(awac_cfg.get("max_vision_tokens", 256)),
        max_text_tokens=int(awac_cfg.get("max_text_tokens", 128)),
    )
    critic.load_state_dict(payload["critic"], strict=False)
    critic.to(device)
    freeze_module(critic, eval_mode=True)
    assert_module_frozen(critic, "Critic (actor phase)")
    return critic


def batch_to_actor_examples(batch, include_action: bool = True):
    examples = []
    batch_size = batch["action"].shape[0]
    for i in range(batch_size):
        ex = {
            "image": batch["image"][i],
            "lang": batch["lang"][i],
        }
        if include_action:
            ex["action"] = batch["action"][i].detach().cpu().numpy()
        if "state" in batch:
            ex["state"] = batch["state"][i].detach().cpu().numpy()
        examples.append(ex)
    return examples


def build_frozen_actor_pi_snapshot(actor_train: Qwen_PI_AWAC) -> Qwen_PI_AWAC:
    """
    Frozen policy snapshot for advantage computation only.

    Taken once after BC weights are loaded; never updated during actor phase.
    """
    actor_pi = copy.deepcopy(actor_train)
    freeze_module(actor_pi, eval_mode=True)
    assert_module_frozen(actor_pi, "Actor_pi (advantage snapshot)")
    return actor_pi


def predict_action_tensor(actor_pi: Qwen_PI_AWAC, batch) -> torch.Tensor:
    """Run frozen actor_pi inference to get a_pi with shape (B, H, action_dim)."""
    infer_examples = batch_to_actor_examples(batch, include_action=False)
    pred = actor_pi.predict_action(infer_examples)
    actions = torch.as_tensor(
        pred["normalized_actions"],
        device=batch["action"].device,
        dtype=batch["action"].dtype,
    )
    horizon = int(actor_pi.action_horizon)
    if actions.ndim == 2:
        actions = actions.unsqueeze(1)
    return actions[:, -horizon:, :]


def compute_awac_weights(actor_pi, critic, batch, awac_cfg) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Advantage: A = Q(s, a_pi) - Q(s, a_data).

    a_pi comes from the frozen actor_pi snapshot; a_data from the offline batch.
    Critic stays frozen (no grad).
    """
    critic.eval()
    with torch.inference_mode():
        action_pi = predict_action_tensor(actor_pi, batch)
        q_pi = critic.min_q(
            batch["image"],
            batch["lang"],
            action_pi,
            state=batch.get("state"),
        ).squeeze(-1)
        q_data = critic.min_q(
            batch["image"],
            batch["lang"],
            batch["action"],
            state=batch.get("state"),
        ).squeeze(-1)
        adv = q_pi - q_data
        weights = torch.exp(adv / float(awac_cfg.awac_lambda))
        weights = torch.clamp(weights, max=float(awac_cfg.awac_weight_max))
    metrics = {
        "awac_adv_mean": float(adv.detach().mean().cpu()),
        "q_pi_mean": float(q_pi.detach().mean().cpu()),
        "q_data_mean": float(q_data.detach().mean().cpu()),
    }
    return weights, metrics


class AWACActorTrainer(TrainerUtils):
    def __init__(self, cfg, actor, actor_pi, critic, dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.actor = actor
        self.actor_pi = actor_pi
        self.critic = critic
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.completed_steps = 0

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        set_seed(self.config.seed + rank)

        freeze_modules = getattr(self.config.trainer, "freeze_modules", None)
        self.actor = self.freeze_backbones(self.actor, freeze_modules=freeze_modules)

        freeze_module(self.critic, eval_mode=True)
        assert_module_frozen(self.critic, "Critic (actor phase)")
        freeze_module(self.actor_pi, eval_mode=True)
        assert_module_frozen(self.actor_pi, "Actor_pi (advantage snapshot)")

        modules = [self.actor, self.optimizer, self.dataloader]
        prepared = self.setup_distributed_training(self.accelerator, *modules)
        self.actor, self.optimizer, self.dataloader = prepared

        frozen_params = {id(p) for p in self.critic.parameters()} | {id(p) for p in self.actor_pi.parameters()}
        for group in self.optimizer.param_groups:
            for param in group["params"]:
                if id(param) in frozen_params:
                    raise RuntimeError(
                        "Optimizer must only include trainable actor parameters "
                        "(not critic or frozen actor_pi)."
                    )

        if self.accelerator.is_main_process:
            logger.info(
                "AWAC actor phase: trainable actor + frozen actor_pi snapshot; "
                "critic frozen; A = Q(s,a_pi) - Q(s,a_data)."
            )
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="awac-actor",
            )

    def _save_checkpoint(self):
        if not self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()
            return
        ckpt_path = os.path.join(
            self.config.output_dir,
            "checkpoints",
            f"steps_{self.completed_steps}_pytorch_model.pt",
        )
        state_dict = self.accelerator.get_state_dict(self.actor)
        torch.save(state_dict, ckpt_path)
        logger.info(f"Saved AWAC actor checkpoint to {ckpt_path}")
        self.accelerator.wait_for_everyone()

    def train(self):
        max_steps = int(self.config.trainer.actor_max_train_steps)
        data_iter = iter(self.dataloader)
        progress = tqdm(
            total=max_steps,
            disable=not self.accelerator.is_local_main_process,
        )

        while self.completed_steps < max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.dataloader)
                batch = next(data_iter)

            for key in ("action", "next_action", "reward", "done", "state", "next_state"):
                if key in batch and torch.is_tensor(batch[key]):
                    batch[key] = batch[key].to(self.accelerator.device)

            actor_train = self.accelerator.unwrap_model(self.actor)
            weights, awac_metrics = compute_awac_weights(
                self.actor_pi,
                self.critic,
                batch,
                self.config.awac,
            )

            examples = batch_to_actor_examples(batch)
            self.optimizer.zero_grad()
            with self.accelerator.accumulate(self.actor):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = actor_train(examples, awac_weights=weights)
                    loss = output["action_loss"]
                self.accelerator.backward(loss)
                if self.config.trainer.get("max_grad_norm"):
                    actor_params = [p for p in self.actor.parameters() if p.requires_grad]
                    self.accelerator.clip_grad_norm_(actor_params, self.config.trainer.max_grad_norm)
                self.optimizer.step()
                if self.accelerator.sync_gradients:
                    self.lr_scheduler.step()

            if self.accelerator.sync_gradients:
                self.completed_steps += 1
                progress.update(1)

            if self.completed_steps % self.config.trainer.logging_frequency == 0:
                metrics = {
                    "action_loss": float(loss.detach().cpu()),
                    "awac_weight_mean": float(weights.detach().mean().cpu()),
                    **awac_metrics,
                }
                if self.accelerator.is_main_process:
                    wandb.log(metrics, step=self.completed_steps)
                progress.set_postfix(metrics)

            if self.completed_steps % self.config.trainer.save_interval == 0:
                self._save_checkpoint()

        self._save_checkpoint()
        if self.accelerator.is_main_process:
            wandb.finish()


def main(cfg):
    cfg = wrap_config(cfg)
    setup_directories(cfg)

    actor = Qwen_PI_AWAC(cfg)
    actor = apply_lora_if_enabled(actor, cfg, sanitize_freeze_modules=True)
    pretrained = getattr(cfg.trainer, "pretrained_checkpoint", None)
    if pretrained:
        TrainerUtils.load_pretrained_backbones(actor, pretrained)

    actor_pi = build_frozen_actor_pi_snapshot(actor)

    device = accelerator.device
    actor_pi.to(device)
    critic = load_awac_critic_checkpoint(actor, cfg, device)

    dataloader = build_awac_dataloader(cfg)
    param_groups = build_param_lr_groups(model=actor, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=float(cfg.trainer.learning_rate.base),
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=float(cfg.trainer.optimizer.weight_decay),
    )
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(cfg.trainer.num_warmup_steps),
        num_training_steps=int(cfg.trainer.actor_max_train_steps),
    )

    trainer = AWACActorTrainer(
        cfg=cfg,
        actor=actor,
        actor_pi=actor_pi,
        critic=critic,
        dataloader=dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )
    trainer.prepare_training()
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, required=True)
    args, clipargs = parser.parse_known_args()
    cfg = OmegaConf.load(args.config_yaml)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(normalize_dotlist_args(clipargs)))
    cfg = apply_config_compat(cfg)
    cfg.config_yaml = args.config_yaml
    main(cfg)
