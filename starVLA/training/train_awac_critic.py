# Copyright 2025 starVLA community. All rights reserved.
"""Phase 1: AWAC Q-critic training (TD loss only, no V network)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_scheduler

from starVLA.dataloader.awac_transition_dataset import build_awac_dataloader
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.model.modules.critic import AWACQCritic, resolve_qwen_visual_module, soft_update_target
from starVLA.training.awac_train_utils import assert_module_frozen, freeze_module
from starVLA.training.trainer_utils.config_tracker import wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, normalize_dotlist_args

deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
logger = get_logger(__name__)


def _count_params(module: torch.nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def log_awac_critic_model_scope(cfg, actor, critic: AWACQCritic) -> None:
    """Rank-0 summary: which parts of Qwen 4B are used vs trained (critic phase)."""
    if dist.is_initialized() and dist.get_rank() != 0:
        return
    base_vlm = str(cfg.framework.qwenvl.base_vlm)
    qwen_iface = critic.qwen_vl_interface
    backbone = getattr(qwen_iface, "model", None)

    visual = getattr(critic, "visual", None)
    visual_src, visual_src_path = resolve_qwen_visual_module(qwen_iface)

    lines = [
        "========== AWAC critic model scope (rank 0) ==========",
        f"base_vlm path: {base_vlm}",
        f"BC checkpoint: {getattr(cfg.trainer, 'pretrained_checkpoint', None)}",
        f"frozen actor (init only): {_count_params(actor) / 1e6:.1f}M params "
        f"(trainable {_count_params(actor, True) / 1e6:.1f}M; includes full QwenPI + action_model)",
        f"qwen visual source path: {visual_src_path} "
        f"(resolved={'yes' if visual_src is not None else 'no'})",
        f"critic.visual deepcopy (frozen vision-only forward): "
        f"{_count_params(visual) / 1e6:.1f}M params"
        if visual is not None
        else "critic.visual: MISSING",
        f"critic trainable (Q head + 6L Transformer + proj): "
        f"{_count_params(critic, True) / 1e6:.1f}M params",
        f"critic total (visual copy + trainable): {_count_params(critic) / 1e6:.1f}M params",
        "Forward path: 2x visual tower / camera + text embed only + 6L critic Transformer.",
        "NOT used for Q forward: LLM decoder, action_model diffusion, policy rollout.",
        "====================================================",
    ]
    for line in lines:
        logger.info(line)
        print(line, flush=True)


def setup_directories(cfg) -> Path:
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)
    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)
    return output_dir


class AWACCriticTrainer(TrainerUtils):
    def __init__(self, cfg, actor, critic, critic_target, dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.actor = actor
        self.critic = critic
        self.critic_target = critic_target
        self.model = critic
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.completed_steps = 0
        self.awac_cfg = cfg.awac
        self.tb_writer: SummaryWriter | None = None

    def _init_tensorboard(self) -> None:
        use_tb = getattr(self.config.trainer, "use_tensorboard", True) not in ["False", False]
        if not use_tb or not self.accelerator.is_main_process:
            return
        log_dir = getattr(self.config.trainer, "tensorboard_log_dir", None)
        if not log_dir:
            log_dir = os.path.join(self.config.output_dir, "tensorboard")
        os.makedirs(log_dir, exist_ok=True)
        self.tb_writer = SummaryWriter(log_dir=log_dir)
        logger.info(f"TensorBoard logging enabled: {log_dir}")

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        if not self.accelerator.is_main_process:
            return
        step = self.completed_steps
        wandb.log(metrics, step=step)
        if self.tb_writer is None:
            return
        for key, value in metrics.items():
            self.tb_writer.add_scalar(key, value, step)
        last_lrs = self.lr_scheduler.get_last_lr()
        self.tb_writer.add_scalar("learning_rate", last_lrs[0], step)

    def _unwrap_critic(self) -> AWACQCritic:
        return self.accelerator.unwrap_model(self.model)

    def _finish_logging(self) -> None:
        if not self.accelerator.is_main_process:
            return
        wandb.finish()
        if self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()
            self.tb_writer = None

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        set_seed(self.config.seed + rank)

        # Phase 1: only train Q critic; BC actor is a fixed initialization source.
        freeze_module(self.actor, eval_mode=True)
        assert_module_frozen(self.actor, "Actor (critic phase)")
        freeze_module(self.critic_target, eval_mode=True)

        modules = [self.model, self.optimizer, self.dataloader]
        prepared = self.setup_distributed_training(self.accelerator, *modules)
        self.model, self.optimizer, self.dataloader = prepared[:3]
        unwrapped_critic = self._unwrap_critic()
        self.critic = unwrapped_critic
        self.critic_target.load_state_dict(unwrapped_critic.state_dict())
        self.critic_target.to(self.accelerator.device)

        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="awac-critic",
            )
        self._init_tensorboard()
        if self.accelerator.is_main_process:
            logger.info("AWAC critic phase: actor frozen; training Q network only.")
            if self.tb_writer is not None:
                logger.info(f"TensorBoard log dir: {self.tb_writer.log_dir}")
                print(f"[tensorboard] writing to {self.tb_writer.log_dir}", flush=True)
            log_awac_critic_model_scope(self.config, self.actor, self._unwrap_critic())

    def _save_checkpoint(self):
        if not self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()
            return
        ckpt_path = os.path.join(
            self.config.output_dir,
            "checkpoints",
            f"steps_{self.completed_steps}_critic.pt",
        )
        payload = {
            "critic": self._unwrap_critic().state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "steps": self.completed_steps,
            "awac": OmegaConf.to_container(self.config.awac, resolve=True),
        }
        torch.save(payload, ckpt_path)
        logger.info(f"Saved AWAC critic checkpoint to {ckpt_path}")
        self.accelerator.wait_for_everyone()

    def train(self):
        max_steps = int(self.config.trainer.critic_max_train_steps)
        gamma = float(self.awac_cfg.gamma)
        horizon = int(self.config.framework.action_model.action_horizon)
        polyak = float(self.awac_cfg.polyak_tau)

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

            self.optimizer.zero_grad()
            with self.accelerator.autocast():
                critic = self._unwrap_critic()
                critic_loss, critic_metrics = critic.td_loss(
                    batch,
                    self.critic_target,
                    gamma=gamma,
                    action_horizon=horizon,
                )

            self.accelerator.backward(critic_loss)
            if self.config.trainer.get("max_grad_norm"):
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.trainer.max_grad_norm,
                )
            self.optimizer.step()
            self.lr_scheduler.step()

            soft_update_target(self.critic_target, self._unwrap_critic(), polyak)

            self.completed_steps += 1
            progress.update(1)

            if self.completed_steps % self.config.trainer.logging_frequency == 0:
                metrics = {**critic_metrics, "total_loss": float(critic_loss.detach().cpu())}
                self._log_metrics(metrics)
                progress.set_postfix(metrics)

            if self.completed_steps % self.config.trainer.save_interval == 0:
                self._save_checkpoint()

        self._save_checkpoint()
        self._finish_logging()


def build_critic_modules(cfg, actor):
    awac_cfg = cfg.awac
    action_cfg = cfg.framework.action_model
    critic_kwargs = dict(
        qwen_vl_interface=actor.qwen_vl_interface,
        action_dim=int(action_cfg.action_dim),
        state_dim=int(action_cfg.state_dim),
        action_horizon=int(action_cfg.action_horizon),
        hidden_dim=int(awac_cfg.hidden_dim),
        num_q_heads=int(awac_cfg.num_q_heads),
        num_layers=int(awac_cfg.transformer_layers),
        nhead=int(awac_cfg.get("nhead", 8)),
        dim_feedforward=int(awac_cfg.get("dim_feedforward", 2048)),
        freeze_visual=awac_cfg.get("freeze_visual", True) not in ["False", False],
        dropout=float(awac_cfg.get("dropout", 0.1)),
        max_vision_tokens=int(awac_cfg.get("max_vision_tokens", 256)),
        max_text_tokens=int(awac_cfg.get("max_text_tokens", 128)),
    )
    critic = AWACQCritic(**critic_kwargs)
    critic_target = AWACQCritic(**critic_kwargs)
    critic_target.load_state_dict(critic.state_dict())
    for param in critic_target.parameters():
        param.requires_grad = False
    return critic, critic_target


def main(cfg):
    cfg = wrap_config(cfg)
    setup_directories(cfg)

    actor = build_framework(cfg)
    pretrained = getattr(cfg.trainer, "pretrained_checkpoint", None)
    if pretrained:
        TrainerUtils.load_pretrained_backbones(actor, pretrained)

    critic, critic_target = build_critic_modules(cfg, actor)
    dataloader = build_awac_dataloader(cfg)

    trainable = [p for p in critic.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("AWAC critic has no trainable parameters.")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg.trainer.learning_rate.critic),
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=float(cfg.trainer.optimizer.weight_decay),
    )
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(cfg.trainer.num_warmup_steps),
        num_training_steps=int(cfg.trainer.critic_max_train_steps),
    )

    trainer = AWACCriticTrainer(
        cfg=cfg,
        actor=actor,
        critic=critic,
        critic_target=critic_target,
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
