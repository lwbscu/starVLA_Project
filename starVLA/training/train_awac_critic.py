# Copyright 2025 starVLA community. All rights reserved.
"""Phase 1: AWAC critic (Q + expectile V) training."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
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
from starVLA.model.framework.base_framework import build_framework
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.model.modules.critic import AWACQCritic, AWACValueNetwork, soft_update_target
from starVLA.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils, normalize_dotlist_args

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


class AWACCriticTrainer(TrainerUtils):
    def __init__(self, cfg, actor, critic, critic_target, value_net, dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.actor = actor
        self.critic = critic
        self.critic_target = critic_target
        self.value_net = value_net
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.completed_steps = 0
        self.awac_cfg = cfg.awac

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        set_seed(self.config.seed + rank)

        self.actor.eval()
        for param in self.actor.parameters():
            param.requires_grad = False

        self.critic_target.eval()
        for param in self.critic_target.parameters():
            param.requires_grad = False

        modules = [self.critic, self.value_net, self.optimizer, self.dataloader]
        prepared = self.setup_distributed_training(self.accelerator, *modules)
        self.critic, self.value_net, self.optimizer, self.dataloader = prepared[:4]
        unwrapped_critic = self.accelerator.unwrap_model(self.critic)
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
            "critic": self.accelerator.unwrap_model(self.critic).state_dict(),
            "critic_target": self.accelerator.unwrap_model(self.critic_target).state_dict(),
            "value_net": self.accelerator.unwrap_model(self.value_net).state_dict(),
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
        expectile_tau = float(self.awac_cfg.expectile_tau)

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
                critic_loss, critic_metrics = self.accelerator.unwrap_model(self.critic).td_loss(
                    batch,
                    self.accelerator.unwrap_model(self.critic_target),
                    gamma=gamma,
                    action_horizon=horizon,
                )
                with torch.no_grad():
                    q_behavior = self.accelerator.unwrap_model(self.critic).min_q(
                        batch["image"],
                        batch["lang"],
                        batch["action"],
                        state=batch.get("state"),
                    ).squeeze(-1)
                value_loss, value_metrics = self.accelerator.unwrap_model(self.value_net).expectile_loss(
                    batch,
                    q_target=q_behavior,
                    tau=expectile_tau,
                )
                total_loss = critic_loss + value_loss

            self.accelerator.backward(total_loss)
            if self.config.trainer.get("max_grad_norm"):
                self.accelerator.clip_grad_norm_(
                    list(self.critic.parameters()) + list(self.value_net.parameters()),
                    self.config.trainer.max_grad_norm,
                )
            self.optimizer.step()
            self.lr_scheduler.step()

            unwrapped_critic = self.accelerator.unwrap_model(self.critic)
            unwrapped_target = self.accelerator.unwrap_model(self.critic_target)
            soft_update_target(unwrapped_target, unwrapped_critic, polyak)

            self.completed_steps += 1
            progress.update(1)

            if self.completed_steps % self.config.trainer.logging_frequency == 0:
                metrics = {**critic_metrics, **value_metrics, "total_loss": float(total_loss.detach().cpu())}
                if self.accelerator.is_main_process:
                    wandb.log(metrics, step=self.completed_steps)
                progress.set_postfix(metrics)

            if self.completed_steps % self.config.trainer.save_interval == 0:
                self._save_checkpoint()

        self._save_checkpoint()
        if self.accelerator.is_main_process:
            wandb.finish()


def build_critic_modules(cfg, actor):
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
        freeze_visual=awac_cfg.get("freeze_visual", True) not in ["False", False],
        dropout=float(awac_cfg.get("dropout", 0.1)),
    )
    critic_target = copy.deepcopy(critic)
    for param in critic_target.parameters():
        param.requires_grad = False
    value_net = AWACValueNetwork(critic=critic, hidden_dim=int(awac_cfg.get("value_hidden_dim", 256)))
    return critic, critic_target, value_net


def main(cfg):
    cfg = wrap_config(cfg)
    setup_directories(cfg)

    actor = build_framework(cfg)
    pretrained = getattr(cfg.trainer, "pretrained_checkpoint", None)
    if pretrained:
        TrainerUtils.load_pretrained_backbones(actor, pretrained)

    critic, critic_target, value_net = build_critic_modules(cfg, actor)
    dataloader = build_awac_dataloader(cfg)

    params = list(critic.parameters()) + list(value_net.parameters())
    trainable = [p for p in params if p.requires_grad]
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
        value_net=value_net,
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
