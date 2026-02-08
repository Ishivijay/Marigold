# Copyright 2023-2025 Marigold Team, ETH Zürich. All rights reserved.
# LoRA adaptation for prosthesis depth estimation

import logging
import numpy as np
import os
import shutil
import torch
from PIL import Image
from datetime import datetime
from diffusers import DDPMScheduler, DDIMScheduler
from omegaconf import OmegaConf
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Union

try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from marigold.marigold_depth_pipeline import MarigoldDepthPipeline, MarigoldDepthOutput
from src.util import metric
from src.util.alignment import align_depth_least_square
from src.util.data_loader import skip_first_batches
from src.util.logging_util import tb_logger, eval_dict_to_text
from src.util.loss import get_loss
from src.util.lr_scheduler import IterExponential
from src.util.metric import MetricTracker
from src.util.multi_res_noise import multi_res_noise_like
from src.util.seeding import generate_seed_sequence

# Import base trainer and copy all methods
from .marigold_depth_trainer import MarigoldDepthTrainer as BaseTrainer


class MarigoldDepthTrainerLoRA(BaseTrainer):
    """LoRA-enabled trainer - inherits from base trainer."""
    
    def __init__(self, cfg, model, train_dataloader, device, out_dir_ckpt, 
                 out_dir_eval, out_dir_vis, accumulation_steps, 
                 val_dataloaders=None, vis_dataloaders=None):
        
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT required: pip install peft")
        
        # Store before parent init
        self.cfg = cfg
        self.model = model
        self.device = device
        
        # Adapt input layers before LoRA
        if 8 != self.model.unet.config["in_channels"]:
            self._replace_unet_conv_in_prelora()
        
        # Configure LoRA
        lora_r = self.cfg.trainer.get("lora_r", 16)
        lora_alpha = self.cfg.trainer.get("lora_alpha", 32)
        lora_dropout = self.cfg.trainer.get("lora_dropout", 0.1)
        lora_target_modules = self.cfg.trainer.get("lora_target_modules", 
                                                   ["to_q", "to_k", "to_v", "to_out.0"])
        
        # Convert OmegaConf objects to plain Python types for JSON serialization
        lora_r = int(lora_r) if lora_r is not None else 16
        lora_alpha = int(lora_alpha) if lora_alpha is not None else 32
        lora_dropout = float(lora_dropout) if lora_dropout is not None else 0.1
        # Convert ListConfig to plain list
        if hasattr(lora_target_modules, '__iter__') and not isinstance(lora_target_modules, str):
            lora_target_modules = list(lora_target_modules)
        
        logging.info(f"LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        
        lora_config = LoraConfig(
            r=lora_r, 
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            # Don't specify task_type for diffusion models - let PEFT be model-agnostic
        )
        
        # Apply LoRA before parent init
        self.model.unet = get_peft_model(self.model.unet, lora_config)
        trainable_params, all_params = self.model.unet.get_nb_trainable_parameters()
        logging.info(f"✓ LoRA: {trainable_params:,}/{all_params:,} ({100*trainable_params/all_params:.4f}%)")
        
        # Call parent init (skipping their unet setup)
        self._init_from_base(cfg, model, train_dataloader, device, out_dir_ckpt,
                            out_dir_eval, out_dir_vis, accumulation_steps,
                            val_dataloaders, vis_dataloaders)
    
    def _replace_unet_conv_in_prelora(self):
        """Replace conv_in before applying LoRA."""
        _weight = self.model.unet.conv_in.weight.clone()
        _bias = self.model.unet.conv_in.bias.clone()
        _weight = _weight.repeat((1, 2, 1, 1)) * 0.5
        _n_convin_out_channel = self.model.unet.conv_in.out_channels
        _new_conv_in = Conv2d(8, _n_convin_out_channel, kernel_size=(3, 3), 
                             stride=(1, 1), padding=(1, 1))
        _new_conv_in.weight = Parameter(_weight)
        _new_conv_in.bias = Parameter(_bias)
        self.model.unet.conv_in = _new_conv_in
        self.model.unet.config["in_channels"] = 8
        logging.info("UNet conv_in replaced (8 channels)")
    
    def _init_from_base(self, cfg, model, train_dataloader, device, out_dir_ckpt,
                       out_dir_eval, out_dir_vis, accumulation_steps,
                       val_dataloaders, vis_dataloaders):
        """Initialize using base class pattern."""
        self.seed = self.cfg.trainer.init_seed
        self.out_dir_ckpt = out_dir_ckpt
        self.out_dir_eval = out_dir_eval
        self.out_dir_vis = out_dir_vis
        self.train_loader = train_dataloader
        self.val_loaders = val_dataloaders
        self.vis_loaders = vis_dataloaders
        self.accumulation_steps = accumulation_steps
        
        # Encode empty text
        self.model.encode_empty_text()
        self.empty_text_embed = self.model.empty_text_embed.detach().clone().to(device)
        self.model.unet.enable_xformers_memory_efficient_attention()
        
        # Freeze non-LoRA parts
        self.model.vae.requires_grad_(False)
        self.model.text_encoder.requires_grad_(False)
        self.model.unet.enable_gradient_checkpointing()
        
        # Optimizer
        lr = self.cfg.lr
        try:
            import bitsandbytes as bnb
            self.optimizer = bnb.optim.AdamW8bit(self.model.unet.parameters(), lr=lr,
                                                weight_decay=self.cfg.get("weight_decay", 0.01))
        except ImportError:
            self.optimizer = Adam(self.model.unet.parameters(), lr=lr,
                                 weight_decay=self.cfg.get("weight_decay", 0.01))
        
        # LR scheduler
        lr_func = IterExponential(
            total_iter_length=self.cfg.lr_scheduler.kwargs.total_iter,
            final_ratio=self.cfg.lr_scheduler.kwargs.final_ratio,
            warmup_steps=self.cfg.lr_scheduler.kwargs.warmup_steps,
        )
        self.lr_scheduler = LambdaLR(optimizer=self.optimizer, lr_lambda=lr_func)
        
        # Loss
        self.loss = get_loss(loss_name=self.cfg.loss.name, **self.cfg.loss.kwargs)
        
        # Schedulers
        self.training_noise_scheduler = DDPMScheduler.from_config(
            self.model.scheduler.config,
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
        )
        self.prediction_type = self.training_noise_scheduler.config.prediction_type
        self.scheduler_timesteps = self.training_noise_scheduler.config.num_train_timesteps
        self.model.scheduler = DDIMScheduler.from_config(self.training_noise_scheduler.config)
        
        # Metrics
        self.metric_funcs = [getattr(metric, _met) for _met in cfg.eval.eval_metrics]
        self.train_metrics = MetricTracker(*["loss"])
        self.val_metrics = MetricTracker(*[m.__name__ for m in self.metric_funcs])
        
        self.main_val_metric = cfg.validation.main_val_metric
        self.main_val_metric_goal = cfg.validation.main_val_metric_goal
        self.best_metric = 1e8 if "minimize" == self.main_val_metric_goal else -1e8
        
        # Settings
        self.max_epoch = self.cfg.max_epoch
        self.max_iter = self.cfg.max_iter
        self.gradient_accumulation_steps = accumulation_steps
        self.gt_depth_type = self.cfg.gt_depth_type
        self.gt_mask_type = self.cfg.gt_mask_type
        self.save_period = self.cfg.trainer.save_period
        self.backup_period = self.cfg.trainer.backup_period
        self.val_period = self.cfg.trainer.validation_period
        self.vis_period = self.cfg.trainer.visualization_period
        
        # Multi-res noise
        self.apply_multi_res_noise = self.cfg.multi_res_noise is not None
        if self.apply_multi_res_noise:
            self.mr_noise_strength = self.cfg.multi_res_noise.strength
            self.annealed_mr_noise = self.cfg.multi_res_noise.annealed
            self.mr_noise_downscale_strategy = self.cfg.multi_res_noise.downscale_strategy
        
        # State
        self.epoch = 1
        self.n_batch_in_epoch = 0
        self.effective_iter = 0
        self.in_evaluation = False
        self.global_seed_sequence = []
    
    def save_checkpoint(self, ckpt_name, save_train_state):
        """Override to save LoRA adapter separately."""
        ckpt_dir = os.path.join(self.out_dir_ckpt, ckpt_name)
        logging.info(f"Saving LoRA checkpoint: {ckpt_name}")
        
        temp_ckpt_dir = None
        if os.path.exists(ckpt_dir):
            temp_ckpt_dir = os.path.join(os.path.dirname(ckpt_dir), f"_old_{os.path.basename(ckpt_dir)}")
            if os.path.exists(temp_ckpt_dir):
                shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            os.rename(ckpt_dir, temp_ckpt_dir)
        
        # Save UNet with LoRA
        unet_path = os.path.join(ckpt_dir, "unet")
        self.model.unet.save_pretrained(unet_path, safe_serialization=True)
        
        # Save LoRA adapter separately
        lora_path = os.path.join(ckpt_dir, "lora_adapter")
        self.model.unet.save_pretrained(lora_path, safe_serialization=True)
        
        # Save scheduler
        scheduler_path = os.path.join(ckpt_dir, "scheduler")
        self.model.scheduler.save_pretrained(scheduler_path)
        
        if save_train_state:
            state = {
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.lr_scheduler.state_dict(),
                "config": self.cfg,
                "effective_iter": self.effective_iter,
                "epoch": self.epoch,
                "n_batch_in_epoch": self.n_batch_in_epoch,
                "best_metric": self.best_metric,
                "in_evaluation": self.in_evaluation,
                "global_seed_sequence": self.global_seed_sequence,
            }
            torch.save(state, os.path.join(ckpt_dir, "trainer.ckpt"))
            open(os.path.join(ckpt_dir, self._get_backup_ckpt_name()), "w").close()
        
        if temp_ckpt_dir and os.path.exists(temp_ckpt_dir):
            shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
