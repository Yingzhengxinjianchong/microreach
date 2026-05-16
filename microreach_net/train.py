"""
microreach_net/train.py

MicroReach 训练主入口（阶段二中期：M0 / M1 单 seed）。

用法:
    python -m microreach_net.train --config configs/m1.yaml
    python -m microreach_net.train --config configs/m0.yaml
    python -m microreach_net.train --config configs/m1.yaml --debug   # 2 epoch 快测

设计要点:
    - 整个模型组装成 MicroReachNet（M0 / M1 共用同一个类，靠 cfg.model.use_pose_decoder 切换）
    - target_mode 同步切换 dataset 的输出 shape（per_point_mean / per_query）
    - 单 GPU 训练，无 DDP（阶段二中期数据小，没必要）
    - W&B 可关：cfg.wandb.mode = disabled
"""

from __future__ import annotations

import argparse
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from microreach_net.dataset import MicroReachDataset, split_dataset
from microreach_net.fusion import CrossScaleFusion
from microreach_net.heads import GeomHead
from microreach_net.losses import microreach_loss_geom_only
from microreach_net.pointnet_backbone import PointNetBackbone
from microreach_net.pose_decoder import PoseConditionedDecoder


# ──────────────────────────────────────────────
# 配置加载（支持 _base_ 字段递归继承）
# ──────────────────────────────────────────────

def load_config(path: str) -> Dict[str, Any]:
    """加载 yaml，处理 _base_ 字段继承。"""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if "_base_" in cfg:
        base_path = (path.parent / cfg["_base_"]).resolve()
        base_cfg = load_config(str(base_path))
        cfg = deep_merge(base_cfg, cfg)

    return cfg


def deep_merge(base: dict, override: dict) -> dict:
    """递归 merge：override 字段覆盖 base，dict 类型字段递归 merge。"""
    out = dict(base)
    for k, v in override.items():
        if k == "_base_":
            continue
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ──────────────────────────────────────────────
# 模型组装
# ──────────────────────────────────────────────

class MicroReachNet(nn.Module):
    """
    完整 MicroReach 网络（阶段二中期版）。

    forward 行为:
        use_pose_decoder=True (M1):
            point_cloud, candidate_p, queries -> logits (B, M, K)
        use_pose_decoder=False (M0):
            point_cloud, candidate_p (queries 忽略) -> logits (B, M)
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        m = cfg["model"]
        self.use_pose_decoder = m["use_pose_decoder"]
        self.feat_dim = m["feat_dim"]

        self.backbone = PointNetBackbone(
            feat_dim=self.feat_dim,
            radius=m["partfield"]["local_radius"],
            k=m["partfield"]["local_k"],
        )
        self.fusion = CrossScaleFusion(
            feat_dim=self.feat_dim,
            use_global_branch=m["use_global_branch"],
        )

        if self.use_pose_decoder:
            self.pose_decoder = PoseConditionedDecoder(
                feat_dim=self.feat_dim,
                pose_token_dim=m["pose_decoder"]["pose_token_dim"],
                n_heads=m["pose_decoder"]["n_heads"],
                n_layers=m["pose_decoder"]["n_layers"],
                n_g=cfg["data"]["n_g"],
            )
        else:
            self.pose_decoder = None

        self.geom_head = GeomHead(feat_dim=self.feat_dim, hidden_dim=64)

    def forward(
        self,
        point_cloud: torch.Tensor,    # (B, N, 3)
        candidate_p: torch.Tensor,    # (B, M, 3)
        queries: Optional[torch.Tensor] = None,  # (B, M, K, 4) 仅 M1 用
    ) -> torch.Tensor:
        local_feat = self.backbone(point_cloud, candidate_p)              # (B, M, D)
        fused = self.fusion(local_feat, point_cloud, candidate_p)         # (B, M, D)

        if self.use_pose_decoder:
            assert queries is not None
            cond = self.pose_decoder(fused, queries)                      # (B, M, K, D)
            logits = self.geom_head(cond)                                 # (B, M, K)
        else:
            logits = self.geom_head(fused)                                # (B, M)
        return logits


# ──────────────────────────────────────────────
# 训练 / 验证循环
# ──────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model, loader, optimizer, device, scheduler=None, grad_clip=1.0, log_every=10,
    use_wandb=False, epoch=0,
    loss_type="bce", focal_alpha=0.75, focal_gamma=2.0,
    composite_weights=None,
):
    model.train()
    total_loss = 0.0
    n_batches = 0
    t0 = time.time()

    for step, batch in enumerate(loader):
        pc = batch["point_cloud"].to(device)
        cp = batch["candidate_p"].to(device)
        q  = batch["queries"].to(device)
        tgt = batch["target"].to(device)
        mask = batch["mask"].to(device)

        # M0: queries 不用；M1: 用 queries
        if model.use_pose_decoder:
            logits = model(pc, cp, q)
        else:
            logits = model(pc, cp)

        out = microreach_loss_geom_only(
            logits, tgt, mask,
            loss_type=loss_type, focal_alpha=focal_alpha, focal_gamma=focal_gamma,
            composite_weights=composite_weights,
        )
        loss = out["loss"]

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if step % log_every == 0:
            print(f"  [epoch {epoch} step {step}/{len(loader)}] loss={loss.item():.4f}")
            if use_wandb:
                import wandb
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                    "step": step + epoch * len(loader),
                })

    if scheduler is not None:
        scheduler.step()

    return {
        "train_loss_avg": total_loss / max(n_batches, 1),
        "epoch_time": time.time() - t0,
    }


@torch.no_grad()
def validate(model, loader, device,
             loss_type="bce", focal_alpha=0.75, focal_gamma=2.0,
             composite_weights=None) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    # 也累积 (preds, targets, masks) 用于后续 Micro-mIoU/Recall（先只算 loss + 简单 acc）
    all_preds, all_targets, all_masks = [], [], []

    for batch in loader:
        pc = batch["point_cloud"].to(device)
        cp = batch["candidate_p"].to(device)
        q  = batch["queries"].to(device)
        tgt = batch["target"].to(device)
        mask = batch["mask"].to(device)

        if model.use_pose_decoder:
            logits = model(pc, cp, q)
        else:
            logits = model(pc, cp)

        out = microreach_loss_geom_only(
            logits, tgt, mask,
            loss_type=loss_type, focal_alpha=focal_alpha, focal_gamma=focal_gamma,
            composite_weights=composite_weights,
        )
        total_loss += out["loss"].item()
        n_batches += 1

        all_preds.append(torch.sigmoid(logits).cpu())
        all_targets.append(tgt.cpu())
        all_masks.append(mask.cpu())

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    masks = torch.cat(all_masks, dim=0)

    # 简单 IoU @ 0.5（与 eval/metrics.py 的 Micro-mIoU 一致的二值化）
    if preds.dim() == 3:
        # M1: (N, M, K)
        mask_e = masks.unsqueeze(-1).expand_as(preds)
    else:
        mask_e = masks
    pred_b = (preds > 0.5) & mask_e.bool()
    gt_b = (targets > 0.5) & mask_e.bool()
    inter = (pred_b & gt_b).sum().item()
    union = (pred_b | gt_b).sum().item()
    iou = inter / max(union, 1)

    # 简单 Recall@1（仅 M1，per_query）
    recall1 = float("nan")
    if preds.dim() == 3:
        top1 = preds.argmax(dim=-1, keepdim=True)                        # (N, M, 1)
        gt_top1 = targets.gather(-1, top1).squeeze(-1)                   # (N, M)
        valid = masks > 0.5
        recall1 = ((gt_top1 > 0.5).float()[valid]).mean().item() if valid.any() else float("nan")

    return {
        "val_loss": total_loss / max(n_batches, 1),
        "val_iou@0.5": iou,
        "val_recall@1": recall1,
    }


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="path to yaml config")
    parser.add_argument("--debug", action="store_true", help="2 epoch 快测")
    args = parser.parse_args()

    cfg = load_config(args.config)
    variant = cfg.get("variant", "unknown")
    print(f"=== Training MicroReach (variant={variant}) ===")
    print(f"Config: {args.config}")

    # 随机种子
    set_seed(cfg["train"]["seed"])

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # 数据
    repo_root = Path(__file__).resolve().parent.parent
    npz_dir = repo_root / cfg["data"]["npz_dir"]
    train_ids, val_ids, test_ids = split_dataset(
        str(npz_dir),
        ratios=tuple(cfg["split"]["ratios"]),
        seed=cfg["split"]["seed"],
    )
    print(f"Split: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

    target_mode = cfg.get("target_mode", "per_query")
    print(f"target_mode = {target_mode}")

    fields = cfg["data"]["fields"]
    num_points = cfg["data"]["num_points"]
    train_ds = MicroReachDataset(str(npz_dir), train_ids, target_mode, fields, num_points=num_points)
    val_ds   = MicroReachDataset(str(npz_dir), val_ids,   target_mode, fields, num_points=num_points)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"],
        shuffle=False, num_workers=cfg["train"]["num_workers"],
    )

    # 模型
    model = MicroReachNet(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")
    print(f"  use_pose_decoder = {model.use_pose_decoder}")

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    n_epochs = 2 if args.debug else cfg["train"]["epochs"]
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    # W&B
    use_wandb = (cfg["wandb"]["mode"] != "disabled") and (not args.debug)
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=cfg["wandb"]["project"],
                name=cfg["wandb"].get("run_name", variant),
                config=cfg,
                mode=cfg["wandb"]["mode"],
            )
        except Exception as e:
            print(f"  [W&B disabled] {e}")
            use_wandb = False

    # ckpt 目录
    ckpt_dir = repo_root / cfg["train"]["ckpt_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Loss 配置（"bce" | "focal" | "composite"）
    loss_cfg = cfg.get("loss", {})
    loss_type = loss_cfg.get("type", "bce")
    focal_alpha = loss_cfg.get("focal_alpha", 0.75)
    focal_gamma = loss_cfg.get("focal_gamma", 2.0)
    composite_weights = loss_cfg.get("composite_weights", None)
    if loss_type == "focal":
        print(f"Loss: type=focal, alpha={focal_alpha}, gamma={focal_gamma}")
    elif loss_type == "composite":
        w = composite_weights or {"bce": 0.3, "dice": 0.3, "focal": 0.2, "iou": 0.2}
        print(f"Loss: type=composite, weights={w}, focal(α={focal_alpha}, γ={focal_gamma})")
    else:
        print(f"Loss: type={loss_type}")

    # 训练循环
    best_iou = -1.0
    for epoch in range(n_epochs):
        train_stat = train_one_epoch(
            model, train_loader, optimizer, device, scheduler,
            grad_clip=cfg["train"]["grad_clip"],
            log_every=cfg["train"]["log_every"],
            use_wandb=use_wandb, epoch=epoch,
            loss_type=loss_type, focal_alpha=focal_alpha, focal_gamma=focal_gamma,
            composite_weights=composite_weights,
        )
        print(f"[epoch {epoch}] train_loss={train_stat['train_loss_avg']:.4f}  time={train_stat['epoch_time']:.1f}s")

        # 验证
        if epoch % cfg["train"]["val_every_epoch"] == 0 or epoch == n_epochs - 1:
            val_stat = validate(model, val_loader, device,
                                loss_type=loss_type, focal_alpha=focal_alpha, focal_gamma=focal_gamma,
                                composite_weights=composite_weights)
            print(f"  val_loss={val_stat['val_loss']:.4f}  iou@0.5={val_stat['val_iou@0.5']:.4f}  recall@1={val_stat['val_recall@1']:.4f}")
            if use_wandb:
                import wandb
                wandb.log({**val_stat, "epoch": epoch})
            if val_stat["val_iou@0.5"] > best_iou:
                best_iou = val_stat["val_iou@0.5"]
                torch.save(
                    {"model": model.state_dict(), "cfg": cfg, "epoch": epoch,
                     "val_stat": val_stat},
                    ckpt_dir / "best.pt",
                )
                print(f"  [ckpt] best.pt updated (iou={best_iou:.4f})")

        # 定期 ckpt
        if (epoch + 1) % cfg["train"]["ckpt_every_epoch"] == 0:
            torch.save(
                {"model": model.state_dict(), "cfg": cfg, "epoch": epoch},
                ckpt_dir / f"epoch_{epoch}.pt",
            )

    print(f"\n=== Done. best_iou={best_iou:.4f} ===")
    print(f"Checkpoints in: {ckpt_dir}")


if __name__ == "__main__":
    main()
