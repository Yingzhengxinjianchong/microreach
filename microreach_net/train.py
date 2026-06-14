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
from microreach_net.heads import ContactHead, ExecHead, GeomHead
from microreach_net.losses import (
    masked_bce_with_logits,
    microreach_loss_geom_only,
    microreach_loss_multihead,
)
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
    完整 MicroReach 网络。

    forward 行为：
        - 单头模式（M0/M1/M1_focal/M1_composite）：返回 tensor (B, M, K) 或 (B, M)
          —— 与阶段二完全向后兼容
        - 多头模式（M2 / M_full，阶段三）：返回 dict {"geom": ..., "contact": ..., "exec": ...}
          每个 value 是 logits (B, M, K)

    head 开关由 cfg["model"]["heads"] 决定：
        heads.geom    = true   始终开
        heads.contact = true   M2 / M_full 开
        heads.exec    = true   仅 M_full 开
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        m = cfg["model"]
        self.use_pose_decoder = m["use_pose_decoder"]
        self.feat_dim = m["feat_dim"]

        # 多头配置（向后兼容：旧 yaml 没有 heads.contact / heads.exec 时默认 False）
        heads_cfg = m.get("heads", {})
        self.use_contact_head = heads_cfg.get("contact", False)
        self.use_exec_head    = heads_cfg.get("exec",    False)
        self.multi_head = self.use_contact_head or self.use_exec_head

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
        # 阶段三：lazily instantiate 多头（不会增加 M0/M1 现有 ckpt 参数）
        self.contact_head = ContactHead(feat_dim=self.feat_dim, hidden_dim=64) \
                            if self.use_contact_head else None
        self.exec_head    = ExecHead(feat_dim=self.feat_dim, hidden_dim=64) \
                            if self.use_exec_head else None

    def forward(
        self,
        point_cloud: torch.Tensor,    # (B, N, 3)
        candidate_p: torch.Tensor,    # (B, M, 3)
        queries: Optional[torch.Tensor] = None,  # (B, M, K, 4) 仅 M1+ 用
    ):
        """
        Returns:
            单头模式 (M0/M1) -> Tensor (logits)
            多头模式 (M2/M_full) -> Dict[str, Tensor] (logits per head)
        """
        local_feat = self.backbone(point_cloud, candidate_p)              # (B, M, D)
        fused = self.fusion(local_feat, point_cloud, candidate_p)         # (B, M, D)

        if self.use_pose_decoder:
            assert queries is not None
            cond = self.pose_decoder(fused, queries)                      # (B, M, K, D)
            geom_logits = self.geom_head(cond)                            # (B, M, K)
        else:
            cond = fused                                                  # M0 走旁路
            geom_logits = self.geom_head(fused)                           # (B, M)

        if not self.multi_head:
            return geom_logits                                            # 向后兼容

        out = {"geom": geom_logits}
        if self.use_contact_head:
            out["contact"] = self.contact_head(cond)                      # (B, M, K)
        if self.use_exec_head:
            out["exec"]    = self.exec_head(cond)                         # (B, M, K)
        return out


# ──────────────────────────────────────────────
# 训练 / 验证循环
# ──────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_cascade_mu(epoch: int, mu_final: float, warmup_epochs: int) -> float:
    """阶段三：cascade mu warmup（Bengio 2009 Curriculum Learning）
        - epoch < warmup        : 0.0
        - warmup <= epoch < 2×warmup : linear ramp 0 → mu_final
        - epoch >= 2×warmup     : mu_final
    """
    if epoch < warmup_epochs:
        return 0.0
    ramp_end = warmup_epochs * 2
    if epoch >= ramp_end:
        return mu_final
    return mu_final * (epoch - warmup_epochs) / max(warmup_epochs, 1)


def train_one_epoch(
    model, loader, optimizer, device, scheduler=None, grad_clip=1.0, log_every=10,
    use_wandb=False, epoch=0,
    loss_type="bce", focal_alpha=0.75, focal_gamma=2.0,
    composite_weights=None,
    # 阶段三：多 head 训练参数
    multi_head_cfg: Optional[Dict[str, Any]] = None,
):
    """
    multi_head_cfg = None → 阶段二行为（单 head L_geom）
    multi_head_cfg = {
        "head_weights": {"geom": 0.5, "contact": 0.3, "exec": 0.3},
        "cascade_consistency": True,
        "cascade_mu":          0.1,
        "cascade_mu_warmup_epochs": 50,
    }
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    t0 = time.time()

    is_multi = multi_head_cfg is not None
    if is_multi:
        cur_mu = _resolve_cascade_mu(
            epoch,
            mu_final=multi_head_cfg.get("cascade_mu", 0.0),
            warmup_epochs=multi_head_cfg.get("cascade_mu_warmup_epochs", 50),
        )

    for step, batch in enumerate(loader):
        pc = batch["point_cloud"].to(device)
        cp = batch["candidate_p"].to(device)
        q  = batch["queries"].to(device)
        tgt = batch["target"].to(device)
        mask = batch["mask"].to(device)

        # M0: queries 不用；M1: 用 queries
        if model.use_pose_decoder:
            output = model(pc, cp, q)
        else:
            output = model(pc, cp)

        if is_multi:
            # output 是 dict {head: logits}
            assert isinstance(output, dict), "multi_head 模式期望 model 返回 dict"
            target_dict = {"geom": tgt}
            valid_dict: Dict[str, torch.Tensor] = {}
            if "contact" in output:
                target_dict["contact"] = batch["target_contact"].to(device)
                valid_dict["contact"]  = batch["valid_contact"].to(device)
            if "exec" in output:
                target_dict["exec"]    = batch["target_exec"].to(device)
                valid_dict["exec"]     = batch["valid_exec"].to(device)
            out = microreach_loss_multihead(
                output, target_dict, mask, valid=valid_dict,
                head_weights=multi_head_cfg.get("head_weights"),
                cascade_consistency=multi_head_cfg.get("cascade_consistency", False),
                cascade_mu=cur_mu,
            )
        else:
            # 单 head：跟阶段二 100% 兼容
            assert isinstance(output, torch.Tensor)
            out = microreach_loss_geom_only(
                output, tgt, mask,
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
            extras = ""
            if is_multi:
                extras = " | " + " ".join(
                    f"{k}={v.item():.3f}" for k, v in out.items() if k.startswith("l_")
                )
                extras += f" | mu={cur_mu:.3f}"
            print(f"  [epoch {epoch} step {step}/{len(loader)}] loss={loss.item():.4f}{extras}")
            if use_wandb:
                import wandb
                log_dict = {
                    "train/loss": loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                    "step": step + epoch * len(loader),
                }
                if is_multi:
                    log_dict["train/cascade_mu"] = cur_mu
                    for k, v in out.items():
                        if k.startswith("l_"):
                            log_dict[f"train/{k}"] = v.item()
                wandb.log(log_dict)

    if scheduler is not None:
        scheduler.step()

    return {
        "train_loss_avg": total_loss / max(n_batches, 1),
        "epoch_time": time.time() - t0,
    }


@torch.no_grad()
def validate(model, loader, device,
             loss_type="bce", focal_alpha=0.75, focal_gamma=2.0,
             composite_weights=None,
             multi_head_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    # 也累积 (preds, targets, masks) 用于后续 Micro-mIoU/Recall（先只算 loss + 简单 acc）
    all_preds, all_targets, all_masks = [], [], []

    is_multi = multi_head_cfg is not None

    for batch in loader:
        pc = batch["point_cloud"].to(device)
        cp = batch["candidate_p"].to(device)
        q  = batch["queries"].to(device)
        tgt = batch["target"].to(device)
        mask = batch["mask"].to(device)

        if model.use_pose_decoder:
            output = model(pc, cp, q)
        else:
            output = model(pc, cp)

        if is_multi:
            assert isinstance(output, dict)
            target_dict = {"geom": tgt}
            valid_dict: Dict[str, torch.Tensor] = {}
            if "contact" in output:
                target_dict["contact"] = batch["target_contact"].to(device)
                valid_dict["contact"]  = batch["valid_contact"].to(device)
            if "exec" in output:
                target_dict["exec"]    = batch["target_exec"].to(device)
                valid_dict["exec"]     = batch["valid_exec"].to(device)
            out = microreach_loss_multihead(
                output, target_dict, mask, valid=valid_dict,
                head_weights=multi_head_cfg.get("head_weights"),
                cascade_consistency=multi_head_cfg.get("cascade_consistency", False),
                cascade_mu=multi_head_cfg.get("cascade_mu", 0.0),  # val 不 warmup，用 final mu
            )
            geom_logits = output["geom"]
        else:
            assert isinstance(output, torch.Tensor)
            out = microreach_loss_geom_only(
                output, tgt, mask,
                loss_type=loss_type, focal_alpha=focal_alpha, focal_gamma=focal_gamma,
                composite_weights=composite_weights,
            )
            geom_logits = output

        total_loss += out["loss"].item()
        n_batches += 1

        # 验证集 IoU/Recall 用 geom head 的 logits 来算（多头 / 单头都一样）
        all_preds.append(torch.sigmoid(geom_logits).cpu())
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
    parser.add_argument("--seed", type=int, default=None,
                        help="覆盖 yaml 里的 train.seed；同时把 ckpt_dir 末尾的 seed42 替换成 seed<N>，"
                             "wandb run_name 也加 _seed<N> 后缀（阶段三多 seed 用）")
    parser.add_argument("--cascade-synthetic", action="store_true",
                        help="（仅 smoke test）当 R_contact / R_exec 全 NaN 时用合成标签 "
                             "R_contact = R_geom × 0.8, R_exec = R_geom × 0.56。"
                             "切勿在真训练里启用。")
    args = parser.parse_args()

    cfg = load_config(args.config)
    variant = cfg.get("variant", "unknown")
    print(f"=== Training MicroReach (variant={variant}) ===")
    print(f"Config: {args.config}")

    # 阶段三：CLI --seed N 覆盖 yaml 里的 train.seed，并自动改 ckpt_dir + wandb name
    if args.seed is not None:
        old_seed = cfg["train"]["seed"]
        cfg["train"]["seed"] = args.seed
        # ckpt_dir: ckpts/m1_seed42 → ckpts/m1_seed43
        # 修复：只有当原 dir 真没 seedN 模式时才追加；
        # 否则即使 new==old（如 yaml seed42 + --seed 42）也保持单后缀，不要叠成 seed42_seed42
        import re
        old_dir = cfg["train"]["ckpt_dir"]
        if re.search(r"seed\d+", old_dir):
            new_dir = re.sub(r"seed\d+", f"seed{args.seed}", old_dir)
        else:
            new_dir = f"{old_dir}_seed{args.seed}"
        cfg["train"]["ckpt_dir"] = new_dir
        # wandb run_name 同步（同样逻辑）
        if "wandb" in cfg and "run_name" in cfg["wandb"]:
            old_name = cfg["wandb"]["run_name"]
            if re.search(r"seed\d+", old_name):
                new_name = re.sub(r"seed\d+", f"seed{args.seed}", old_name)
            else:
                new_name = f"{old_name}_seed{args.seed}"
            cfg["wandb"]["run_name"] = new_name
        print(f"[CLI override] seed: {old_seed} -> {args.seed}")
        print(f"[CLI override] ckpt_dir: {old_dir} -> {new_dir}")

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

    # 阶段三：多 head 训练需要加载 R_contact / R_exec
    data_cfg = cfg["data"]
    load_contact = bool(data_cfg.get("load_contact", False))
    load_exec    = bool(data_cfg.get("load_exec",    False))
    if (load_contact or load_exec) and args.cascade_synthetic:
        print("[WARNING] --cascade-synthetic 已启用：R_contact/R_exec 缺失时用合成标签。"
              "切勿用于生成真实结果！")

    train_ds = MicroReachDataset(
        str(npz_dir), train_ids, target_mode, fields, num_points=num_points,
        load_contact=load_contact, load_exec=load_exec,
        cascade_synthetic=args.cascade_synthetic,
    )
    val_ds   = MicroReachDataset(
        str(npz_dir), val_ids,   target_mode, fields, num_points=num_points,
        load_contact=load_contact, load_exec=load_exec,
        cascade_synthetic=args.cascade_synthetic,
    )

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

    # 阶段三：装配 multi_head 配置（仅当 ContactHead / ExecHead 启用时）
    multi_head_cfg = None
    if model.multi_head:
        multi_head_cfg = {
            "head_weights":         loss_cfg.get("head_weights",
                                                 {"geom": 0.5, "contact": 0.3, "exec": 0.3}),
            "cascade_consistency":  loss_cfg.get("cascade_consistency", False),
            "cascade_mu":           loss_cfg.get("cascade_mu", 0.0),
            "cascade_mu_warmup_epochs": loss_cfg.get("cascade_mu_warmup_epochs", 50),
        }
        print(f"Multi-head: contact={model.use_contact_head}, exec={model.use_exec_head}")
        print(f"  head_weights={multi_head_cfg['head_weights']}")
        print(f"  cascade_consistency={multi_head_cfg['cascade_consistency']}, "
              f"mu_final={multi_head_cfg['cascade_mu']}, "
              f"warmup={multi_head_cfg['cascade_mu_warmup_epochs']}")

    # 训练循环
    # 阶段三修复：原先按 val_iou@0.5 选 best.pt 在 M0 模式（per_point_mean）下永远 = 0
    # （sigmoid 输出标量几乎不可能 > 0.5 阈值），导致 M0 的 best.pt 卡在 epoch=0。
    # 改为按 val_loss 越低越好（与 M0/M1 都兼容）。
    best_val_loss = float("inf")
    for epoch in range(n_epochs):
        train_stat = train_one_epoch(
            model, train_loader, optimizer, device, scheduler,
            grad_clip=cfg["train"]["grad_clip"],
            log_every=cfg["train"]["log_every"],
            use_wandb=use_wandb, epoch=epoch,
            loss_type=loss_type, focal_alpha=focal_alpha, focal_gamma=focal_gamma,
            composite_weights=composite_weights,
            multi_head_cfg=multi_head_cfg,
        )
        print(f"[epoch {epoch}] train_loss={train_stat['train_loss_avg']:.4f}  time={train_stat['epoch_time']:.1f}s")

        # 验证
        if epoch % cfg["train"]["val_every_epoch"] == 0 or epoch == n_epochs - 1:
            val_stat = validate(model, val_loader, device,
                                loss_type=loss_type, focal_alpha=focal_alpha, focal_gamma=focal_gamma,
                                composite_weights=composite_weights,
                                multi_head_cfg=multi_head_cfg)
            print(f"  val_loss={val_stat['val_loss']:.4f}  iou@0.5={val_stat['val_iou@0.5']:.4f}  recall@1={val_stat['val_recall@1']:.4f}")
            if use_wandb:
                import wandb
                wandb.log({**val_stat, "epoch": epoch})
            cur_val_loss = val_stat["val_loss"]
            # 阶段三 P3 严谨修复：禁止 epoch=0 被选为 best.pt。
            # 原因：M0 (per_point_mean) 模式下，随机初始化网络的 sigmoid 输出 ≈ 0.5，
            # 偶然让 val_loss 比训练几个 epoch 之后还低（M0 学的 R_geom 均值 ≈ 0.155
            # 反而推 sigmoid 远离 0.5），导致 best.pt 永远停在未训练的 epoch=0。
            # 之前 6.13 M0 训练已经踩过这个坑（与 5 月 22 日 bugfix 是同一类问题）。
            if epoch > 0 and cur_val_loss < best_val_loss:
                best_val_loss = cur_val_loss
                torch.save(
                    {"model": model.state_dict(), "cfg": cfg, "epoch": epoch,
                     "val_stat": val_stat},
                    ckpt_dir / "best.pt",
                )
                print(f"  [ckpt] best.pt updated (val_loss={best_val_loss:.4f}, "
                      f"val_iou={val_stat['val_iou@0.5']:.4f}, "
                      f"val_recall@1={val_stat['val_recall@1']})")
            elif epoch == 0:
                # 仍保存一个 epoch=0 ckpt 作为初始 baseline，但不计入 best
                print(f"  [ckpt] epoch=0 不计入 best.pt（避免随机初始化干扰）")

        # 定期 ckpt
        if (epoch + 1) % cfg["train"]["ckpt_every_epoch"] == 0:
            torch.save(
                {"model": model.state_dict(), "cfg": cfg, "epoch": epoch},
                ckpt_dir / f"epoch_{epoch}.pt",
            )

    print(f"\n=== Done. best_val_loss={best_val_loss:.4f} ===")
    print(f"Checkpoints in: {ckpt_dir}")


if __name__ == "__main__":
    main()
