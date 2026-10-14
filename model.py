"""
model.py — 增强版（对齐论文 MSTCN 架构）
========================================
改进点：
1. 引入 MSTCNBlock：使用膨胀卷积（Dilation）增大感受野，捕捉长程频率演变。
2. 金字塔特征提取：通道数从底层到高层逐步压缩（32->16->8），过滤背景噪声。
3. 激活函数对齐：全线改用 ELU，对 EEG 信号的正负偏移更具鲁棒性。
4. 残差连接：在卷积块中加入 Shortcut，确保深层特征在跨被试迁移时不丢失。
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Callable

# ============================================================================
# 1. Cross-Center Alignment Module (CCAM)
# ============================================================================

class CrossCenterAlignmentModule(nn.Module):
    def __init__(self, in_channels: int = 18, feature_dim: int = 64) -> None:
        super().__init__()
        self.proj = nn.Linear(in_channels, feature_dim, bias=True)
        self.norm = nn.LayerNorm(feature_dim)
        self.act = nn.ELU() # 改为 ELU

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, T) -> (B, T, C)
        x = x.permute(0, 2, 1)
        x = self.act(self.norm(self.proj(x)))
        return x

# ============================================================================
# 2. MSTCN 核心组件：多尺度膨胀卷积块
# ============================================================================

class MSTCNBlock(nn.Module):
    """
    论文核心：膨胀因果卷积块
    通过不同的膨胀率（Dilation）在不增加参数的情况下增大感受野
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        # 对称填充以保持时间序列长度不变
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ELU()

        # 残差连接
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)) + self.res(x))

class SpectralFeatureExtractor(nn.Module):
    """
    改进版：多尺度时间卷积网络 (MSTCN)
    采用金字塔结构：3层叠加，每层膨胀率递增 (1, 2, 4)
    """
    def __init__(self, in_channels: int = 18, base_ch: int = 32):
        super().__init__()

        # Level 1: 基础特征提取 (Dilation=1)
        self.level1 = nn.ModuleList([
            MSTCNBlock(in_channels, base_ch, kernel_size=3, dilation=1),
            MSTCNBlock(in_channels, base_ch, kernel_size=5, dilation=1)
        ])

        # Level 2: 中程特征捕捉 (Dilation=2), 通道数压缩
        self.level2 = nn.ModuleList([
            MSTCNBlock(base_ch * 2, base_ch // 2, kernel_size=3, dilation=2),
            MSTCNBlock(base_ch * 2, base_ch // 2, kernel_size=5, dilation=2)
        ])

        # Level 3: 长程演变捕捉 (Dilation=4), 进一步压缩
        self.level3 = nn.ModuleList([
            MSTCNBlock(base_ch, base_ch // 4, kernel_size=3, dilation=4),
            MSTCNBlock(base_ch, base_ch // 4, kernel_size=5, dilation=4)
        ])

        # 最终输出通道数: (32 // 4) * 2 = 16
        self.out_channels = (base_ch // 4) * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Level 1: (B, 18, T) -> (B, 64, T)
        out = torch.cat([conv(x) for conv in self.level1], dim=1)
        # Level 2: (B, 64, T) -> (B, 32, T)
        out = torch.cat([conv(out) for conv in self.level2], dim=1)
        # Level 3: (B, 32, T) -> (B, 16, T)
        out = torch.cat([conv(out) for conv in self.level3], dim=1)
        return out

# ============================================================================
# 3. Dynamic Graph Builder & GAT Layers
# ============================================================================

class DynamicGraphBuilder(nn.Module):
    def __init__(self, num_nodes: int = 18, embed_dim: int = 32) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.node_emb = nn.Embedding(num_nodes, embed_dim)
        self.lam = nn.Parameter(torch.tensor(0.5))

        prior = self._build_prior(num_nodes)
        self.register_buffer("prior", prior)

    @staticmethod
    def _build_prior(n: int) -> torch.Tensor:
        A = torch.zeros(n, n)
        for i in range(n):
            A[i, (i - 1) % n] = 1.0
            A[i, (i + 1) % n] = 1.0
        deg = A.sum(dim=1, keepdim=True).clamp(min=1.0)
        return A / deg.sqrt() / deg.sqrt().t()

    def forward(self, batch_size: int) -> torch.Tensor:
        idx = torch.arange(self.num_nodes, device=self.node_emb.weight.device)
        emb = self.node_emb(idx)
        sim = torch.mm(emb, emb.t()) / math.sqrt(emb.size(1))
        A_adapt = torch.softmax(sim, dim=-1)

        lam = torch.sigmoid(self.lam)
        A = lam * self.prior + (1 - lam) * A_adapt
        return A.unsqueeze(0).expand(batch_size, -1, -1)

class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a = nn.Linear(2 * out_dim, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ELU()

    def forward(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        B, N, _ = h.shape
        Wh = self.W(h)
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1)
        Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1)
        e = self.act(self.a(torch.cat([Wh_i, Wh_j], dim=-1)).squeeze(-1))

        e = e.masked_fill(A == 0, -1e9)
        alpha = self.drop(F.softmax(e, dim=-1))
        return torch.bmm(alpha, Wh)

class MultiHeadGAT(nn.Module):
    def __init__(self, in_dim: int, head_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.heads = nn.ModuleList([GATLayer(in_dim, head_dim, dropout) for _ in range(num_heads)])
        self.out_dim = head_dim * num_heads
        self.norm = nn.LayerNorm(self.out_dim)

    def forward(self, h: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        out = torch.cat([head(h, A) for head in self.heads], dim=-1)
        return self.norm(out)

# ============================================================================
# 4. FDGAT – 主模型（集成 MSTCN）
# ============================================================================

class FDGAT(nn.Module):
    def __init__(
        self,
        in_time: int = 2048,
        num_nodes: int = 18,
        hidden_dim: int = 64,
        band_channels: int = 32, # 增大基础通道
        gat_heads: int = 4,
        gat_head_dim: int = 32,
        clf_hidden: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes

        # 1. 跨中心对齐
        self.ccam = CrossCenterAlignmentModule(num_nodes, hidden_dim)

        # 2. MSTCN 频谱特征提取
        self.spectral = SpectralFeatureExtractor(num_nodes, band_channels)
        spec_out = self.spectral.out_channels

        # 特征投影
        self.spec_proj = nn.Sequential(
            nn.Linear(spec_out, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
        )

        # 3. 动态图
        self.graph_builder = DynamicGraphBuilder(num_nodes, embed_dim=32)

        # 4. GAT 层
        self.gat1 = MultiHeadGAT(hidden_dim, gat_head_dim, gat_heads, dropout)
        self.gat2 = MultiHeadGAT(gat_head_dim * gat_heads, gat_head_dim, gat_heads, dropout)

        # 5. 分类器
        gat_out = gat_head_dim * gat_heads
        self.classifier = nn.Sequential(
            nn.Linear(gat_out, clf_hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(clf_hidden, 1),
        )

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape

        # 频谱特征提取 (B, 16, T)
        spec = self.spectral(x)
        # 时间池化：将 T 压缩到 N 个节点 bins
        spec_pooled = F.adaptive_avg_pool1d(spec, self.num_nodes).permute(0, 2, 1) # (B, N, 16)
        node_feats = self.spec_proj(spec_pooled) # (B, N, 64)

        # 融合 CCAM 全局上下文
        ccam_mean = self.ccam(x).mean(dim=1, keepdim=True) # (B, 1, 64)
        node_feats = self.dropout(node_feats + ccam_mean)

        # 图计算
        A = self.graph_builder(B)
        h = self.gat1(node_feats, A)
        h = self.gat2(h, A)

        # 池化与分类
        logits = self.classifier(h.mean(dim=1)).squeeze(-1)
        return torch.sigmoid(logits)

    @staticmethod
    def loss(pred: torch.Tensor, target: torch.Tensor, pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        pred = pred.clamp(1e-7, 1 - 1e-7)
        bce = -(target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
        if pos_weight is not None:
            # 这里的 pos_weight 会极大影响灵敏度
            weight = torch.where(target == 1, pos_weight.squeeze(), torch.ones_like(target))
            bce = bce * weight
        return bce.mean()

# ============================================================================
# 5. Privacy Module (保持兼容)
# ============================================================================

class LayerImportanceDP(nn.Module):
    _TRACKED_TYPES: Tuple[type, ...] = (nn.Linear, nn.Conv1d)

    def __init__(self, model: FDGAT, total_eps: float = 1.0, clip_norm: float = 1.0, warmup_batches: int = 5) -> None:
        super().__init__()
        self.model = model
        self.total_eps = total_eps
        self.clip_norm = clip_norm
        self.warmup_batches = warmup_batches
        self.tracked_names = [n for n, m in model.named_modules() if isinstance(m, self._TRACKED_TYPES)]
        self.lic = {n: 1.0 for n in self.tracked_names}
        self.layer_eps = {n: total_eps for n in self.tracked_names}
        self._hooks = []
        self._pre_activations = {}

    def _make_pre_activation_hook(self, name: str):
        def hook(module, input, output):
            if isinstance(input, tuple) and len(input) > 0:
                self._pre_activations[name].append(input[0].detach().cpu().reshape(-1))
        return hook

    def compute_lic(self, data_loader, device="cuda"):
        self.model.eval()
        self._pre_activations = {n: [] for n in self.tracked_names}
        for name, mod in self.model.named_modules():
            if name in self.tracked_names:
                self._hooks.append(mod.register_forward_hook(self._make_pre_activation_hook(name)))

        with torch.no_grad():
            for i, (X, _) in enumerate(data_loader):
                if i >= self.warmup_batches: break
                self.model(X.to(device))

        for h in self._hooks: h.remove()
        for n in self.tracked_names:
            if self._pre_activations[n]:
                self.lic[n] = float(torch.cat(self._pre_activations[n]).abs().mean().item())

        self._allocate_budgets()
        self.model.train()
        return self.lic

    def _allocate_budgets(self):
        lic_vals = np.array([self.lic[n] for n in self.tracked_names])
        norm_imp = np.exp(lic_vals - np.max(lic_vals))
        norm_imp /= (norm_imp.sum() + 1e-12)
        for n, ni in zip(self.tracked_names, norm_imp):
            self.layer_eps[n] = float(np.clip(self.total_eps * (1.0 - ni), 0.01, self.total_eps))

    @staticmethod
    def pm_isub(tensor, eps, clip_val):
        C = clip_val
        if eps < 1e-6: return np.random.uniform(-C, C, size=tensor.shape).astype(tensor.dtype)
        exp_half = math.exp(eps / 2.0)
        t = (exp_half + 1.0) / (exp_half - 1.0)
        v = np.clip(tensor, -C, C)
        l_v = ((t + 1.0) / 2.0) * (v / C) - (t - 1.0) / 2.0
        r_v = l_v + (t - 1.0)
        p_inner = (exp_half - 1.0) / (2.0 * exp_half - 2.0) * (t - 1.0)
        u = np.random.random(v.shape).astype(np.float32)
        inner_mask = u < p_inner
        res = np.where(inner_mask,
                       np.random.uniform(np.maximum(l_v, -t), np.minimum(r_v, t), size=v.shape),
                       np.random.uniform(-t, t, size=v.shape)) # 简化版采样逻辑
        return (res * (C / t)).astype(tensor.dtype)

    def perturb_gradients(self):
        nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
        for name, mod in self.model.named_modules():
            if name in self.tracked_names:
                eps = self.layer_eps.get(name, self.total_eps)
                for p in mod.parameters():
                    if p.grad is not None:
                        p.grad.copy_(torch.from_numpy(self.pm_isub(p.grad.cpu().numpy(), eps, self.clip_norm)).to(p.device))

    def summary(self) -> str:
        res = ["LIDP Summary:", f"{'Layer':<40} {'LIC':>10} {'Eps':>10}"]
        for n in self.tracked_names:
            res.append(f"{n:<40} {self.lic[n]:>10.4f} {self.layer_eps[n]:>10.4f}")
        return "\n".join(res)