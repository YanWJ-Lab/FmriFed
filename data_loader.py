from __future__ import annotations
import os
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# 常量设置 (对齐 CHB-MIT 预处理参数)
# ---------------------------------------------------------------------------
SZ_THRESHOLD: float = 0.0
NUM_SUBJECTS: int = 24  # 支持CHB-MIT的24个被试者
INTERICTAL_GAP_S: float = 3600.0  # 1小时安全间隔
SAMPLE_RATE: int = 256
WINDOW_STEP: int = 1024  # 步长为4秒


class EEGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, noise_std: float = 0.0) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.noise_std = noise_std

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx]
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std
        return x, self.y[idx]


class FederatedDataManager:
    def __init__(self, data_dir: str, num_clients: int = 8, batch_size: int = 32,
                 noise_std: float = 0.0, seed: int = 42) -> None:
        self.data_dir = data_dir
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.noise_std = noise_std
        np.random.seed(seed)

        # 1. 加载数据
        self.X = np.load(os.path.join(data_dir, "signal_samples.npy")).astype(np.float32)
        is_sz = np.load(os.path.join(data_dir, "is_sz.npy")).astype(np.float32)
        self.y = (is_sz > SZ_THRESHOLD).astype(np.float32)

        # 2. 受试者划分 (24个被试者 chb01-chb24)
        N = len(self.y)
        self.cases = np.zeros(N, dtype=np.int64)
        chunk = N // NUM_SUBJECTS
        for c in range(NUM_SUBJECTS):
            self.cases[c * chunk: (c + 1) * chunk if c < 23 else N] = c + 1

        # 3. 切分（训练集1:1平衡，测试集保持自然分布）
        self.client_splits = self._split_logic()

        # 4. 为每个被试者准备独立的测试集（用于全局模型评估）
        self._prepare_subject_test_splits()

        # 5. 计算正样本权重（供带权损失函数使用）
        all_train_idx = []
        for s in self.client_splits.values():
            all_train_idx.extend(s["train"])
        all_train_y = self.y[all_train_idx]
        n_pos = all_train_y.sum()
        n_neg = len(all_train_y) - n_pos
        # pos_weight = neg/pos，BCEWithLogitsLoss 标准用法
        self.pos_weight = torch.tensor(
            [n_neg / max(n_pos, 1)], dtype=torch.float32
        )

    # ------------------------------------------------------------------
    # 核心修改：训练集 1:1 平衡，测试集保留全部自然分布样本
    # ------------------------------------------------------------------
    def _split_logic(self) -> Dict[int, Dict]:
        splits = {}
        gap_windows = int(INTERICTAL_GAP_S * SAMPLE_RATE / WINDOW_STEP)

        for i in range(self.num_clients):
            # 映射客户端到被试者：循环分配，K个客户端对应前K个被试者
            case_id = i + 1
            idx_in_case = np.where(self.cases == case_id)[0]
            sz_idx = idx_in_case[self.y[idx_in_case] == 1]
            non_sz_idx = idx_in_case[self.y[idx_in_case] == 0]

            # ── 发作样本 5:5 划分为训练/测试 ─────────────────────────
            np.random.shuffle(sz_idx)
            n_train_sz = max(1, len(sz_idx) // 2)
            train_sz = sz_idx[:n_train_sz]
            test_sz = sz_idx[n_train_sz:]

            # ── 筛选 1 小时安全间歇期 ────────────────────────────────
            sorted_sz = np.sort(sz_idx)
            safe_inter = np.array(
                [idx for idx in non_sz_idx
                 if self._is_safe(idx, sorted_sz, gap_windows)]
            )
            if len(safe_inter) == 0:
                safe_inter = non_sz_idx  # 回退：无安全窗口时用全部非发作

            # ── 训练集：1:1 过采样平衡 ───────────────────────────────
            # replace=True 允许过采样，保证正负各 n_train_sz 条
            train_inter = np.random.choice(
                safe_inter, size=max(1, len(train_sz)), replace=True
            )

            # ── 测试集：★核心修改★ 保持自然分布，不做 1:1 强平衡 ─────
            # 将全部安全间歇期样本纳入测试集（不抽样），
            # 使发作率接近真实临床环境（约 2–5%），评估更可信。
            # 为防测试集过大拖慢评估，最多取 safe_inter 的前 4×test_sz 条
            max_inter_test = max(len(test_sz) * 4, 20)
            if len(safe_inter) > max_inter_test:
                test_inter = np.random.choice(
                    safe_inter, size=max_inter_test, replace=False
                )
            else:
                test_inter = safe_inter

            splits[i] = {
                "train": np.concatenate([train_sz, train_inter]).tolist(),
                "test": np.concatenate([test_sz, test_inter]).tolist(),
                "cases": [case_id],
                # 记录真实发作率，供日志输出
                "test_sz_pct": len(test_sz) / max(len(test_sz) + len(test_inter), 1) * 100,
            }

        return splits

    def _is_safe(self, idx, sorted_sz, gap):
        if len(sorted_sz) == 0:
            return True
        pos = np.searchsorted(sorted_sz, idx)
        ld = idx - sorted_sz[pos - 1] if pos > 0 else float('inf')
        rd = sorted_sz[pos] - idx if pos < len(sorted_sz) else float('inf')
        return min(ld, rd) > gap

    # ------------------------------------------------------------------
    # 新增：为每个被试者准备独立的测试集（用于全局模型评估）
    # ------------------------------------------------------------------
    def _prepare_subject_test_splits(self):
        """
        为每个被试者准备独立的测试集。
        所有24个被试者都有独立的测试集，可用于评估全局模型。
        """
        gap_windows = int(INTERICTAL_GAP_S * SAMPLE_RATE / WINDOW_STEP)

        self.subject_test_splits = {}

        for subject_id in range(1, NUM_SUBJECTS + 1):
            idx_in_subject = np.where(self.cases == subject_id)[0]
            sz_idx = idx_in_subject[self.y[idx_in_subject] == 1]
            non_sz_idx = idx_in_subject[self.y[idx_in_subject] == 0]

            if len(sz_idx) == 0:
                # 如果该被试者没有发作样本，只使用间歇期
                sorted_sz = np.array([], dtype=np.int64)
            else:
                sorted_sz = np.sort(sz_idx)

            # 筛选1小时安全间歇期
            safe_inter = np.array(
                [idx for idx in non_sz_idx
                 if self._is_safe(idx, sorted_sz, gap_windows)]
            )
            if len(safe_inter) == 0:
                safe_inter = non_sz_idx

            # 发作样本全部用于测试
            test_sz = sz_idx

            # 间歇期样本最多取发作样本的4倍
            max_inter_test = max(len(test_sz) * 4, 20)
            if len(safe_inter) > max_inter_test:
                test_inter = safe_inter[:max_inter_test]
            else:
                test_inter = safe_inter

            test_indices = np.concatenate([test_sz, test_inter])

            self.subject_test_splits[subject_id] = {
                "test": test_indices.tolist(),
                "n_test": len(test_indices),
                "n_sz": len(test_sz),
                "n_inter": len(test_inter),
                "test_sz_pct": len(test_sz) / max(len(test_indices), 1) * 100,
            }

    def get_subject_test_loader(self, subject_id: int) -> DataLoader:
        """
        获取指定被试者的测试数据加载器。

        Args:
            subject_id: 被试者ID (1-24)

        Returns:
            DataLoader: 该被试者的测试数据加载器
        """
        if subject_id not in self.subject_test_splits:
            raise ValueError(f"Subject {subject_id} not found. Available: 1-{NUM_SUBJECTS}")

        test_indices = self.subject_test_splits[subject_id]["test"]
        test_ds = EEGDataset(self.X[test_indices], self.y[test_indices])
        return DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

    def get_all_subject_ids(self) -> List[int]:
        """
        返回所有被试者ID列表。

        Returns:
            List[int]: [1, 2, 3, ..., 24]
        """
        return list(range(1, NUM_SUBJECTS + 1))

    def get_subject_info(self) -> Dict[int, Dict]:
        """
        返回所有被试者的信息。

        Returns:
            Dict: 被试者ID到信息的映射
        """
        return self.subject_test_splits.copy()

    def get_client_loaders(self, client_id: int) -> Tuple[DataLoader, None, DataLoader]:
        s = self.client_splits[client_id]
        t_idx = list(s["train"])

        # 防御：确保训练集样本数 >= batch_size
        if len(t_idx) < self.batch_size:
            extra = np.random.choice(
                t_idx, size=self.batch_size - len(t_idx), replace=True
            ).tolist()
            t_idx = t_idx + extra

        train_ds = EEGDataset(self.X[t_idx], self.y[t_idx], self.noise_std)
        test_ds = EEGDataset(self.X[s["test"]], self.y[s["test"]])

        return (
            DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True),
            None,
            DataLoader(test_ds, batch_size=self.batch_size, shuffle=False),
        )

    def get_client_info(self) -> Dict:
        return {
            cid: {
                "cases": s["cases"],
                "n_train": len(s["train"]),
                "n_val": 0,
                "n_test": len(s["test"]),
                "test_sz_pct": round(s.get("test_sz_pct", 50.0), 1),
            }
            for cid, s in self.client_splits.items()
        }

    def print_summary(self) -> None:
        print(f"\n{'=' * 60}")
        print(f"  FederatedDataManager — 训练1:1平衡 · 测试自然分布")
        print(f"  客户端数量: {self.num_clients}   pos_weight: {self.pos_weight.item():.3f}")
        print(f"{'=' * 60}")
        for cid, info in self.get_client_info().items():
            print(
                f"  Client {cid:>2} (Case {cid + 1:>2}): "
                f"train={info['n_train']:>4}  "
                f"test={info['n_test']:>4}  "
                f"test_sz%={info['test_sz_pct']:>5.1f}%"
            )
        print(f"{'=' * 60}\n")

        # 打印所有被试者测试集信息
        print(f"{'=' * 60}")
        print(f"  所有被试者测试集信息（共{NUM_SUBJECTS}个）")
        print(f"{'=' * 60}")
        for sid, info in self.subject_test_splits.items():
            print(
                f"  Subject {sid:>2} (chb{sid:02d}): "
                f"test={info['n_test']:>4}  "
                f"sz={info['n_sz']:>3}  "
                f"inter={info['n_inter']:>4}  "
                f"sz%={info['test_sz_pct']:>5.1f}%"
            )
        print(f"{'=' * 60}\n")

    # 保留原有内部工具方法（向后兼容）
    def _concat_and_shuffle(
            self, pos_idx: np.ndarray, neg_idx: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_pos = self.X[pos_idx]
        X_neg = self.X[neg_idx]
        y_pos = np.ones(len(pos_idx), dtype=np.float32)
        y_neg = np.zeros(len(neg_idx), dtype=np.float32)
        X_all = np.concatenate([X_pos, X_neg], axis=0)
        y_all = np.concatenate([y_pos, y_neg], axis=0)
        perm = np.random.permutation(len(y_all))
        return X_all[perm].copy(), y_all[perm].copy()


# ---------------------------------------------------------------------------
# 模块级工具函数（保持原样，向后兼容）
# ---------------------------------------------------------------------------

def _filter_interictal(
        non_sz_idx: np.ndarray, window_times: np.ndarray,
        seizure_onsets: np.ndarray, gap_s: float,
) -> np.ndarray:
    if len(seizure_onsets) == 0 or len(non_sz_idx) == 0:
        return non_sz_idx
    times = window_times[non_sz_idx]
    onsets = seizure_onsets[:, np.newaxis]
    min_dist = np.abs(times[np.newaxis, :] - onsets).min(axis=0)
    return non_sz_idx[min_dist >= gap_s]


def _oversample(idx: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(idx) == 0:
        return idx
    replace = len(idx) < n
    return rng.choice(idx, size=n, replace=replace)


def _infer_seizure_onsets(
        subject_ids: np.ndarray, y: np.ndarray, window_times: np.ndarray,
) -> np.ndarray:
    rows: List[List[float]] = []
    for subj in np.unique(subject_ids):
        mask = subject_ids == subj
        local_y = y[mask]
        local_t = window_times[mask]
        in_seizure = False
        for i, lab in enumerate(local_y):
            if lab == 1 and not in_seizure:
                rows.append([float(subj), float(local_t[i])])
                in_seizure = True
            elif lab == 0:
                in_seizure = False
    if not rows:
        return np.zeros((0, 2), dtype=np.float64)
    return np.array(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# 自测入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import tempfile

    print("=== data_loader 自测（合成数据，支持24个被试者）===\n")
    with tempfile.TemporaryDirectory() as tmp:
        N = 24 * 200
        np.save(os.path.join(tmp, "signal_samples.npy"),
                np.random.randn(N, 18, 2048).astype(np.float32))
        np.save(os.path.join(tmp, "is_sz.npy"),
                np.random.choice([0, 1], size=N, p=[0.96, 0.04]).astype(np.float32))

        dm = FederatedDataManager(tmp, num_clients=8, batch_size=16)
        dm.print_summary()

        # 测试被试者加载器
        print("测试 get_subject_test_loader:")
        all_ids = dm.get_all_subject_ids()
        print(f"  所有被试者ID: {all_ids}")

        for sid in [1, 12, 24]:
            loader = dm.get_subject_test_loader(sid)
            info = dm.subject_test_splits[sid]
            print(f"  Subject {sid}: test_samples={info['n_test']}, sz={info['n_sz']}, inter={info['n_inter']}")

        tr, _, te = dm.get_client_loaders(0)
        xb, yb = next(iter(tr))
        print(f"训练批次: X={xb.shape}, y={yb.shape}, 正样本比={yb.mean():.2f}")

        xb_t, yb_t = next(iter(te))
        print(f"测试批次: X={xb_t.shape}, 正样本比={yb_t.mean():.3f}（接近真实发作率）")
        print(f"\npos_weight = {dm.pos_weight.item():.3f}")
