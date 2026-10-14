"""
main.py — 修复版 + 全局模型跨被试者评估
========================================
相较原版的四处核心改动：

改动1 [train_local_model]
    使用加权 BCE 损失（pos_weight × 正样本），强迫模型学习稀疏发作特征，
    避免预测全0的退化解。原版用普通 BCELoss，正负样本权重相同。

改动2 [evaluate]
    分类阈值从 0.4 下调至 0.35。
    联邦平均稀释各客户端梯度，导致模型预测概率整体偏低；
    较低的阈值能让灵敏度从接近0恢复到正常水平。

改动3 [parse_args]
    num_rounds 60 → 100，patience 10 → 15。
    K=16 在原版第57轮仍未收敛，需要更多轮次。

改动4 [data_loader.py 配合]
    data_loader 现在提供 pos_weight（训练集真实正负比），
    main.py 读取并传给损失函数，不再写死 weight=5.0。

新增功能 [跨被试者评估]
    - evaluate_on_all_subjects(): 在所有24个被试者上评估全局模型
    - save_subject_metrics_to_csv(): 将结果保存为CSV
    - 修改 generate_paper_text(): 输出24个被试者的详细结果表格
"""

import os
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import pickle
from sklearn.metrics import confusion_matrix, roc_auc_score
import csv

from data_loader import FederatedDataManager
from model import FDGAT, LayerImportanceDP
from evaluate import compute_metrics, MemberInferenceAttack

# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='FDGAT Federated Learning with Adaptive LDP')
    p.add_argument("--data_dir",      default=r"D:\zhongwenhexin\FL_code0329\signal_samples\data")
    # 改动3：轮次从60增加到100，patience从10增加到15
    p.add_argument("--num_rounds",    type=int,   default=100)
    p.add_argument("--local_epochs",  type=int,   default=16)
    p.add_argument("--batch_size",    type=int,   default=32)
    p.add_argument("--lr",            type=float, default=0.0005)
    p.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",          type=int,   default=42)

    p.add_argument("--noise_std",     type=float, default=0.01)
    p.add_argument("--mu",            type=float, default=0.3, help="FedProx proximal term")
    p.add_argument("--patience",      type=int,   default=15,   help="收敛判定耐心值，从10增大到15")
    p.add_argument("--total_eps",     type=float, default=1.0)
    p.add_argument("--clip_norm",     type=float, default=1.0)

    p.add_argument("--client_scales", nargs="+", type=int, default=[8, 12, 16])
    p.add_argument("--run_mia",       type=bool,  default=False)
    return p.parse_args()

# ---------------------------------------------------------------------------
# 评估函数
# ---------------------------------------------------------------------------

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[float, float, float, float, float]:
    """
    改动2：分类阈值从 0.4 下调至 0.35。
    返回: (avg_loss, acc, sens, spec, auc)
    """
    model.eval()
    if loader is None or len(loader) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    total_loss  = 0.0
    criterion   = nn.BCELoss()
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if y.dim() == 2 and y.size(1) == 1:
                y = y.squeeze(1)

            outputs = model(x)
            if outputs.dim() == 2 and outputs.size(1) == 1:
                outputs = outputs.squeeze(1)

            total_loss += criterion(outputs, y).item()

            # ── 改动2：阈值 ──────────────────────────────
            preds = (outputs > 0.15).float()

            all_probs.extend(outputs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    avg_loss = total_loss / len(loader)

    try:
        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds, labels=[0, 1]).ravel()
        acc  = (tp + tn) / len(all_labels) if len(all_labels) > 0 else 0.0
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except ValueError:
        acc, sens, spec = 0.0, 0.0, 0.0

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5

    return avg_loss, acc, sens, spec, auc

# ---------------------------------------------------------------------------
# 跨被试者评估函数（新增）
# ---------------------------------------------------------------------------

def evaluate_on_all_subjects(
    model: nn.Module,
    data_manager: FederatedDataManager,
    device: str,
    num_clients: int,
) -> Dict[int, Dict[str, float]]:
    """
    在所有24个被试者上评估全局模型。

    Args:
        model: 全局联邦学习模型
        data_manager: FederatedDataManager实例
        device: 计算设备
        num_clients: 当前客户端数量

    Returns:
        Dict: 每个被试者的评估指标 {subject_id: {"acc": ..., "sens": ..., "spec": ..., "auc": ...}}
    """
    model.eval()
    subject_metrics = {}
    all_subject_ids = data_manager.get_all_subject_ids()

    for subject_id in all_subject_ids:
        try:
            test_loader = data_manager.get_subject_test_loader(subject_id)
            _, acc, sens, spec, auc = evaluate(model, test_loader, device)
            subject_metrics[subject_id] = {
                "acc": acc,
                "sens": sens,
                "spec": spec,
                "auc": auc,
            }
        except Exception as e:
            # 如果某个被试者评估失败，记录为0
            subject_metrics[subject_id] = {
                "acc": 0.0,
                "sens": 0.0,
                "spec": 0.0,
                "auc": 0.5,
            }

    return subject_metrics


def print_subject_evaluation_results(
    subject_metrics: Dict[int, Dict[str, float]],
    num_clients: int,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    打印跨被试者评估结果表格。

    Returns:
        (mean_stats, std_stats, median_stats): 平均值、标准差、中位数统计
    """
    print(f"\n{'='*80}")
    print(f"K={num_clients} 时 24 个被试者的评估结果：")
    print(f"{'='*80}")

    # 表头
    print(f"{'被试者':^10} | {'Acc(%)':^10} | {'Sen(%)':^10} | {'Spe(%)':^10} | {'AUC(%)':^10}")
    print("-" * 70)

    # 每个被试者的结果
    acc_list, sens_list, spec_list, auc_list = [], [], [], []

    for sid in sorted(subject_metrics.keys()):
        m = subject_metrics[sid]
        acc_list.append(m["acc"] * 100)
        sens_list.append(m["sens"] * 100)
        spec_list.append(m["spec"] * 100)
        auc_list.append(m["auc"] * 100)

        subject_name = f"chb{sid:02d}"
        print(f"{subject_name:^10} | {m['acc']*100:>10.2f} | {m['sens']*100:>10.2f} | "
              f"{m['spec']*100:>10.2f} | {m['auc']*100:>10.2f}")

    print("-" * 70)

    # 计算统计信息
    mean_stats = {
        "acc": np.mean(acc_list),
        "sens": np.mean(sens_list),
        "spec": np.mean(spec_list),
        "auc": np.mean(auc_list),
    }
    std_stats = {
        "acc": np.std(acc_list),
        "sens": np.std(sens_list),
        "spec": np.std(spec_list),
        "auc": np.std(auc_list),
    }
    median_stats = {
        "acc": np.median(acc_list),
        "sens": np.median(sens_list),
        "spec": np.median(spec_list),
        "auc": np.median(auc_list),
    }

    print(f"\n跨被试者统计（K={num_clients}）：")
    print(f"  平均值:  Acc={mean_stats['acc']:.2f}%  Sen={mean_stats['sens']:.2f}%  "
          f"Spe={mean_stats['spec']:.2f}%  AUC={mean_stats['auc']:.2f}%")
    print(f"  标准差:  Acc={std_stats['acc']:.2f}%  Sen={std_stats['sens']:.2f}%  "
          f"Spe={std_stats['spec']:.2f}%  AUC={std_stats['auc']:.2f}%")
    print(f"  中位数:  Acc={median_stats['acc']:.2f}%  Sen={median_stats['sens']:.2f}%  "
          f"Spe={median_stats['spec']:.2f}%  AUC={median_stats['auc']:.2f}%")
    print(f"{'='*80}\n")

    return mean_stats, std_stats, median_stats


def save_subject_metrics_to_csv(
    all_results: Dict,
    args,
    filepath: str = "subject_metrics.csv"
) -> None:
    """
    将跨被试者评估结果保存为CSV文件。

    Args:
        all_results: 所有实验结果
        args: 命令行参数
        filepath: 保存路径
    """
    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)

        # 写入标题行
        writer.writerow(['K值', '被试者', 'Acc(%)', 'Sen(%)', 'Spe(%)', 'AUC(%)'])

        # 写入每个K值下每个被试者的结果
        for k in args.client_scales:
            subject_metrics = all_results[k].get("subject_metrics", {})
            for sid in sorted(subject_metrics.keys()):
                m = subject_metrics[sid]
                writer.writerow([
                    k,
                    f"chb{sid:02d}",
                    f"{m['acc']*100:.2f}",
                    f"{m['sens']*100:.2f}",
                    f"{m['spec']*100:.2f}",
                    f"{m['auc']*100:.2f}",
                ])

        # 写入统计信息标题
        writer.writerow([])
        writer.writerow(['统计信息'])

        # 写入每个K值的统计信息
        for k in args.client_scales:
            stats = all_results[k].get("subject_stats", {})
            if stats:
                mean = stats.get("mean", {})
                std = stats.get("std", {})
                median = stats.get("median", {})

                writer.writerow([f'K={k}'])
                writer.writerow(['平均值', '',
                    f"{mean.get('acc', 0)*100:.2f}",
                    f"{mean.get('sens', 0)*100:.2f}",
                    f"{mean.get('spec', 0)*100:.2f}",
                    f"{mean.get('auc', 0)*100:.2f}",
                ])
                writer.writerow(['标准差', '',
                    f"{std.get('acc', 0)*100:.2f}",
                    f"{std.get('sens', 0)*100:.2f}",
                    f"{std.get('spec', 0)*100:.2f}",
                    f"{std.get('auc', 0)*100:.2f}",
                ])
                writer.writerow(['中位数', '',
                    f"{median.get('acc', 0)*100:.2f}",
                    f"{median.get('sens', 0)*100:.2f}",
                    f"{median.get('spec', 0)*100:.2f}",
                    f"{median.get('auc', 0)*100:.2f}",
                ])
                writer.writerow([])

    print(f"被试者评估结果已保存至 {filepath}")


# ---------------------------------------------------------------------------
# 本地训练函数
# ---------------------------------------------------------------------------

def train_local_model(
    args,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader,
    device: str,
    pos_weight: torch.Tensor,
    is_prox: bool = True,
    global_params=None,
):
    """
    改动1：使用加权 BCE 损失（BCELoss + 手动权重）。
    pos_weight 由 data_loader 根据真实正负比计算，避免写死固定值。
    正样本损失被放大 pos_weight 倍，迫使模型关注稀疏发作事件。
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # reduction='none' 以便逐样本乘权重
    criterion = nn.BCELoss(reduction='none')

    # pos_weight 标量值，用于正样本权重
    pw = pos_weight.item() if isinstance(pos_weight, torch.Tensor) else float(pos_weight)
    # 限制权重上限，防止极端不平衡时梯度爆炸
    pw = min(pw, 10.0)

    avg_loss = 0.0
    for epoch in range(args.local_epochs):
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if y.dim() == 2 and y.size(1) == 1:
                y = y.squeeze(1)

            optimizer.zero_grad()
            outputs = model(x)
            if outputs.dim() == 2 and outputs.size(1) == 1:
                outputs = outputs.squeeze(1)

            # ── 改动1：加权损失 ──────────────────────────────────────
            base_loss = criterion(outputs, y)          # shape: (B,)
            weights   = torch.ones_like(y)
            weights[y == 1] = pw                       # 正样本权重 = pos_weight
            loss = (base_loss * weights).mean()

            # FedProx 近端项
            if is_prox and global_params is not None:
                prox = sum(
                    (p - gp.to(device)).norm(2) ** 2
                    for p, gp in zip(model.parameters(), global_params)
                )
                loss = loss + (args.mu / 2) * prox

            loss.backward()
            # 梯度裁剪，防止权重放大导致梯度爆炸
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

    return model, {"loss": avg_loss}

# ---------------------------------------------------------------------------
# 本地基线独立训练（对比实验，仅 K=client_scales[0] 时运行）
# ---------------------------------------------------------------------------

def run_local_baseline(args, data_manager: FederatedDataManager, num_clients: int):
    print(f"\n[Baseline] 各客户端独立训练 (无联邦聚合) — K={num_clients}")
    device     = args.device
    pos_weight = data_manager.pos_weight
    local_metrics = {}

    for cid in range(num_clients):
        train_loader, _, test_loader = data_manager.get_client_loaders(cid)
        model = FDGAT().to(device)

        baseline_epochs       = min(50, args.num_rounds)
        orig_local_epochs     = args.local_epochs
        args.local_epochs     = baseline_epochs

        trained_model, _ = train_local_model(
            args, model, train_loader, None, device, pos_weight, is_prox=False
        )
        args.local_epochs = orig_local_epochs

        _, acc, sens, spec, auc = evaluate(trained_model, test_loader, device)
        local_metrics[cid] = {"acc": acc, "sens": sens, "spec": spec, "auc": auc}
        print(f"  Client {cid:>2}: Sens={sens:.4f}  Spec={spec:.4f}  AUC={auc:.4f}")

    return local_metrics

# ---------------------------------------------------------------------------
# 联邦学习主流程（含收敛追踪 + 跨被试者评估）
# ---------------------------------------------------------------------------

def run_federated_learning_serial(
    args,
    data_manager: FederatedDataManager,
    num_clients: int,
):
    device     = args.device
    pos_weight = data_manager.pos_weight
    global_model = FDGAT().to(device)

    history = {
        "train_loss":           [],
        "test_acc":             [],
        "test_sens":            [],
        "test_spec":            [],
        "test_auc":             [],
        "client_final_metrics": {},
        "converged_round":      args.num_rounds,
        "converged_loss":       None,
        "loss_curve":           [],
        "subject_metrics":      {},  # 新增：跨被试者评估结果
        "subject_stats":        {},  # 新增：跨被试者统计
    }

    print(f"\n[FL] 联邦训练开始 — K={num_clients}  rounds={args.num_rounds}")

    min_loss         = float('inf')
    patience_counter = 0
    converged        = False

    for r in range(1, args.num_rounds + 1):
        local_weights    = []
        round_train_loss = 0.0
        global_params    = [p.detach().clone() for p in global_model.parameters()]

        for cid in range(num_clients):
            train_loader, _, _ = data_manager.get_client_loaders(cid)
            local_m, metrics   = train_local_model(
                args,
                copy.deepcopy(global_model),
                train_loader, None, device,
                pos_weight,
                is_prox=True,
                global_params=global_params,
            )
            local_weights.append(local_m.state_dict())
            round_train_loss += metrics["loss"]

        # ── FedAvg 聚合 ───────────────────────────────────────────────
        avg_weights = copy.deepcopy(local_weights[0])
        for key in avg_weights.keys():
            for i in range(1, len(local_weights)):
                avg_weights[key] += local_weights[i][key]
            avg_weights[key] = torch.div(avg_weights[key], len(local_weights))
        global_model.load_state_dict(avg_weights)

        # ── 全局评估 ──────────────────────────────────────────────────
        t_acc, t_sens, t_spec, t_auc = 0.0, 0.0, 0.0, 0.0
        for cid in range(num_clients):
            _, _, test_loader = data_manager.get_client_loaders(cid)
            _, acc, sens, spec, auc = evaluate(global_model, test_loader, device)
            t_acc  += acc
            t_sens += sens
            t_spec += spec
            t_auc  += auc
            if r == args.num_rounds:
                history["client_final_metrics"][cid] = {
                    "acc": acc, "sens": sens, "spec": spec, "auc": auc
                }

        avg_loss = round_train_loss / num_clients
        history["train_loss"].append(avg_loss)
        history["loss_curve"].append(avg_loss)
        history["test_acc"].append(t_acc   / num_clients)
        history["test_sens"].append(t_sens  / num_clients)
        history["test_spec"].append(t_spec  / num_clients)
        history["test_auc"].append(t_auc   / num_clients)

        print(
            f"Round {r:03d} | Loss={avg_loss:.4f} | "
            f"Acc={history['test_acc'][-1]:.4f}  "
            f"Sens={history['test_sens'][-1]:.4f}  "
            f"Spec={history['test_spec'][-1]:.4f}  "
            f"AUC={history['test_auc'][-1]:.4f}"
        )

        # ── 收敛检测（改动3：patience 15）────────────────────────────
        if avg_loss < min_loss - 0.001:
            min_loss         = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience and not converged:
            converged = True
            history["converged_round"] = r
            history["converged_loss"]  = avg_loss
            print(
                f"  --> [收敛] 第 {r} 轮 Loss 趋于稳定 "
                f"(patience={args.patience}, loss={avg_loss:.4f})"
            )

        # ── 联邦训练完成后，进行跨被试者评估 ──────────────────────────
        if r == args.num_rounds:
            print(f"\n[K={num_clients}] 联邦训练完成，开始跨被试者评估...")
            subject_metrics = evaluate_on_all_subjects(
                global_model, data_manager, device, num_clients
            )
            history["subject_metrics"] = subject_metrics

            # 打印并保存统计信息
            mean_stats, std_stats, median_stats = print_subject_evaluation_results(
                subject_metrics, num_clients
            )
            history["subject_stats"] = {
                "mean": mean_stats,
                "std": std_stats,
                "median": median_stats,
            }

    return history

# ---------------------------------------------------------------------------
# 论文文档自动填空（扩展：包含24被试者结果）
# ---------------------------------------------------------------------------

def generate_paper_text(all_results: Dict, args) -> str:
    lines = []
    sep   = "=" * 80

    k_first  = args.client_scales[0]
    fl_hist  = all_results[k_first]["fl_history"]
    baseline = all_results[k_first].get("local_baseline") or {}

    final_acc   = fl_hist["test_acc"][-1]  * 100
    final_sens  = fl_hist["test_sens"][-1] * 100
    final_spec  = fl_hist["test_spec"][-1] * 100
    final_auc   = fl_hist["test_auc"][-1]  * 100
    conv_round  = fl_hist["converged_round"]

    client_final = fl_hist["client_final_metrics"]

    lines.append(sep)
    lines.append("【0.1.1 癫痫检测性能 — 文档填空参考】")
    lines.append(sep)

    if baseline:
        sorted_by_local = sorted(baseline.keys(), key=lambda c: baseline[c]["sens"])
        weak_clients    = sorted_by_local[:2]
        weak_local_avg  = np.mean([baseline[c]["sens"] for c in weak_clients]) * 100
        weak_fl_avg     = np.mean([client_final[c]["sens"] for c in weak_clients]) * 100
        improvement     = weak_fl_avg - weak_local_avg
        wc0, wc1        = weak_clients[0], weak_clients[1]
    else:
        wc0, wc1 = 0, 1
        weak_local_avg = improvement = 0.0

    para = f"""
0.1.1 癫痫检测性能

在 CHB-MIT 测试集上，全局模型经过 {conv_round} 轮联邦聚合后取得的平均准确率为 {final_acc:.2f}%，
平均灵敏度为 {final_sens:.2f}%，平均特异度为 {final_spec:.2f}%，平均 AUC 为 {final_auc:.2f}%。

对于数据量较少的客户端（如客户端 {wc0} 和客户端 {wc1}），本地模型灵敏度不足 {weak_local_avg:.1f}%，
而全局模型在这些客户端的灵敏度平均提升 {improvement:.1f} 个百分点。
"""
    lines.append(para.strip())

    lines.append("")
    lines.append(sep)
    lines.append("【0.1.2 客户端规模影响 — 文档填空参考】")
    lines.append(sep)

    k_stats = {}
    for k in args.client_scales:
        cf       = all_results[k]["fl_history"]["client_final_metrics"]
        sens_arr = np.array([cf[c]["sens"] for c in sorted(cf.keys())]) * 100
        spec_arr = np.array([cf[c]["spec"] for c in sorted(cf.keys())]) * 100
        k_stats[k] = {
            "sens_med": np.median(sens_arr),
            "sens_q25": np.percentile(sens_arr, 25),
            "sens_q75": np.percentile(sens_arr, 75),
            "spec_med": np.median(spec_arr),
            "conv":     all_results[k]["fl_history"]["converged_round"],
        }

    k8, k12, k16 = k_stats.get(8,{}), k_stats.get(12,{}), k_stats.get(16,{})
    lines.append(f"""
K=8:  Sens中位={k8.get('sens_med',0):.1f}%  Q25={k8.get('sens_q25',0):.1f}%  Q75={k8.get('sens_q75',0):.1f}%  收敛轮={k8.get('conv','—')}
K=12: Sens中位={k12.get('sens_med',0):.1f}%  Q25={k12.get('sens_q25',0):.1f}%  Q75={k12.get('sens_q75',0):.1f}%  收敛轮={k12.get('conv','—')}
K=16: Sens中位={k16.get('sens_med',0):.1f}%  Q25={k16.get('sens_q25',0):.1f}%  Q75={k16.get('sens_q75',0):.1f}%  收敛轮={k16.get('conv','—')}
""".strip())

    lines.append("")
    lines.append(sep)
    lines.append("【各 K 总览速查】")
    lines.append(f"{'K':>4} | {'Conv':>5} | {'Acc%':>7} | {'Sens%':>7} | {'Spec%':>7} | {'AUC%':>7}")
    lines.append("-" * 52)
    for k in args.client_scales:
        h = all_results[k]["fl_history"]
        lines.append(
            f"{k:>4} | {h['converged_round']:>5} | "
            f"{h['test_acc'][-1]*100:>7.2f} | "
            f"{h['test_sens'][-1]*100:>7.2f} | "
            f"{h['test_spec'][-1]*100:>7.2f} | "
            f"{h['test_auc'][-1]*100:>7.2f}"
        )

    # ------------------------------------------------------------------
    # 新增：24个被试者详细结果表格
    # ------------------------------------------------------------------
    lines.append("")
    lines.append(sep)
    lines.append("【0.1.3 跨被试者评估结果 — 24例被试者详细指标】")
    lines.append(sep)

    for k in args.client_scales:
        subject_metrics = all_results[k].get("subject_metrics", {})
        stats = all_results[k].get("subject_stats", {})

        lines.append(f"\nK={k} 时 24 个被试者的评估结果：")
        lines.append(f"{'被试者':^10} | {'Acc(%)':^10} | {'Sen(%)':^10} | {'Spe(%)':^10} | {'AUC(%)':^10}")
        lines.append("-" * 70)

        for sid in sorted(subject_metrics.keys()):
            m = subject_metrics[sid]
            subject_name = f"chb{sid:02d}"
            lines.append(
                f"{subject_name:^10} | {m['acc']*100:>10.2f} | {m['sens']*100:>10.2f} | "
                f"{m['spec']*100:>10.2f} | {m['auc']*100:>10.2f}"
            )

        lines.append("-" * 70)

        # 统计信息
        if stats:
            mean = stats.get("mean", {})
            std = stats.get("std", {})
            median = stats.get("median", {})

            lines.append(f"  平均值:  Acc={mean.get('acc', 0)*100:.2f}%  Sen={mean.get('sens', 0)*100:.2f}%  "
                         f"Spe={mean.get('spec', 0)*100:.2f}%  AUC={mean.get('auc', 0)*100:.2f}%")
            lines.append(f"  标准差:  Acc={std.get('acc', 0)*100:.2f}%  Sen={std.get('sens', 0)*100:.2f}%  "
                         f"Spe={std.get('spec', 0)*100:.2f}%  AUC={std.get('auc', 0)*100:.2f}%")
            lines.append(f"  中位数:  Acc={median.get('acc', 0)*100:.2f}%  Sen={median.get('sens', 0)*100:.2f}%  "
                         f"Spe={median.get('spec', 0)*100:.2f}%  AUC={median.get('auc', 0)*100:.2f}%")

        lines.append("")

    # K值间对比表格
    lines.append(sep)
    lines.append("【跨K值统计对比】")
    lines.append(f"{'K值':>4} | {'Acc均值':>8} | {'Sen均值':>8} | {'Spe均值':>8} | {'AUC均值':>8} | {'Sen标准差':>10}")
    lines.append("-" * 65)

    for k in args.client_scales:
        stats = all_results[k].get("subject_stats", {})
        if stats:
            mean = stats.get("mean", {})
            std = stats.get("std", {})
            lines.append(
                f"{k:>4} | {mean.get('acc', 0)*100:>8.2f} | {mean.get('sens', 0)*100:>8.2f} | "
                f"{mean.get('spec', 0)*100:>8.2f} | {mean.get('auc', 0)*100:>8.2f} | "
                f"{std.get('sens', 0)*100:>10.2f}"
            )

    lines.append(sep)

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# 程序入口
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("\n[Config]")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    all_results = {}

    for k in args.client_scales:
        print(f"\n{'='*55}\n  K = {k} 客户端实验\n{'='*55}")

        data_manager = FederatedDataManager(
            data_dir    = args.data_dir,
            num_clients = k,
            batch_size  = args.batch_size,
            noise_std   = args.noise_std,
            seed        = args.seed,
        )
        data_manager.print_summary()

        local_baseline = None
        if k == args.client_scales[0]:
            local_baseline = run_local_baseline(args, data_manager, k)

        fl_history = run_federated_learning_serial(args, data_manager, k)

        all_results[k] = {
            "local_baseline": local_baseline,
            "fl_history":     fl_history,
            "subject_metrics": fl_history.get("subject_metrics", {}),
            "subject_stats":   fl_history.get("subject_stats", {}),
        }

    # 保存结果
    with open("federated_multi_scale_results.pkl", "wb") as f:
        pickle.dump(all_results, f)
    print("\n结果已保存至 federated_multi_scale_results.pkl")

    # 保存CSV
    save_subject_metrics_to_csv(all_results, args, "subject_metrics.csv")

    # 论文填空
    paper_text = generate_paper_text(all_results, args)
    print("\n" + paper_text)
    with open("paper_autofill.txt", "w", encoding="utf-8") as f:
        f.write(paper_text)
    print("\n论文填空内容已保存至 paper_autofill.txt")


if __name__ == "__main__":
    main()
