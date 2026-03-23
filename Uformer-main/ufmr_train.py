"""
Uformer 光流估计 - 统一训练版 (与NAFNet/Restormer完全相同)
"""

import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 如果想用CPU

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import cv2
from tqdm import tqdm
import glob
import re
import time
import csv
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
import json

# ==================== 多卡相关导入 ====================
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp

# 导入修改后的Uformer模型
from model import Uformer  # 假设您的文件名为uformer.py


class BlurToFlowDataset(Dataset):
    """与之前三个模型完全相同的数据集类"""

    def __init__(self, root_dir, split='Train', max_samples=None):
        self.img_dir = os.path.join(root_dir, split, 'img')
        self.flow_dir = os.path.join(root_dir, split, 'Displacement')

        self.blur_files = [f for f in os.listdir(self.img_dir) if f.endswith('_blur.jpg')]
        self.blur_files.sort()

        if max_samples is not None and len(self.blur_files) > max_samples:
            self.blur_files = self.blur_files[:max_samples]
            print(f"使用 {max_samples} 个样本 (从 {len(self.blur_files)} 总数中选取)")
        else:
            print(f"使用全部 {len(self.blur_files)} 个样本")

    def __len__(self):
        return len(self.blur_files)

    def __getitem__(self, idx):
        blur_file = self.blur_files[idx]
        base_name = blur_file.replace('_blur.jpg', '')

        # 加载模糊图像 (归一化到[0,1])
        blur_path = os.path.join(self.img_dir, blur_file)
        blur_img = cv2.imread(blur_path)
        blur_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB)
        blur_img = blur_img.astype(np.float32) / 255.0

        # 加载光流 (保持原始数值)
        flow_x = np.loadtxt(os.path.join(self.flow_dir, base_name + '_flowx.csv'), delimiter=',').astype(np.float32)
        flow_y = np.loadtxt(os.path.join(self.flow_dir, base_name + '_flowy.csv'), delimiter=',').astype(np.float32)
        flow = np.stack([flow_x, flow_y], axis=0)

        # 转换为tensor
        blur_img = torch.from_numpy(blur_img).permute(2, 0, 1)
        flow = torch.from_numpy(flow)

        return blur_img, flow, base_name


def setup(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12360'  # 用不同端口避免冲突
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()


def evaluate_on_test(model, test_loader, device, epoch, rank):
    """在测试集上评估模型"""
    model.eval()
    test_losses = []
    test_epes = []
    test_predictions = []
    test_targets = []
    test_names = []

    criterion = nn.L1Loss()

    with torch.no_grad():
        for blur, flow, name in tqdm(test_loader, desc=f'测试集评估 Epoch {epoch}', disable=(rank != 0)):
            blur, flow = blur.to(device), flow.to(device)
            pred = model(blur)

            loss = criterion(pred, flow)
            epe = torch.sqrt(((pred - flow) ** 2).sum(dim=1)).mean()

            test_losses.append(loss.item())
            test_epes.append(epe.item())

            # 保存前8个样本用于可视化（只在rank 0）
            if rank == 0 and len(test_predictions) < 8:
                test_predictions.append(pred[0].cpu())
                test_targets.append(flow[0].cpu())
                test_names.append(name[0])

    return {
        'mean_loss': np.mean(test_losses),
        'std_loss': np.std(test_losses),
        'mean_epe': np.mean(test_epes),
        'std_epe': np.std(test_epes),
        'predictions': test_predictions if rank == 0 else [],
        'targets': test_targets if rank == 0 else [],
        'names': test_names if rank == 0 else []
    }


def plot_training_curves(history, save_path):
    """绘制完整的训练曲线（6个子图）"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Uformer Training History', fontsize=16)

    # 1. Loss曲线
    axes[0, 0].plot(history['epoch'], history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['epoch'], history['val_loss'], 'r-', label='Val Loss', linewidth=2)

    # 修复：只画有测试数据的点
    if 'test_loss' in history and any(v is not None for v in history['test_loss']):
        test_epochs = []
        test_losses = []
        test_epes = []

        for i, epoch in enumerate(history['epoch']):
            if i < len(history['test_loss']) and history['test_loss'][i] is not None:
                test_epochs.append(epoch)
                test_losses.append(history['test_loss'][i])
                test_epes.append(history['test_epe'][i])

        if test_epochs:
            axes[0, 0].scatter(test_epochs, test_losses, c='g', marker='o', s=50, label='Test Loss', zorder=5)
            axes[0, 1].scatter(test_epochs, test_epes, c='g', marker='o', s=50, label='Test EPE', zorder=5)

    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. EPE曲线
    axes[0, 1].plot(history['epoch'], history['train_epe'], 'b-', label='Train EPE', linewidth=2)
    axes[0, 1].plot(history['epoch'], history['val_epe'], 'r-', label='Val EPE', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('EPE (pixels)')
    axes[0, 1].set_title('Endpoint Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 可学习参数
    axes[0, 2].plot(history['epoch'], history['init_scale'], 'g-', label='init_scale', linewidth=2)
    axes[0, 2].plot(history['epoch'], history['residual_scale'], 'orange', label='residual_scale', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Scale Value')
    axes[0, 2].set_title('Learnable Parameters')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. 预测范围
    axes[1, 0].plot(history['epoch'], history['pred_max'], 'r--', label='Pred Max', linewidth=2)
    axes[1, 0].plot(history['epoch'], history['pred_min'], 'b--', label='Pred Min', linewidth=2)
    axes[1, 0].plot(history['epoch'], history['true_max'], 'r-', label='True Max', linewidth=2)
    axes[1, 0].plot(history['epoch'], history['true_min'], 'b-', label='True Min', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Flow Range')
    axes[1, 0].set_title('Prediction Range vs Ground Truth')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. 学习率
    axes[1, 1].plot(history['epoch'], history['learning_rate'], 'purple', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].grid(True, alpha=0.3)

    # 6. 过拟合检测
    val_train_ratio = [v / t for v, t in zip(history['val_epe'], history['train_epe'])]
    axes[1, 2].plot(history['epoch'], val_train_ratio, 'k-', linewidth=2)
    axes[1, 2].axhline(y=1.0, color='g', linestyle='-', alpha=0.3, label='Ideal (1.0)')
    axes[1, 2].axhline(y=1.1, color='r', linestyle='--', label='Overfitting threshold (1.1)')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Val EPE / Train EPE')
    axes[1, 2].set_title('Overfitting Detection')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    if len(history['val_epe']) > 0:
        latest_ratio = val_train_ratio[-1]
        status = "✓ Normal" if latest_ratio < 1.1 else "⚠️ Possible overfitting"
        axes[1, 2].text(0.02, 0.98, f'Latest ratio: {latest_ratio:.3f}\nStatus: {status}',
                        transform=axes[1, 2].transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    if dist.get_rank() == 0:
        print(f"  📊 训练曲线已保存: {save_path}")


def plot_test_predictions(predictions, targets, names, epoch, save_dir='test_predictions_uformer'):
    """绘制测试集预测结果"""
    os.makedirs(save_dir, exist_ok=True)

    n_samples = len(predictions)
    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))

    for i in range(n_samples):
        axes[i, 0].imshow(predictions[i][0].numpy(), cmap='RdBu', vmin=-5, vmax=5)
        axes[i, 0].set_title(f'Pred U - {names[i]}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(targets[i][0].numpy(), cmap='RdBu', vmin=-5, vmax=5)
        axes[i, 1].set_title(f'GT U - {names[i]}')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(predictions[i][1].numpy(), cmap='RdBu', vmin=-5, vmax=5)
        axes[i, 2].set_title(f'Pred V - {names[i]}')
        axes[i, 2].axis('off')

        axes[i, 3].imshow(targets[i][1].numpy(), cmap='RdBu', vmin=-5, vmax=5)
        axes[i, 3].set_title(f'GT V - {names[i]}')
        axes[i, 3].axis('off')

    plt.suptitle(f'Uformer Test Set Predictions - Epoch {epoch}', fontsize=14)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'test_predictions_epoch_{epoch}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  🖼️ 测试集预测图已保存: {save_path}")


def main_worker(rank, world_size):
    """每个GPU上运行的函数"""

    # 初始化分布式环境
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    device = rank

    # 只有rank 0打印信息
    if rank == 0:
        print("=" * 70)
        print("Uformer 光流估计 - 统一训练版 (与NAFNet完全相同)")
        print("=" * 70)
        print("训练时间: ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(f"使用设备: {rank} / {world_size} GPUs")
        print("=" * 70)

    # ==================== 统一参数设置 ====================
    BATCH_SIZE = 2  # Uformer是Transformer，显存消耗大，先设1试试
    NUM_EPOCHS = 60  # 与NAFNet一致
    LEARNING_RATE = 1e-4
    TARGET_MAG = 5.0
    PENALTY_COEF = 0.3
    DATA_ROOT = '/home/dell/DATA/hzq/BS/dataset/generated_dataset/'

    CHECKPOINT_DIR = 'checkpoints_uformer'
    PLOTS_DIR = 'plots_uformer'
    TEST_PRED_DIR = 'test_predictions_uformer'
    HISTORY_DIR = 'history_uformer'

    # 只在rank 0创建目录
    if rank == 0:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        os.makedirs(TEST_PRED_DIR, exist_ok=True)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(HISTORY_DIR, exist_ok=True)

    # ==================== 数据集 ====================
    full_train_dataset = BlurToFlowDataset(DATA_ROOT, 'Train', max_samples=None)
    val_dataset = BlurToFlowDataset(DATA_ROOT, 'Val', max_samples=None)
    test_dataset = BlurToFlowDataset(DATA_ROOT, 'Test', max_samples=100)

    # 训练集/验证集划分
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, extra_val = random_split(full_train_dataset, [train_size, val_size])
    combined_val_dataset = torch.utils.data.ConcatDataset([val_dataset, extra_val])

    # 分布式采样器
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(combined_val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        combined_val_dataset,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True
    )

    if rank == 0:
        print(f"\n数据集划分:")
        print(f"  - 训练集: {len(train_dataset)} 个样本")
        print(f"  - 验证集: {len(combined_val_dataset)} 个样本")
        print(f"  - 测试集: {len(test_dataset)} 个样本")

    # ==================== 模型创建 ====================
    model = Uformer(
        img_size=256,
        embed_dim=32,  # 可调整
        depths=[2, 2, 2, 2, 2, 2, 2, 2, 2],
        num_heads=[1, 2, 4, 8, 16, 16, 8, 4, 2],
        win_size=8,
        mlp_ratio=4.,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1,
        use_checkpoint=False
    ).to(device)

    # 包装为DDP模型
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    total_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"\n模型参数量: {total_params / 1e6:.2f}M")

    # ==================== 损失函数和优化器 ====================
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=(rank == 0)
    )

    # ==================== 训练历史记录 ====================
    history = {
        'epoch': [],
        'train_loss': [],
        'train_epe': [],
        'val_loss': [],
        'val_epe': [],
        'test_loss': [],
        'test_epe': [],
        'init_scale': [],
        'residual_scale': [],
        'pred_min': [],
        'pred_max': [],
        'true_min': [],
        'true_max': [],
        'learning_rate': []
    }

    # ==================== 断点续训 ====================
    start_epoch = 1
    best_epe = float('inf')

    if rank == 0:
        checkpoint_files = glob.glob(os.path.join(CHECKPOINT_DIR, 'checkpoint_epoch_*.pth'))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(re.search(r'epoch_(\d+)', x).group(1)))
            checkpoint = torch.load(latest_checkpoint, map_location=f'cuda:{device}')
            model.module.load_state_dict(checkpoint['model_state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_epe = checkpoint['best_epe']
            history = checkpoint.get('history', history)
            print(f"🔄 从 epoch {checkpoint['epoch']} 继续训练")

    # 广播start_epoch到所有GPU
    start_epoch = torch.tensor(start_epoch).to(device)
    dist.broadcast(start_epoch, src=0)
    start_epoch = start_epoch.item()

    # ==================== CSV记录初始化 ====================
    if rank == 0:
        record_file = 'uformer_training_record.csv'
        if not os.path.isfile(record_file):
            with open(record_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss', 'train_epe', 'val_loss', 'val_epe',
                                 'test_epe', 'init_scale', 'residual_scale', 'pred_min', 'pred_max',
                                 'true_min', 'true_max', 'lr', 'val_train_ratio'])

    # ==================== 训练循环 ====================
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        epoch_start_time = time.time()

        # 设置sampler的epoch（保证shuffle正确）
        train_sampler.set_epoch(epoch)

        # ----- 训练 -----
        model.train()
        train_loss = 0
        train_epe = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{NUM_EPOCHS} [Train]', disable=(rank != 0))
        for batch_idx, (blur, flow, _) in enumerate(pbar):
            blur, flow = blur.to(device), flow.to(device)

            pred = model(blur)
            l1_loss = criterion(pred, flow)
            pred_mag = pred.abs().mean()

            if pred_mag < TARGET_MAG:
                mag_loss = (TARGET_MAG - pred_mag.detach()) * PENALTY_COEF
                loss = l1_loss + mag_loss
            else:
                loss = l1_loss

            epe = torch.sqrt(((pred - flow) ** 2).sum(dim=1)).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss.item()
            train_epe += epe.item()

            if rank == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'epe': f'{epe.item():.4f}',
                    'mag': f'{pred_mag:.2f}'
                })

        # 同步所有GPU的统计
        train_loss_tensor = torch.tensor(train_loss).to(device)
        train_epe_tensor = torch.tensor(train_epe).to(device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_epe_tensor, op=dist.ReduceOp.SUM)
        train_loss = train_loss_tensor.item() / world_size / len(train_loader)
        train_epe = train_epe_tensor.item() / world_size / len(train_loader)

        # ----- 验证 -----
        model.eval()
        val_loss = 0
        val_epe = 0

        with torch.no_grad():
            for blur, flow, _ in tqdm(val_loader, desc=f'Epoch {epoch}/{NUM_EPOCHS} [Val]', disable=(rank != 0)):
                blur, flow = blur.to(device), flow.to(device)
                pred = model(blur)

                loss = criterion(pred, flow)
                epe = torch.sqrt(((pred - flow) ** 2).sum(dim=1)).mean()

                val_loss += loss.item()
                val_epe += epe.item()

        # 同步验证统计
        val_loss_tensor = torch.tensor(val_loss).to(device)
        val_epe_tensor = torch.tensor(val_epe).to(device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_epe_tensor, op=dist.ReduceOp.SUM)
        val_loss = val_loss_tensor.item() / world_size / len(val_loader)
        val_epe = val_epe_tensor.item() / world_size / len(val_loader)

        # 每5个epoch在测试集上评估（只在rank 0）
        test_results = None
        if rank == 0 and (epoch % 5 == 0 or epoch == 1 or epoch == NUM_EPOCHS):
            test_results = evaluate_on_test(model, test_loader, device, epoch, rank)
            if test_results['predictions']:
                plot_test_predictions(test_results['predictions'], test_results['targets'],
                                      test_results['names'], epoch, TEST_PRED_DIR)

        # 获取样本用于范围监控（只在rank 0）
        if rank == 0:
            sample_blur, sample_flow, _ = next(iter(val_loader))
            sample_blur = sample_blur.to(device)
            with torch.no_grad():
                if hasattr(model, 'module'):
                    # DDP模型需要用module访问原始属性
                    init_out = model.module.initial_flow(sample_blur[:1]) * model.module.init_scale
                    sample_pred = model(sample_blur[:1])
                    init_scale_val = model.module.init_scale.item()
                    residual_scale_val = model.module.residual_scale.item() if hasattr(model.module,
                                                                                       'residual_scale') else 1.0
                else:
                    init_out = model.initial_flow(sample_blur[:1]) * model.init_scale
                    sample_pred = model(sample_blur[:1])
                    init_scale_val = model.init_scale.item()
                    residual_scale_val = model.residual_scale.item() if hasattr(model, 'residual_scale') else 1.0
        else:
            init_scale_val = 0
            residual_scale_val = 0
            sample_pred_min = 0
            sample_pred_max = 0
            true_min = 0
            true_max = 0

        # 广播范围监控数据
        if rank == 0:
            monitor_data = torch.tensor([
                init_scale_val,
                residual_scale_val,
                sample_pred.min().item(),
                sample_pred.max().item(),
                sample_flow[:1].min().item(),
                sample_flow[:1].max().item()
            ]).to(device)
        else:
            monitor_data = torch.zeros(6).to(device)

        dist.broadcast(monitor_data, src=0)

        # 更新历史记录（只在rank 0）
        if rank == 0:
            history['epoch'].append(epoch)
            history['train_loss'].append(train_loss)
            history['train_epe'].append(train_epe)
            history['val_loss'].append(val_loss)
            history['val_epe'].append(val_epe)

            if test_results:
                history['test_loss'].append(test_results['mean_loss'])
                history['test_epe'].append(test_results['mean_epe'])
            else:
                history['test_loss'].append(None)
                history['test_epe'].append(None)

            history['init_scale'].append(monitor_data[0].item())
            history['residual_scale'].append(monitor_data[1].item())
            history['pred_min'].append(monitor_data[2].item())
            history['pred_max'].append(monitor_data[3].item())
            history['true_min'].append(monitor_data[4].item())
            history['true_max'].append(monitor_data[5].item())
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        # 调整学习率
        scheduler.step(val_loss)

        # 计算耗时（只在rank 0）
        if rank == 0:
            epoch_time = time.time() - epoch_start_time
            remaining_epochs = NUM_EPOCHS - epoch
            estimated_time = remaining_epochs * epoch_time / 3600

            # 过拟合检测
            val_train_ratio = val_epe / train_epe

            # 写入CSV记录
            with open(record_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    f'{train_loss:.4f}',
                    f'{train_epe:.4f}',
                    f'{val_loss:.4f}',
                    f'{val_epe:.4f}',
                    f'{test_results["mean_epe"]:.4f}' if test_results else 'N/A',
                    f'{monitor_data[0].item():.4f}',
                    f'{monitor_data[1].item():.4f}',
                    f'{monitor_data[2].item():.2f}',
                    f'{monitor_data[3].item():.2f}',
                    f'{monitor_data[4].item():.2f}',
                    f'{monitor_data[5].item():.2f}',
                    f'{optimizer.param_groups[0]["lr"]:.2e}',
                    f'{val_train_ratio:.4f}'
                ])

            # 打印训练信息
            test_epe_str = ''
            if test_results:
                test_epe_str = f'Test EPE: {test_results["mean_epe"]:.4f} | '

            print(f"\nEpoch {epoch:3d}/{NUM_EPOCHS} | "
                  f"Train Loss: {train_loss:.4f} | Train EPE: {train_epe:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val EPE: {val_epe:.4f} | "
                  f"{test_epe_str}"
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {epoch_time:.1f}s | Est: {estimated_time:.1f}h")

            print(f"      init_scale: {monitor_data[0].item():.4f} | "
                  f"预测范围: [{monitor_data[2].item():.2f}, {monitor_data[3].item():.2f}]")

            if val_train_ratio > 1.1:
                print(f"  ⚠️ 警告: 可能过拟合! Val/Train ratio = {val_train_ratio:.3f} > 1.1")

            # 保存最佳模型
            if val_epe < best_epe:
                best_epe = val_epe
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_epe': best_epe,
                    'history': history,
                }, os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
                print(f"  🏆 保存最佳模型 (EPE={best_epe:.4f})")

            # 每1个epoch绘制训练曲线和保存checkpoint
            if epoch % 1 == 0:
                plot_training_curves(history, os.path.join(PLOTS_DIR, f'training_curves_epoch_{epoch}.png'))

                # 保存历史记录
                with open(os.path.join(HISTORY_DIR, f'history_epoch_{epoch}.json'), 'w') as f:
                    history_json = {k: [None if v is None else v for v in vals]
                                    for k, vals in history.items()}
                    json.dump(history_json, f, indent=4)

                # 保存checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_epe': best_epe,
                    'history': history,
                }, os.path.join(CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pth'))

    # ==================== 训练完成 ====================
    if rank == 0:
        print("\n" + "=" * 70)
        print("🎉 训练完成！")
        print(f"最佳验证 EPE: {best_epe:.4f}")

        # 最终测试集评估
        print("\n📊 最终测试集评估...")
        final_test = evaluate_on_test(model, test_loader, device, 'final', rank)
        print(f"  测试损失: {final_test['mean_loss']:.4f} ± {final_test['std_loss']:.4f}")
        print(f"  测试EPE: {final_test['mean_epe']:.4f} ± {final_test['std_epe']:.4f}")

        if final_test['predictions']:
            plot_test_predictions(final_test['predictions'], final_test['targets'],
                                  final_test['names'], 'final', TEST_PRED_DIR)

        # 添加最终测试结果到历史记录
        history['final_test_epe'] = final_test['mean_epe']
        history['final_test_loss'] = final_test['mean_loss']

        # 绘制最终训练曲线
        plot_training_curves(history, os.path.join(PLOTS_DIR, 'final_training_curves.png'))

        # 保存完整历史记录
        with open(os.path.join(HISTORY_DIR, 'complete_history.json'), 'w') as f:
            history_json = {k: [None if v is None else v for v in vals]
                            for k, vals in history.items()}
            json.dump(history_json, f, indent=4)

        print("\n" + "=" * 70)
        print("✅ 所有结果已保存：")
        print(f"  - {PLOTS_DIR}/final_training_curves.png")
        print(f"  - {TEST_PRED_DIR}/final_predictions.png")
        print(f"  - {HISTORY_DIR}/complete_history.json")
        print(f"  - {CHECKPOINT_DIR}/best_model.pth")
        print(f"  - uformer_training_record.csv")
        print("=" * 70)

    cleanup()


def train():
    world_size = torch.cuda.device_count()
    print(f"检测到 {world_size} 张GPU，启动分布式训练...")
    mp.spawn(main_worker, args=(world_size,), nprocs=world_size, join=True)


if __name__ == '__main__':
    train()