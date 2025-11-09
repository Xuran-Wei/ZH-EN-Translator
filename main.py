# 此为主函数（即入口函数）
import jieba
import nltk
from torch.utils.data import DataLoader
import torch

from Transformer import Transformer, create_mask
# from model3 import TransformerNoFFN, create_mask

from dataset import LoadDataset, build_vocab, data_process, generate_batch, english_tokenizer, chinese_tokenizer
import time
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import warnings
import matplotlib.pyplot as plt
import numpy as np
import os

# 禁止所有 UserWarning
warnings.filterwarnings("ignore", category=UserWarning)


def plot_loss_curve(train_losses, val_losses, save_path='loss_curve.png'):
    """绘制训练和验证损失曲线"""
    plt.figure(figsize=(10, 6))

    # 绘制训练损失
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', linewidth=2, label='Train Loss', alpha=0.7)

    # 绘制验证损失
    if val_losses:
        plt.plot(epochs, val_losses, 'r-', linewidth=2, label='Val Loss', alpha=0.7)

    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"损失曲线已保存: {save_path}")


def plot_lr_curve(learning_rates, save_path='lr_curve.png'):
    """绘制学习率变化曲线"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(learning_rates) + 1)
    plt.plot(epochs, learning_rates, 'g-', linewidth=2, label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 使用对数坐标更清晰
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"学习率曲线已保存: {save_path}")


def validate_model(model, val_data, criterion, device, PAD_IDX, vocab_ch, vocab_en):
    """在验证集上评估模型"""
    model.eval()
    total_val_loss = 0
    val_count = 0

    val_iter = DataLoader(val_data, batch_size=8, shuffle=False, collate_fn=generate_batch)

    with torch.no_grad():
        for src, tgt in val_iter:
            src = src.to(device)
            tgt = tgt.to(device)

            # 创建掩码
            src_mask, tgt_mask = create_mask(src, tgt, PAD_IDX)

            # 目标序列处理
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # 调整目标掩码以匹配输入序列长度
            tgt_mask = tgt_mask[:, :, :tgt_input.size(1), :tgt_input.size(1)]

            # 前向传播
            output = model(src, tgt_input, src_mask, tgt_mask)

            # 计算损失
            loss = criterion(output.reshape(-1, len(vocab_en)), tgt_output.reshape(-1))

            total_val_loss += loss.item()
            val_count += 1

    model.train()
    return total_val_loss / val_count if val_count > 0 else 0


def save_model_checkpoint(checkpoint_data, save_path, model_type="checkpoint"):
    """
    统一保存模型检查点的函数

    参数:
    - checkpoint_data: 包含所有需要保存的数据的字典
    - save_path: 保存路径
    - model_type: 模型类型描述，用于日志输出
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

    # 构建完整的检查点
    checkpoint = {
        'epoch': checkpoint_data['epoch'],
        'model_state_dict': checkpoint_data['model_state_dict'],
        'optimizer_state_dict': checkpoint_data['optimizer_state_dict'],
        'scheduler_state_dict': checkpoint_data['scheduler_state_dict'],
        'train_loss': checkpoint_data['train_loss'],
        'val_loss': checkpoint_data['val_loss'],
        'vocab_ch': checkpoint_data['vocab_ch'],
        'vocab_en': checkpoint_data['vocab_en'],
        'model_config': checkpoint_data['model_config']
    }

    # 可选字段
    if 'best_val_loss' in checkpoint_data:
        checkpoint['best_val_loss'] = checkpoint_data['best_val_loss']
    if 'train_losses' in checkpoint_data:
        checkpoint['train_losses'] = checkpoint_data['train_losses']
    if 'val_losses' in checkpoint_data:
        checkpoint['val_losses'] = checkpoint_data['val_losses']
    if 'learning_rates' in checkpoint_data:
        checkpoint['learning_rates'] = checkpoint_data['learning_rates']

    # 保存模型
    torch.save(checkpoint, save_path)
    print(f"{model_type}已保存: {save_path}")


def initialize_model(model):
    """初始化模型参数"""
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)
    return model


if __name__ == '__main__':
    # 创建必要的目录
    os.makedirs('./ckpt', exist_ok=True)
    os.makedirs('./result', exist_ok=True)

    # 数据集准备和预处理
    train_ch, val_ch, train_en, val_en, test_ch, test_en = LoadDataset(
        train_path="/data/fzy/xuran_Reproduction/LLM_midterm/dataset/train.txt",
        test_path="/data/fzy/xuran_Reproduction/LLM_midterm/dataset/test.txt",
    )

    # 输出训练集前三行示例
    print("=== 训练集前三行示例 ===")
    print("中文 (train_ch):")
    for i, ch_text in enumerate(train_ch[:3]):
        print(f"第{i + 1}行: {ch_text}")

    print("\n英文 (train_en):")
    for i, en_text in enumerate(train_en[:3]):
        print(f"第{i + 1}行: {en_text}")

    vocab_ch = build_vocab(train_ch, chinese_tokenizer)
    vocab_en = build_vocab(train_en, english_tokenizer)

    print("=== 分词结果示例 ===")
    for i in range(3):
        print(f"样本 {i + 1}:")
        print(f"  英文原文: {train_en[i]}")
        print(f"  英文分词: {english_tokenizer(train_en[i])}")
        print(f"  中文原文: {train_ch[i]}")
        print(f"  中文分词: {chinese_tokenizer(train_ch[i])}")
        print()

    # 打印词汇表信息
    print(f"中文词汇表大小: {len(vocab_ch)}")
    print(f"英文词汇表大小: {len(vocab_en)}")
    print(f"特殊标记: {vocab_ch.get_itos()[:4]}")  # 显示前4个特殊标记

    train_data = data_process(train_ch, train_en, vocab_ch, vocab_en)
    val_data = data_process(val_ch, val_en, vocab_ch, vocab_en)  # 添加验证集

    BATCH_SIZE = 8

    train_iter = DataLoader(train_data, batch_size=BATCH_SIZE,
                            shuffle=True, collate_fn=generate_batch)

    # 首先固定随机数种子，方便后续复现结果
    torch.manual_seed(42)

    # 模型参数配置
    SRC_VOCAB_SIZE = len(vocab_ch)  # 中文词汇表大小
    TGT_VOCAB_SIZE = len(vocab_en)  # 英文词汇表大小
    D_MODEL = 256  # 模型维度
    N_LAYERS = 3  # 编码器/解码器层数
    N_HEADS = 4  # 多头注意力头数
    D_FF = 1024  # 前馈网络维度
    MAX_LEN = 50  # 最大序列长度
    DROPOUT = 0.2  # dropout率
    PAD_IDX = 1  # 填充符索引（根据Vocabulary定义，<pad>是索引1）

    # 验证参数
    assert D_MODEL % N_HEADS == 0, f"d_model({D_MODEL})必须能被n_heads({N_HEADS})整除"

    # 初始化模型
    model = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        max_len=MAX_LEN,
        dropout=DROPOUT
    )

    # 消融实验模型
    # model = TransformerNoFFN(
    #     src_vocab_size=SRC_VOCAB_SIZE,
    #     tgt_vocab_size=TGT_VOCAB_SIZE,
    #     d_model=D_MODEL,
    #     n_layers=N_LAYERS,
    #     n_heads=N_HEADS,
    #     max_len=MAX_LEN,
    #     dropout=DROPOUT
    # )

    # 初始化模型参数
    model = initialize_model(model)

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params:,}")

    # 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    criterion = CrossEntropyLoss(ignore_index=PAD_IDX)  # 忽略填充位置的损失

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # 监控验证损失最小化
        factor=0.5,  # 学习率减半
        patience=3,  # 3个epoch没有改善就调整
        verbose=True,  # 打印调整信息
        min_lr=1e-6  # 最小学习率
    )

    # 训练设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model.to(device)

    # 训练参数
    NUM_EPOCHS = 100
    PRINT_EVERY = 100

    # 添加损失记录
    train_epoch_losses = []  # 记录每个epoch的平均训练损失
    val_epoch_losses = []  # 记录每个epoch的验证损失
    learning_rates = []  # 记录每个epoch的学习率

    # 模型配置字典
    model_config = {
        'src_vocab_size': SRC_VOCAB_SIZE,
        'tgt_vocab_size': TGT_VOCAB_SIZE,
        'd_model': D_MODEL,
        'n_layers': N_LAYERS,
        'n_heads': N_HEADS,
        'd_ff': D_FF,
        'max_len': MAX_LEN,
        'dropout': DROPOUT
    }

    type = 'train'
    if type == 'train':
        print("开始训练...")
        model.train()

        # 添加早停机制
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0

        for epoch in range(NUM_EPOCHS):
            start_time = time.time()
            epoch_total_loss = 0
            epoch_batch_count = 0

            # 用于打印的临时变量
            step_total_loss = 0
            step_count = 0

            for batch_id, (src, tgt) in enumerate(train_iter):
                src = src.to(device)
                tgt = tgt.to(device)

                # 创建掩码
                src_mask, tgt_mask = create_mask(src, tgt, PAD_IDX)

                # 目标序列处理
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                # 调整目标掩码以匹配输入序列长度
                tgt_mask = tgt_mask[:, :, :tgt_input.size(1), :tgt_input.size(1)]

                # 前向传播
                optimizer.zero_grad()
                output = model(src, tgt_input, src_mask, tgt_mask)

                # 计算损失
                loss = criterion(output.reshape(-1, TGT_VOCAB_SIZE), tgt_output.reshape(-1))

                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                current_loss = loss.item()
                epoch_total_loss += current_loss
                epoch_batch_count += 1
                step_total_loss += current_loss
                step_count += 1

                # 打印训练信息
                if (batch_id + 1) % PRINT_EVERY == 0:
                    avg_loss = step_total_loss / step_count
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f'Epoch: {epoch + 1:02d} | Batch: {batch_id + 1:04d} | '
                          f'Loss: {avg_loss:.4f} | LR: {current_lr:.6f}')
                    step_total_loss = 0
                    step_count = 0

            # 每个epoch结束后计算平均训练损失
            epoch_train_loss = epoch_total_loss / epoch_batch_count if epoch_batch_count > 0 else 0
            train_epoch_losses.append(epoch_train_loss)  # 只记录epoch平均损失

            # 在验证集上评估
            epoch_val_loss = validate_model(model, val_data, criterion, device, PAD_IDX, vocab_ch, vocab_en)
            val_epoch_losses.append(epoch_val_loss)

            # 记录当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)

            epoch_time = time.time() - start_time

            print(f'Epoch: {epoch + 1:02d} completed | Time: {epoch_time:.2f}s | '
                  f'Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | '
                  f'LR: {current_lr:.6f}')

            # 早停检查
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0

                # 保存最佳模型
                checkpoint_data = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': epoch_train_loss,
                    'val_loss': epoch_val_loss,
                    'best_val_loss': best_val_loss,
                    'train_losses': train_epoch_losses.copy(),
                    'val_losses': val_epoch_losses.copy(),
                    'learning_rates': learning_rates.copy(),
                    'vocab_ch': vocab_ch,
                    'vocab_en': vocab_en,
                    'model_config': model_config
                }
                save_model_checkpoint(checkpoint_data, './ckpt/best_model.pth', "最佳模型")

            else:
                patience_counter += 1
                print(f"验证损失未改善: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"早停触发! 在epoch {epoch + 1}停止训练")
                print(f"最佳验证损失: {best_val_loss:.4f}")
                break

            # 更新学习率
            scheduler.step(epoch_val_loss)  # 基于验证损失调整学习率

            # 实时更新损失曲线和学习率曲线（每5个epoch保存一次）
            if (epoch + 1) % 5 == 0:
                plot_loss_curve(train_epoch_losses, val_epoch_losses, f'./result/loss_curve_epoch_{epoch + 1}.png')
                plot_lr_curve(learning_rates, f'./result/lr_curve_epoch_{epoch + 1}.png')

            # 保存模型检查点（每5个epoch保存一次）
            if (epoch + 1) % 5 == 0:
                checkpoint_data = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': epoch_train_loss,
                    'val_loss': epoch_val_loss,
                    'train_losses': train_epoch_losses.copy(),
                    'val_losses': val_epoch_losses.copy(),
                    'learning_rates': learning_rates.copy(),
                    'vocab_ch': vocab_ch,
                    'vocab_en': vocab_en,
                    'model_config': model_config
                }
                save_model_checkpoint(checkpoint_data, f'./ckpt/epoch_{epoch + 1}.pth', f"第{epoch + 1}轮检查点")

        print("训练完成!")

        # 绘制最终的损失曲线和学习率曲线
        plot_loss_curve(train_epoch_losses, val_epoch_losses, './result/final_loss_curve.png')
        plot_lr_curve(learning_rates, './result/final_lr_curve.png')

        # 保存最终模型
        final_checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_epoch_losses[-1] if train_epoch_losses else 0,
            'val_loss': val_epoch_losses[-1] if val_epoch_losses else 0,
            'train_losses': train_epoch_losses.copy(),
            'val_losses': val_epoch_losses.copy(),
            'learning_rates': learning_rates.copy(),
            'vocab_ch': vocab_ch,
            'vocab_en': vocab_en,
            'model_config': model_config
        }
        save_model_checkpoint(final_checkpoint_data, './ckpt/final_model.pth', "最终模型")

        # 打印训练总结
        print("\n 训练总结:")
        print(f"   最终训练损失: {train_epoch_losses[-1]:.4f}")
        print(f"   最终验证损失: {val_epoch_losses[-1]:.4f}")
        print(f"   最佳验证损失: {best_val_loss:.4f}")
        print(f"   最终学习率: {learning_rates[-1]:.6f}")
        print(f"   总训练轮数: {len(train_epoch_losses)}")