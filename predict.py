# inference.py
import torch
import torch.nn as nn
from Transformer import Transformer
# from model1 import TransformerNoPE
# from model2 import TransformerSingleHead
# from model3 import TransformerNoFFN
import os
import jieba
import nltk
from nltk.tokenize import word_tokenize

# 下载nltk所需数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class Translator:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

        # 加载检查点
        checkpoint = torch.load(model_path, map_location=device)

        # 提取词汇表
        if 'vocab_ch' in checkpoint and 'vocab_en' in checkpoint:
            self.vocab_ch = checkpoint['vocab_ch']
            self.vocab_en = checkpoint['vocab_en']
            print("词汇表加载成功")
        else:
            raise ValueError("检查点中未找到词汇表")

        # 提取模型配置
        if 'model_config' in checkpoint:
            self.model_config = checkpoint['model_config']
        else:
            self.model_config = {
                'src_vocab_size': len(self.vocab_ch),
                'tgt_vocab_size': len(self.vocab_en),
                'd_model': 256,
                'n_layers': 3,
                'n_heads': 4,
                'd_ff': 1024,
                'max_len': 50,
                'dropout': 0.2
            }
        # self.model_config = {
        #     'src_vocab_size': len(self.vocab_ch),
        #     'tgt_vocab_size': len(self.vocab_en),
        #     'd_model': 256,
        #     'n_layers': 3,
        #     'n_heads': 4,
        #     'max_len': 50,
        #     'dropout': 0.2
        # }

        print(f"中文词汇表大小: {len(self.vocab_ch)}")
        print(f"英文词汇表大小: {len(self.vocab_en)}")

        # 初始化模型
        self.model = Transformer(**self.model_config)
        # self.model = TransformerNoFFN(**self.model_config)

        # 加载权重
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(device)
        self.model.eval()

        # 特殊标记
        self.PAD_IDX = 1
        self.BOS_IDX = 2
        self.EOS_IDX = 3

        print("翻译器初始化完成！")

    def chinese_tokenizer(self, text):
        """使用与训练时相同的中文分词器（jieba）"""
        tokens = jieba.cut(text, cut_all=False)
        return list(tokens)

    def english_tokenizer(self, text):
        """使用与训练时相同的英文分词器（nltk）"""
        tokens = word_tokenize(text.lower())
        return tokens

    def translate(self, chinese_text, max_length=50):
        """将中文翻译成英文"""
        # 使用与训练时相同的分词方式
        tokens = self.chinese_tokenizer(chinese_text)
        if not tokens:
            return "输入为空"

        print(f"中文分词: {tokens}")

        # 转换为索引
        indices = []
        for token in tokens:
            if token in self.vocab_ch:
                indices.append(self.vocab_ch[token])
                print(f"  '{token}' -> {self.vocab_ch[token]}")
            else:
                indices.append(self.vocab_ch['<unk>'])
                print(f"  '{token}' -> <unk> ({self.vocab_ch['<unk>']})")

        # 创建源序列 [BOS, tokens, EOS]
        src = torch.tensor([self.BOS_IDX] + indices + [self.EOS_IDX], device=self.device).unsqueeze(0)
        print(f"源序列: {src[0].tolist()}")

        # 初始化目标序列
        tgt_indices = [self.BOS_IDX]

        with torch.no_grad():
            # 编码
            src_mask = (src != self.PAD_IDX).unsqueeze(1).unsqueeze(2)
            enc_output = self.model.encoder(src, src_mask)

            # 逐步解码
            for i in range(max_length):
                tgt = torch.tensor(tgt_indices, device=self.device).unsqueeze(0)

                # 创建目标掩码
                tgt_len = len(tgt_indices)
                tgt_mask = (tgt != self.PAD_IDX).unsqueeze(1).unsqueeze(2)
                tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=self.device)).bool()
                tgt_sub_mask = tgt_sub_mask.unsqueeze(0).unsqueeze(0)
                tgt_mask = tgt_mask & tgt_sub_mask

                # 解码
                dec_output = self.model.decoder(tgt, enc_output, src_mask, tgt_mask)
                output = self.model.linear(dec_output)

                # 取最后一个token的预测
                next_token_logits = output[0, -1, :]
                next_token_probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.argmax(next_token_logits, dim=-1).item()

                # 调试信息
                if i == 0:
                    topk_probs, topk_indices = torch.topk(next_token_probs, 5)
                    print("第一步预测top5:")
                    for j, (idx, prob) in enumerate(zip(topk_indices.tolist(), topk_probs.tolist())):
                        token_str = self.vocab_en.get_itos()[idx] if idx < len(
                            self.vocab_en.get_itos()) else f'<out_of_range:{idx}>'
                        print(f"  {j + 1}. {token_str} (prob: {prob:.4f}, idx: {idx})")

                token_str = self.vocab_en.get_itos()[next_token] if next_token < len(
                    self.vocab_en.get_itos()) else f'<out_of_range:{next_token}>'
                print(f"第{i + 1}步预测: 索引={next_token}, token='{token_str}'")

                # 遇到EOS停止
                if next_token == self.EOS_IDX:
                    print("遇到EOS，停止生成")
                    break

                tgt_indices.append(next_token)

                # 长度限制
                if len(tgt_indices) >= max_length:
                    print("达到最大长度，停止生成")
                    break

        # 转换为英文文本
        english_tokens = []
        for idx in tgt_indices[1:]:  # 跳过BOS
            if idx == self.EOS_IDX:
                break
            if idx < len(self.vocab_en.get_itos()):
                token = self.vocab_en.get_itos()[idx]
                english_tokens.append(token)
            else:
                english_tokens.append(f'<out_of_range:{idx}>')

        result = ' '.join(english_tokens)
        print(f"最终结果: {result}")
        return result


def main():
    model_path = './ckpt/best_model.pth'

    if not os.path.exists(model_path):
        print(f"错误：模型文件 {model_path} 不存在！")
        return

    try:
        translator = Translator(model_path)
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    print("=" * 50)
    print("中文到英文翻译器")
    print("输入 'quit' 退出")
    print("=" * 50)

    while True:
        try:
            text = input("\n请输入中文句子: ").strip()

            if text.lower() in ['quit', 'exit', '退出']:
                break

            if not text:
                continue

            print("翻译中...")
            result = translator.translate(text)
            print(f"英文翻译: {result}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"错误: {e}")


if __name__ == "__main__":
    main()