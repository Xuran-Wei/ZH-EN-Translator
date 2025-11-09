# 完成数据集的准备和预处理任务
import pandas as pd
from sklearn.model_selection import train_test_split
import jieba
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
from typing import List, Dict, Optional
from torch.nn.utils.rnn import pad_sequence
import torch
import zhconv
import re

# 下载nltk所需数据（第一次运行需要）
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


def contains_chinese(text):
    """检查文本是否包含中文字符"""
    return bool(re.search(r'[\u4e00-\u9fff]', text))


def split_english_chinese(line):
    """分割英文和中文部分 - 针对制表符分隔的格式"""
    line = line.strip()
    if not line:
        return None, None

    # 方法1：使用制表符分割（主要方法）
    if '\t' in line:
        parts = line.split('\t', 1)  # 只分割第一个制表符
        if len(parts) == 2:
            en_part, ch_part = parts[0].strip(), parts[1].strip()
            # 验证英文部分不为空，中文部分包含中文
            if en_part and ch_part and contains_chinese(ch_part):
                return en_part, ch_part

    # 方法2：如果制表符分割失败，尝试其他分隔符
    separators = ['\t', '  ', '    ']  # 制表符，多个空格
    for sep in separators:
        if sep in line:
            parts = line.split(sep, 1)
            if len(parts) == 2:
                en_part, ch_part = parts[0].strip(), parts[1].strip()
                if en_part and ch_part and contains_chinese(ch_part):
                    return en_part, ch_part

    # 方法3：最后尝试智能分割
    return split_english_chinese_fallback(line)


def split_english_chinese_fallback(line):
    """备用分割方法"""
    # 查找第一个中文字符的位置
    chinese_match = re.search(r'[\u4e00-\u9fff]', line)
    if not chinese_match:
        return None, None

    first_chinese_idx = chinese_match.start()

    # 英文部分是第一个中文字符之前的所有内容
    english_part = line[:first_chinese_idx].strip()
    chinese_part = line[first_chinese_idx:].strip()

    # 清理英文部分末尾的标点符号
    english_part = re.sub(r'[\.\s]*$', '', english_part)

    if english_part and chinese_part:
        return english_part, chinese_part

    return None, None


def LoadDataset(train_path, test_path, train_size=None):
    """分别读取训练集和测试集txt文件"""

    def read_txt_file(file_path, max_size=None):
        """读取txt文件，返回英文和中文列表"""
        en_list = []
        ch_list = []
        skipped_lines = 0

        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                # 使用分割函数
                en, ch = split_english_chinese(line)

                if en is None or ch is None:
                    skipped_lines += 1
                    if skipped_lines <= 3:  # 只打印前3个跳过的行
                        print(f"跳过第{line_num}行（无法分割）: {repr(line)}")
                    continue

                # 繁体转简体
                ch_simplified = zhconv.convert(ch, 'zh-cn')
                en_list.append(en)
                ch_list.append(ch_simplified)

                # 限制数据大小
                if max_size and len(en_list) >= max_size:
                    break

        if skipped_lines > 0:
            print(f"文件 {file_path} 总共跳过了 {skipped_lines} 行无法分割的数据")

        return en_list, ch_list

    # 读取训练数据
    print("正在读取训练数据...")
    train_en, train_ch = read_txt_file(train_path, train_size)

    # 读取测试数据
    print("正在读取测试数据...")
    test_en, test_ch = read_txt_file(test_path)

    print(f"训练集大小: {len(train_en)}")
    print(f"测试集大小: {len(test_en)}")

    # 从训练集中划分验证集（20%）
    train_ch, val_ch, train_en, val_en = train_test_split(
        train_ch, train_en, test_size=0.2, random_state=42
    )

    print(f"划分后 - 训练集: {len(train_en)}, 验证集: {len(val_en)}, 测试集: {len(test_en)}")

    return train_ch, val_ch, train_en, val_en, test_ch, test_en


# 中文分词器（使用jieba）
def chinese_tokenizer(text):
    """使用jieba进行中文分词"""
    tokens = jieba.cut(text, cut_all=False)
    return list(tokens)


# 英文分词器（使用nltk）
def english_tokenizer(text):
    """使用nltk进行英文分词"""
    tokens = word_tokenize(text.lower())
    return tokens


# 构建词汇表
class Vocabulary:
    def __init__(self, counter: Counter, specials: List[str] = None, min_freq: int = 1, max_size: int = None):
        if specials is None:
            specials = ['<unk>', '<pad>', '<bos>', '<eos>']

        self.specials = specials
        self.unk_index = 0

        self.itos = specials.copy()
        sorted_tokens = []
        for token, freq in counter.most_common():
            if freq >= min_freq:
                sorted_tokens.append(token)
            if max_size and len(sorted_tokens) >= (max_size - len(specials)):
                break

        for token in sorted_tokens:
            if token not in self.itos:
                self.itos.append(token)

        self.stoi = {token: idx for idx, token in enumerate(self.itos)}

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.unk_index)

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, token: str) -> bool:
        return token in self.stoi

    def lookup_indices(self, tokens: List[str]) -> List[int]:
        return [self[token] for token in tokens]

    def lookup_tokens(self, indices: List[int]) -> List[str]:
        return [self.itos[idx] if idx < len(self.itos) else self.itos[self.unk_index]
                for idx in indices]

    def get_stoi(self) -> Dict[str, int]:
        return self.stoi.copy()

    def get_itos(self) -> List[str]:
        return self.itos.copy()


def build_vocab(sentences, tokenizer, min_freq=1, max_size=50000):
    counter = Counter()
    for sentence in sentences:
        tokens = tokenizer(sentence)
        counter.update(tokens)

    return Vocabulary(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'],
                      min_freq=min_freq, max_size=max_size)


def data_process(ch_texts, en_texts, vocab_ch, vocab_en):
    data = []
    for raw_ch, raw_en in zip(ch_texts, en_texts):
        raw_ch = str(raw_ch).strip()
        raw_en = str(raw_en).strip()

        ch_tokens = chinese_tokenizer(raw_ch)
        ch_indices = [vocab_ch[token] for token in ch_tokens]
        ch_tensor = torch.tensor(ch_indices, dtype=torch.long)

        en_tokens = english_tokenizer(raw_en)
        en_indices = [vocab_en[token] for token in en_tokens]
        en_tensor = torch.tensor(en_indices, dtype=torch.long)

        data.append((ch_tensor, en_tensor))

    return data


def generate_batch(data_batch):
    PAD_IDX = 1
    BOS_IDX = 2
    EOS_IDX = 3

    ch_batch, en_batch = [], []
    for (ch_item, en_item) in data_batch:
        ch_batch.append(torch.cat([torch.tensor([BOS_IDX]), ch_item, torch.tensor([EOS_IDX])], dim=0))
        en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))

    ch_batch = pad_sequence(ch_batch, padding_value=PAD_IDX, batch_first=True)
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX, batch_first=True)

    return ch_batch, en_batch


# 测试函数
def test_split_function():
    """测试分割函数的效果"""
    test_cases = [
        "Anyone can do that.\t任何人都可以做到。",
        "Hello world!\t你好世界！",
        "This is a test.\t这是一个测试。",
        "How are you?\t你好吗？",
        "I love programming.\t我喜欢编程。"
    ]

    print("=== 测试分割函数 ===")
    for i, case in enumerate(test_cases, 1):
        en, ch = split_english_chinese(case)
        print(f"测试 {i}:")
        print(f"  原始: {repr(case)}")
        print(f"  英文: {repr(en)}")
        print(f"  中文: {repr(ch)}")
        print()


if __name__ == "__main__":
    test_split_function()