"""
Module xử lý dữ liệu cho bài toán dịch máy Seq2Seq
- Thu thập và làm sạch dữ liệu
- Tokenization
- Xây dựng Vocabulary
- Padding/Truncation
- Tạo DataLoader
"""

import os
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pandas as pd
from typing import List, Tuple, Dict
import pickle
from tqdm import tqdm


class Vocabulary:
    """Xây dựng và quản lý từ điển"""
    
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = Counter()
        
        # Special tokens
        self.PAD_TOKEN = '<pad>'
        self.UNK_TOKEN = '<unk>'
        self.SOS_TOKEN = '<sos>'
        self.EOS_TOKEN = '<eos>'
        
        # Thêm special tokens vào từ điển
        self.add_word(self.PAD_TOKEN)
        self.add_word(self.UNK_TOKEN)
        self.add_word(self.SOS_TOKEN)
        self.add_word(self.EOS_TOKEN)
    
    def add_word(self, word):
        """Thêm từ vào từ điển"""
        if word not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def build_vocab(self, sentences: List[List[str]]):
        """Xây dựng từ điển từ danh sách câu"""
        # Đếm tần suất từ
        for sentence in sentences:
            for word in sentence:
                self.word_count[word] += 1
        
        # Thêm các từ có tần suất >= min_freq
        for word, count in self.word_count.items():
            if count >= self.min_freq:
                self.add_word(word)
    
    def encode(self, sentence: List[str], max_len: int = None) -> List[int]:
        """Chuyển câu thành chuỗi indices"""
        indices = [self.word2idx.get(word, self.word2idx[self.UNK_TOKEN]) 
                  for word in sentence]
        
        if max_len:
            if len(indices) > max_len:
                indices = indices[:max_len]
            else:
                indices.extend([self.word2idx[self.PAD_TOKEN]] * (max_len - len(indices)))
        
        return indices
    
    def decode(self, indices: List[int]) -> List[str]:
        """Chuyển chuỗi indices thành câu"""
        words = []
        for idx in indices:
            if idx == self.word2idx[self.EOS_TOKEN]:
                break
            if idx != self.word2idx[self.PAD_TOKEN]:
                words.append(self.idx2word.get(idx, self.UNK_TOKEN))
        return words
    
    def __len__(self):
        return len(self.word2idx)
    
    def save(self, path):
        """Lưu từ điển"""
        with open(path, 'wb') as f:
            pickle.dump({
                'word2idx': self.word2idx,
                'idx2word': self.idx2word,
                'word_count': dict(self.word_count),
                'min_freq': self.min_freq
            }, f)
    
    def load(self, path):
        """Tải từ điển"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.word2idx = data['word2idx']
            self.idx2word = data['idx2word']
            self.word_count = Counter(data['word_count'])
            self.min_freq = data['min_freq']


def normalize_text(text: str) -> str:
    """Chuẩn hóa văn bản"""
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r'\s+', ' ', text)
    # Loại bỏ khoảng trắng ở đầu và cuối
    text = text.strip()
    return text


def tokenize(text: str, lang='vi') -> List[str]:
    """Tokenization đơn giản (có thể thay thế bằng SentencePiece, spaCy, etc.)"""
    text = normalize_text(text.lower())
    # Tách theo khoảng trắng
    tokens = text.split()
    return tokens


class TranslationDataset(Dataset):
    """Dataset cho bài toán dịch máy"""
    
    def __init__(self, src_sentences: List[List[str]], tgt_sentences: List[List[str]],
                 src_vocab: Vocabulary, tgt_vocab: Vocabulary, max_len: int = 128):
        self.src_sentences = src_sentences
        self.tgt_sentences = tgt_sentences
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
    
    def __len__(self):
        return len(self.src_sentences)
    
    def __getitem__(self, idx):
        src_sentence = self.src_sentences[idx]
        tgt_sentence = self.tgt_sentences[idx]
        
        # Encode source sentence
        src_indices = [self.src_vocab.word2idx[self.src_vocab.SOS_TOKEN]] + \
                     self.src_vocab.encode(src_sentence, max_len=self.max_len-2) + \
                     [self.src_vocab.word2idx[self.src_vocab.EOS_TOKEN]]
        
        # Encode target sentence (input và output)
        tgt_input = [self.tgt_vocab.word2idx[self.tgt_vocab.SOS_TOKEN]] + \
                   self.tgt_vocab.encode(tgt_sentence, max_len=self.max_len-2)
        tgt_output = self.tgt_vocab.encode(tgt_sentence, max_len=self.max_len-2) + \
                    [self.tgt_vocab.word2idx[self.tgt_vocab.EOS_TOKEN]]
        
        # Đảm bảo cùng độ dài
        if len(tgt_input) < len(tgt_output):
            tgt_input.append(self.tgt_vocab.word2idx[self.tgt_vocab.PAD_TOKEN])
        elif len(tgt_input) > len(tgt_output):
            tgt_output.append(self.tgt_vocab.word2idx[self.tgt_vocab.PAD_TOKEN])
        
        return {
            'src': torch.tensor(src_indices, dtype=torch.long),
            'tgt_input': torch.tensor(tgt_input, dtype=torch.long),
            'tgt_output': torch.tensor(tgt_output, dtype=torch.long)
        }


def load_iwslt_data(data_dir='data', split='train', iwslt_dir='iwslt_en_vi'):
    """
    Tải dữ liệu IWSLT Vi-En từ file
    """
    src_sentences = []
    tgt_sentences = []
    
    # Map split names to file names
    split_map = {
        'train': ('train.vi', 'train.en'),
        'validation': ('tst2012.vi', 'tst2012.en'),
        'test': ('tst2013.vi', 'tst2013.en')
    }
    
    if split not in split_map:
        split = 'train'
    
    src_file = os.path.join(iwslt_dir, split_map[split][0])
    tgt_file = os.path.join(iwslt_dir, split_map[split][1])
    
    # Thử load từ file IWSLT
    if os.path.exists(src_file) and os.path.exists(tgt_file):
        print(f"Đang tải dữ liệu từ {src_file} và {tgt_file}...")
        with open(src_file, 'r', encoding='utf-8') as f_src, \
             open(tgt_file, 'r', encoding='utf-8') as f_tgt:
            for src_line, tgt_line in zip(f_src, f_tgt):
                src_line = src_line.strip()
                tgt_line = tgt_line.strip()
                if src_line and tgt_line:
                    src_sentences.append(tokenize(src_line, 'vi'))
                    tgt_sentences.append(tokenize(tgt_line, 'en'))
        
        print(f"Đã tải {len(src_sentences)} cặp câu từ IWSLT file")
        return src_sentences, tgt_sentences
    
    # Thử tải từ datasets library
    try:
        from datasets import load_dataset
        dataset = load_dataset('iwslt2017', 'iwslt2017-vi-en', split=split)
        
        for item in dataset:
            src_sentences.append(tokenize(item['translation']['vi'], 'vi'))
            tgt_sentences.append(tokenize(item['translation']['en'], 'en'))
        
        print(f"Đã tải {len(src_sentences)} cặp câu từ IWSLT dataset")
        return src_sentences, tgt_sentences
    
    except Exception as e:
        print(f"Không thể tải từ datasets: {e}")
        print("Tạo dữ liệu mẫu để demo...")
        
        # Tạo dữ liệu mẫu
        sample_data = [
            ("xin chào", "hello"),
            ("tôi là sinh viên", "i am a student"),
            ("hôm nay trời đẹp", "today is a beautiful day"),
            ("tôi thích học máy tính", "i like learning computer science"),
            ("bạn có khỏe không", "how are you"),
        ]
        
        for src, tgt in sample_data:
            src_sentences.append(tokenize(src, 'vi'))
            tgt_sentences.append(tokenize(tgt, 'en'))
        
        return src_sentences, tgt_sentences


def prepare_data(data_dir='data', max_len=128, min_freq=2, batch_size=32, 
                 train_split=0.8, val_split=0.1, iwslt_dir='iwslt_en_vi', num_workers=0):
    """
    Chuẩn bị dữ liệu hoàn chỉnh: tải, tokenize, build vocab, tạo dataloader
    """
    print("=" * 50)
    print("BƯỚC 1: Tải dữ liệu")
    print("=" * 50)
    
    # Kiểm tra cache
    processed_data_path = os.path.join(data_dir, 'processed_data.pkl')
    
    if os.path.exists(processed_data_path):
        print(f"✓ Đã tìm thấy cache tại {processed_data_path}. Đang tải...")
        with open(processed_data_path, 'rb') as f:
            cached_data = pickle.load(f)
            train_src = cached_data['train_src']
            train_tgt = cached_data['train_tgt']
            val_src = cached_data['val_src']
            val_tgt = cached_data['val_tgt']
            test_src = cached_data['test_src']
            test_tgt = cached_data['test_tgt']
    else:
        # Tải dữ liệu train
        train_src, train_tgt = load_iwslt_data(data_dir, split='train', iwslt_dir=iwslt_dir)
        
        # Tải dữ liệu validation và test (nếu có)
        try:
            val_src, val_tgt = load_iwslt_data(data_dir, split='validation', iwslt_dir=iwslt_dir)
            test_src, test_tgt = load_iwslt_data(data_dir, split='test', iwslt_dir=iwslt_dir)
        except:
            # Chia dữ liệu train thành train/val/test
            total = len(train_src)
            val_size = int(total * val_split)
            test_size = int(total * (1 - train_split - val_split))
            
            val_src = train_src[:val_size]
            val_tgt = train_tgt[:val_size]
            test_src = train_src[val_size:val_size+test_size]
            test_tgt = train_tgt[val_size:val_size+test_size]
            train_src = train_src[val_size+test_size:]
            train_tgt = train_tgt[val_size+test_size:]
        
        # Lưu cache
        print(f"Đang lưu cache vào {processed_data_path}...")
        with open(processed_data_path, 'wb') as f:
            pickle.dump({
                'train_src': train_src, 'train_tgt': train_tgt,
                'val_src': val_src, 'val_tgt': val_tgt,
                'test_src': test_src, 'test_tgt': test_tgt
            }, f)
        print("✓ Đã lưu cache dữ liệu.")
    
    print(f"Số lượng cặp câu train: {len(train_src)}")
    print(f"Số lượng cặp câu validation: {len(val_src)}")
    print(f"Số lượng cặp câu test: {len(test_src)}")
    
    print("\n" + "=" * 50)
    print("BƯỚC 2: Xây dựng Vocabulary")
    print("=" * 50)
    
    # Xây dựng hoặc tải vocabulary
    src_vocab_path = os.path.join(data_dir, 'src_vocab.pkl')
    tgt_vocab_path = os.path.join(data_dir, 'tgt_vocab.pkl')
    
    src_vocab = Vocabulary(min_freq=min_freq)
    tgt_vocab = Vocabulary(min_freq=min_freq)
    
    if os.path.exists(src_vocab_path) and os.path.exists(tgt_vocab_path):
        print(f"Đã tìm thấy vocabulary tại {data_dir}. Đang tải lại...")
        src_vocab.load(src_vocab_path)
        tgt_vocab.load(tgt_vocab_path)
        print(f"Đã tải Source vocabulary size: {len(src_vocab)}")
        print(f"Đã tải Target vocabulary size: {len(tgt_vocab)}")
    else:
        print("Không tìm thấy vocabulary có sẵn. Đang xây dựng mới...")
        # Xây dựng vocabulary cho source (Vietnamese)
        src_vocab.build_vocab(train_src)
        print(f"Source vocabulary size: {len(src_vocab)}")
        print(f"Top 10 từ phổ biến: {src_vocab.word_count.most_common(10)}")
        
        # Xây dựng vocabulary cho target (English)
        tgt_vocab.build_vocab(train_tgt)
        print(f"Target vocabulary size: {len(tgt_vocab)}")
        print(f"Top 10 từ phổ biến: {tgt_vocab.word_count.most_common(10)}")
        
        # Lưu vocabulary
        os.makedirs(data_dir, exist_ok=True)
        src_vocab.save(src_vocab_path)
        tgt_vocab.save(tgt_vocab_path)
    
    print("\n" + "=" * 50)
    print("BƯỚC 3: Tạo DataLoader")
    print("=" * 50)
    
    # Tạo datasets
    train_dataset = TranslationDataset(train_src, train_tgt, src_vocab, tgt_vocab, max_len)
    val_dataset = TranslationDataset(val_src, val_tgt, src_vocab, tgt_vocab, max_len)
    test_dataset = TranslationDataset(test_src, test_tgt, src_vocab, tgt_vocab, max_len)
    
    # Tạo dataloaders với tối ưu hóa
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Thống kê độ dài câu
    src_lens = [len(s) for s in train_src]
    tgt_lens = [len(s) for s in train_tgt]
    
    print("\n" + "=" * 50)
    print("THỐNG KÊ DỮ LIỆU")
    print("=" * 50)
    print(f"Độ dài câu source - Min: {min(src_lens)}, Max: {max(src_lens)}, Avg: {sum(src_lens)/len(src_lens):.2f}")
    print(f"Độ dài câu target - Min: {min(tgt_lens)}, Max: {max(tgt_lens)}, Avg: {sum(tgt_lens)/len(tgt_lens):.2f}")
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'src_vocab': src_vocab,
        'tgt_vocab': tgt_vocab,
        'train_stats': {
            'src_lens': src_lens,
            'tgt_lens': tgt_lens,
            'num_samples': len(train_src)
        }
    }


if __name__ == '__main__':
    # Test data processing
    data = prepare_data(data_dir='data', max_len=128, batch_size=32)
    print("\n✓ Hoàn thành xử lý dữ liệu!")

