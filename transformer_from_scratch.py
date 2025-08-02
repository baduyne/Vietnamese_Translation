import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import math
import random
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
import numpy as np
import pandas as pd
from torch import optim
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch.autograd as Variable
from sklearn.model_selection import train_test_split
import argparse
import copy
import os
from transformers.models.t5.modeling_t5 import T5Attention 


Refered_Model = "VietAI/vit5-base"




Refered_Model = "VietAI/vit5-base"


# ==== Embedding & Positional ====
class Embedding_Layer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) / math.sqrt(self.d_model)


class Position_Layer(nn.Module):
    def __init__(self, max_len, d_model, dropout=0.1):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
        x = x + self.pe(pos)
        return self.dropout(x)


# ==== Layer Norm ====
class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.a = nn.Parameter(torch.ones(d_model))
        self.b = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b


# ==== FeedForward ====
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.gate = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x_ff = self.linear_1(x)
        x_gate = self.gate(x)
        x = F.silu(x_ff) * x_gate
        x = self.linear_2(self.dropout(x))
        return x


# ==== Relative Position Bias ====
class RelativePositionBias(nn.Module):
    def __init__(self, num_heads, num_buckets=32, max_distance=200):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    def _relative_position_bucket(self, relative_position):
        relative_position = torch.clamp(relative_position, min=1)  # tránh log(0)
        num_buckets = self.num_buckets // 2
        relative_buckets = (relative_position > 0).long() * num_buckets
        relative_position = torch.abs(relative_position)
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(self.max_distance / max_exact)
            * (num_buckets - max_exact)
        ).long()
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def forward(self, seq_len, device):
        context_position = torch.arange(seq_len, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(seq_len, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        rp_bucket = self._relative_position_bucket(relative_position)
        values = self.relative_attention_bias(rp_bucket)
        return values.permute(2, 0, 1).unsqueeze(0)  # [1,H,T,T]


# ==== Multi-Query Attention ====
class MultiQuery_attention(T5Attention):
    def __init__(self, num_heads, d_model, dropout_rate, num_buckets=32, max_distance=128):
        super().__init__()
        self.head = num_heads
        self.d_model = d_model
        self.d_h = self.d_model // self.head
        self.q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k = nn.Linear(self.d_model, self.d_h, bias=False)
        self.v = nn.Linear(self.d_model, self.d_h, bias=False)
        self.o = nn.Linear(self.d_model, self.d_model, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hidden_states, mask=None, key_value_states=None, is_causal=False, ):
        batch_size, seq_len = hidden_states.shape[:2]
        device = hidden_states.device

        q = self.q(hidden_states)
        key_value_encoder = hidden_states if key_value_states is None else key_value_states
        k = self.k(key_value_encoder)
        v = self.v(key_value_encoder)

        # Reshape Q: [B, H, T, D/H]
        q = q.view(batch_size, seq_len, self.head, self.d_h).transpose(1, 2)  # [B,H,T,D/H]
        # MQA -> K,V shared across heads
        k = k.unsqueeze(1).expand(-1, self.head, -1, -1)  # [B,H,T,D/H]
        v = v.unsqueeze(1).expand(-1, self.head, -1, -1)

        # === Flash Attention API ===
        if mask is not None:
            # mask [B, 1, 1, T] or [B,1,T,T], convert về bool
            if mask.dtype != torch.bool:
                mask = mask > 0
            attn_mask = mask  # [B,1,1,T] cho encoder, [B,1,T,T] cho decoder causal
        else:
            attn_mask = None

        try:
            # PyTorch >= 2.0 hỗ trợ scaled_dot_product_attention
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=is_causal
            )
        except RuntimeError:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_h)
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, float("-inf"))
            attn_weights = F.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn_weights, v)

        # Merge heads
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        attn_output = self.o(attn_output)
        return self.dropout(attn_output)


# ==== Encoder/Decoder Layers ====
class Encoder_layer(nn.Module):
    def __init__(self, d_model, head, dropout=0.1):
        super().__init__()
        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)
        self.multi_head = MultiQuery_attention(head, d_model, dropout)
        self.ffn = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x2 = self.norm1(x)
        x = x + self.dropout_1(self.multi_head(x2, mask))
        x2 = self.norm2(x)
        x = x + self.dropout_2(self.ffn(x2))
        return x


class Decoder_Layer(nn.Module):
    def __init__(self, d_model, head, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        self.attn = MultiQuery_attention(head, d_model, dropout)
        self.cross_attention = MultiQuery_attention(head, d_model, dropout)
        self.ffn = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.cross_attention(x2, src_mask, encoder_output))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ffn(x2))
        return x


# ==== Encoder/Decoder ====
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout, max_len=200):
        super().__init__()
        self.N = N
        self.embed = Embedding_Layer(vocab_size, d_model)
        self.pe = Position_Layer(max_len, d_model, dropout=dropout)
        self.layers = get_clones(Encoder_layer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout, max_len=200):
        super().__init__()
        self.N = N
        self.embed = Embedding_Layer(vocab_size, d_model)
        self.pe = Position_Layer(max_len, d_model, dropout=dropout)
        self.layers = get_clones(Decoder_Layer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


# ==== Mask ====
def Nopeak_Mask(size, device):
    mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
    np_mask = mask.masked_fill(mask == 1, float("-inf")).unsqueeze(0)
    return np_mask


def Create_Masks(src, trg, src_pad, trg_pad, device):
    src_mask = (src != src_pad).unsqueeze(1).unsqueeze(2)
    trg_mask = (trg != trg_pad).unsqueeze(1).unsqueeze(2)
    size = trg.size(1)
    np_mask = Nopeak_Mask(size, device)
    trg_mask = trg_mask & (np_mask == 0)
    return src_mask, trg_mask


# ==== Transformer ====
class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        return self.out(d_output)


def load_tokenizer(Refered_Model):
    tokenizer = AutoTokenizer.from_pretrained(Refered_Model)
    return tokenizer

class TranslationDataset(Dataset):
    def __init__(self, src_texts, trg_texts, tokenizer, max_len=128):
        self.src_texts = src_texts
        self.trg_texts = trg_texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_enc = self.tokenizer(
            self.src_texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        trg_enc = self.tokenizer(
            self.trg_texts[idx],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return src_enc["input_ids"].squeeze(0), trg_enc["input_ids"].squeeze(0)


def train_model(model, train_loader, val_loader, optimizer, criterion, device, pad_id, tokenizer, num_epochs=40):
    
    model = model.to(device)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        for src_batch, tgt_batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            src_batch, tgt_batch = src_batch.long().to(device), tgt_batch.long().to(device)

            optimizer.zero_grad()

            trg_input = tgt_batch[:, :-1]
            trg_output = tgt_batch[:, 1:]

            src_mask, trg_mask = Create_Masks(src_batch, trg_input, pad_id, pad_id, device)
            output = model(src_batch, trg_input, src_mask, trg_mask)

            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg_output = trg_output.contiguous().view(-1)

            loss = criterion(output, trg_output)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)

        # -------- VALIDATION --------
        model.eval()
        val_loss = 0
        bleu_scores = []

        with torch.no_grad():
            for src_batch, tgt_batch in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}"):
                src_batch, tgt_batch = src_batch.long().to(device), tgt_batch.long().to(device)

                trg_input = tgt_batch[:, :-1]
                trg_output = tgt_batch[:, 1:]

                src_mask, trg_mask = Create_Masks(src_batch, trg_input, pad_id, pad_id , device)
                output = model(src_batch, trg_input, src_mask, trg_mask)

                output_dim = output.shape[-1]
                output_logits = output.contiguous().view(-1, output_dim)
                trg_output = trg_output.contiguous().view(-1)

                loss = criterion(output_logits, trg_output)
                val_loss += loss.item()

                # ---- BLEU Calculation ----
                pred_tokens = F.softmax(output, dim=-1).argmax(dim=-1)  # [B, T]
                for pred, target in zip(pred_tokens, tgt_batch[:, 1:]):  # ignore <bos>
                    pred_text = tokenizer.decode(pred, skip_special_tokens=True)
                    target_text = tokenizer.decode(target, skip_special_tokens=True)
                    ref = [target_text.split()]
                    hyp = pred_text.split()
                    bleu = sentence_bleu(ref, hyp, smoothing_function=SmoothingFunction().method1)
                    bleu_scores.append(bleu)

        avg_val_loss = val_loss / len(val_loader)
        avg_bleu = np.mean(bleu_scores)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, BLEU = {avg_bleu:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_model.pt")
            print("Saved best model.")


def get_args():
    parser = argparse.ArgumentParser(description="Transformer Training Script")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--d_model", type=int, default=512, help="Dimension of model")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--N", type=int, default=6, help="Number of encoder/decoder layers")
    return parser.parse_args()

def main():
    args = get_args()

    train_path = "data/train.csv"
    test_path = "data/test.csv"

    train_df = pd.read_csv(train_path)
    
    val_df = pd.read_csv(test_path)

    print(f"Train Size : {train_df.shape}, Val Size: {val_df.shape}")
    tokenizer = load_tokenizer(Refered_Model)
    pad_id = tokenizer.pad_token_id

    train_dataset = TranslationDataset(train_df["English"].values, train_df["Vietnamese"].values, tokenizer)
    val_dataset = TranslationDataset(val_df["English"].values, val_df["Vietnamese"].values, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Model is trained on {device}")
    
    model = Transformer(tokenizer.vocab_size, tokenizer.vocab_size, args.d_model, args.N, args.heads, 0.1).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    model_path = "best_model.pt"
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        print("Model loaded successfully!")
    else:
        print("No previous model found. Training from scratch.")

    train_model(model, train_loader, val_loader, optimizer, criterion, device, pad_id, tokenizer, num_epochs=40)


if __name__ == "__main__":
    main()

