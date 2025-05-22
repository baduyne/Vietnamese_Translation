import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import math
import random
import torch.nn as nn
import numpy as np
import pandas as pd
from torch import optim
from tqdm import tqdm

import torch.autograd as Variable
from sklearn.model_selection import train_test_split


def Create_Dictionary(src_sents, trg_sents):
    src_words = set()
    trg_words = set()

    # create vocab
    for sent in src_sents:
        src_words.update(sent.split()) # update function allows us to add many items in the same thing

    for sent in trg_sents:
        trg_words.update(sent.split())

    #
    src_vocab = {"<sos>":0, "<eos>": 1, "<pad>":2, "<unk>":3}
    for i, word in enumerate(src_words, start= 4):
        src_vocab[word] = i


    trg_vocab = {"<sos>":0, "<eos>": 1, "<pad>":2, "<unk>": 3}
    for i, word in enumerate(trg_words, start = 4):
        trg_vocab[word] = i

    return  src_vocab, trg_vocab




class Read_Dataset(Dataset):
    def __init__(self, src_sents, tgt_sents, src_vocab, trg_vocab, max_leng = 50):
        super().__init__()

        self.src = src_sents
        self.tgt = tgt_sents
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_leng = max_leng

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        src_sentence = self.src[index]
        trg_sentence = self.tgt[index]

        src_indices = [self.src_vocab['<sos>']] + [self.src_vocab.get(word.lower(), self.src_vocab['<unk>']) for word in src_sentence.split()] + [self.src_vocab['<eos>']]
        trg_indices = [self.trg_vocab['<sos>']] + [self.trg_vocab.get(word.lower(), self.trg_vocab['<unk>']) for word in trg_sentence.split()] + [self.trg_vocab['<eos>']]

        # Sửa lại padding để đảm bảo đúng max_length
        src_indices = src_indices[:self.max_leng] + [self.src_vocab['<pad>']] * max(0, self.max_leng - len(src_indices))
        trg_indices = trg_indices[:self.max_leng] + [self.trg_vocab['<pad>']] * max(0, self.max_leng - len(trg_indices))

        return torch.LongTensor(src_indices), torch.LongTensor(trg_indices)

class Embedding_Layer(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)



class Position_Layer(nn.Module):
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)





def Attention(q,k,v, mask = None, dropout = None):

    # q,k,v shape [batch_size, seq_leng, d_model]
    d_model = q.size(-1)
    score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_model)

    if mask is not None:
        score = score.masked_fill(mask ==0, -1e10)

    sort_max = F.softmax(score, -1)

    if dropout is not None:
        sort_max = dropout(sort_max)

    return torch.matmul(sort_max,v), sort_max




# Multi-Head Attention
class Multihead_Attention(nn.Module):
    def __init__(self, head, d_model, dropout=0.1):
        super().__init__()
        assert d_model % head == 0
        self.d_k = d_model // head
        self.h = head

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Linear projections
        q = self.q_linear(q).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        k = self.k_linear(k).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        # Adjust mask shape
        if mask is not None:
            # mask: [batch_size, 1, 1, seq_len] -> [batch_size, h, seq_len, seq_len]
            mask = mask.unsqueeze(1)  # Add head dimension
            # This ensures it can be broadcasted over attention scores

        # Calculate attention
        scores, attn = Attention(q, k, v, mask=mask, dropout=self.dropout)

        # Concatenate heads
        concat = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        # Final linear layer
        output = self.out(concat)
        return output




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




class FeedForward(nn.Module):
    """ Trong kiến trúc của chúng ta có tầng linear
    """
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Encoder_layer(nn.Module):
    def __init__(self, d_model, head, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Layer normalization
        self.norm1 = Norm(d_model)
        self.norm2 = Norm(d_model)

        # Multi-head attention
        self.multi_head = Multihead_Attention(head=head, d_model=d_model, dropout=dropout)

        # Feed-forward network
        self.ffn = FeedForward(d_model, dropout=dropout)

        # Dropout layers
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
      x2 = self.norm1(x)
      x = x + self.dropout_1(self.multi_head(x2, x2, x2, mask))
      x2 = self.norm2(x)
      x = x + self.dropout_2(self.ffn(x2))
      return x


class Decoder_Layer(nn.Module):
    def __init__(self, d_model, head, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # 3 norm layers
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        # 3 dropout layers
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        # 2 multi-head attention blocks
        self.attn_1 = Multihead_Attention(head, d_model, dropout=dropout)  
        self.attn_2 = Multihead_Attention(head, d_model, dropout=dropout)

        # Feedforward network
        self.ffn = FeedForward(d_model, dropout=dropout)

    def forward(self, x, encoder_output, src_mask, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, encoder_output, encoder_output, src_mask))
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ffn(x2))
        return x


import copy

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Encoder(nn.Module):
    """Một encoder có nhiều encoder layer nhé !!!
    """
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedding_Layer(vocab_size, d_model)
        self.pe = Position_Layer(d_model, dropout=dropout)
        self.layers = get_clones(Encoder_layer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        """
        src: batch_size x seq_length
        mask: batch_size x 1 x seq_length
        output: batch_size x seq_length x d_model
        """
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedding_Layer(vocab_size, d_model)
        self.pe = Position_Layer(d_model, dropout=dropout)
        self.layers = get_clones(Decoder_Layer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)


# Mask functions
def Nopeak_Mask(size, device):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask = (torch.from_numpy(np_mask) == 0).to(device)
    return np_mask

def Create_Masks(src, trg, src_pad, trg_pad, device):
    src_mask = (src != src_pad).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != trg_pad).unsqueeze(-2)
        size = trg.size(1)
        np_mask = Nopeak_Mask(size, device)
        trg_mask = trg_mask & np_mask
    else:
        trg_mask = None

    return src_mask, trg_mask





# Transformer Model
class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout = dropout)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output

def train_model(model, train_loader, val_loader,optimizer, criterion, device, num_epochs=10, batch_size=32):

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

            src_mask, trg_mask = Create_Masks(src_batch, trg_input, 2, 2 , device)

            output = model(src_batch, trg_input, src_mask, trg_mask)

            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg_output = trg_output.contiguous().view(-1)

            loss = criterion(output, trg_output)

            loss.backward()
            optimizer.step()


            train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0

        with torch.no_grad():
            for src_batch, tgt_batch in tqdm(val_loader, desc=f"Validating Epoch {epoch+1}"):
                src_batch, tgt_batch = src_batch.long().to(device), tgt_batch.long().to(device)

                trg_input = tgt_batch[:, :-1]
                trg_output = tgt_batch[:, 1:]

                src_mask, trg_mask = Create_Masks(src_batch, trg_input, 2, 2 , device)

                output = model(src_batch, trg_input, src_mask, trg_mask)

                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                trg_output = trg_output.contiguous().view(-1)

                loss = criterion(output, trg_output)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_transformer_model.pt")
            print("Saved best model.")


def main():
    path = "data/translation_dataset.csv"
    # path = "/content/drive/MyDrive/Dataset_of_Colab/translation_dataset.csv"


    df = pd.read_csv(path)

    src_vocab, trg_vocab = Create_Dictionary(df["English"].values, df["Vietnamese"].values)
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_dataset = Read_Dataset(train_df["English"].values, train_df["Vietnamese"].values, src_vocab, trg_vocab, max_leng=200)
    val_dataset = Read_Dataset(val_df["English"].values, val_df["Vietnamese"].values, src_vocab, trg_vocab, max_leng=200)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer(
        len(src_vocab),
        len(trg_vocab),
        128,
        4,
        4,
        0.1,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr = 0.001,  betas=(0.9, 0.98))


    criterion = nn.CrossEntropyLoss(ignore_index = trg_vocab["<pad>"]) # trg_vocab["<pad>"] equals 2
    train_model(model, train_loader, val_loader, optimizer, criterion, device)


if __name__ == "__main__":
    main()
