import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    pack_padded_sequence, pad_packed_sequence,
)

class Encoder(nn.Module):
    def __init__(self, vocab_size = 5000, embedding_dim = 256, hidden_size = 256, layers = 3, dropout = 0.3):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=vocab_size, 
                                embedding_dim=embedding_dim, 
                                padding_idx=0)
        self.drop = nn.Dropout(dropout)
        self.rnn = nn.LSTM(input_size=embedding_dim, 
                           hidden_size=hidden_size, 
                           num_layers=layers, 
                           batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)

    def forward(self, src, lengths):
        packed = pack_padded_sequence(
            self.drop(self.emb(src)), lengths.cpu(),
            batch_first=True, enforce_sorted=False)
        packed_out, state = self.rnn(packed)
        memory, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=src.size(1))
        return memory, state


# Hai kiểu tính điểm: Bahdanau (cộng) và Luong general (nhân)
class Attention(nn.Module):
    def __init__(self, hidden, mode):
        super().__init__()
        self.mode = mode
        if mode == 'bahdanau':
            self.Wq = nn.Linear(hidden, hidden, bias=False)
            self.Wk = nn.Linear(hidden, hidden, bias=False)
            self.v = nn.Linear(hidden, 1, bias=False)

        elif mode == 'luong':
            self.W = nn.Linear(hidden, hidden, bias=False)
        else:
            raise ValueError(mode)

    def forward(self, decoder_hidden, encoder_outputs, mask):
        if self.mode == 'bahdanau':
            energy = torch.tanh(self.Wq(decoder_hidden).unsqueeze(1)
                                + self.Wk(encoder_outputs))
            scores = self.v(energy).squeeze(-1)
        else:
            scores = torch.bmm(self.W(encoder_outputs),
                               decoder_hidden.unsqueeze(2)).squeeze(2)
            
        scores = scores.masked_fill(~mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights

# Decoder chạy từng bước; mode = 'none', 'bahdanau' hoặc 'luong'
class Decoder(nn.Module):
    def __init__(self, vocab_size, mode, embedding_dim, hidden_size, layers = 3, dropout = 0.3):
        super().__init__()
        self.mode = mode
        self.emb = nn.Embedding(num_embeddings=vocab_size, 
                                embedding_dim=embedding_dim, 
                                padding_idx=0)
        self.drop = nn.Dropout(dropout)
        width = embedding_dim if mode == 'none' else embedding_dim + hidden_size
        self.rnn = nn.LSTM(input_size=width, 
                           hidden_size=hidden_size, 
                           num_layers=layers, 
                           batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        if mode != 'none':
            self.attn = Attention(hidden_size, mode)
            self.fuse = nn.Linear(2 * hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size)

    def step(self, input_token, state, memory, mask, feeding):
        embedded = self.drop(self.emb(input_token))
        weights = None
        if self.mode == 'bahdanau':
            context, weights = self.attn(state[0][-1], memory, mask)
            rnn_input = torch.cat([embedded, context], dim=-1)
        elif self.mode == 'luong':
            rnn_input = torch.cat([embedded, feeding], dim=-1)
        else:
            rnn_input = embedded
        out, state = self.rnn(rnn_input.unsqueeze(1), state)
        query = out[:, 0]
        if self.mode == 'luong':
            context, weights = self.attn(query, memory, mask)
        if self.mode != 'none':
            feeding = torch.tanh(self.fuse(torch.cat([query, context], -1)))
            query = feeding
        logits = self.head(self.drop(query))
        return logits, state, feeding, weights

# Seq2Seq = Encoder + Decoder; teacher_ratio là xác suất dùng token đúng
class Seq2Seq(nn.Module):
    def __init__(self, mode, src_size, tgt_size, **kwargs):
        super().__init__()
        self.mode = mode
        self.encoder = Encoder(src_size, **kwargs)
        self.decoder = Decoder(tgt_size, mode, **kwargs)
        self.hidden = kwargs['hidden_size']

    def forward(self, src, lengths, tgt, teacher_ratio=0.5):
        memory, state = self.encoder(src, lengths)
        mask = src != 0
        token = tgt[:, 0]
        feeding = memory.new_zeros(src.size(0), self.hidden)
        logits = []
        for t in range(1, tgt.size(1)):
            step, state, feeding, _ = self.decoder.step(
                token, state, memory, mask, feeding)
            logits.append(step)
            coin = torch.rand(src.size(0), device=src.device)
            predicted = step.argmax(-1).detach()
            token = torch.where(coin < teacher_ratio, tgt[:, t], predicted)
        return torch.stack(logits, dim=1)
