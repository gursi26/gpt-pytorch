from dataset import prepare_mask
from torch import nn
import torch
import math

class Attention(nn.Module):

    def __init__(self, input_dim, output_dim, num_heads, p):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.c_attn = nn.Linear(input_dim, output_dim * 3)
        self.c_proj = nn.Linear(output_dim, output_dim)
        self.attn_dropout = nn.Dropout(p=p, inplace=False)
        self.resid_dropout = nn.Dropout(p=p, inplace=False)

    def scaled_dot_product_attention(self, q, k, v, mask):
        """
        q: [batch_size, num_heads, head_dim, seq1_len]
        k: [batch_size, num_heads, head_dim, seq2_len]
        v: [batch_size, num_heads, head_dim, seq2_len]
        mask: [batch_size, num_heads, seq1_len, seq1_len]
        (seq1_len = seq2_len for self attention)
        """
        qk = q.matmul(k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
        if mask is not None:
            qk = qk.masked_fill(~mask, -torch.inf)
        attn_weights = self.attn_dropout(qk.softmax(dim=-1))
        return attn_weights.matmul(v)
    
    def qkv_reshape(self, x):
        return x.view(x.shape[0], x.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)
    
    def output_reshape(self, x):
        x = x.permute(0, 2, 1, 3)
        return x.reshape(x.shape[0], x.shape[1], -1)
    
    def forward(self, x, mask):
        q, k, v = self.c_attn(x).chunk(3, dim=-1)
        q, k, v = self.qkv_reshape(q), self.qkv_reshape(k), self.qkv_reshape(v)
        attn_outputs = self.output_reshape(self.scaled_dot_product_attention(q, k, v, mask))
        return self.resid_dropout(self.c_proj(attn_outputs))
    

class MLP(nn.Module):

    def __init__(self, input_dim, p) -> None:
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(input_dim, input_dim * 4)
        self.c_proj = nn.Linear(input_dim * 4, input_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=p, inplace=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.dropout(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, d_model, num_heads, p):
        super(Block, self).__init__()
        self.attn = Attention(d_model, d_model, num_heads, p)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, p)
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        skip_x = x
        x = self.attn(x, mask=mask)
        x = self.ln_1(x + skip_x)
        skip_x = x
        x = self.mlp(x)
        x = self.ln_2(x + skip_x)
        return x
    

class GPT(nn.Module):

    def __init__(self, vocab_size, max_seq_len, n_layers, d_model, num_heads, p):
        super(GPT, self).__init__()
        self.d_model, self.max_seq_len = d_model, max_seq_len
        self.tokens_embed = nn.Embedding(vocab_size, d_model)
        self.positions_embed = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(p=p, inplace=False)
        self.h = nn.ModuleList([Block(d_model, num_heads, p) for _ in range(n_layers)])

    def forward(self, x, mask=None):
        """
        x: [batch_size, seq_len]
        """
        x = self.tokens_embed(x) * math.sqrt(self.d_model)
        position_tokens = torch.arange(x.shape[-2]).unsqueeze(0).repeat(x.shape[0], 1).to(x.device)
        x = self.drop(x + self.positions_embed(position_tokens))
        for layer in self.h:
            x = layer(x, mask=mask)
        return x
    

class GPTSemanticSimilarity(nn.Module):

    def __init__(self, gpt_base: GPT):
        super(GPTSemanticSimilarity, self).__init__()
        self.gpt_base = gpt_base
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.classifier = nn.Linear(self.gpt_base.d_model * self.gpt_base.max_seq_len, 1)

    def forward(self, x1, x2, x1_mask, x2_mask):
        x1 = self.gpt_base(x1, prepare_mask(x1_mask))
        x2 = self.gpt_base(x2, prepare_mask(x2_mask))
        x = self.dropout(x1 + x2)
        padding = torch.zeros(x.shape[0], self.gpt_base.max_seq_len - x.shape[1], x.shape[-1]).to(x.device)
        x = torch.cat([x, padding], dim=1).view(x.shape[0], -1)
        return self.classifier(x).view(-1)