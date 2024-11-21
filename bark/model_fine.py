"""
Much of this code is adapted from Andrej Karpathy's NanoGPT
(https://github.com/karpathy/nanoGPT)
"""
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from .model import GPT, GPTConfig, MLP


class NonCausalSelfAttention(nn.Module):
    """
    Non-causal self-attention layer for the FineGPT model.

    This layer computes the attention scores for each token in the input sequence,
    allowing each token to attend to all other tokens (non-causal attention).
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = (
            hasattr(torch.nn.functional, "scaled_dot_product_attention")
        )

    def forward(self, x):
        """
        Forward pass for the non-causal self-attention layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, embedding_dim).
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=False
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class FineBlock(nn.Module):
    """
    Transformer block for the FineGPT model.

    This block consists of a non-causal self-attention layer followed by an MLP layer.
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = NonCausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        """
        Forward pass for the transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, embedding_dim).
        """
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class FineGPT(GPT):
    """
    FineGPT (Fine-tuned Generative Pre-trained Transformer) model.

    This model consists of multiple transformer blocks, each containing a non-causal self-attention layer
    and an MLP layer. The model also includes token and position embeddings, as well as a final linear
    layer for generating logits.
    """

    def __init__(self, config):
        super().__init__(config)
        del self.lm_head
        self.config = config
        self.n_codes_total = config.n_codes_total
        self.transformer = nn.ModuleDict(
            dict(
                wtes=nn.ModuleList(
                    [
                        nn.Embedding(config.input_vocab_size, config.n_embd)
                        for _ in range(config.n_codes_total)
                    ]
                ),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([FineBlock(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_heads = nn.ModuleList(
            [
                nn.Linear(config.n_embd, config.output_vocab_size, bias=False)
                for _ in range(config.n_codes_given, self.n_codes_total)
            ]
        )
        for i in range(self.n_codes_total - config.n_codes_given):
            self.transformer.wtes[i + 1].weight = self.lm_heads[i].weight

    def forward(self, pred_idx, idx):
        """
        Forward pass for the FineGPT model.

        Args:
            pred_idx (int): Index of the codebook to predict.
            idx (torch.Tensor): Input tensor of shape (batch_size, sequence_length, n_codes_total).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, output_vocab_size).
        """
        device = idx.device
        b, t, codes = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        assert pred_idx > 0, "cannot predict 0th codebook"
        assert codes == self.n_codes_total, (b, t, codes)
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)

        # forward the GPT model itself
        tok_embs = [
            wte(idx[:, :, i]).unsqueeze(-1) for i, wte in enumerate(self.transformer.wtes)
        ]  # token embeddings of shape (b, t, n_embd)
        tok_emb = torch.cat(tok_embs, dim=-1)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        x = tok_emb[:, :, :, : pred_idx + 1].sum(dim=-1)
        x = self.transformer.drop(x + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_heads[pred_idx - self.config.n_codes_given](x)
        return logits

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.

        Args:
            non_embedding (bool, optional): Whether to exclude the embedding parameters. Defaults to True.

        Returns:
            int: The number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            for wte in self.transformer.wtes:
                n_params -= wte.weight.numel()
            n_params -= self.transformer.wpe.weight.numel()
        return n_params


@dataclass
class FineGPTConfig(GPTConfig):
    """
    Configuration class for the FineGPT model.

    Attributes:
        n_codes_total (int): The total number of codebooks.
        n_codes_given (int): The number of given codebooks.
    """
    n_codes_total: int = 8
    n_codes_given: int = 1
