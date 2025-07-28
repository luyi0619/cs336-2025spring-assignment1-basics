import torch
import torch.nn as nn
import math
from jaxtyping import Float, Int
from einops import einsum
from torch import Tensor


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    max_values, _ =  torch.max(in_features, dim=dim, keepdims=True)
    in_features = in_features - max_values
    exp_values = torch.exp(in_features)
    sum_values = torch.sum(exp_values, dim=dim, keepdims=True)
    return exp_values / sum_values
    
def _init_linear_w(d_out: int, d_in: int):
    w = torch.empty(d_out, d_in)
    std = math.sqrt(2.0 / (d_in + d_out))
    torch.nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)
    return nn.Parameter(w)
    
    
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.w = _init_linear_w(out_features, in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(self.w, x, 'o i, ... i -> ... o')
        

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        w = torch.empty(num_embeddings, embedding_dim)
        torch.nn.init.trunc_normal_(w, mean=0.0, std=1, a=-3.0, b=3.0)
        self.w = nn.Parameter(w)
        
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # num_embeddings, embedding_dim
        # batch_size, sequence_length
        # batch_size, sequence_length, embedding_dim
        return self.w[token_ids]
        

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.w = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        # Your code here performing RMSNorm
        rms = torch.sqrt(einsum(x, x, 'b s d, b s d -> b s') / self.d_model + self.eps)
        result = x / rms.unsqueeze(-1) * self.w
        # Return the result in the original dtype
        return result.to(in_dtype)
        
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = self._init_w(d_ff, d_model)
        self.w2 = self._init_w(d_model, d_ff)
        self.w3 = self._init_w(d_ff, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = einsum(self.w1, x, 'd_ff d_model, ... d_model -> ... d_ff')
        w3x = einsum(self.w3, x, 'd_ff d_model, ... d_model -> ... d_ff')
        silu_w1x = silu(w1x)
        silu_w1x_w3x = silu_w1x * w3x
        return einsum(self.w2, silu_w1x_w3x, 'd_model d_ff, ... d_ff -> ... d_model')
        
    def _init_w(self, out_features: int, in_features: int) -> torch.Tensor:
        w = torch.empty(out_features, in_features)
        std = math.sqrt(2.0 / (in_features + out_features))
        torch.nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)
        return nn.Parameter(w)


# X (..., seq_len, d_model)
# WQ (d_model, d_k).  Q = WQ * x -> (..., seq_len, d_k)
# WK (d_model, d_k).  K = WK * x -> (..., seq_len, d_k)
# WV (d_model, d_v).  V = WV * x -> (..., seq_len, d_v)
#
# softmax(Q * KT / sqrt(d_k)) * V.  (..., seq_len, d_v)

def attention(Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = Q.shape[-1]
    qk = einsum(Q, K, '... n d_k, ... m d_k -> ... n m') / math.sqrt(d_k)

    if mask is not None:
      # Ensure mask is broadcastable to qk's last two dimensions
      if mask.shape != qk.shape:
          expanded_mask_shape = list(qk.shape[:-2]) + list(mask.shape)
          mask = mask.expand(expanded_mask_shape)
      qk[~mask] = float('-inf')

    assert qk.ndim >= 1
    sqk = softmax(qk, qk.ndim - 1)    
    attention = einsum(sqk, V, '... n m, ... m d_v -> ... n d_v')
    return attention

        
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.register_buffer('R', self.r(), persistent=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        return einsum(self.R[token_positions], x, '... s i j, ... s j -> ... s i')
    
    def r(self) -> torch.Tensor:
        R = torch.zeros(self.max_seq_len, self.d_k, self.d_k)
        for i in range(self.max_seq_len):
            R[i] = self.r_i(i)
        return R
    
    def r_i(self, i: int) -> torch.Tensor:
        Ri = torch.zeros(self.d_k, self.d_k)
        assert self.d_k % 2 == 0
        for k in range(self.d_k // 2):
            b = self.r_ik(i, k)
            offset = 2 * k
            Ri[offset:offset + 2, offset:offset + 2] = b
        return Ri
    
    def r_ik(self, i: int, k:int) -> torch.Tensor:
        s = math.sin(self._theta_ik(i,k))
        c = math.cos(self._theta_ik(i,k))
        return torch.tensor([[c, -s], [s, c]])
    
    def _theta_ik(self, i: int, k:int) -> float:
        return i / (self.theta ** (2 * k / self.d_k))
        
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RotaryPositionalEmbedding | None = None, device=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.wq = _init_linear_w(self.num_heads * self.d_k, self.d_model)
        self.wk = _init_linear_w(self.num_heads * self.d_k, self.d_model)
        self.wv = _init_linear_w(self.num_heads * self.d_v, self.d_model)
        self.wo = _init_linear_w(self.d_model, self.num_heads * self.d_v)
        self.rope = rope

    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        Q = einsum(self.wq, x, 'hdk d_model, ... seq d_model -> ... seq hdk')
        K = einsum(self.wk, x, 'hdk d_model, ... seq d_model -> ... seq hdk')
        V = einsum(self.wv, x, 'hdv d_model, ... seq d_model -> ... seq hdv')
        assert Q.shape[-1] == self.num_heads * self.d_k
        assert K.shape[-1] == self.num_heads * self.d_k
        assert V.shape[-1] == self.num_heads * self.d_v
         
        all_true_matrix = torch.ones((seq_len, seq_len), dtype=torch.bool)
        # This creates a causal mask (look-ahead). If full attention is needed,
        # remove this mask or pass a mask of all True.
        mask = ~torch.triu(all_true_matrix, diagonal=1)
        
        hs = []
        for i in range(self.num_heads):
            q = Q[..., i * self.d_k: (i+1) * self.d_k] # (batch, seq_len, d_k)
            k = K[..., i * self.d_k: (i+1) * self.d_k] # (batch, seq_len, d_k)
            v = V[..., i * self.d_v: (i+1) * self.d_v] # (batch, seq_len, d_v)
            
            # Apply RoPE here!
            if self.rope is not None and token_positions is not None:
                q = self.rope(q, token_positions)
                k = self.rope(k, token_positions)
            
            h = attention(q, k, v, mask)
            hs.append(h)
        
        # Concatenate results from all heads along the last dimension
        multi_head = torch.cat(hs, dim=-1) # (batch, seq_len, num_heads * d_v)
        return einsum(self.wo, multi_head, 'd_model hdv, b seq hdv -> b seq d_model')


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: RotaryPositionalEmbedding | None = None, device=None):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, num_heads, rope, device)
        self.rms_norm_1 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.rms_norm_2 = RMSNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        token_positions = torch.arange(seq_len).unsqueeze(0)
        
        residual = x
        x = self.rms_norm_1.forward(x)
        x = self.attention.forward(x, token_positions)
        x = residual + x
        
        residual = x
        x = self.rms_norm_2.forward(x)
        x = self.ffn.forward(x)
        return x + residual


class TransformerLM(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, vocab_size: int, context_length: int, num_layers: int, rope: RotaryPositionalEmbedding | None = None, device=None):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.transformer_blocks =  nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, rope) for i in range(num_layers)])
        self.rms_norm = RMSNorm(d_model)
        self.linear = Linear(d_model, vocab_size)
    
    def forward(self, in_indices: Int[Tensor, "b s"]) -> torch.Tensor:
        x = self.embedding.forward(in_indices)
        for i in range(len(self.transformer_blocks)):
            x = self.transformer_blocks[i].forward(x)
        x = self.rms_norm.forward(x)
        x = self.linear.forward(x)
        return x