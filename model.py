import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048
    device: str = None

    # 添加新字段来存储原始配置
    intermediate_size: Optional[int] = None

    @classmethod
    def from_dict(cls, config_dict: dict):
        """从配置字典创建 ModelArgs"""
        # 首先设置默认值
        filtered_config = {}

        # 处理每个可能的参数
        for key, value in config_dict.items():
            if key == 'hidden_size':
                filtered_config['dim'] = value
            elif key == 'num_hidden_layers':
                filtered_config['n_layers'] = value
            elif key == 'num_attention_heads':
                filtered_config['n_heads'] = value
            elif key == 'num_key_value_heads':
                filtered_config['n_kv_heads'] = value
            elif key == 'rms_norm_eps':
                filtered_config['norm_eps'] = value
            elif key == 'max_position_embeddings':
                filtered_config['max_seq_len'] = value
            elif key == 'vocab_size':
                filtered_config['vocab_size'] = value
            elif key == 'intermediate_size':
                # 保存但不作为必需参数
                filtered_config['intermediate_size'] = value
            elif key in ['dim', 'n_layers', 'n_heads', 'n_kv_heads', 'vocab_size',
                         'multiple_of', 'ffn_dim_multiplier', 'norm_eps',
                         'max_batch_size', 'max_seq_len', 'device']:
                filtered_config[key] = value
            # 忽略其他参数

        # 确保必要的参数有值
        if 'multiple_of' not in filtered_config and 'intermediate_size' in filtered_config:
            # 使用 intermediate_size 作为 multiple_of 的参考
            intermediate_size = filtered_config['intermediate_size']
            # 找到最接近的 256 的倍数
            filtered_config['multiple_of'] = 256

        return cls(**filtered_config)


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    # 预先计算 RoPE 中需要的 mθ
    assert head_dim % 2 == 0, "必须可以被2整除,因为公式中 d/2"

    # 构建 theta 参数
    # 根据论文中的公式实现
    theta_numerator = torch.arange(0, head_dim, 2).float()  # 2(i-1)
    theta = 1.0 / (theta ** (theta_numerator / head_dim))  # 10000^(-2(i-1)/d)

    # 构建 m 参数, 代表着 positions 位置
    m = torch.arange(seq_len, device=device)

    # 接下来 mθ 两个序列内积, 这里我们要得到所有的排列组合, 用 torch.outer
    # 这样每个 position 都有一组 mθ 值
    freqs = torch.outer(m, theta).float()

    # 我们可以用极坐标形式计算复数
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_embedding(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # 1, 将 x token 向量中的 dimension 个值分组, 2个值一组
    # 2, 然后将其转换为复数形式
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (Seq_Len, Head_Dim / 2) -> (1, Seq_Len, 1, Head_Dim / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)

    # 3, 乘上我们准备好的矩阵
    x_rotated = x_complex * freqs_complex

    # 4, 将复数 a+ib 形式中的 a 和 b 提取出来
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 公式中的 g 参数
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, Seq_Len, Dim)
        # torch.rsqrt() 简单来说就是对每个元素开根号后再取倒数
        # -1 是对最后一个维度求平均
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        return self._norm(x.float()).type_as(x) * self.weight


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        # MHA
        return x
    else:
        return (
            x[:, :, :, None, :].expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads

        # 指名一个 query of head 对应多少个重复的 repeated keys and values of head
        self.n_req = self.n_heads_q // self.n_kv_heads  # 32 / 8 = 4

        # 4096 / 32 = 128
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (B, 1, Dim)
        batch_size, seq_len, _ = x.shape

        # (B, 1, Dim) -> (B, 1, H_Q * Head_Dim)
        xq = self.wq(x)
        # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        xk = self.wk(x)
        xv = self.wv(x)

        # (B, 1, H_Q * Head_Dim) -> (B, 1, H_Q, Head_Dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # 应用 RoPE
        xq = apply_rotary_embedding(xq, freqs_complex, device=x.device)
        xk = apply_rotary_embedding(xk, freqs_complex, device=x.device)

        # 因为前面把 cache 全部初始化为 0, 所以这里"append"其实就是将对应位置赋值
        self.cache_k[:batch_size, start_pos:start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos + seq_len] = xv

        # 为了后面去计算 self attention
        keys = self.cache_k[:batch_size, 0:start_pos + seq_len]
        values = self.cache_v[:batch_size, 0:start_pos + seq_len]

        # 重复 keys and values 以达到 queries 的数量
        keys = repeat_kv(keys, self.n_req)
        values = repeat_kv(values, self.n_req)

        # (B, 1, H_Q, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # (B, H_Q, 1, Head_Dim) @ (B, H_Q, Head_Dim, Seq_Len_KV) --> (B, H_Q, 1, Seq_Len_KV)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, H_Q, 1, Seq_Len_KV) @ (B, H_Q, Seq_Len_KV, Head_Dim) --> (B, H_Q, 1, Head_Dim)
        output = torch.matmul(scores, values)
        # (B, H_Q, 1, Head_Dim) -> (B, 1, H_Q, Head_Dim) -> (B, 1, Dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.wo(output)  # (B, 1, Dim)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        # hidden_size = 7; multiple = 5; 现在是7,但是想要比7大的第一个5的倍数
        # (7 + 5 - 1) // 5 = 2 --> 5 * 2 = 10
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        # SwiGLU 是 Swish 激活函数和 GLU 函数的结合
        # SwiGLU(A, B) = A * Swish(B)
        swish = F.silu(self.w1(x))
        x_v = self.w3(x)
        x = swish * x_v
        x = self.w2(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_heads = args.n_heads  # 32
        self.dim = args.dim  # 4096
        # 4096 / 32 = 128
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # attention 之前需要归一化
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # feedward 之前需要归一化
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1, "必须设定词表大小"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        # input embedding
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            # EncoderBlock 是之后要去实现的 LLaMA transformer block
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        # 稍后会展开去讲, 现在先写在这里
        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads,
                                                              self.args.max_seq_len * 2,
                                                              device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # 这里实现的是 inference, 所以每次传入一个token, 那么 seq_len 一直都是 1
        # tokens 的形状是 (B, seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "每次只处理一个token"

        # (B, seq_len) -> (B, seq_len, Dim)
        h = self.tok_embeddings(tokens)
        # 先去获取对应的 positional encoding 相关信息
        # 根据位置 [start_pos, start_pos+seq_len] 获取 (m, theta)
        freqs_complex = self.freqs_complex[start_pos: start_pos + seq_len]

        # 连续应用 encoder layers / transformer block
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output

