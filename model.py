import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# 加了注释之后，会自动为类生成一些特殊方法，减少样板代码
@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32  # transformer block 堆叠数量
    n_heads: int = 32  # heads中Q的数量，MQA/GQA
    n_kv_heads: Optional[int] = None  # 可选类型，heads中 K 和V的数量
    vocab_size: int = -1  # 这个值在加载分词器的时候才会被设置，此时不知道
    multiple_of: int = 256  # FFN网络中隐藏神经元的个数
    ffn_dim_multiplier: Optional[int] = None  # 当使用GQA之后，K和V的数量会减少，KV中的参数减少，但是会增加FFN中神经元的数量
    norm_eps: float = 1e-5

    # 参数给KV cache 使用
    max_batch_size: int = 32  # 最大批次数量
    max_seq_len: int = 2048  # 最大句子长度

    device: str = None  # 给 pytorch用的，是用GPU还是用CPU


def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    # 预先计算RoPE中需要的mθ，将需要提前定义的位置编码计算出来
    assert head_dim % 2 == 0, "必须可以被2整除，因为公式中 d/2"

    # 构建theta参数
    # 根据论文中的公式实现
    theta_numerator = torch.arrange(0, head_dim, 2).float()  # 序列左闭，右开，步长为2,得到的就是公式中的 2(i-1)
    theta = 1.0 / (theta ** (theta_numerator / head_dim))  # 10000^(-2(i-1)/d)

    # 构建 m 参数，代表着 positions位置
    m = torch.arange(seq_len, device=device)

    # 接下来 要计算m θ两个序列内积，这里我们要的到所有的排列组合，用torch.outer
    # 这样每个position都有一组mθ值
    freqs = torch.outer(m, theta).float()

    # 可以用极坐标形式计算复数 torch.polar(abs, angle,*,out=None): out = abs·cos(angle)+abs·sin(angle)·j
    # torch.ones_like(freqs) 是和 freqs形状一样的矩阵，只是每个位置都是1， 那前面公式中的abs就是1
    # 那些面得到的就是cos(mθ)+i sin(mθ)的矩阵
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotary_embedding(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # 1. 将 x token 向量中的dimension个值进行分组，2个值一组
    # x.float().reshape(*x.shape[:-1], -1, 2) x.shape[:-1] 是除了最后一列，前面的所有列，左闭右开，前面加了*号后，去掉了外面的一层小括号，加入原本的形状为(1,2,3,6)->就变成了(1,2,3,3,2)
    # 2. 将其转换为 复数形式
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # freqs_complex 是个二维矩阵，现在需要升高维度，进行转换(Seq_Len, Head_Dim/2)->(1, Seq_Len, 1, Head_Dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)  # 在维度为0和2的位置增加维度
    # 3. 乘上我们准备好的矩阵
    x_rotated = x_complex * freqs_complex
    # 4. 将复数a+ib形式中的 a和b提取出来
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)  # 加* 去掉小括号
    return x_out.type_as(x).to(device)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 公式中的 g 参数
        self.weight = nn.Parameter(torch.ones(dim))  # 初始化参数值为1

    def _norm(self, x: torch.Tensor):  # x需要归一化的数据
        # x的形状(B, Seq_Len, Dim) batch size, 每条样本的长度，每条样本每个token对应维度的大小
        # torch.rsqrt() 简单来说就是对每个元素开根号，再去取倒数
        # mean 参数中的-1，是值对最后一个维度求平均，就是对Dim这个维度来求平均,加 + self.eps，防止分母为0
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        return self._norm(x.float()).type_as(x) * self.weight


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        # MHA
        return x
    else:
        # GQA
        return (
            x[:, :, :, None, :].expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )


class SelfAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads_q = args.n_heads  # 当 n_kv_heads 和n_heads_q 都是n_heads，那同一组里就都有QKV，就是MHA，如果不同就是GQA
        # 算出一个query of head 对应多少个重复的 repeated keys and values of heads
        self.n_req = self.n_heads_q // self.n_kv_heads  # 相等时，就是MHA，不等时，是GQA，算出每组中对应几个Q
        self.head_dim = args.dim // args.n_heads  # 4096/32
        self.device = args.device

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)  # 第二参数就是 args.dim,是个方形矩阵，这样写在之后reshape的时候比较好理解
        self.wk = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.cache_k = torch.zeros(args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)
        self.cache_v = torch.zeros(args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # 把x的形状接收一下 (B, 1, Dim)
        batch_size, seq_len, _ = x.shape

        # x的形状(B, 1, Dim)->xq的形状(B, 1, H_Q* Head_Dim)
        xq = self.wq(x)
        # x的形状(B, 1, Dim)->xk的形状(B, 1, H_KV* Head_Dim)
        xk = self.wk(x)
        xv = self.wv(x)

        # reshape: xq (B, 1, H_Q* Head_Dim)->(B,1 H_Q, Head_Dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # reshape: xk/xv (B, 1, H_KV* Head_Dim)->(B,1 H_KV, Head_Dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # 应用RoPE
        xq = apply_rotary_embedding(xq, freqs_complex, device=self.device)
        xk = apply_rotary_embedding(xk, freqs_complex, device=self.device)

        # 前面把cache初始化为0， 所以这里 append 其实就是将对应位置赋值就可以了
        self.cache_k[:batch_size, start_pos:start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos + seq_len] = xv

        # 为了后面去计算 self attention，需要取出前面的数据

        keys = self.cache_k[:batch_size, 0:start_pos + seq_len]
        values = self.cache_v[:batch_size, 0:start_pos + seq_len]

        # 重复 keys 和 values，以达到与 query的数量匹配，再进行矩阵相乘时，不会出错
        keys = repeat_kv(keys, self.n_req)
        values = repeat_kv(values, self.n_req)

        # (B, 1, H_Q, Head_Dim)-> (B, H_Q, 1, Head_Dim) 维度转换一下
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # 把 K转换成keys.transpose(2, 3)
        # 然后用 Q 和K的转置进行矩阵相乘 (B, H_Q, 1, Head_Dim) @ (B,H_Q,Head_Dim,Seq_Len_KV) --> (B, H_Q, 1, Seq_Len_KV)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores, dim=-1).type_as(xq)  # 变成概率分布

        # (B, H_Q, 1, Seq_Len_KV) @ ((B,H_Q,Seq_Len_KV,Head_Dim)-->(B, H_Q, 1, Head_Dim)
        output = torch.matmul(scores, values)

        # (B, H_Q, 1, Head_Dim)->(B, 1, H_Q, Head_Dim) -->(B, 1, Dim)下面的代码先调换1，2位置，然后把最后两列称道一起
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output)


class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads  # default value 32
        self.dim = args.dim  # 编码维度 default value 4096
        self.head_dim = args.dim // args.n_heads  # 4096/32
        self.attention = SelfAttention(args)  # MHA或者是GQA
        self.feed_forward = FeedForward(args)
        # attention之前需要归一化
        self.attention_norm = RMSNorm(self.dim, eps=args.norm_eps)
        # feedforward之前需要归一化
        self.ffn_norm = RMSNorm(self.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)  # self attention 的正向传播, h是中间结果
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1, "必须设定词表大小"  # 如果没有设置vocab_size，就会报错，如果设置了，就会继续执行
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embedding = nn.Embedding(self.vocab_size, self.args.dim)  # 词表有多少个词，和转换成多少维的向量

        self.layers = nn.ModuleList()  # 装载神经网络的多层

        for _ in range(args.n_layers):
            # EncoderBlock 是之后实现的LLaMa中的transformer block
            self.layers.append(EncoderBlock(args))  # 列表追加

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size,
                                bias=False)  # 输出转换成词表中词的个数,bias=False，现在很多大模型不会用bias，不用截距项，所以设置为False，所以Linear 中就只有W矩阵

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads,
                                                              self.args.max_seq_len * 2,
                                                              device=self.args.device)  # 预先计算 theta和pos的旋转位置编码

    def forward(self, tokens: torch.Tensor, start_pos: int):  # 正向传播
        # 这里实现的是inference，所以每次传入的是一个token,那么seq_len一直都是1
        # tokens 的形式是(B, seq_len) Batch size * 每条样本的长度
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "每次只处理一个token"

        # (B, seq_len)-> (B, seq_len, Dim)是input embedding做的事情
        h = self.tok_embedding(tokens)  # 本事是调用的nn.Embedding

        # 先去计算 positional encoding相关的信息
        # 根据位置 [start_pos, start_pos+seq_len] 获取(m, theta)， m代表position，theta代表角度
        freqs_complex = self.freqs_complex[
            start_pos:start_pos + seq_len]  # freqs_complex 已经在前面通过precompute_theta_pos_frequencies 计算出来了

        # 连续应用 encoder layers / transformer block

        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)  # h会一直往下传，再被赋值回来

        h = self.norm(h)  # 整体归一化
        output = self.output(h).float()
        return output
