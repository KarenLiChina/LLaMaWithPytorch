import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional

from torch.nn import RMSNorm


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
