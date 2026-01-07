import json
import time
from pathlib import Path
from typing import Optional

import torch
from safetensors.torch import load_file
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import Transformer, ModelArgs


# 加载 LLaMa的模型
class LLaMa:
    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    # 加了static之后不需要每次都new模型
    @staticmethod
    def build(checkpoint_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device: str):
        # checkpoint_dir 存放下载模型的路径
        prev_time = time.time()  # 记录时间
        if load_model:
            # 获取所有safetensors文件并按分片编号排序
            checkpoint_files = sorted(
                Path(checkpoint_dir).glob('model-*.safetensors'),
                key=lambda x: int(x.stem.split('-')[1])  # 按分片编号排序
            )
            assert len(checkpoint_files) > 0, f" checkpoint文件没有在{checkpoint_dir}中找到"
            print(f"找到 {len(checkpoint_files)} 个模型分片文件")

            # 合并所有分片文件
            checkpoint = {}
            for i, chk_path in enumerate(checkpoint_files):
                print(f"加载模型文件 {i + 1}/{len(checkpoint_files)}: {chk_path.name}")
                shard_checkpoint = load_file(chk_path, device='cpu')
                checkpoint.update(shard_checkpoint)
            print(f"checkpoint 文件加载完毕，耗时{(time.time() - prev_time):.2f}s")
            prev_time = time.time()
        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)  # 半精度模型
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)  # cpu的时候设置为BFloat16Tensor

        with open('params.json', 'r') as f:
            params = json.load(f)

        # 获取模型参数
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        model = Transformer(model_args).to(device)

        if load_model:
            # 从checkpoint中删除 rope.freqs，因为我们前面已经通过 precompute_theta_pos_frequencies 函数自己计算了，用自己写的替换一下
            if "rope.freqs" in checkpoint:
                del checkpoint["rope.freqs"]
            # strict=True, 模型是字典，键值对形式，所以参数名字要对上，如果对不上就抛异常
            model.load_state_dict(checkpoint, strict=True)
            print(f"加载 state dict 耗时{(time.time() - prev_time):.2f}s")

        return LLaMa(model, tokenizer, model_args)

    def text_completion(self, prompt: list[str], sample: bool = True, temperature: float = 0.6,
                        top_p: float = 0.9, max_gen_len: Optional[int] = None):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len
        # 转换每个prompt 为token ids
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]  # 任务是补全句子，所以eos设置为false
        # 确保batch size 不要太大，太大容易撑爆模型，我们指明了KV cache的大小
        batch_size = len(prompts)
        assert batch_size <= self.args.max_batch_size, f"batch size必须要小于 {self.args.max_batch_size}"
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        assert max_prompt_len <= self.args.max_seq_len, f"prompt 长度必须小于等于{self.args.max_seq_len}"
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id()  # 获取分词器对应填充使用的pad在词表中的int 值
        # 我们会把输入的提示词和模型的输出结果统统放到tokens 变量中。
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(prompt_tokens):
            # 将tokens 矩阵中提示词对应的token进行替换，没有被替换的位置依然是padding token
            tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        # 初始化 eos_reached 对于每个 prompt 都是False，后面每个prompt的eos_reached 都变为True，模型就可以结束，任务完成，不需要再generate。
        eos_reached = torch.tensor([False] * batch_size, device=device)
        # 计算一个矩阵prompt_tokens_mask，指明tokens矩阵中哪些位置对应的输入，哪些是padding_id.
        prompt_tokens_mask = tokens != pad_id  # true 为原本输入 token，false为新生成的或者补全的padding

        # 接下来就可以把准备好的输入交给模型进行预测
        for cur_pos in tqdm(range(1, total_len), desc="生成的tokens"):
            with torch.no_grad():
                # 每一时刻传入一个token
                logits = self.model.forward(tokens[:cur_pos - 1:cur_pos], cur_pos)

            if sample:  # 用户传进来的超参数
                # 基于 Top P的随机采样策略
                probs = torch.softmax(logits[:-1] / temperature, dim=-1)  # dim 对哪个 维度的数据进行softmax
                next_token = self._sample_top_p(probs, top_p)
            else:
                # 取当前这一时刻概率分布最大值的索引, Greedy Search
                next_token = torch.argmax(logits[:-1], dim=-1)

            next_token = next_token.reshape(-1)  # 将矩阵变成1维
            # prompt_tokens_mask 中对应维True，那么next_token就来自于提示词，否则就用LLM预测出来的token作为next_token
            next_token = torch.where(prompt_tokens_mask[:cur_pos], tokens[:, cur_pos], next_token)

            tokens[:, cur_pos] = next_token

            # EOS is reached 仅当我们发现新生成token位置中包含 EOS token
            eos_reached |= (~prompt_tokens_mask[:cur_pos]) & (next_token == self.tokenizer.eos_id())

            # 对于每一个prompt提示词 LLM都已经生成过 EOS标识符，那就可以停止推理
            if all(eos_reached):
                break
        output_tokens = []
        output_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # 只打印到EOS token，如果句子中包含 EOS，输入的提示词prompt 不能包含EOS
            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_index = current_prompt_tokens.index(self.tokenizer.eos_id())
                current_prompt_tokens = current_prompt_tokens[:eos_index]
            output_tokens.append(current_prompt_tokens)
            output_text.append(self.tokenizer.decode(current_prompt_tokens))

        return output_tokens, output_text

    def _sample_top_p(self, probs, p):
        probs_sort, probs_inx = torch.sort(probs, dim=-1, descending=True)  # 降序排序
        probs_sum = torch.cumsum(probs_sort, dim=-1)  # 将概率进行累加
        mask = probs_sum - probs_sort > p
        # 把没有被 top p 选中的 tokens的概率值设置为0.0
        probs_sort[mask] = 0.0
        # 重新获得一个加起来是1 的概率分布
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))  # 重新获得概率分布
        # 进行随机采样
        # 从 top p distribution 中采样一个token的index，此时的index 并不是词表中的token id
        next_token = torch.multinomial(probs_sort, num_samples=1)  # 随机采样一个
        # 因为一开始 概率分布进行了排序， 所以采样出来的token index 并非词典中的token id
        # sort 排序的时候要接收两个结果， probs_sort 是根据排序的具体的概率值
        # probs_idx 是排序后概率值对应原有词典中的顺序
        next_token = torch.gather(probs_inx, -1, next_token)  # 对应词典中的token id
        return next_token


if __name__ == '__main__':
    allow_cuda = True
    device = "cuda" if torch.cuda.is_available() and allow_cuda else "cpu"

    prompts = [
        "Simply put, the theory of relativity states that"
    ]

    model = LLaMa.build(
        checkpoint_dir="c:/model/llama-2-7b/",
        tokenizer_path="c:/model/llama-2-7b/tokenizer.model",
        load_model=True,
        max_seq_len=1024,
        max_batch_size=1,
        device=device
    )

    print("Model is running")

    out_tokens, out_texts = (model.text_completion(prompts, max_gen_len=64))
    assert len(out_texts) == len(prompts)
    for i in range(len(out_texts)):
        print(f'{out_texts[i]}')
        print('-' * 50)
