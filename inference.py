import json
from typing import Optional, Dict

import torch
from pathlib import Path

from safetensors import safe_open
from sentencepiece import SentencePieceProcessor
import time

from tqdm import tqdm

from model import Transformer, ModelArgs


class LLaMA:
    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int,
              device: str):
        prev_time = time.time()
        checkpoint = {}

        if load_model:

            # 查找 .safetensors 文件
            safetensor_files = sorted(Path(checkpoints_dir).glob('*.safetensors'))
            # 如果没有找到 .safetensors，尝试查找 .pth 文件
            if len(safetensor_files) == 0:
                safetensor_files = sorted(Path(checkpoints_dir).glob('*.pth'))
                if len(safetensor_files) > 0:
                    print(f"使用 .pth 文件: {safetensor_files[0]}")

            assert len(safetensor_files) > 0, f"checkpoint 文件没有在 {checkpoints_dir} 找到"
            chk_path = safetensor_files[0]
            print(f"加载模型文件... {chk_path}")

            if str(chk_path).endswith('.safetensors'):
                # 使用 safetensors 加载
                with safe_open(chk_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        checkpoint[key] = f.get_tensor(key)
            else:
                # 使用 torch 加载 .pth 文件
                checkpoint = torch.load(chk_path, map_location='cpu')

            print(f"checkpoint 文件加载完成, 耗时 {(time.time() - prev_time):.2f}s")
            prev_time = time.time()

        # 设置 tensor 类型（避免警告）
        if device == 'cuda':
            torch.set_default_device('cuda')
            torch.set_default_dtype(torch.float16)
        else:
            torch.set_default_dtype(torch.bfloat16)

        # 加载参数文件
        params_path = Path(checkpoints_dir) / "params.json"
        if not params_path.exists():
            # 尝试其他可能的文件名
            alt_paths = [
                Path(checkpoints_dir) / "config.json",
                Path(checkpoints_dir) / "model_config.json",
                Path(checkpoints_dir) / "configuration.json",
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    params_path = alt_path
                    print(f"使用配置文件: {alt_path}")
                    break

        assert params_path.exists(), f"参数文件不存在: {checkpoints_dir}"

        with open(params_path, 'r') as f:
            params = json.loads(f.read())

        # 打印配置信息用于调试
        print(f"\n配置文件包含以下键: {list(params.keys())}")
        print("关键参数:")
        for key in ['hidden_size', 'dim', 'num_hidden_layers', 'n_layers',
                    'num_attention_heads', 'n_heads', 'vocab_size']:
            if key in params:
                print(f"  {key}: {params[key]}")

        # 创建模型参数，使用 from_dict 方法过滤额外参数
        model_args = ModelArgs.from_dict({
            **params,
            'max_seq_len': max_seq_len,
            'max_batch_size': max_batch_size,
            'device': device,
        })

        # 加载分词器
        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        # 创建模型
        model = Transformer(model_args).to(device)

        if load_model:
            # 从checkpoint中删除 rope.freqs, 因为我们前面通过 precompute_theta_pos_frequencies 函数自己计算了
            if "rope.freqs" in checkpoint:
                del checkpoint["rope.freqs"]
            if "freqs_complex" in checkpoint:
                del checkpoint["freqs_complex"]

            # 转换键名以匹配模型结构
            checkpoint = LLaMA._convert_checkpoint_keys(checkpoint)

            # 加载模型参数，使用宽松模式
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)

            # 打印警告信息
            if missing_keys:
                print(f"\n警告: 以下键缺失 ({len(missing_keys)} 个):")
                for key in missing_keys[:10]:  # 只显示前10个
                    print(f"  - {key}")
                if len(missing_keys) > 10:
                    print(f"  ... 还有 {len(missing_keys) - 10} 个缺失键")

            if unexpected_keys:
                print(f"\n警告: 以下键多余 ({len(unexpected_keys)} 个):")
                for key in unexpected_keys[:10]:  # 只显示前10个
                    print(f"  - {key}")
                if len(unexpected_keys) > 10:
                    print(f"  ... 还有 {len(unexpected_keys) - 10} 个多余键")

            print(f"\n加载 state dict 耗时 {time.time() - prev_time:.2f}s")

        return LLaMA(model, tokenizer, model_args)

    @staticmethod
    def _convert_checkpoint_keys(checkpoint: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """转换 checkpoint 的键名以匹配模型结构"""
        converted = {}

        for key, tensor in checkpoint.items():
            new_key = key

            # 常见的前缀移除
            prefixes_to_remove = ["model.", "transformer.", "base_model.model.", "_orig_mod."]
            for prefix in prefixes_to_remove:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]

            # LLaMA 2 格式转换
            if new_key.startswith("layers."):
                # 转换层编号格式
                if "layers." in new_key and not new_key.startswith("layers."):
                    parts = new_key.split(".")
                    for i, part in enumerate(parts):
                        if part.isdigit():
                            # 找到层编号，重新构建键
                            new_key = "layers." + ".".join(parts[i:])
                            break

                # 转换注意力层
                if "self_attn.q_proj" in new_key:
                    new_key = new_key.replace("self_attn.q_proj", "attention.wq")
                elif "self_attn.k_proj" in new_key:
                    new_key = new_key.replace("self_attn.k_proj", "attention.wk")
                elif "self_attn.v_proj" in new_key:
                    new_key = new_key.replace("self_attn.v_proj", "attention.wv")
                elif "self_attn.o_proj" in new_key:
                    new_key = new_key.replace("self_attn.o_proj", "attention.wo")

                # 转换前馈层 (LLaMA 2 使用不同的命名)
                elif "mlp.gate_proj" in new_key:
                    new_key = new_key.replace("mlp.gate_proj", "feed_forward.w1")
                elif "mlp.down_proj" in new_key:
                    new_key = new_key.replace("mlp.down_proj", "feed_forward.w2")
                elif "mlp.up_proj" in new_key:
                    new_key = new_key.replace("mlp.up_proj", "feed_forward.w3")

                # 转换归一化层
                elif "input_layernorm" in new_key:
                    new_key = new_key.replace("input_layernorm", "attention_norm")
                elif "post_attention_layernorm" in new_key:
                    new_key = new_key.replace("post_attention_layernorm", "ffn_norm")

            # 嵌入层转换
            elif "embed_tokens" in new_key:
                new_key = new_key.replace("embed_tokens", "tok_embeddings")

            # 归一化层转换
            elif new_key == "norm.weight":
                new_key = "norm.weight"
            elif "norm.weight" in new_key and "layers" not in new_key:
                new_key = "norm.weight"

            # 输出层转换
            elif new_key == "lm_head.weight":
                new_key = "output.weight"
            elif "lm_head" in new_key:
                new_key = new_key.replace("lm_head", "output")

            # 旧的 LLaMA 格式
            elif "attention.wq" in new_key or "attention.wk" in new_key or "attention.wv" in new_key or "attention.wo" in new_key:
                # 保持原样
                pass
            elif "feed_forward.w1" in new_key or "feed_forward.w2" in new_key or "feed_forward.w3" in new_key:
                # 保持原样
                pass
            elif "attention_norm" in new_key or "ffn_norm" in new_key:
                # 保持原样
                pass

            converted[new_key] = tensor

        return converted

    def text_completion(self, prompts: list[str], device: str, do_sample: bool = True, temperature: float = 0.6,
                        top_p: float = 0.9, max_gen_len: Optional[int] = None):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len

        # 转换每个 prompt 为 token ids
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]

        # 确保 batch size 不要太大, 因为我们指明了 KV cache 的大小
        batch_size = len(prompts)
        assert batch_size <= self.args.max_batch_size, f"batch size 必须小于等于 {self.args.max_batch_size}"
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        assert max_prompt_len <= self.args.max_seq_len, f"prompt 长度必须小于等于 {self.args.max_seq_len}"

        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id()  # 获取分词器对应填充所使用的pad在词表中的int值
        # 我们会把输入的提示词和模型输出的结果统统放到 tokens 变量中
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(prompt_tokens):
            # 将 tokens 矩阵中提示词对应的token替换，没有被替换的位置依然是padding token
            tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        # 初始化 eos_reached 对于每个prompt 都是 False, 后面每个 prompt的 eos_reached 都为 True, 就无需再去 generate
        eos_reached = torch.tensor([False] * batch_size, device=device)
        # 计算一个矩阵 prompt_tokens_mask 指明在 tokens 矩阵中哪些位置对应的提示词 token id, 哪些位置是 padding id
        prompt_tokens_mask = tokens != pad_id  # True 如果token是输入的提示词token, 否则就是 False

        # 接下来就可以把准备好的输入交给模型进行推理预测
        for cur_pos in tqdm(range(1, total_len), desc="生成 tokens"):
            with torch.no_grad():
                # 每次传入一个token
                logits = self.model.forward(tokens[:, cur_pos - 1:cur_pos], cur_pos)

            if do_sample:
                # 基于 Top P 的随机采样策略
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                # 取当前这一个时刻最大的值对应索引, Greedy Search
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # prompt_tokens_mask 中对应位置是 True, 那么next_token就来自于提示词, 否则就用 LLM 预测出来的 token 作为 next_token
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)

            tokens[:, cur_pos] = next_token

            # EOS is reached 仅当我们发现在新生成token位置中包含 EOS token
            # 反过来说就是, 因为提示词对于LLM给出的next_token如果是EOS的话, 我们不关系
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id())

            # 对于每一个prompt提示词LLM都已经生成过EOS标识, 就可以停止继续inference
            if all(eos_reached):
                break

        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # 只保留到 EOS token, 如果句子中包含 EOS 的话, 输入的提示词 prompt 不能包含 EOS
            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id())
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        return out_tokens, out_text

    def _sample_top_p(self, probs, p):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > p
        # 没有被 top p 选择的 tokens 的概率值 全部设置为 0.0
        probs_sort[mask] = 0.0
        # 重新获得一个加起来是 1 的概率分布
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        # 下面进行随机采样
        # 从 top p distribution 中采样一个 token 的 index, 注意此时 index 并不是词典中对于的 token_id
        next_token = torch.multinomial(probs_sort, num_samples=1)
        # 因为一开始概率分布probs进行了排序, 所以采样出来的 token index 并非词典中对应的 Token id
        # 这就是为什么一开始 sort 要接收两个东西的原因, probs_sort 是根据顺序排序的概率值,
        # probs_idx 是排序后概率值对应的原有的词典中的顺序
        next_token = torch.gather(probs_idx, -1, next_token)  # 对应在词典中的 token id
        return next_token


if __name__ == '__main__':
    allow_cuda = True
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'

    prompts = [
        "Simply put, the theory of relativity states that ",
        "1+1=2 2+3=4 4+10= "
    ]

    model = LLaMA.build(
        checkpoints_dir="c:/model/llama-2-7b/",
        tokenizer_path="c:/model/llama-2-7b/tokenizer.model",
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(prompts),
        device=device
    )

    print("Model is running")

    out_tokens, out_texts = (model.text_completion(prompts, device=device, max_gen_len=64, do_sample=False))
    assert len(out_texts) == len(prompts)
    for i in range(len(out_texts)):
        print(f'{out_texts[i]}')
        print('-' * 50)

    out_tokens, out_texts = (model.text_completion(prompts, device=device, max_gen_len=64, do_sample=True))
    assert len(out_texts) == len(prompts)
    for i in range(len(out_texts)):
        print(f'{out_texts[i]}')
        print('-' * 50)
