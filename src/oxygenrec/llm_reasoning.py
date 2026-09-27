"""生成结构化 Slow-LLM reasoning，并显式限制可用证据与输出格式。"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

from .llm_features import LLMFeatureBatch


REQUIRED_REASONING_KEYS = (
    "intent", "evidence", "retrieval_strategy", "retrieval_plan", "constraints",
)


@dataclass(frozen=True)
class GeneratedReasoning:
    """同时保留模型原始文本和通过严格 schema 校验后的字典。"""
    raw_text: str
    parsed: dict[str, object]


class ReasoningGenerationError(ValueError):
    """携带失败样本和原始输出，供上层做可审计的逐条重试。"""

    def __init__(
        self, *, case_index: int, raw_text: str, hit_token_limit: bool,
        max_new_tokens: int, cause: Exception,
    ) -> None:
        self.case_index = case_index
        self.raw_text = raw_text
        self.hit_token_limit = hit_token_limit
        self.max_new_tokens = max_new_tokens
        self.cause = cause
        preview = raw_text[:240].replace("\n", "\\n")
        super().__init__(
            f"reasoning case {case_index} failed schema parsing; "
            f"hit_token_limit={hit_token_limit}; max_new_tokens={max_new_tokens}; "
            f"generated_chars={len(raw_text)}; raw_prefix={preview!r}; {cause}"
        )


def contextual_instruction_text(reasoning: Mapping[str, object]) -> str:
    """把已校验Reasoning转换成论文主线消费的自然语言指令。

    这里只使用intent、evidence和retrieval_strategy，不读取retrieval_plan。
    因而paper_igr与agentic_plan可以共享同一份LLM输出，并把“是否执行Plan”
    保持为唯一实验变量。
    """

    intent = reasoning.get("intent")
    evidence = reasoning.get("evidence")
    strategy = reasoning.get("retrieval_strategy")
    if not isinstance(intent, str) or not intent.strip():
        raise ValueError("reasoning intent must be a non-empty string")
    if not isinstance(strategy, str) or not strategy.strip():
        raise ValueError("reasoning retrieval_strategy must be a non-empty string")
    if not isinstance(evidence, list) or not evidence or not all(
        isinstance(item, str) and item.strip() for item in evidence
    ):
        raise ValueError("reasoning evidence must contain non-empty strings")
    return (
        f"当前意图：{intent.strip()}\n"
        f"推理依据：{'；'.join(item.strip() for item in evidence)}\n"
        f"推荐指令：{strategy.strip()}"
    )


def reasoning_system_prompt() -> str:
    """返回约束 Qwen 输出可审计 Retrieval Plan 的 system prompt。"""
    return (
        "你是电商检索规划器。只能使用用户提供的历史证据，不得猜测下一次真实行为、"
        "目标商品或未提供的属性。只输出一个JSON对象，不输出Markdown。JSON必须包含"
        "intent、evidence、retrieval_strategy、retrieval_plan、constraints五个字段。"
        "evidence必须是1到6条聚合事实，不能逐条重复view；constraints必须是字符串数组；"
        "intent和retrieval_strategy必须是简短字符串。retrieval_plan必须是JSON对象，且只包含"
        "priority_behaviors（只能从view/addtocart/transaction选择的数组）、recency"
        "（recent/long_term/balanced之一）、prefer_repeated_items（布尔值）和diversity"
        "（low/medium/high之一）。"
    )


def parse_reasoning_json(text: str) -> dict[str, object]:
    """提取第一个完整 JSON 对象，并严格校验 Reasoning/Plan schema。

    允许完整 JSON 前后出现少量多余文本，但不会猜测或修补残缺 JSON。
    """

    start = text.find("{")
    if start < 0:
        raise ValueError("generated text does not contain a JSON object")
    try:
        parsed, _ = json.JSONDecoder().raw_decode(text, idx=start)
    except json.JSONDecodeError as error:
        raise ValueError(
            "generated JSON is incomplete or invalid: "
            f"{error.msg} at line {error.lineno} column {error.colno}"
        ) from error
    if not isinstance(parsed, dict):
        raise ValueError("generated JSON root must be an object")
    missing = [key for key in REQUIRED_REASONING_KEYS if key not in parsed]
    if missing:
        raise ValueError(f"generated reasoning is missing keys: {missing}")
    if not isinstance(parsed["intent"], str) or not isinstance(
        parsed["retrieval_strategy"], str
    ):
        raise ValueError("intent and retrieval_strategy must be strings")
    for key in ("evidence", "constraints"):
        if not isinstance(parsed[key], list) or not all(
            isinstance(value, str) for value in parsed[key]
        ):
            raise ValueError(f"{key} must be a list of strings")
    if not 1 <= len(parsed["evidence"]) <= 6:
        raise ValueError("evidence must contain 1 to 6 aggregated facts")
    plan = parsed["retrieval_plan"]
    if not isinstance(plan, dict) or set(plan) != {
        "priority_behaviors", "recency", "prefer_repeated_items", "diversity",
    }:
        raise ValueError("retrieval_plan has the wrong fields")
    priorities = plan["priority_behaviors"]
    if not isinstance(priorities, list) or not priorities or any(
        value not in {"view", "addtocart", "transaction"} for value in priorities
    ):
        raise ValueError("retrieval_plan priority_behaviors is invalid")
    if plan["recency"] not in {"recent", "long_term", "balanced"}:
        raise ValueError("retrieval_plan recency is invalid")
    if not isinstance(plan["prefer_repeated_items"], bool):
        raise ValueError("retrieval_plan prefer_repeated_items must be boolean")
    if plan["diversity"] not in {"low", "medium", "high"}:
        raise ValueError("retrieval_plan diversity is invalid")
    return parsed


class FrozenLLMReasoningGenerator:
    """使用冻结的本地因果语言模型做确定性结构化生成。"""

    def __init__(
        self, model_path: str | Path, *, device: str = "cuda",
        dtype: str = "bfloat16", max_input_length: int = 512,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = Path(model_path)
        if not path.is_dir():
            raise FileNotFoundError(f"local model directory not found: {path}")
        dtype_by_name = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.device = torch.device(device)
        self.max_input_length = max_input_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False,
        )
        self.tokenizer.padding_side = "left"
        # Causal chat过长时必须保留尾部user约束和assistant JSON prefill。
        self.tokenizer.truncation_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, trust_remote_code=False,
            dtype=dtype_by_name[dtype], low_cpu_mem_usage=True,
        ).to(self.device).eval()
        self.model.requires_grad_(False)

    def generate(
        self, evidence_prompts: Sequence[str], *, max_new_tokens: int = 192,
        generation_seed: int | None = None,
    ) -> list[GeneratedReasoning]:
        """按Qwen官方采样配置生成，并用assistant prefill强制JSON起始。"""
        import torch

        if not evidence_prompts:
            raise ValueError("evidence_prompts must not be empty")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        # assistant prefill 让模型直接续写JSON内容，避免先输出说明或长篇复述。
        # Instruct-2507官方仅支持non-thinking；enable_thinking=False作为兼容性显式门。
        rendered = [
            self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": reasoning_system_prompt()},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "{"},
                ],
                tokenize=False,
                continue_final_message=True,
                enable_thinking=False,
            )
            for prompt in evidence_prompts
        ]
        tokens = self.tokenizer(
            rendered, padding=True, truncation=True,
            max_length=self.max_input_length, return_tensors="pt",
        )
        tokens = {name: value.to(self.device) for name, value in tokens.items()}
        # Instruct-2507官方generation_config为sampling；greedy在当前失败样本上
        # 发生了无JSON的长输出。固定batch seed，在保持可复现的同时使用官方参数。
        with torch.inference_mode():
            if generation_seed is not None:
                torch.manual_seed(generation_seed)
            generation_config = copy.deepcopy(self.model.generation_config)
            generation_config.do_sample = True
            generation_config.temperature = 0.7
            generation_config.top_p = 0.8
            generation_config.top_k = 20
            generated = self.model.generate(
                **tokens, generation_config=generation_config,
                max_new_tokens=max_new_tokens, use_cache=True,
            )
        # decoder-only generate 返回“输入+新 token”，所以解码时先切掉输入部分。
        prompt_length = tokens["input_ids"].shape[1]
        results = []
        for index, row in enumerate(generated):
            continuation = self.tokenizer.decode(
                row[prompt_length:], skip_special_tokens=True,
            ).strip()
            raw = continuation if continuation.startswith("{") else "{" + continuation
            try:
                parsed = parse_reasoning_json(raw)
            except ValueError as error:
                hit_token_limit = row.shape[0] >= prompt_length + max_new_tokens
                raise ReasoningGenerationError(
                    case_index=index,
                    raw_text=raw,
                    hit_token_limit=hit_token_limit,
                    max_new_tokens=max_new_tokens,
                    cause=error,
                ) from error
            results.append(GeneratedReasoning(raw_text=raw, parsed=parsed))
        return results

    def encode_instruction_texts(
        self, texts: Sequence[str], *, pooling: str = "last_token",
    ) -> LLMFeatureBatch:
        """复用同一冻结Qwen，把生成的指令文本编码为Tensor[B,H]。

        论文部署将LLM生成和Adapter文本编码拆成两个服务。本公开复现为了避免
        在单卡上同时加载两份4B权重，复用同一Qwen backbone提取hidden state；
        输出随后仍由OxygenREC的可训练instruction adapter完成特征映射。
        """
        import torch
        import torch.nn.functional as F

        if not texts or any(not text.strip() for text in texts):
            raise ValueError("texts must contain non-empty strings")
        if pooling not in {"mean", "last_token"}:
            raise ValueError("pooling must be mean or last_token")
        tokens = self.tokenizer(
            list(texts), padding=True, truncation=True,
            max_length=self.max_input_length, return_tensors="pt",
        )
        tokens = {name: value.to(self.device) for name, value in tokens.items()}
        with torch.inference_mode():
            outputs = self.model(
                **tokens, output_hidden_states=True, return_dict=True,
                use_cache=False,
            )
            hidden = outputs.hidden_states[-1]
            mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            if pooling == "mean":
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            else:
                positions = torch.arange(hidden.shape[1], device=self.device)
                last_indices = (
                    tokens["attention_mask"] * positions.unsqueeze(0)
                ).argmax(dim=1)
                pooled = hidden[
                    torch.arange(hidden.shape[0], device=self.device), last_indices
                ]
            features = F.normalize(pooled.float(), dim=-1)
        features = features.clone()
        counts = tuple(
            int(value) for value in tokens["attention_mask"].sum(dim=1).tolist()
        )
        return LLMFeatureBatch(features=features, token_counts=counts)
