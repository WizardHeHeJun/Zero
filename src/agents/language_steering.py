"""SteeringLanguageModel：表示空间 steering 的 LanguageModel 适配器（开放权重）。

承接 notes/2026-06-24-text-output-emotion.md 路径 2：LLM 激活空间里情绪排成 VA 环
（circumplex），可线性操控。给定目标 e*=(valence, arousal)，按强度 α 把
    steering = α·(V·w_V + A·w_A)
加到目标层隐状态（VA 子空间环状几何，arXiv 2604.03147；历史源头 = sentiment neuron）。
内核 e*=(v,a)∈ℝ² 即天然 steering 坐标，与 affect 数学内核同构。

分层设计（与项目「torch 隔离」一致）：
- 纯函数（list[float] 运算、torch-free、可单测）：`axis_from_contrast`（对比法求轴，eq.1）、
  `l2_normalize`、`orthogonalize`（Gram-Schmidt 令 arousal 轴 ⟂ valence 轴）、`steering_delta`。
- `SteerBackend` 协议（注入缝）：默认 `_TransformersSteerBackend` 延迟 import torch/transformers、
  注册前向 hook 加 delta；测试/自定义注入 fake backend 即可在无 torch 下覆盖编排逻辑。
- 反推情感：无 LLM 打分时用 `emotion_lexicon.appraise_text`（词典法）满足 LanguageModel 协议。

真接入需 `pip install -e ".[steer]"` + 指定 HF 模型（`model_name`）。编排层与默认路径不依赖 torch。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from src.agents.emotion_lexicon import appraise_text
from src.agents.language import LanguageDraft

# steering 默认强度（arXiv 2604.03147 用 α≈0.45：可靠且不过度伤语法）
DEFAULT_ALPHA = 0.45


def _mean_vec(rows: list[list[float]]) -> list[float]:
    """逐维均值；空输入返回空向量。"""
    if not rows:
        return []
    n = len(rows)
    dim = len(rows[0])
    return [sum(r[d] for r in rows) / n for d in range(dim)]


def axis_from_contrast(
    emotion_acts: list[list[float]], neutral_acts: list[list[float]]
) -> list[float]:
    """对比法求情绪向量（eq.1）：mean(情绪激活) − mean(中性激活)，逐维。"""
    e = _mean_vec(emotion_acts)
    z = _mean_vec(neutral_acts)
    if len(e) != len(z):
        raise ValueError("情绪/中性激活维度不一致")
    return [ei - zi for ei, zi in zip(e, z, strict=True)]


def l2_normalize(vec: list[float]) -> list[float]:
    """单位化；零向量原样返回（避免除零）。"""
    norm = sum(x * x for x in vec) ** 0.5
    if norm == 0.0:
        return list(vec)
    return [x / norm for x in vec]


def orthogonalize(target: list[float], reference: list[float]) -> list[float]:
    """Gram-Schmidt：从 target 去掉在 reference 方向上的分量（令两轴正交）。"""
    ref = l2_normalize(reference)
    proj = sum(t * r for t, r in zip(target, ref, strict=True))
    return [t - proj * r for t, r in zip(target, ref, strict=True)]


def steering_delta(
    valence: float,
    arousal: float,
    w_valence: list[float],
    w_arousal: list[float],
    *,
    alpha: float = DEFAULT_ALPHA,
) -> list[float]:
    """steering 向量：α·(V·w_V + A·w_A)，逐维。w_V/w_A 应已单位化且正交。"""
    if len(w_valence) != len(w_arousal):
        raise ValueError("valence/arousal 轴维度不一致")
    return [
        alpha * (valence * wv + arousal * wa) for wv, wa in zip(w_valence, w_arousal, strict=True)
    ]


def _build_prompt(context: str, retrieved: str, feedback: str | None, appraisal: str) -> str:
    """把上下文/检索/反馈/评价拼成生成提示（情感由 steering 注入，不靠提示词描述坐标）。"""
    parts = [f"关于：{context or '（无）'}"]
    if appraisal:
        parts.append(f"认知评价：{appraisal}")
    if retrieved:
        parts.append(f"记忆：{retrieved}")
    if feedback:
        parts.append(f"上一轮反馈：{feedback}")
    return "\n".join(parts)


class SteerBackend(Protocol):
    """生成后端协议：给定提示与 steering delta，产出文本（默认实现注册 hook 加 delta）。"""

    def generate_steered(self, prompt: str, delta: list[float]) -> str: ...


class SteeringLanguageModel:
    """满足 LanguageModel 协议的 steering 适配器：e* → 隐状态 steering → 文本 → 词典反推情感。"""

    def __init__(
        self,
        *,
        w_valence: list[float],
        w_arousal: list[float],
        alpha: float = DEFAULT_ALPHA,
        backend: SteerBackend | None = None,
        model_name: str | None = None,
        target_layer: int = -1,
        appraiser: Callable[[str], tuple[float, float]] = appraise_text,
    ) -> None:
        # 单位化 valence 轴；arousal 轴先正交化再单位化（paper：Gram-Schmidt 令两轴 ⟂）
        self.w_valence = l2_normalize(w_valence)
        self.w_arousal = l2_normalize(orthogonalize(w_arousal, self.w_valence))
        self.alpha = alpha
        self.appraiser = appraiser
        self.backend = backend
        self.model_name = model_name
        self.target_layer = target_layer

    def _ensure_backend(self) -> SteerBackend:
        if self.backend is None:
            self.backend = _TransformersSteerBackend(self.model_name, self.target_layer)
        return self.backend

    async def generate(
        self,
        *,
        affect: tuple[float, float],
        context: str,
        retrieved: str,
        feedback: str | None,
        appraisal: str = "",
    ) -> LanguageDraft:
        delta = steering_delta(
            affect[0], affect[1], self.w_valence, self.w_arousal, alpha=self.alpha
        )
        prompt = _build_prompt(context, retrieved, feedback, appraisal)
        text = self._ensure_backend().generate_steered(prompt, delta)
        return LanguageDraft(text=text, affect=self.appraiser(text))


class _TransformersSteerBackend:
    """默认后端：延迟 import torch/transformers，前向 hook 把 delta 加到目标层隐状态。"""

    def __init__(self, model_name: str | None, target_layer: int) -> None:
        if not model_name:
            raise ValueError("需提供 model_name（HF 因果 LM）或注入 backend")
        import torch  # 延迟 import：注入 backend 时无需安装 torch/transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch: Any = torch
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        self.model: Any = AutoModelForCausalLM.from_pretrained(model_name)
        self.target_layer = target_layer

    def generate_steered(self, prompt: str, delta: list[float]) -> str:
        torch = self.torch
        vec = torch.tensor(delta, dtype=self.model.dtype)
        layer = self.model.model.layers[self.target_layer]

        def hook(_module: Any, _inp: Any, out: Any) -> Any:
            hidden = out[0] if isinstance(out, tuple) else out
            hidden = hidden + vec.to(hidden.device)
            return (hidden, *out[1:]) if isinstance(out, tuple) else hidden

        handle = layer.register_forward_hook(hook)
        try:
            ids = self.tokenizer(prompt, return_tensors="pt")
            gen = self.model.generate(**ids, max_new_tokens=64)
            prompt_len = ids["input_ids"].shape[1]
            text = self.tokenizer.decode(gen[0][prompt_len:], skip_special_tokens=True)
        finally:
            handle.remove()
        return text.strip()
