"""Language 回路集成：affect↔language 双向收敛、终止上限、默认关零回归。"""

from __future__ import annotations

from src.agents.language import LanguageDraft
from src.orchestration.runner import run
from src.orchestration.state import Stimulus


class _DriftModel:
    """语言情感固定偏向 (1,1)，制造与内核 e* 的不一致以驱动双向回路。"""

    async def generate(
        self,
        *,
        affect: tuple[float, float],
        context: str,
        retrieved: str,
        feedback: str | None,
    ) -> LanguageDraft:
        return LanguageDraft(text=f"drift:{context}", affect=(1.0, 1.0))


async def test_language_loop_bounded_and_woven_into_expression() -> None:
    traj = await run(
        [Stimulus(name="evt", goal_congruence=0.2, intensity=0.5)],
        thread_id="t-lang",
        language_enabled=True,
        language_max_iters=3,
        language_model=_DriftModel(),
        rng_seed=7,
    )
    step = traj[0]
    assert step["language_text"] is not None
    assert 1 <= step["language_iter"] <= 3  # 回路跑过且不超过终止上限
    assert "language" in step["expression"]  # 语言并入最终表现


async def test_language_disabled_zero_regression() -> None:
    traj = await run(
        [Stimulus(name="evt", goal_congruence=0.2, intensity=0.5)],
        thread_id="t-nolang",
        language_enabled=False,
        rng_seed=7,
    )
    step = traj[0]
    assert step["language_text"] is None
    assert step["language_iter"] == 0
    assert "language" not in step["expression"]
