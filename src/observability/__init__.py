"""可观测性设施（入口无关的横切工具）：统一日志初始化等。

不依赖任何业务层（agents / memory / orchestration / storage），可被任意启动入口安全
import；业务层按需用 ``logging.getLogger(__name__)`` 产生日志，无需 import 本包。
"""

from src.observability.logging_setup import setup_logging

__all__ = ["setup_logging"]
