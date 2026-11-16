# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""LWD edge compute host with the native V1 model-runner lifecycle."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm_ascend.worker.lwd_base_model.lwd_base_model_runner import LwdBaseModelRunner
from vllm_ascend.worker.lwd_edge_model.lwd_edge_execution import (
    LwdEdgeExecutionMixin,
    select_token_batch,
)

__all__ = ["LwdEdgeModelRunner", "select_token_batch"]

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from vllm_ascend.worker.lwd_base_model.lwd_worker_base import LwdWorkerBase


class LwdEdgeModelRunner(LwdEdgeExecutionMixin, LwdBaseModelRunner):
    """LWD edge direct execution using the native V1 model runner."""

    runner_kind = "v1"

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        worker: LwdWorkerBase | None = None,
    ) -> None:
        super().__init__(vllm_config, device, worker)
        self._init_lwd_edge_execution(worker)

    def load_model(self) -> None:
        super().load_model()
        self._load_lwd_edge_hash_tables()
