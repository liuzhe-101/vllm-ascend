# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Any

import torch
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.lwd_control.lwd_comm.lwd_link_protocol import LwdRunnerKind

from vllm_ascend.worker.lwd_base_model.lwd_pinned_staging import LwdPinnedStaging
from vllm_ascend.worker.lwd_base_model.lwd_worker_base import LwdWorkerBase
from vllm_ascend.worker.lwd_edge_model.lwd_edge_model_runner import (
    LwdEdgeModelRunner,
)
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


class LwdEdgeModelRunnerV2(LwdEdgeModelRunner, NPUModelRunner):
    """LWD direct execution with native MRV2 construction and model loading."""

    runner_kind: LwdRunnerKind = "v2"

    def __init__(
        self,
        vllm_config: Any,
        device: torch.device,
        worker: LwdWorkerBase | None = None,
    ) -> None:
        NPUModelRunner.__init__(self, vllm_config, device)
        self.worker = worker
        self.pin_memory = PIN_MEMORY
        self._lwd_staging_rings: dict[str, LwdPinnedStaging] = {}
        self._init_lwd_edge_execution(worker)

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        NPUModelRunner.load_model(self, *args, **kwargs)
        self._load_lwd_edge_hash_tables()
