# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing import Any

import torch
from vllm.v1.lwd_control.lwd_comm.lwd_link_protocol import LwdRunnerKind

from vllm_ascend.worker.lwd_base_model.lwd_worker_base import LwdWorkerBase
from vllm_ascend.worker.lwd_edge_model.lwd_edge_execution import LwdEdgeExecutionMixin
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


class LwdEdgeModelRunnerV2(LwdEdgeExecutionMixin, NPUModelRunner):
    """LWD direct execution with native MRV2 construction and model loading."""

    runner_kind: LwdRunnerKind = "v2"

    def __init__(
        self,
        vllm_config: Any,
        device: torch.device,
        worker: LwdWorkerBase | None = None,
    ) -> None:
        super().__init__(vllm_config, device)
        self._init_lwd_edge_execution(worker)

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        super().load_model(*args, **kwargs)
        self._load_lwd_edge_hash_tables()
