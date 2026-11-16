# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Common base for the LWD edge/cloud model runners.

Owns what both data-plane runners share: the worker back-reference
(the comm service and the LWD config live on the worker) and the named
lazy staging factory that three call sites used to hand-roll.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm_ascend.worker.lwd_base_model.lwd_pinned_staging import LwdPinnedStaging
from vllm_ascend.worker.lwd_base_model.lwd_staging_mixin import LwdStagingMixin
from vllm_ascend.worker.v1.model_runner import NPUModelRunner

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from vllm_ascend.worker.lwd_base_model.lwd_worker_base import LwdWorkerBase


class LwdBaseModelRunner(LwdStagingMixin, NPUModelRunner):
    """NPUModelRunner base shared by the edge/cloud LWD runners."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        worker: LwdWorkerBase | None = None,
    ) -> None:
        super().__init__(vllm_config, device)
        self.worker = worker  # LwdWorkerBase ref (comm service lives there)
        # Named lazy staging rings; see _lwd_staging.
        self._lwd_staging_rings: dict[str, LwdPinnedStaging] = {}
