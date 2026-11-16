# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import torch

from vllm_ascend.worker.lwd_base_model.lwd_pinned_staging import LwdPinnedStaging


class LwdStagingMixin:
    device: torch.device
    _lwd_staging_rings: dict[str, LwdPinnedStaging]

    def _lwd_staging(
        self,
        name: str,
        *,
        dtype: torch.dtype,
        min_capacity: int,
        device_mirror: bool = True,
    ) -> LwdPinnedStaging:
        """Return a named lazy staging ring shared by the V1 and V2 runners."""
        ring = self._lwd_staging_rings.get(name)
        if ring is None:
            ring = LwdPinnedStaging(
                dtype=dtype,
                min_capacity=min_capacity,
                device=self.device if device_mirror else None,
            )
            self._lwd_staging_rings[name] = ring
        return ring
