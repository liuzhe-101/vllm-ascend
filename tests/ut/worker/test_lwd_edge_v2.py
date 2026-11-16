# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest

from vllm_ascend.worker.lwd_base_model.lwd_base_model_runner import (
    LwdBaseModelRunner,
)
from vllm_ascend.worker.lwd_edge_model.lwd_edge_model_runner import (
    LwdEdgeModelRunner,
)
from vllm_ascend.worker.lwd_edge_model.lwd_edge_worker import (
    _resolve_lwd_edge_model_runner_cls,
)
from vllm_ascend.worker.lwd_edge_model.v2.model_runner import (
    LwdEdgeModelRunnerV2,
)
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


@pytest.mark.parametrize(
    ("use_v2", "expected_cls", "expected_kind"),
    [
        (False, LwdEdgeModelRunner, "v1"),
        (True, LwdEdgeModelRunnerV2, "v2"),
    ],
)
def test_edge_worker_selects_runner_matching_engine(
    use_v2: bool, expected_cls: type[LwdEdgeModelRunner], expected_kind: str
) -> None:
    runner_cls = _resolve_lwd_edge_model_runner_cls(use_v2)

    assert runner_cls is expected_cls
    assert runner_cls.runner_kind == expected_kind


def test_edge_v2_loads_model_through_native_v2_once(monkeypatch) -> None:
    runner = LwdEdgeModelRunnerV2.__new__(LwdEdgeModelRunnerV2)
    assert isinstance(runner, NPUModelRunner)
    calls = []

    monkeypatch.setattr(
        NPUModelRunner,
        "load_model",
        lambda self, *args, **kwargs: calls.append("v2"),
    )
    monkeypatch.setattr(
        LwdBaseModelRunner,
        "load_model",
        lambda self, *args, **kwargs: calls.append("v1"),
    )
    monkeypatch.setattr(
        LwdEdgeModelRunner,
        "_load_lwd_edge_hash_tables",
        lambda self: calls.append("hash"),
    )

    runner.load_model()

    assert calls == ["v2", "hash"]
