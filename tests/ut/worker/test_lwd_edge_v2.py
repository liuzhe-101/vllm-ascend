# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import pytest
import torch
from transformers import PreTrainedConfig
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.config.device import DeviceConfig
from vllm.config.model import ModelConfig
from vllm.distributed.parallel_state import (
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)

from vllm_ascend.ascend_config import init_ascend_config
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


@pytest.fixture
def cpu_device_boundaries(monkeypatch):
    from vllm.platforms import current_platform

    monkeypatch.setattr(type(current_platform), "device_type", "cpu")
    monkeypatch.setattr("vllm.v1.worker.gpu.buffer_utils.is_uva_available", lambda: False)
    for name in ("empty", "zeros", "tensor"):
        original = getattr(torch, name)

        def allocate(*args, _original=original, **kwargs):
            kwargs.pop("pin_memory", None)
            return _original(*args, **kwargs)

        monkeypatch.setattr(torch, name, allocate)
    yield


class _Qwen3Config(PreTrainedConfig):
    model_type = "qwen3"


def _model_config(tmp_path) -> ModelConfig:
    hf_config = _Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        hidden_size=2,
        vocab_size=8,
        num_hidden_layers=0,
        num_attention_heads=1,
        num_key_value_heads=1,
    )
    hf_config.model_type = "qwen3"
    hf_config.save_pretrained(tmp_path)
    return ModelConfig(model=str(tmp_path), max_model_len=16)


def test_edge_v2_constructs_native_runner(tmp_path, monkeypatch, cpu_device_boundaries) -> None:
    monkeypatch.chdir(tmp_path)
    vllm_config = VllmConfig(
        model_config=_model_config(tmp_path),
        device_config=DeviceConfig(device="cpu"),
    )
    init_ascend_config(vllm_config)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("MASTER_ADDR", "localhost")
    monkeypatch.setenv("MASTER_PORT", "29517")
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(backend="gloo")
        initialize_model_parallel()
        try:
            runner = LwdEdgeModelRunnerV2(vllm_config, torch.device("cpu"))
        finally:
            destroy_model_parallel()

    assert runner.runner_kind == "v2"
    assert runner.use_dcp is False
    assert runner._lwd_staging_rings == {}
