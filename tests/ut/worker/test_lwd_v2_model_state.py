# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from transformers import Qwen3_5TextConfig
from vllm.config import VllmConfig
from vllm.config.device import DeviceConfig
from vllm.config.lwd import resolve_lwd_config
from vllm.config.model import ModelConfig
from vllm.v1.worker.gpu.lwd.cloud_input import LwdCloudBatchView
from vllm.v1.worker.gpu.lwd.model_state import LwdCloudModelStateMixin

from vllm_ascend.ascend_config import init_ascend_config
from vllm_ascend.worker.lwd_cloud_model.v2.model_state import LwdAscendCloudModelState
from vllm_ascend.worker.v2.model_states import _resolve_ascend_model_state_cls
from vllm_ascend.worker.v2.model_states.mamba_hybrid import AscendMambaHybridModelState


def _cloud_config(tmp_path: Path, layer_type: str) -> VllmConfig:
    hf_config = Qwen3_5TextConfig(
        architectures=["Qwen3_5ForCausalLM"],
        hidden_size=2,
        vocab_size=8,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=2,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        layer_types=[layer_type],
        max_position_embeddings=16,
    )
    hf_config.save_pretrained(tmp_path)
    topology_path = tmp_path / "topology.yaml"
    topology_path.write_text(
        yaml.safe_dump(
            {
                "deployment": {
                    "mode": 0,
                    "scene": "single_instance",
                    "hccl_world_size": 2,
                    "edges_num": 1,
                    "clouds_num": 1,
                },
                "feature_ctrl": {
                    "enable_early_recv": False,
                    "scramble": {
                        "enable": False,
                        "device_id": None,
                        "custom_factor": None,
                    },
                },
                "edges": [{"id": 0, "dp": [{"dp_idx": 0, "addr": "10.0.0.1", "ranks": [0]}]}],
                "clouds": [
                    {
                        "id": 0,
                        "dp": [
                            {
                                "dp_idx": 0,
                                "addr": "10.0.0.2",
                                "ranks": [1],
                                "ctrl_port": 5550,
                            }
                        ],
                    }
                ],
                "instance_links": [{"edge": 0, "cloud": 0}],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    vllm_config = VllmConfig(
        model_config=ModelConfig(model=str(tmp_path), max_model_len=16),
        device_config=DeviceConfig(device="cpu"),
    )
    init_ascend_config(vllm_config)
    vllm_config.lwd_config = resolve_lwd_config(
        {
            "lwd_config": {
                "path": str(topology_path),
                "role": "cloud",
                "instance_id": 0,
            }
        }
    )
    return vllm_config


def test_qwen35_linear_attention_composes_lwd_and_mamba_state(tmp_path: Path) -> None:
    vllm_config = _cloud_config(tmp_path, "linear_attention")
    assert vllm_config.model_config.is_hybrid

    state_cls = _resolve_ascend_model_state_cls(vllm_config, torch.nn.Module())

    assert state_cls.__name__ == "LwdAscendCloudMambaHybridModelState"
    assert issubclass(state_cls, LwdCloudModelStateMixin)
    assert issubclass(state_cls, AscendMambaHybridModelState)
    assert state_cls.prepare_inputs_embeds is LwdCloudModelStateMixin.prepare_inputs_embeds
    assert state_cls.prepare_attn is AscendMambaHybridModelState.prepare_attn
    assert state_cls.compute_sample_logits is LwdCloudModelStateMixin.compute_sample_logits

    state = state_cls(vllm_config, torch.nn.Module(), None, torch.device("cpu"))
    assert state.prompt_embeds_state is None
    assert state.dummy_inputs_embeds(1).data_ptr() == state.remote_inputs_embeds.data_ptr()


def test_qwen35_full_attention_keeps_default_lwd_state(tmp_path: Path) -> None:
    vllm_config = _cloud_config(tmp_path, "full_attention")
    assert not vllm_config.model_config.is_hybrid

    state_cls = _resolve_ascend_model_state_cls(vllm_config, torch.nn.Module())

    assert state_cls is LwdAscendCloudModelState


def test_model_specific_state_is_not_replaced_by_lwd_hybrid(tmp_path: Path) -> None:
    vllm_config = _cloud_config(tmp_path, "linear_attention")

    class ModelWithStateOverride(torch.nn.Module):
        def get_model_state_cls(self) -> type[AscendMambaHybridModelState]:
            return AscendMambaHybridModelState

    state_cls = _resolve_ascend_model_state_cls(vllm_config, ModelWithStateOverride())

    assert state_cls is AscendMambaHybridModelState


class _PromptStage:
    def __init__(self, host: torch.Tensor) -> None:
        self.host = host
        self.consumed = 0

    def take_req(self, req_id: str) -> "_PromptStage":
        return self

    def recv_rows(self, req_id: str) -> int:
        return self.host.numel() // 2

    def consumed_rows(self, req_id: str) -> int:
        return self.consumed

    def segment_for(self, req_id: str, n_rows: int) -> tuple[int, int, int]:
        return self.consumed * 2, n_rows * 2, 0

    def consume(self, req_id: str, n_rows: int) -> None:
        self.consumed += n_rows


def test_cloud_leader_broadcasts_staged_prompt_window(tmp_path: Path, monkeypatch) -> None:
    vllm_config = _cloud_config(tmp_path, "full_attention")
    state = LwdAscendCloudModelState(vllm_config, torch.nn.Module(), None, torch.device("cpu"))
    stage = _PromptStage(torch.arange(1, 9, dtype=torch.bfloat16))
    state.bind_lwd_prompt_stage(stage)
    view = LwdCloudBatchView(
        req_ids=["r1"],
        idx_mapping_np=np.array([3]),
        num_scheduled_tokens=np.array([2], dtype=np.int32),
        num_computed_tokens_np=np.array([0], dtype=np.int32),
        prefill_len_np=np.array([4], dtype=np.int32),
        query_start_loc_np=np.array([0, 2], dtype=np.int32),
    )
    broadcasts = []

    def broadcast(tensor, *, src, group, async_op):
        broadcasts.append((tensor.clone(), src, group, async_op))
        return SimpleNamespace(wait=lambda: None)

    class _Stream:
        def wait_event(self, event) -> None:
            pass

    class _Event:
        def record(self, stream=None) -> None:
            pass

    current_stream = _Stream()

    @contextmanager
    def stream_context(stream) -> None:
        yield

    monkeypatch.setattr("torch.distributed.broadcast", broadcast)
    monkeypatch.setattr(
        "vllm_ascend.worker.lwd_cloud_model.v2.cloud_input.get_tp_group",
        lambda: SimpleNamespace(world_size=2, ranks=[4, 5], device_group="hccl"),
    )
    monkeypatch.setattr("torch.cuda.Stream", lambda device: _Stream())
    monkeypatch.setattr("torch.cuda.Event", _Event)
    monkeypatch.setattr("torch.cuda.stream", stream_context)
    monkeypatch.setattr("torch.cuda.current_stream", lambda: current_stream)
    inputs_embeds = torch.zeros((2, 2), dtype=torch.bfloat16)
    next_prompt_embeds = torch.zeros((4, 2), dtype=torch.bfloat16)

    state.lwd_input_assembler.stage_prompt_window(view, inputs_embeds, next_prompt_embeds)

    assert inputs_embeds.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert stage.consumed == 2
    assert len(broadcasts) == 1
    assert broadcasts[0][0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert broadcasts[0][1:] == (4, "hccl", True)
