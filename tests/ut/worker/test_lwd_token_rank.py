# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Regression tests for deterministic LWD cloud ranking and edge replay."""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="LWD rank replay needs torch")

from vllm_ascend.worker.lwd_cloud_model.lwd_cloud_model_runner import (  # noqa: E402
    LwdCloudModelRunner,
)
from vllm_ascend.worker.lwd_cloud_model.lwd_token_rank import (  # noqa: E402
    LwdTokenRankEncoder,
)
from vllm_ascend.worker.lwd_edge_model.lwd_edge_model_runner import (  # noqa: E402
    select_token_batch,
)


def assert_round_trip(
    encoder: LwdTokenRankEncoder,
    logits: torch.Tensor,
    selected: torch.Tensor,
) -> torch.Tensor:
    original_logits = logits.clone()
    original_selected = selected.clone()
    ranks = encoder.encode(logits, selected)
    assert select_token_batch(logits, ranks.tolist()) == selected.tolist()
    assert torch.equal(logits, original_logits)
    assert torch.equal(selected, original_selected)
    return ranks


def make_runner() -> LwdCloudModelRunner:
    runner = LwdCloudModelRunner.__new__(LwdCloudModelRunner)
    runner._lwd_token_rank_encoder = LwdTokenRankEncoder()
    return runner


def test_every_candidate_in_tied_and_non_max_groups() -> None:
    encoder = LwdTokenRankEncoder()
    values = torch.tensor(
        [12.0, 10.0, 10.0, 10.0, -2.0, -2.0],
        dtype=torch.bfloat16,
    )
    selected = torch.arange(len(values))
    ranks = assert_round_trip(encoder, values.repeat(len(values), 1), selected)
    assert ranks.tolist() == list(range(len(values)))


def test_round_trip_many_quantized_ties() -> None:
    generator = torch.Generator().manual_seed(7)
    logits = torch.randint(-4, 5, (31, 257), generator=generator).to(torch.bfloat16)
    selected = torch.randint(0, 257, (31,), generator=generator)
    assert_round_trip(LwdTokenRankEncoder(), logits, selected)


def test_production_vocab_and_reported_tied_ids() -> None:
    logits = torch.zeros(1, 129280, dtype=torch.bfloat16)
    logits[0, 9213] = 42.75
    logits[0, 9828] = 42.75
    ranks = assert_round_trip(
        LwdTokenRankEncoder(),
        logits,
        torch.tensor([9828]),
    )
    assert ranks.tolist() == [1]


def test_masked_logits_and_all_equal_scores() -> None:
    logits = torch.tensor(
        [
            [float("-inf"), 1.0, 1.0, float("-inf")],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    ranks = assert_round_trip(
        LwdTokenRankEncoder(),
        logits,
        torch.tensor([3, 2]),
    )
    assert ranks.tolist() == [3, 2]


def test_rejected_slots_are_safe_and_vocab_index_is_reused() -> None:
    encoder = LwdTokenRankEncoder()
    logits = torch.tensor([[2.0, 2.0, 1.0]]).repeat(3, 1)
    ranks = encoder.encode(logits, torch.tensor([1, -1, 2]))
    assert ranks.tolist() == [1, 0, 2]
    cached = encoder.vocab_ids
    encoder.encode(logits[:1], torch.tensor([0]))
    assert encoder.vocab_ids is cached
    assert encoder.encode(torch.ones(1, 5), torch.tensor([4])).tolist() == [4]
    assert encoder.vocab_ids is not None
    assert encoder.vocab_ids.numel() == 5


def test_plain_collector_uses_deterministic_ranking() -> None:
    runner = make_runner()
    logits = torch.tensor([[10.0, 10.0, 0.0]]).repeat(2, 1)
    hidden = torch.arange(4).reshape(2, 2).float()
    selected = torch.tensor([[2], [0]])
    selection = runner._lwd_pack_plain_down_rows(
        hidden,
        logits,
        selected,
        np.array([True, True]),
        ["a", "b"],
    )
    assert selection is not None
    assert selection.hidden_packet is hidden
    assert selection.ranks[0].tolist() == [2, 0]
    assert select_token_batch(logits, selection.ranks[0].tolist()) == [2, 0]


def test_plain_collector_partial_cover_keeps_selected_rows() -> None:
    runner = make_runner()
    logits = torch.tensor([[10.0, 10.0, 0.0]]).repeat(3, 1)
    hidden = torch.arange(6).reshape(3, 2).float()
    selected = torch.tensor([[2], [0], [1]])
    selection = runner._lwd_pack_plain_down_rows(
        hidden,
        logits,
        selected,
        np.array([True, False, True]),
        ["a", "b", "c"],
    )
    assert selection is not None
    assert torch.equal(selection.hidden_packet, hidden[[0, 2]])
    assert select_token_batch(logits[[0, 2]], selection.ranks[0].tolist()) == [2, 1]


@pytest.mark.parametrize(
    ("sampled_values", "expected_ranks", "expected_count"),
    [
        ([1, 2, -1], [1, 2, 0], 2),
        ([1, 2, 0], [1, 2, 2], 3),
    ],
    ids=["masked-rejected-slot", "all-accepted"],
)
def test_spec_collector_single_request_full_cover_skips_index_staging(
    sampled_values: list[int], expected_ranks: list[int], expected_count: int
) -> None:
    runner = make_runner()

    def fail_index_staging(_arr: np.ndarray) -> torch.Tensor:
        raise AssertionError("full-cover single-request path must not stage indices")

    runner._lwd_h2d_stage_i64 = fail_index_staging
    logits = torch.tensor([[10.0, 10.0, 0.0]]).repeat(3, 1)
    hidden = torch.arange(6).reshape(3, 2).float()
    selection = runner._lwd_pack_spec_down_rows(
        hidden,
        logits,
        torch.tensor([sampled_values]),
        SimpleNamespace(num_draft_tokens=[2]),
        np.array([True]),
        ["a"],
    )
    assert selection is not None
    assert selection.hidden_packet.data_ptr() == hidden.data_ptr()
    assert selection.ranks[0].tolist() == expected_ranks
    assert selection.counts_dev is not None
    assert selection.counts_dev.tolist() == [expected_count]
    assert selection.seg_lens_host == [3]
    assert (
        select_token_batch(logits[:expected_count], selection.ranks[0][:expected_count].tolist())
        == sampled_values[:expected_count]
    )


def test_spec_collector_multiple_requests_keep_general_index_path() -> None:
    runner = make_runner()
    staging_calls = 0

    def index_staging(arr: np.ndarray) -> torch.Tensor:
        nonlocal staging_calls
        staging_calls += 1
        return torch.from_numpy(arr)

    runner._lwd_h2d_stage_i64 = index_staging
    logits = torch.tensor([[10.0, 10.0, 0.0]]).repeat(4, 1)
    hidden = torch.arange(8).reshape(4, 2).float()
    selection = runner._lwd_pack_spec_down_rows(
        hidden,
        logits,
        torch.tensor([[1, 2], [2, -1]]),
        SimpleNamespace(num_draft_tokens=[1, 1]),
        np.array([True, True]),
        ["a", "b"],
    )
    assert selection is not None
    assert staging_calls == 1
    assert torch.equal(selection.hidden_packet, hidden)
    assert selection.counts_dev is not None
    assert selection.counts_dev.tolist() == [2, 1]
    assert selection.seg_lens_host == [2, 2]
    assert select_token_batch(logits[:3], selection.ranks[0][:3].tolist()) == [1, 2, 2]


def test_spec_collector_partial_cover_keeps_general_index_path() -> None:
    runner = make_runner()
    staging_calls = 0

    def index_staging(arr: np.ndarray) -> torch.Tensor:
        nonlocal staging_calls
        staging_calls += 1
        return torch.from_numpy(arr)

    runner._lwd_h2d_stage_i64 = index_staging
    logits = torch.tensor([[10.0, 10.0, 0.0]]).repeat(3, 1)
    hidden = torch.arange(6).reshape(3, 2).float()
    selection = runner._lwd_pack_spec_down_rows(
        hidden,
        logits,
        torch.tensor([[9, 9, 9], [1, 2, -1]]),
        SimpleNamespace(num_draft_tokens=[0, 2]),
        np.array([False, True]),
        ["a", "b"],
    )
    assert selection is not None
    assert staging_calls == 1
    assert torch.equal(selection.hidden_packet, hidden[1:3])
    assert selection.counts_dev is not None
    assert selection.counts_dev.tolist() == [2]
    assert selection.seg_lens_host == [2]
    assert select_token_batch(logits[1:3], selection.ranks[0].tolist()) == [1, 2]
