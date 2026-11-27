# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Table-driven tests for LwdEdgeWorker.select_token_batch (edge rank replay).

The replay contract: for row ``i``, the returned token id is the
``ranks[i]``-th largest logit of that row (rank 0 = argmax, rank V-1 = the
smallest logit). Ties are resolved by ascending token ID, matching the cloud
rank encoder.

Execution needs the training stack; skipped on hosts without it.
Rerun condition: python>=3.10 with vllm / vllm-ascend installed (CPU tensors
suffice, no NPU required).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip(
    "torch",
    reason="select_token_batch needs torch; rerun on a host with the training stack",
)

from vllm_ascend.worker.lwd_edge_model.lwd_edge_execution import (  # noqa: E402
    replay_accepted_tokens,
)
from vllm_ascend.worker.lwd_edge_model.lwd_edge_model_runner import (  # noqa: E402
    select_token_batch,
)  # noqa: E402


@pytest.mark.parametrize(
    ("logits", "ranks", "expected"),
    [
        # rank 0 picks the argmax token id.
        ([[10, 30, 20]], [0], [1]),
        # rank V-1 picks the smallest logit.
        ([[10, 30, 20]], [2], [0]),
        # Ranks are resolved independently per row.
        ([[10, 30, 20], [3, 1, 2]], [1, 2], [2, 1]),
        # Smallest logit sits at the last column here.
        ([[5, 9, 1]], [2], [2]),
        # Four rows, mixed ranks, distinct values in every row.
        (
            [[7, 5, 3], [1, 2, 3], [9, 0, 5], [4, 6, 8]],
            [1, 0, 2, 1],
            [1, 2, 1, 1],
        ),
        # Equal logits are ordered by ascending token ID.
        ([[10, 10, 10], [5, 5, 5]], [0, 2], [0, 2]),
        ([[10, 10, 0], [10, 10, 0]], [1, 2], [1, 2]),
    ],
    ids=[
        "rank0-argmax",
        "rank-vocab-minus-one",
        "per-row-ranks",
        "min-at-last-column",
        "four-rows-mixed",
        "all-equal",
        "tied-max-and-lower-score",
    ],
)
def test_select_token_batch(logits: list[list[int]], ranks: list[int], expected: list[int]) -> None:
    assert select_token_batch(torch.tensor(logits), ranks) == expected


def replay_all_rows(
    logits: torch.Tensor,
    top_id_ths: list[list[int]],
    num_accept_tokens: list[int],
) -> list[list[int]]:
    flat_ids = select_token_batch(logits, [r for ranks in top_id_ths for r in ranks])
    sampled = []
    offset = 0
    for accept, ranks in zip(num_accept_tokens, top_id_ths):
        sampled.append(flat_ids[offset : offset + accept])
        offset += len(ranks)
    return sampled


@pytest.mark.parametrize(
    ("logits", "top_id_ths", "num_accept_tokens"),
    [
        # Single request: rejected rows must not affect accepted replay.
        (
            [[10, 10, 0], [3, 2, 1], [7, 5, 6]],
            [[1, 0, 2]],
            [2],
        ),
        # Full acceptance keeps the original all-row path.
        (
            [[10, 10, 0], [3, 2, 1], [7, 5, 6], [4, 3, 2]],
            [[1, 0], [2, 1]],
            [2, 2],
        ),
        # Only the middle request has accepted rows.
        (
            [[10, 10, 0], [3, 2, 1], [7, 5, 6], [4, 4, 4], [9, 8, 7], [6, 4, 2]],
            [[0, 1], [0, 2, 1], [1]],
            [0, 2, 0],
        ),
        # Multiple nonempty partial segments take the compatibility path.
        (
            [[10, 10, 0], [3, 2, 1], [7, 5, 6], [4, 4, 4], [8, 6, 4]],
            [[0, 1, 2], [2, 1]],
            [2, 1],
        ),
        # Ties and masked logits retain stable token-ID ordering.
        (
            [[float("-inf"), 1, 1], [5, 5, 5], [2, 2, 2], [0, 1, 0]],
            [[2, 1], [0, 2]],
            [1, 2],
        ),
    ],
    ids=[
        "single-partial",
        "all-accepted",
        "single-nonempty-among-empty",
        "multiple-nonempty-partial",
        "ties-and-negative-inf",
    ],
)
def test_replay_accepted_tokens_matches_all_row_replay(
    logits: list[list[float]],
    top_id_ths: list[list[int]],
    num_accept_tokens: list[int],
) -> None:
    tensor = torch.tensor(logits)
    assert replay_accepted_tokens(tensor, top_id_ths, num_accept_tokens) == replay_all_rows(
        tensor, top_id_ths, num_accept_tokens
    )


def test_replay_all_rejected_returns_aligned_empty_rows() -> None:
    logits = torch.tensor([[3, 2, 1], [2, 1, 0], [1, 0, -1]])
    assert replay_accepted_tokens(logits, [[0, 1], [2]], [0, 0]) == [[], []]
