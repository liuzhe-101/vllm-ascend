# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Bounded, low-overhead timing aggregation for LWD NPU workers."""

from __future__ import annotations

import time

from vllm.logger import init_logger

logger = init_logger("vllm.lwd_meter")

_METER_INTERVAL_S = 300.0
_CLOUD_METRICS = (
    "execute",
    "prepare_total",
    "up_harvest",
    "up_other",
    "native_prepare",
    "inject",
    "hash_engram_prepare",
    "prepare_other",
    "forward",
    "execute_other",
    "sample",
    "native_sample",
    "collect",
    "send",
    "stage",
)
_EDGE_METRICS = (
    "embed_forward",
    "embed_send",
    "unembed_post",
    "wait_tensor",
    "lm_head",
    "select",
)


def _residual(total: float, *parts: float) -> float:
    # Host wall-clock intervals can overlap through asynchronous work; these
    # residuals attribute host time only and must not be read as device time.
    return max(0.0, total - sum(parts))


class LwdWorkerSpan:
    __slots__ = ("count", "_max_s", "_total_s")

    def __init__(self) -> None:
        self.count = 0
        self._total_s = 0.0
        self._max_s = 0.0

    def add(self, seconds: float) -> None:
        self.count += 1
        self._total_s += seconds
        self._max_s = max(self._max_s, seconds)

    def __str__(self) -> str:
        if not self.count:
            return "-/-"
        return f"{self._total_s / self.count * 1000:.1f}/{self._max_s * 1000:.1f}"


class LwdWorkerMeter:
    """Aggregate worker timings without logging on the per-step hot path."""

    def __init__(self, label: str, edge: bool = False) -> None:
        self._label = label
        self._edge = edge
        self._last_report = time.monotonic()
        self._execute_prepare_total = 0.0
        self._execute_forward = 0.0
        self._prepare_up_stage = 0.0
        self._reset()

    def _reset(self) -> None:
        names = _EDGE_METRICS if self._edge else _CLOUD_METRICS
        self._spans = {name: LwdWorkerSpan() for name in names}

    def _note(self, name: str, seconds: float) -> None:
        self._note_many(((name, seconds),))

    def _note_many(self, samples: tuple[tuple[str, float], ...]) -> None:
        for name, seconds in samples:
            self._spans[name].add(seconds)
        now = time.monotonic()
        if now - self._last_report < _METER_INTERVAL_S:
            return
        self._last_report = now
        self._report()
        self._reset()

    def begin_execute(self) -> None:
        self._execute_prepare_total = 0.0
        self._execute_forward = 0.0
        self._prepare_up_stage = 0.0

    def note_execute(self, seconds: float) -> None:
        execute_other = _residual(seconds, self._execute_prepare_total, self._execute_forward)
        self._execute_prepare_total = 0.0
        self._execute_forward = 0.0
        self._prepare_up_stage = 0.0
        self._note_many((("execute", seconds), ("execute_other", execute_other)))

    def note_prepare(
        self,
        prepare_total: float,
        up_harvest: float,
        native_prepare: float,
        inject: float,
        hash_engram_prepare: float,
    ) -> None:
        up_other = _residual(up_harvest, self._prepare_up_stage)
        prepare_other = _residual(
            prepare_total,
            up_harvest,
            native_prepare,
            inject,
            hash_engram_prepare,
        )
        self._prepare_up_stage = 0.0
        self._execute_prepare_total += prepare_total
        self._note_many(
            (
                ("prepare_total", prepare_total),
                ("up_harvest", up_harvest),
                ("up_other", up_other),
                ("native_prepare", native_prepare),
                ("inject", inject),
                ("hash_engram_prepare", hash_engram_prepare),
                ("prepare_other", prepare_other),
            )
        )

    def note_forward(self, seconds: float) -> None:
        self._execute_forward += seconds
        self._note("forward", seconds)

    def note_native_sample(self, seconds: float) -> None:
        self._note("native_sample", seconds)

    def note_sample(self, sample: float, send: float) -> None:
        self._note_many((("sample", sample), ("send", send)))

    def note_collect(self, seconds: float) -> None:
        self._note("collect", seconds)

    def note_stage(self, seconds: float) -> None:
        self._prepare_up_stage += seconds
        self._note("stage", seconds)

    def note_embed(self, forward: float, submit_send: float) -> None:
        self._note("embed_forward", forward)
        self._note("embed_send", submit_send)

    def note_unembed(self, post_recv: float, wait_tensor: float, lm_head: float, select: float) -> None:
        self._note("unembed_post", post_recv)
        self._note("wait_tensor", wait_tensor)
        self._note("lm_head", lm_head)
        self._note("select", select)

    def _report(self) -> None:
        if all(not span.count for span in self._spans.values()):
            return
        metrics = " ".join(f"{name}={span}(avg/max ms)" for name, span in self._spans.items())
        logger.info("[lwd][worker] %s %s", self._label, metrics)

    def flush(self) -> None:
        self._report()
        self._reset()
        self._last_report = time.monotonic()
