"""Tests for real-time input monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np
import pytest

from tape_machine.audio import AudioDevice, AudioSettings
from tape_machine.engine import (
    AudioEngine,
    AudioEngineError,
    build_monitor_matrix,
    build_playback_matrix,
    db_to_gain,
    stream_channel_counts,
)
from tape_machine.mixer import MIN_LEVEL_DB, MixerState


def device(
    *, inputs: int = 8, outputs: int = 8, index: int = 1
) -> AudioDevice:
    return AudioDevice(
        index=index,
        name="Interface",
        host_api="Test",
        max_input_channels=inputs,
        max_output_channels=outputs,
        default_sample_rate=48_000,
    )


def settings(
    *,
    track_inputs: tuple[int | None, ...] = (
        0,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ),
    bus_outputs: tuple[int | None, ...] = (0, 1),
) -> AudioSettings:
    return AudioSettings(1, 2, 48_000, track_inputs, bus_outputs)


@dataclass
class FakeStream:
    kwargs: dict[str, Any]
    start_error: str | None = None
    active: bool = False
    closed: bool = False
    stop_calls: int = 0

    def start(self) -> None:
        if self.start_error is not None:
            raise RuntimeError(self.start_error)
        self.active = True

    def stop(self) -> None:
        self.stop_calls += 1
        self.active = False

    def close(self) -> None:
        self.closed = True
        self.active = False


class FakeBackend:
    def __init__(self, *, fail_start: bool = False) -> None:
        self.fail_start = fail_start
        self.streams: list[FakeStream] = []

    def RawStream(self, **kwargs: Any) -> FakeStream:
        stream = FakeStream(
            kwargs,
            start_error="device busy" if self.fail_start else None,
        )
        self.streams.append(stream)
        return stream


def monitored_state(*inputs: int | None) -> MixerState:
    routes = tuple(inputs) + (None,) * (8 - len(inputs))
    state = MixerState.from_track_inputs(routes)
    for index, channel in enumerate(inputs):
        state.tracks[index].input_monitoring = channel is not None
    return state


def test_stream_uses_low_latency_adaptive_duplex_configuration() -> None:
    backend = FakeBackend()
    engine = AudioEngine(backend)
    audio_settings = settings(
        track_inputs=(3, None, None, None, None, None, None, None),
        bus_outputs=(1, 3),
    )
    state = monitored_state(3)

    engine.start(
        audio_settings,
        state,
        device(inputs=4, index=1),
        device(outputs=4, index=2),
    )

    assert engine.running is True
    assert backend.streams[0].kwargs == {
        "samplerate": 48_000,
        "blocksize": 0,
        "device": (1, 2),
        "channels": (4, 4),
        "dtype": "float32",
        "latency": "low",
        "callback": engine._callback,
    }

    engine.stop()

    assert engine.running is False
    assert backend.streams[0].stop_calls == 1
    assert backend.streams[0].closed is True


def test_stream_start_failure_is_descriptive_and_closes_stream() -> None:
    backend = FakeBackend(fail_start=True)
    engine = AudioEngine(backend)

    with pytest.raises(AudioEngineError, match="device busy"):
        engine.start(
            settings(), monitored_state(0), device(index=1), device(index=2)
        )

    assert engine.running is False
    assert backend.streams[0].closed is True


def test_transient_core_audio_start_error_is_retried() -> None:
    class FlakyBackend(FakeBackend):
        def RawStream(self, **kwargs: Any) -> FakeStream:
            stream = FakeStream(
                kwargs,
                start_error=(
                    "PaMacCore (AUHAL) error -50"
                    if not self.streams
                    else None
                ),
            )
            self.streams.append(stream)
            return stream

    delays: list[float] = []
    backend = FlakyBackend()
    engine = AudioEngine(backend, sleep_function=delays.append)

    engine.start(
        settings(), monitored_state(0), device(index=1), device(index=2)
    )

    assert engine.running is True
    assert len(backend.streams) == 2
    assert backend.streams[0].closed is True
    assert delays == [0.05]


def test_stream_counts_ignore_routes_into_the_void_but_remain_duplex() -> None:
    audio_settings = settings(
        track_inputs=(9, None, None, None, None, None, None, None),
        bus_outputs=(7, None),
    )

    assert stream_channel_counts(
        audio_settings,
        device(inputs=2),
        device(outputs=2),
    ) == (1, 1)


def test_center_pan_uses_constant_power_and_hard_pan_is_unity() -> None:
    audio_settings = settings()
    state = monitored_state(0)

    matrix = build_monitor_matrix(audio_settings, state, 1, 2)

    assert matrix[0] == pytest.approx([sqrt(0.5), sqrt(0.5)])

    state.tracks[0].pan = -1
    assert build_monitor_matrix(audio_settings, state, 1, 2)[0] == pytest.approx(
        [1, 0]
    )
    state.tracks[0].pan = 1
    assert build_monitor_matrix(audio_settings, state, 1, 2)[0] == pytest.approx(
        [0, 1]
    )


def test_track_and_bus_faders_apply_amplitude_gain() -> None:
    state = monitored_state(0)
    state.tracks[0].pan = -1
    state.tracks[0].level_db = -6
    state.bus_level_db = -6

    matrix = build_monitor_matrix(settings(), state, 1, 2)

    assert matrix[0, 0] == pytest.approx(db_to_gain(-6) ** 2)
    assert matrix[0, 1] == 0

    state.tracks[0].level_db = MIN_LEVEL_DB
    assert not build_monitor_matrix(settings(), state, 1, 2).any()


def test_mute_and_solo_gate_monitored_tracks() -> None:
    audio_settings = settings(
        track_inputs=(0, 1, None, None, None, None, None, None)
    )
    state = monitored_state(0, 1)
    state.tracks[0].pan = -1
    state.tracks[1].pan = 1
    state.tracks[1].soloed = True

    matrix = build_monitor_matrix(audio_settings, state, 2, 2)

    assert not matrix[0].any()
    assert matrix[1] == pytest.approx([0, 1])

    state.tracks[1].muted = True
    assert not build_monitor_matrix(audio_settings, state, 2, 2).any()


def test_repeated_inputs_sum_and_partial_outputs_drop_missing_side() -> None:
    audio_settings = settings(
        track_inputs=(0, 0, None, None, None, None, None, None),
        bus_outputs=(1, None),
    )
    state = monitored_state(0, 0)
    state.tracks[0].pan = -1
    state.tracks[1].pan = -1

    matrix = build_monitor_matrix(audio_settings, state, 1, 2)

    assert matrix[0] == pytest.approx([0, 2])


def test_playback_matrix_uses_track_mix_without_monitor_buttons() -> None:
    state = MixerState()
    state.tracks[0].pan = -1
    state.tracks[1].pan = 1
    state.tracks[1].soloed = True

    matrix = build_playback_matrix(settings(), state, 2)

    assert not matrix[0].any()
    assert matrix[1] == pytest.approx([0, 1])


def test_callback_renders_current_matrix_and_clips_final_output() -> None:
    backend = FakeBackend()
    engine = AudioEngine(backend)
    state = monitored_state(0)
    state.tracks[0].pan = -1
    state.tracks[0].level_db = 6
    engine.start(settings(), state, device(index=1), device(index=2))
    callback = backend.streams[0].kwargs["callback"]
    indata = np.array([[0.25], [0.75]], dtype=np.float32)
    outdata = np.full((2, 2), 99, dtype=np.float32)

    callback(indata, outdata, 2, None, None)

    assert outdata[:, 0] == pytest.approx([0.25 * db_to_gain(6), 1.0])
    assert outdata[:, 1] == pytest.approx([0, 0])

    state.tracks[0].pan = 1
    engine.update_mix(state)
    callback(indata, outdata, 2, None, None)
    assert outdata[:, 0] == pytest.approx([0, 0])
    assert outdata[:, 1] == pytest.approx([0.25 * db_to_gain(6), 1.0])


def test_callback_routes_project_playback_through_the_mixer() -> None:
    class FakeTransport:
        def process_audio(
            self, indata: np.ndarray, frames: int, status: object
        ) -> np.ndarray:
            playback = np.zeros((frames, 8), dtype=np.float32)
            playback[:, 0] = 0.25
            return playback

    backend = FakeBackend()
    engine = AudioEngine(backend)
    state = MixerState()
    state.tracks[0].pan = -1
    engine.set_transport(FakeTransport())
    engine.start(settings(), state, device(index=1), device(index=2))
    callback = backend.streams[0].kwargs["callback"]
    indata = np.zeros((2, 1), dtype=np.float32)
    outdata = np.zeros((2, 2), dtype=np.float32)

    callback(indata, outdata, 2, None, None)

    assert outdata[:, 0] == pytest.approx([0.25, 0.25])
    assert not outdata[:, 1].any()


def test_callback_accepts_raw_stream_buffers() -> None:
    engine = AudioEngine(FakeBackend())
    indata = np.ones((2, 1), dtype=np.float32)
    outdata = np.ones((2, 1), dtype=np.float32)

    engine._callback(indata, outdata, 2, None, None)

    assert not outdata.any()
