"""Tests for real-time input monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tape_machine.audio import (
    AudioDevice,
    AudioSettings,
    StereoBusInput,
    TrackInputRoute,
)
from tape_machine.engine import (
    AudioEngine,
    AudioEngineError,
    build_monitor_matrix,
    build_playback_matrix,
    db_to_gain,
    stream_channel_counts,
)
from tape_machine.mixer import MIN_LEVEL_DB, MixerState
from tape_machine.project import AudioProject, ProjectMetadata
from tape_machine.transport import TransportController


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
    track_inputs: tuple[TrackInputRoute, ...] = (
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
    buffer_size: int = 0,
) -> AudioSettings:
    return AudioSettings(
        1, 2, 48_000, track_inputs, bus_outputs, buffer_size
    )


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


def test_stream_uses_selected_fixed_buffer_size() -> None:
    backend = FakeBackend()
    engine = AudioEngine(backend)

    engine.start(
        settings(buffer_size=256),
        monitored_state(0),
        device(index=1),
        device(index=2),
    )

    assert backend.streams[0].kwargs["blocksize"] == 256
    engine.stop()


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
        def prepare_audio(
            self,
            frames: int,
            status: object,
            destination: np.ndarray | None = None,
        ) -> tuple[np.ndarray, None]:
            playback = (
                destination
                if destination is not None
                else np.zeros((frames, 8), dtype=np.float32)
            )
            playback.fill(0)
            playback[:, 0] = 0.25
            return playback, None

        def submit_capture(
            self,
            context: object,
            input_data: np.ndarray,
            stereo_bus: np.ndarray,
        ) -> None:
            raise AssertionError("playback did not request capture")

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


def test_callback_reuses_transport_and_bus_scratch_buffers() -> None:
    class ScratchTransport:
        def __init__(self) -> None:
            self.destinations: list[np.ndarray] = []

        def prepare_audio(
            self,
            frames: int,
            status: object,
            destination: np.ndarray | None = None,
        ) -> tuple[np.ndarray, None]:
            assert destination is not None
            destination.fill(0)
            self.destinations.append(destination)
            return destination, None

        def submit_capture(
            self,
            context: object,
            input_data: np.ndarray,
            stereo_bus: np.ndarray,
        ) -> None:
            raise AssertionError("playback did not request capture")

    transport = ScratchTransport()
    backend = FakeBackend()
    engine = AudioEngine(backend)
    engine.set_transport(transport)
    engine.start(
        settings(), MixerState(), device(index=1), device(index=2)
    )
    callback = backend.streams[0].kwargs["callback"]

    for _ in range(2):
        callback(
            np.zeros((4, 1), dtype=np.float32),
            np.zeros((4, 2), dtype=np.float32),
            4,
            None,
            None,
        )

    assert transport.destinations[0].base is transport.destinations[1].base


def test_callback_submits_actual_bus_even_without_hardware_outputs() -> None:
    capture_token = object()

    class CapturingTransport:
        def __init__(self) -> None:
            self.captures: list[tuple[np.ndarray, np.ndarray]] = []

        def prepare_audio(
            self,
            frames: int,
            status: object,
            destination: np.ndarray | None = None,
        ) -> tuple[np.ndarray, object]:
            playback = (
                destination
                if destination is not None
                else np.zeros((frames, 8), dtype=np.float32)
            )
            playback.fill(0)
            playback[:, 3] = 0.25
            return playback, capture_token

        def submit_capture(
            self,
            context: object,
            input_data: np.ndarray,
            stereo_bus: np.ndarray,
        ) -> None:
            assert context is capture_token
            self.captures.append((input_data.copy(), stereo_bus.copy()))

    routes = (
        0,
        StereoBusInput.LEFT,
        StereoBusInput.RIGHT,
    ) + (None,) * 5
    audio_settings = settings(track_inputs=routes, bus_outputs=(None, None))
    state = MixerState.from_track_inputs(routes)
    state.tracks[0].input_monitoring = True
    state.tracks[0].pan = -1
    state.tracks[0].level_db = -6
    state.tracks[3].pan = 1
    state.bus_level_db = -6
    transport = CapturingTransport()
    backend = FakeBackend()
    engine = AudioEngine(backend)
    engine.set_transport(transport)
    engine.start(
        audio_settings,
        state,
        device(inputs=1, index=1),
        device(outputs=1, index=2),
    )
    callback = backend.streams[0].kwargs["callback"]
    indata = np.full((2, 1), 0.5, dtype=np.float32)
    outdata = np.ones((2, 1), dtype=np.float32)

    callback(indata, outdata, 2, None, None)

    assert not outdata.any()
    assert len(transport.captures) == 1
    captured_input, captured_bus = transport.captures[0]
    assert captured_input == pytest.approx(indata)
    assert captured_bus[:, 0] == pytest.approx(
        [0.5 * db_to_gain(-6) ** 2] * 2
    )
    assert captured_bus[:, 1] == pytest.approx(
        [0.25 * db_to_gain(-6)] * 2
    )


def test_engine_and_transport_record_stereo_bus_loopback_end_to_end(
    tmp_path: Path,
) -> None:
    project = AudioProject.create(
        tmp_path / "loopback.wav", 48_000, ProjectMetadata()
    )
    original = np.zeros((4, 8), dtype=np.float32)
    original[:, 0] = 0.25
    original[:, 1] = -0.25
    project.audio_file.write(original)
    project.flush_audio()
    routes = (None,) * 6 + (
        StereoBusInput.LEFT,
        StereoBusInput.RIGHT,
    )
    armed = (False,) * 6 + (True, True)
    state = MixerState.from_track_inputs(routes)
    state.tracks[0].pan = -1
    state.tracks[1].pan = 1
    transport = TransportController(project)
    transport.toggle_record()
    assert transport.play(routes, armed) is True
    backend = FakeBackend()
    engine = AudioEngine(backend)
    engine.set_transport(transport)
    engine.start(
        settings(track_inputs=routes, bus_outputs=(None, None)),
        state,
        device(inputs=1, index=1),
        device(outputs=1, index=2),
    )
    callback = backend.streams[0].kwargs["callback"]

    callback(
        np.zeros((4, 1), dtype=np.float32),
        np.zeros((4, 1), dtype=np.float32),
        4,
        None,
        None,
    )
    transport.stop()
    engine.stop()
    recorded = project.read_audio_block(0, 4)

    assert recorded[:, 6] == pytest.approx([0.25] * 4, abs=1e-6)
    assert recorded[:, 7] == pytest.approx([-0.25] * 4, abs=1e-6)
    assert recorded[:, :6] == pytest.approx(original[:, :6], abs=1e-6)
    project.close()


def test_callback_accepts_raw_stream_buffers() -> None:
    engine = AudioEngine(FakeBackend())
    indata = np.ones((2, 1), dtype=np.float32)
    outdata = np.ones((2, 1), dtype=np.float32)

    engine._callback(indata, outdata, 2, None, None)

    assert not outdata.any()
