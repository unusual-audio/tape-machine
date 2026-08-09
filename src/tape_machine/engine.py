"""Real-time input monitoring engine."""

from __future__ import annotations

from math import sqrt
from time import sleep
from typing import Any, Callable, Protocol

import numpy as np
import sounddevice

from tape_machine.audio import (
    STEREO_BUS_CHANNEL_COUNT,
    AudioDevice,
    AudioSettings,
    is_physical_input,
)
from tape_machine.mixer import MIN_LEVEL_DB, MixerState


class AudioEngineError(RuntimeError):
    """Raised when a monitoring stream cannot be opened or started."""


class AudioStream(Protocol):
    """Subset of a SoundDevice stream used by the engine."""

    active: bool

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def close(self) -> None: ...


class AudioStreamBackend(Protocol):
    """Factory protocol used to isolate SoundDevice in tests."""

    def RawStream(self, **kwargs: Any) -> AudioStream: ...


class TransportAudioSource(Protocol):
    """Real-time transport interface consumed by the stream callback."""

    def prepare_audio(
        self, frames: int, status: object
    ) -> tuple[np.ndarray | None, object | None]: ...

    def submit_capture(
        self,
        context: object,
        input_data: np.ndarray,
        stereo_bus: np.ndarray,
    ) -> None: ...


def db_to_gain(value: float) -> float:
    """Convert the mixer dB scale to linear amplitude gain."""
    if value <= MIN_LEVEL_DB:
        return 0.0
    return 10.0 ** (value / 20.0)


def stream_channel_counts(
    settings: AudioSettings,
    input_device: AudioDevice,
    output_device: AudioDevice,
) -> tuple[int, int]:
    """Return minimal non-zero device channel counts for a duplex stream."""
    available_inputs = [
        channel
        for channel in settings.track_inputs
        if is_physical_input(channel)
        and channel < input_device.max_input_channels
    ]
    available_outputs = [
        channel
        for channel in settings.bus_outputs
        if channel is not None and channel < output_device.max_output_channels
    ]
    return (
        max(1, max(available_inputs, default=-1) + 1),
        max(1, max(available_outputs, default=-1) + 1),
    )


def build_monitor_matrix(
    settings: AudioSettings,
    mixer_state: MixerState,
    input_channels: int,
    output_channels: int,
) -> np.ndarray:
    """Build the immutable input-to-device-output monitoring matrix."""
    return build_monitor_bus_matrix(
        settings, mixer_state, input_channels
    ) @ build_bus_output_matrix(settings, output_channels)


def build_monitor_bus_matrix(
    settings: AudioSettings,
    mixer_state: MixerState,
    input_channels: int,
) -> np.ndarray:
    """Build the device-input-to-internal-stereo-bus monitoring matrix."""
    matrix = np.zeros(
        (input_channels, STEREO_BUS_CHANNEL_COUNT), dtype=np.float32
    )
    bus_gain = db_to_gain(mixer_state.bus_level_db)
    if bus_gain == 0.0:
        return matrix

    any_solo = any(track.soloed for track in mixer_state.tracks)
    for track, input_channel in zip(
        mixer_state.tracks, settings.track_inputs, strict=True
    ):
        if (
            not track.input_monitoring
            or track.muted
            or (any_solo and not track.soloed)
            or not is_physical_input(input_channel)
            or input_channel >= input_channels
        ):
            continue

        track_gain = db_to_gain(track.level_db) * bus_gain
        if track_gain == 0.0:
            continue
        pan = max(-1.0, min(1.0, track.pan))
        left_gain = sqrt((1.0 - pan) / 2.0) * track_gain
        right_gain = sqrt((1.0 + pan) / 2.0) * track_gain
        matrix[input_channel, 0] += left_gain
        matrix[input_channel, 1] += right_gain
    return matrix


def build_playback_matrix(
    settings: AudioSettings,
    mixer_state: MixerState,
    output_channels: int,
) -> np.ndarray:
    """Build the project-track-to-device-output playback matrix."""
    return build_playback_bus_matrix(
        mixer_state
    ) @ build_bus_output_matrix(settings, output_channels)


def build_playback_bus_matrix(mixer_state: MixerState) -> np.ndarray:
    """Build the project-track-to-internal-stereo-bus playback matrix."""
    matrix = np.zeros(
        (len(mixer_state.tracks), STEREO_BUS_CHANNEL_COUNT),
        dtype=np.float32,
    )
    bus_gain = db_to_gain(mixer_state.bus_level_db)
    if bus_gain == 0.0:
        return matrix

    any_solo = any(track.soloed for track in mixer_state.tracks)
    for track_index, track in enumerate(mixer_state.tracks):
        if track.muted or (any_solo and not track.soloed):
            continue
        track_gain = db_to_gain(track.level_db) * bus_gain
        pan = max(-1.0, min(1.0, track.pan))
        matrix[track_index, 0] = sqrt((1.0 - pan) / 2.0) * track_gain
        matrix[track_index, 1] = sqrt((1.0 + pan) / 2.0) * track_gain
    return matrix


def build_bus_output_matrix(
    settings: AudioSettings, output_channels: int
) -> np.ndarray:
    """Build the internal-stereo-bus-to-device-output routing matrix."""
    matrix = np.zeros(
        (STEREO_BUS_CHANNEL_COUNT, output_channels), dtype=np.float32
    )
    for bus_channel, output_channel in enumerate(settings.bus_outputs):
        if output_channel is not None and output_channel < output_channels:
            matrix[bus_channel, output_channel] = 1.0
    return matrix


class AudioEngine:
    """Own a duplex SoundDevice stream and its lock-free mix snapshot."""

    def __init__(
        self,
        backend: AudioStreamBackend = sounddevice,
        sleep_function: Callable[[float], None] = sleep,
    ) -> None:
        self.backend = backend
        self._sleep = sleep_function
        self._stream: AudioStream | None = None
        self._settings: AudioSettings | None = None
        self._input_channels = 1
        self._output_channels = 1
        self._monitor_bus_matrix = np.zeros((1, 2), dtype=np.float32)
        self._playback_bus_matrix = np.zeros((8, 2), dtype=np.float32)
        self._bus_output_matrix = np.zeros((2, 1), dtype=np.float32)
        self._transport: TransportAudioSource | None = None

    @property
    def running(self) -> bool:
        return self._stream is not None and bool(self._stream.active)

    @property
    def settings(self) -> AudioSettings | None:
        return self._settings

    def start(
        self,
        settings: AudioSettings,
        mixer_state: MixerState,
        input_device: AudioDevice,
        output_device: AudioDevice,
    ) -> None:
        """Open and start a low-latency duplex monitoring stream."""
        replacing_stream = self._stream is not None
        self.stop()
        if replacing_stream:
            self._sleep(0.05)
        input_channels, output_channels = stream_channel_counts(
            settings, input_device, output_device
        )
        monitor_bus_matrix = build_monitor_bus_matrix(
            settings, mixer_state, input_channels
        )
        playback_bus_matrix = build_playback_bus_matrix(mixer_state)
        bus_output_matrix = build_bus_output_matrix(
            settings, output_channels
        )
        retry_delays = (0.0, 0.05, 0.2)
        for attempt, retry_delay in enumerate(retry_delays):
            if retry_delay:
                self._sleep(retry_delay)
            try:
                stream = self.backend.RawStream(
                    samplerate=settings.sample_rate,
                    blocksize=settings.buffer_size,
                    device=(
                        settings.input_device_id,
                        settings.output_device_id,
                    ),
                    channels=(input_channels, output_channels),
                    dtype="float32",
                    latency="low",
                    callback=self._callback,
                )
                self._settings = settings
                self._input_channels = input_channels
                self._output_channels = output_channels
                self._monitor_bus_matrix = monitor_bus_matrix
                self._playback_bus_matrix = playback_bus_matrix
                self._bus_output_matrix = bus_output_matrix
                self._stream = stream
                stream.start()
                return
            except Exception as exc:
                self._discard_failed_stream()
                should_retry = (
                    attempt < len(retry_delays) - 1
                    and _is_transient_core_audio_error(exc)
                )
                if not should_retry:
                    raise AudioEngineError(
                        f"Unable to start input monitoring: {exc}"
                    ) from exc

    def _discard_failed_stream(self) -> None:
        """Close a partly started stream and reset the engine snapshot."""
        failed_stream = self._stream
        self._stream = None
        self._settings = None
        self._monitor_bus_matrix = np.zeros((1, 2), dtype=np.float32)
        self._playback_bus_matrix = np.zeros((8, 2), dtype=np.float32)
        self._bus_output_matrix = np.zeros((2, 1), dtype=np.float32)
        if failed_stream is not None:
            try:
                failed_stream.close()
            except Exception:
                pass

    def reconfigure(
        self,
        settings: AudioSettings,
        mixer_state: MixerState,
        input_device: AudioDevice,
        output_device: AudioDevice,
    ) -> None:
        """Restart the stream with a new device or routing configuration."""
        self.start(settings, mixer_state, input_device, output_device)

    def update_mix(self, mixer_state: MixerState) -> None:
        """Atomically replace the matrix consumed by the callback."""
        if self._settings is None:
            return
        self._monitor_bus_matrix = build_monitor_bus_matrix(
            self._settings,
            mixer_state,
            self._input_channels,
        )
        self._playback_bus_matrix = build_playback_bus_matrix(mixer_state)

    def set_transport(self, transport: TransportAudioSource | None) -> None:
        """Attach the project transport consumed by future callbacks."""
        self._transport = transport

    def stop(self) -> None:
        """Stop and close the current stream, tolerating device removal."""
        stream = self._stream
        self._stream = None
        self._settings = None
        self._monitor_bus_matrix = np.zeros((1, 2), dtype=np.float32)
        self._playback_bus_matrix = np.zeros((8, 2), dtype=np.float32)
        self._bus_output_matrix = np.zeros((2, 1), dtype=np.float32)
        if stream is None:
            return
        try:
            if stream.active:
                stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass

    def _callback(
        self,
        input_buffer: object,
        output_buffer: object,
        frames: int,
        time: Any,
        status: Any,
    ) -> None:
        """Render one monitoring block without locks or intermediate arrays."""
        indata = np.frombuffer(input_buffer, dtype=np.float32).reshape(
            frames, self._input_channels
        )
        outdata = np.frombuffer(output_buffer, dtype=np.float32).reshape(
            frames, self._output_channels
        )
        monitor_bus_matrix = self._monitor_bus_matrix
        stereo_bus = np.zeros(
            (frames, STEREO_BUS_CHANNEL_COUNT), dtype=np.float32
        )
        outdata.fill(0)
        if monitor_bus_matrix.shape == (
            indata.shape[1],
            STEREO_BUS_CHANNEL_COUNT,
        ):
            np.matmul(indata, monitor_bus_matrix, out=stereo_bus)
        transport = self._transport
        capture_context: object | None = None
        if transport is not None:
            playback, capture_context = transport.prepare_audio(frames, status)
            playback_bus_matrix = self._playback_bus_matrix
            if (
                playback is not None
                and playback.shape[1] == playback_bus_matrix.shape[0]
            ):
                stereo_bus += playback @ playback_bus_matrix
        np.clip(stereo_bus, -1.0, 1.0, out=stereo_bus)
        bus_output_matrix = self._bus_output_matrix
        if bus_output_matrix.shape == (
            STEREO_BUS_CHANNEL_COUNT,
            outdata.shape[1],
        ):
            np.matmul(stereo_bus, bus_output_matrix, out=outdata)
        if transport is not None and capture_context is not None:
            transport.submit_capture(capture_context, indata, stereo_bus)
        np.clip(outdata, -1.0, 1.0, out=outdata)


def _is_transient_core_audio_error(exc: Exception) -> bool:
    """Return whether CoreAudio reported its transient parameter error."""
    message = str(exc)
    return "-50" in message or "PaMacCore (AUHAL)" in message
