"""Real-time input monitoring engine."""

from __future__ import annotations

from dataclasses import dataclass
from math import log10, sqrt
from time import sleep
from typing import Any, Callable, Protocol

import numpy as np
import sounddevice

from tape_machine.audio import (
    PROJECT_TRACK_COUNT,
    STEREO_BUS_CHANNEL_COUNT,
    AudioDevice,
    AudioSettings,
    StereoBusInput,
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


class CaptureAudioContext(Protocol):
    """Recording state returned for one callback audio block."""

    armed_tracks: tuple[bool, ...]


class TransportAudioSource(Protocol):
    """Real-time transport interface consumed by the stream callback."""

    def prepare_audio(
        self,
        frames: int,
        status: object,
        destination: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, CaptureAudioContext | None]: ...

    def submit_capture(
        self,
        context: object,
        input_data: np.ndarray,
        stereo_bus: np.ndarray,
    ) -> None: ...


METER_FLOOR_DB = -60.0
METER_CEILING_DB = 0.0
METER_FALL_DB_PER_SECOND = 20.0
_METER_VALUE_COUNT = PROJECT_TRACK_COUNT + STEREO_BUS_CHANNEL_COUNT
_NO_RECORDING_TRACKS = (False,) * PROJECT_TRACK_COUNT


@dataclass(frozen=True, slots=True)
class MeterSnapshot:
    """Coherent UI-facing track and stereo-bus peak levels in dBFS."""

    track_db: tuple[float, ...]
    bus_db: tuple[float, float]


def _silent_meter_snapshot() -> MeterSnapshot:
    return MeterSnapshot(
        (METER_FLOOR_DB,) * PROJECT_TRACK_COUNT,
        (METER_FLOOR_DB,) * STEREO_BUS_CHANNEL_COUNT,
    )


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
    return build_monitor_pre_bus_matrix(
        settings, mixer_state, input_channels
    ) * db_to_gain(mixer_state.bus_level_db)


def build_monitor_pre_bus_matrix(
    settings: AudioSettings,
    mixer_state: MixerState,
    input_channels: int,
) -> np.ndarray:
    """Build monitoring gains before the stereo-bus fader."""
    matrix = np.zeros(
        (input_channels, STEREO_BUS_CHANNEL_COUNT), dtype=np.float32
    )
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

        track_gain = db_to_gain(track.level_db)
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
    return build_playback_pre_bus_matrix(mixer_state) * db_to_gain(
        mixer_state.bus_level_db
    )


def build_playback_pre_bus_matrix(mixer_state: MixerState) -> np.ndarray:
    """Build project-track gains before the stereo-bus fader."""
    matrix = np.zeros(
        (len(mixer_state.tracks), STEREO_BUS_CHANNEL_COUNT),
        dtype=np.float32,
    )
    any_solo = any(track.soloed for track in mixer_state.tracks)
    for track_index, track in enumerate(mixer_state.tracks):
        if track.muted or (any_solo and not track.soloed):
            continue
        track_gain = db_to_gain(track.level_db)
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
        self._mix_snapshot = (
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((8, 2), dtype=np.float32),
            0.0,
        )
        self._bus_output_matrix = np.zeros((2, 1), dtype=np.float32)
        self._transport: TransportAudioSource | None = None
        self._pre_bus_scratch = np.zeros(
            (0, STEREO_BUS_CHANNEL_COUNT), dtype=np.float32
        )
        self._stereo_bus_scratch = np.zeros(
            (0, STEREO_BUS_CHANNEL_COUNT), dtype=np.float32
        )
        self._playback_bus_scratch = np.zeros(
            (0, STEREO_BUS_CHANNEL_COUNT), dtype=np.float32
        )
        self._track_playback_scratch = np.zeros(
            (0, PROJECT_TRACK_COUNT), dtype=np.float32
        )
        self._meter_abs_scratch = np.zeros(0, dtype=np.float32)
        self._meter_peak_scratch = np.zeros((), dtype=np.float32)
        self._meter_levels_db = np.full(
            _METER_VALUE_COUNT, METER_FLOOR_DB, dtype=np.float32
        )
        self._meter_generation = 0
        self._meter_ui_snapshot = _silent_meter_snapshot()

    @property
    def running(self) -> bool:
        return self._stream is not None and bool(self._stream.active)

    @property
    def settings(self) -> AudioSettings | None:
        return self._settings

    @property
    def meter_snapshot(self) -> MeterSnapshot:
        """Return the latest coherent callback meter envelope."""
        generation = self._meter_generation
        if generation % 2:
            return self._meter_ui_snapshot
        values = self._meter_levels_db
        snapshot = MeterSnapshot(
            tuple(float(value) for value in values[:PROJECT_TRACK_COUNT]),
            (
                float(values[PROJECT_TRACK_COUNT]),
                float(values[PROJECT_TRACK_COUNT + 1]),
            ),
        )
        if generation != self._meter_generation:
            return self._meter_ui_snapshot
        self._meter_ui_snapshot = snapshot
        return snapshot

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
        monitor_bus_matrix = build_monitor_pre_bus_matrix(
            settings, mixer_state, input_channels
        )
        playback_bus_matrix = build_playback_pre_bus_matrix(mixer_state)
        bus_gain = db_to_gain(mixer_state.bus_level_db)
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
                self._mix_snapshot = (
                    monitor_bus_matrix,
                    playback_bus_matrix,
                    bus_gain,
                )
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
        self._mix_snapshot = (
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((8, 2), dtype=np.float32),
            0.0,
        )
        self._bus_output_matrix = np.zeros((2, 1), dtype=np.float32)
        self._reset_audio_scratch()
        self._reset_meter_levels()
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
        self._mix_snapshot = (
            build_monitor_pre_bus_matrix(
                self._settings,
                mixer_state,
                self._input_channels,
            ),
            build_playback_pre_bus_matrix(mixer_state),
            db_to_gain(mixer_state.bus_level_db),
        )

    def set_transport(self, transport: TransportAudioSource | None) -> None:
        """Attach the project transport consumed by future callbacks."""
        self._transport = transport

    def stop(self) -> None:
        """Stop and close the current stream, tolerating device removal."""
        stream = self._stream
        self._stream = None
        self._settings = None
        self._mix_snapshot = (
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((8, 2), dtype=np.float32),
            0.0,
        )
        self._bus_output_matrix = np.zeros((2, 1), dtype=np.float32)
        self._reset_audio_scratch()
        if stream is None:
            self._reset_meter_levels()
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
        self._reset_meter_levels()

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
        monitor_bus_matrix, playback_bus_matrix, bus_gain = self._mix_snapshot
        (
            pre_bus,
            stereo_bus,
            playback_bus,
            track_playback,
        ) = self._audio_scratch(frames)
        pre_bus.fill(0)
        outdata.fill(0)
        if monitor_bus_matrix.shape == (
            indata.shape[1],
            STEREO_BUS_CHANNEL_COUNT,
        ):
            np.matmul(indata, monitor_bus_matrix, out=pre_bus)
        transport = self._transport
        playback: np.ndarray | None = None
        capture_context: CaptureAudioContext | None = None
        if transport is not None:
            playback, capture_context = transport.prepare_audio(
                frames, status, track_playback
            )
            if (
                playback is not None
                and playback.shape[1] == playback_bus_matrix.shape[0]
            ):
                np.matmul(
                    playback, playback_bus_matrix, out=playback_bus
                )
                pre_bus += playback_bus
        np.multiply(pre_bus, bus_gain, out=stereo_bus)
        np.clip(stereo_bus, -1.0, 1.0, out=stereo_bus)
        self._update_meter_levels(
            indata,
            playback,
            capture_context,
            pre_bus,
            stereo_bus,
            frames,
        )
        bus_output_matrix = self._bus_output_matrix
        if bus_output_matrix.shape == (
            STEREO_BUS_CHANNEL_COUNT,
            outdata.shape[1],
        ):
            np.matmul(stereo_bus, bus_output_matrix, out=outdata)
        if transport is not None and capture_context is not None:
            transport.submit_capture(capture_context, indata, stereo_bus)
        np.clip(outdata, -1.0, 1.0, out=outdata)

    def _audio_scratch(
        self, frames: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return reusable callback buffers sized for the current block."""
        if len(self._stereo_bus_scratch) < frames:
            self._pre_bus_scratch = np.zeros(
                (frames, STEREO_BUS_CHANNEL_COUNT), dtype=np.float32
            )
            self._stereo_bus_scratch = np.zeros(
                (frames, STEREO_BUS_CHANNEL_COUNT), dtype=np.float32
            )
            self._playback_bus_scratch = np.zeros(
                (frames, STEREO_BUS_CHANNEL_COUNT), dtype=np.float32
            )
            self._track_playback_scratch = np.zeros(
                (frames, PROJECT_TRACK_COUNT), dtype=np.float32
            )
            self._meter_abs_scratch = np.zeros(frames, dtype=np.float32)
        return (
            self._pre_bus_scratch[:frames],
            self._stereo_bus_scratch[:frames],
            self._playback_bus_scratch[:frames],
            self._track_playback_scratch[:frames],
        )

    def _update_meter_levels(
        self,
        indata: np.ndarray,
        playback: np.ndarray | None,
        capture_context: CaptureAudioContext | None,
        pre_bus: np.ndarray,
        stereo_bus: np.ndarray,
        frames: int,
    ) -> None:
        """Publish source-aware peaks with immediate attack and smooth fall."""
        settings = self._settings
        if settings is None:
            return
        recording_tracks = getattr(
            capture_context, "armed_tracks", _NO_RECORDING_TRACKS
        )
        if len(recording_tracks) != PROJECT_TRACK_COUNT:
            recording_tracks = _NO_RECORDING_TRACKS
        decay_db = (
            METER_FALL_DB_PER_SECOND * frames / settings.sample_rate
        )

        self._meter_generation += 1
        try:
            for track_index, route in enumerate(settings.track_inputs):
                use_input = playback is None or recording_tracks[track_index]
                source: np.ndarray | None
                if not use_input:
                    source = (
                        playback[:, track_index]
                        if playback is not None
                        and playback.shape[1] > track_index
                        else None
                    )
                elif (
                    is_physical_input(route)
                    and route < indata.shape[1]
                ):
                    source = indata[:, route]
                elif route is StereoBusInput.LEFT:
                    source = stereo_bus[:, 0]
                elif route is StereoBusInput.RIGHT:
                    source = stereo_bus[:, 1]
                else:
                    source = None
                self._update_meter_channel(
                    track_index, source, decay_db
                )

            for bus_channel in range(STEREO_BUS_CHANNEL_COUNT):
                self._update_meter_channel(
                    PROJECT_TRACK_COUNT + bus_channel,
                    pre_bus[:, bus_channel],
                    decay_db,
                )
        finally:
            self._meter_generation += 1

    def _update_meter_channel(
        self,
        meter_index: int,
        source: np.ndarray | None,
        decay_db: float,
    ) -> None:
        target_db = METER_FLOOR_DB
        if source is not None and len(source):
            absolute = self._meter_abs_scratch[: len(source)]
            np.abs(source, out=absolute)
            np.max(absolute, out=self._meter_peak_scratch)
            peak = float(self._meter_peak_scratch)
            if peak > 0.0:
                target_db = max(
                    METER_FLOOR_DB,
                    min(METER_CEILING_DB, 20.0 * log10(peak)),
                )
        falling_db = max(
            METER_FLOOR_DB,
            float(self._meter_levels_db[meter_index]) - decay_db,
        )
        self._meter_levels_db[meter_index] = max(target_db, falling_db)

    def _reset_meter_levels(self) -> None:
        self._meter_generation += 1
        self._meter_levels_db.fill(METER_FLOOR_DB)
        self._meter_generation += 1
        self._meter_ui_snapshot = _silent_meter_snapshot()

    def _reset_audio_scratch(self) -> None:
        self._pre_bus_scratch = np.zeros(
            (0, STEREO_BUS_CHANNEL_COUNT), dtype=np.float32
        )
        self._stereo_bus_scratch = np.zeros(
            (0, STEREO_BUS_CHANNEL_COUNT), dtype=np.float32
        )
        self._playback_bus_scratch = np.zeros(
            (0, STEREO_BUS_CHANNEL_COUNT), dtype=np.float32
        )
        self._track_playback_scratch = np.zeros(
            (0, PROJECT_TRACK_COUNT), dtype=np.float32
        )
        self._meter_abs_scratch = np.zeros(0, dtype=np.float32)


def _is_transient_core_audio_error(exc: Exception) -> bool:
    """Return whether CoreAudio reported its transient parameter error."""
    message = str(exc)
    return "-50" in message or "PaMacCore (AUHAL)" in message
