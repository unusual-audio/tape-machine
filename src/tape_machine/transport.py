"""Project playback and recording transport."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from math import ceil
from queue import Full
from threading import Event, Thread
from time import monotonic

import numpy as np
import soundfile

from tape_machine.audio import (
    PROJECT_TRACK_COUNT,
    STEREO_BUS_CHANNEL_COUNT,
    StereoBusInput,
    TrackInputRoute,
    is_physical_input,
)
from tape_machine.project import AudioProject, ProjectError


DISK_BLOCK_FRAMES = 8192
PLAYBACK_BUFFER_SECONDS = 1.0
PLAYBACK_PREBUFFER_SECONDS = 0.25
CAPTURE_BUFFER_SECONDS = 3.0
CAPTURE_BATCH_FRAMES = 8192
# At the lowest supported sample rate, 8192 frames arrive in about 186 ms.
# Waiting this long lets the disk worker write full blocks during steady-state
# recording instead of turning small device callbacks into frequent seeks.
CAPTURE_COALESCE_SECONDS = 0.200
RECORDING_CHECKPOINT_SECONDS = 5.0
SHUTTLE_SPEED = 10
SHUTTLE_GAIN_DB = -9.0
SHUTTLE_GAIN = 10 ** (SHUTTLE_GAIN_DB / 20.0)
SHUTTLE_OUTPUT_BLOCK_FRAMES = 1024
SHUTTLE_FILTER_TAPS = 81
_NO_ARMED_TRACKS = (False,) * PROJECT_TRACK_COUNT


class TransportMode(Enum):
    """Current direction and speed of the project transport."""

    STOPPED = "stopped"
    PLAYING = "playing"
    FAST_FORWARD = "fast_forward"
    REWIND = "rewind"

    @property
    def shuttling(self) -> bool:
        return self in {self.FAST_FORWARD, self.REWIND}


class TransportLifecycle(Enum):
    """Ownership state for disk workers and their project handles."""

    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    FAULTED = "faulted"


class RealtimeFault(IntEnum):
    """Allocation-free fault codes published by the audio callback."""

    NONE = 0
    PLAYBACK_UNDERRUN = 1
    SHUTTLE_UNDERRUN = 2
    RECORDING_DROPOUT = 3
    CAPTURE_OVERFLOW = 4
    INVALID_CAPTURE = 5


class TransportError(RuntimeError):
    """Raised when playback or recording cannot continue safely."""


@dataclass(frozen=True, slots=True)
class CaptureContext:
    """Callback-local recording state awaiting the rendered stereo bus."""

    position: int
    frames: int
    armed_tracks: tuple[bool, ...]


@dataclass(slots=True)
class TransportRenderState:
    """Reusable callback result populated in place by the transport."""

    playback_active: bool = False
    capture_active: bool = False
    position: int = 0
    frames: int = 0
    armed_tracks: tuple[bool, ...] = _NO_ARMED_TRACKS


@dataclass(frozen=True, slots=True)
class _CaptureRingBatch:
    """A worker-side lease over contiguous capture descriptors."""

    position: int
    frames: int
    descriptor_count: int
    armed_mask: int


class _PlaybackRing:
    """Preallocated single-producer/single-consumer playback storage."""

    def __init__(self, max_frames: int) -> None:
        self.max_frames = max(1, max_frames)
        self.data = np.zeros(
            (self.max_frames, PROJECT_TRACK_COUNT), dtype=np.float32
        )
        self._written_frames = 0
        self._read_frames = 0

    @property
    def available_frames(self) -> int:
        return self._written_frames - self._read_frames

    @property
    def free_frames(self) -> int:
        return self.max_frames - self.available_frames

    def clear(self) -> None:
        self._written_frames = 0
        self._read_frames = 0

    def write(self, block: np.ndarray) -> None:
        frames = len(block)
        if frames > self.free_frames:
            raise Full
        start = self._written_frames % self.max_frames
        first = min(frames, self.max_frames - start)
        self.data[start : start + first] = block[:first]
        if first < frames:
            self.data[: frames - first] = block[first:]
        # Publish only after every sample has been copied.
        self._written_frames += frames

    def read_into(self, destination: np.ndarray) -> int:
        frames = min(len(destination), self.available_frames)
        if frames <= 0:
            return 0
        start = self._read_frames % self.max_frames
        first = min(frames, self.max_frames - start)
        destination[:first] = self.data[start : start + first]
        if first < frames:
            destination[first:frames] = self.data[: frames - first]
        self._read_frames += frames
        return frames


class _CaptureRing:
    """Preallocated callback-to-disk ring containing routed track audio."""

    def __init__(self, max_frames: int) -> None:
        self.max_frames = max(1, max_frames)
        self.data = np.zeros(
            (self.max_frames, PROJECT_TRACK_COUNT), dtype=np.float32
        )
        # One descriptor per frame is deliberately conservative: even a host
        # that supplies one-frame callbacks cannot exhaust metadata first.
        self._positions = np.zeros(self.max_frames, dtype=np.int64)
        self._lengths = np.zeros(self.max_frames, dtype=np.int32)
        self._armed_masks = np.zeros(self.max_frames, dtype=np.uint8)
        self._written_frames = 0
        self._read_frames = 0
        self._written_descriptors = 0
        self._read_descriptors = 0

    @property
    def buffered_frames(self) -> int:
        return self._written_frames - self._read_frames

    def empty(self) -> bool:
        return self._written_descriptors == self._read_descriptors

    def clear(self) -> None:
        self._written_frames = 0
        self._read_frames = 0
        self._written_descriptors = 0
        self._read_descriptors = 0

    def put_routed_nowait(
        self,
        position: int,
        input_data: np.ndarray,
        stereo_bus: np.ndarray,
        track_inputs: tuple[TrackInputRoute, ...],
        armed_tracks: tuple[bool, ...],
    ) -> None:
        """Route one callback block into the ring without locks or allocation."""
        frames = len(input_data)
        if (
            frames > self.max_frames - self.buffered_frames
            or self._written_descriptors - self._read_descriptors
            >= self.max_frames
        ):
            raise Full

        start = self._written_frames % self.max_frames
        first = min(frames, self.max_frames - start)
        second = frames - first
        armed_mask = 0
        for track_index, (route, armed) in enumerate(
            zip(track_inputs, armed_tracks, strict=True)
        ):
            if not armed:
                continue
            armed_mask |= 1 << track_index
            if is_physical_input(route) and route < input_data.shape[1]:
                source = input_data[:, route]
            elif route is StereoBusInput.LEFT:
                source = stereo_bus[:, 0]
            elif route is StereoBusInput.RIGHT:
                source = stereo_bus[:, 1]
            else:
                source = None
            if source is None:
                self.data[start : start + first, track_index].fill(0)
                if second:
                    self.data[:second, track_index].fill(0)
            else:
                self.data[start : start + first, track_index] = source[:first]
                if second:
                    self.data[:second, track_index] = source[first:]

        descriptor = self._written_descriptors % self.max_frames
        self._positions[descriptor] = position
        self._lengths[descriptor] = frames
        self._armed_masks[descriptor] = armed_mask
        self._written_frames += frames
        # Descriptor publication is last so the consumer cannot observe a
        # partially copied audio block.
        self._written_descriptors += 1

    def wait_for_batch(
        self,
        target_frames: int,
        coalesce_seconds: float,
        stop_event: Event,
    ) -> _CaptureRingBatch | None:
        while self.empty():
            if stop_event.is_set():
                return None
            stop_event.wait(0.001)

        deadline = monotonic() + coalesce_seconds
        while self.buffered_frames < target_frames and not stop_event.is_set():
            remaining = deadline - monotonic()
            if remaining <= 0:
                break
            stop_event.wait(min(remaining, 0.001))

        descriptor = self._read_descriptors % self.max_frames
        position = int(self._positions[descriptor])
        armed_mask = int(self._armed_masks[descriptor])
        frames = int(self._lengths[descriptor])
        descriptor_count = 1
        next_position = position + frames
        available = self._written_descriptors - self._read_descriptors
        while descriptor_count < available:
            candidate = (self._read_descriptors + descriptor_count) % self.max_frames
            candidate_frames = int(self._lengths[candidate])
            if (
                int(self._armed_masks[candidate]) != armed_mask
                or int(self._positions[candidate]) != next_position
                or frames + candidate_frames > target_frames
            ):
                break
            frames += candidate_frames
            next_position += candidate_frames
            descriptor_count += 1
        return _CaptureRingBatch(
            position, frames, descriptor_count, armed_mask
        )

    def views(self, batch: _CaptureRingBatch) -> tuple[np.ndarray, ...]:
        start = self._read_frames % self.max_frames
        first = min(batch.frames, self.max_frames - start)
        if first == batch.frames:
            return (self.data[start : start + first],)
        return (
            self.data[start : start + first],
            self.data[: batch.frames - first],
        )

    def release(self, batch: _CaptureRingBatch) -> None:
        self._read_frames += batch.frames
        self._read_descriptors += batch.descriptor_count


_ARM_MASK_TRACKS = tuple(
    tuple(bool(mask & (1 << track)) for track in range(PROJECT_TRACK_COUNT))
    for mask in range(1 << PROJECT_TRACK_COUNT)
)


def format_transport_time(position_frames: int, sample_rate: int) -> str:
    """Format a frame position as unbounded MM:SS.mmm."""
    total_ms = max(0, position_frames) * 1000 // sample_rate
    minutes, remainder = divmod(total_ms, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


class TransportController:
    """Coordinate an audio callback with independent disk I/O workers."""

    def __init__(self, project: AudioProject) -> None:
        self.project = project
        self.position_frames = 0
        self.mode = TransportMode.STOPPED
        self.lifecycle = TransportLifecycle.IDLE
        self.record_armed = False
        self.end_requested = False
        self._terminal_error: str | None = None
        self._realtime_fault = RealtimeFault.NONE

        self._track_inputs: tuple[TrackInputRoute, ...] = (
            None,
        ) * PROJECT_TRACK_COUNT
        self._armed_tracks: tuple[bool, ...] = (False,) * PROJECT_TRACK_COUNT
        self._known_frames = project.frames
        self._playback_end_frames = project.frames
        self._read_cursor = 0
        self._worker_mode = TransportMode.STOPPED
        self._playback_ring = _PlaybackRing(
            max(1, round(project.sample_rate * PLAYBACK_BUFFER_SECONDS))
        )
        self._capture_ring = _CaptureRing(
            max(1, round(project.sample_rate * CAPTURE_BUFFER_SECONDS))
        )
        self._render_state = TransportRenderState()
        self._stop_event = Event()
        self._ready_event = Event()
        self._prebuffer_target_frames = 0
        self._prebuffered_frames = 0
        self._playback_reader: soundfile.SoundFile | None = None
        self._playback_worker_thread: Thread | None = None
        self._capture_worker_thread: Thread | None = None

    @property
    def running(self) -> bool:
        return self.mode is not TransportMode.STOPPED

    @property
    def busy(self) -> bool:
        """Whether workers still own transport or project resources."""
        return self.lifecycle is not TransportLifecycle.IDLE

    @property
    def terminal_error(self) -> str | None:
        if self._terminal_error is not None:
            return self._terminal_error
        messages = {
            RealtimeFault.PLAYBACK_UNDERRUN: (
                "Project playback could not keep up with the audio stream."
            ),
            RealtimeFault.SHUTTLE_UNDERRUN: (
                "Project shuttle playback could not keep up with the audio stream."
            ),
            RealtimeFault.RECORDING_DROPOUT: "Audio dropout while recording.",
            RealtimeFault.CAPTURE_OVERFLOW: (
                "Recording buffer overflowed after "
                f"{CAPTURE_BUFFER_SECONDS:g} seconds; storage could not keep up."
            ),
            RealtimeFault.INVALID_CAPTURE: (
                "Recording received an invalid audio block."
            ),
        }
        return messages.get(self._realtime_fault)

    @terminal_error.setter
    def terminal_error(self, value: str | None) -> None:
        self._terminal_error = value

    @running.setter
    def running(self, value: bool) -> None:
        """Retain compatibility with callers that set the old running flag."""
        if value:
            if self.mode is TransportMode.STOPPED:
                self.mode = TransportMode.PLAYING
        else:
            self.mode = TransportMode.STOPPED

    @property
    def recording(self) -> bool:
        return (
            self.mode is TransportMode.PLAYING
            and self.record_armed
            and any(self._armed_tracks)
        )

    @property
    def armed_tracks(self) -> tuple[bool, ...]:
        return self._armed_tracks

    def toggle_record(self) -> bool:
        """Toggle record mode for the next callback boundary."""
        if self.mode.shuttling:
            self.record_armed = False
            return False
        self.record_armed = not self.record_armed
        return self.record_armed

    def set_armed_tracks(self, armed_tracks: tuple[bool, ...]) -> None:
        """Replace the track-arm snapshot used by the next audio callback."""
        if len(armed_tracks) != PROJECT_TRACK_COUNT:
            raise TransportError(
                "Transport requires exactly eight record-enable states."
            )
        self._armed_tracks = armed_tracks

    def play(
        self,
        track_inputs: tuple[TrackInputRoute, ...],
        armed_tracks: tuple[bool, ...],
    ) -> bool:
        """Prebuffer and start transport at the current frame position."""
        if self.running:
            return self.mode is TransportMode.PLAYING
        if len(track_inputs) != PROJECT_TRACK_COUNT or len(armed_tracks) != (
            PROJECT_TRACK_COUNT
        ):
            raise TransportError("Transport requires exactly eight project tracks.")
        if self.record_armed and any(armed_tracks) and not self.project.writable:
            raise TransportError("This project file is read-only and cannot record.")
        if self.position_frames >= self.project.frames and not self.record_armed:
            return False

        self._track_inputs = track_inputs
        self._armed_tracks = armed_tracks
        return self._start(TransportMode.PLAYING)

    def fast_forward(self) -> bool:
        """Start audible ten-times-speed forward shuttle."""
        return self._start_shuttle(TransportMode.FAST_FORWARD)

    def rewind(self) -> bool:
        """Start audible ten-times-speed reverse shuttle."""
        return self._start_shuttle(TransportMode.REWIND)

    def _start_shuttle(self, mode: TransportMode) -> bool:
        if self.running:
            return self.mode is mode
        self.record_armed = False
        if self.project.frames == 0:
            return False
        if mode is TransportMode.FAST_FORWARD:
            if self.position_frames >= self.project.frames:
                return False
        elif mode is TransportMode.REWIND:
            if self.position_frames <= 0:
                return False
        else:
            raise TransportError("Invalid shuttle transport mode.")
        return self._start(mode)

    def _start(self, mode: TransportMode) -> bool:
        """Prebuffer and start one transport mode."""
        if self.lifecycle is not TransportLifecycle.IDLE:
            raise TransportError(
                "The previous transport operation has not finished yet."
            )
        self.lifecycle = TransportLifecycle.STARTING
        self._known_frames = self.project.frames
        self._playback_end_frames = self._known_frames
        self._read_cursor = self.position_frames
        self.end_requested = False
        self._terminal_error = None
        self._realtime_fault = RealtimeFault.NONE
        self._playback_ring.clear()
        self._capture_ring.clear()
        self._stop_event.clear()
        self._ready_event.clear()
        self._worker_mode = mode
        source_frames = (
            self._playback_end_frames - self._read_cursor
            if mode is not TransportMode.REWIND
            else self._read_cursor
        )
        available_output_frames = max(0, source_frames)
        if mode.shuttling:
            available_output_frames = ceil(
                available_output_frames / SHUTTLE_SPEED
            )
        self._prebuffer_target_frames = min(
            available_output_frames,
            max(
                1,
                round(
                    self.project.sample_rate * PLAYBACK_PREBUFFER_SECONDS
                ),
            ),
        )
        self._prebuffered_frames = 0
        try:
            self._playback_reader = self.project.open_playback_reader()
            if (
                mode is TransportMode.PLAYING
                and self.record_armed
                and any(self._armed_tracks)
                and not self.project.writable
            ):
                raise TransportError(
                    "This project became read-only and cannot record."
                )
            self._capture_worker_thread = Thread(
                target=self._capture_worker_main,
                name="TapeMachineCapture",
                daemon=True,
            )
            self._playback_worker_thread = Thread(
                target=self._playback_worker_main,
                args=(self._playback_reader,),
                name="TapeMachinePlayback",
                daemon=True,
            )
            self._capture_worker_thread.start()
            self._playback_worker_thread.start()
        except Exception as exc:
            self._stop_event.set()
            stopped = self._finish_workers(timeout=2.0)
            self.lifecycle = (
                TransportLifecycle.IDLE
                if stopped
                else TransportLifecycle.FAULTED
            )
            raise TransportError(
                f"Unable to prepare project playback: {exc}"
            ) from exc
        if not self._ready_event.wait(timeout=2.0):
            stopped = self._finish_workers(timeout=2.0)
            self.lifecycle = (
                TransportLifecycle.IDLE
                if stopped
                else TransportLifecycle.FAULTED
            )
            raise TransportError("Timed out while prebuffering project audio.")
        if self.terminal_error is not None:
            stopped = self._finish_workers(timeout=2.0)
            self.lifecycle = (
                TransportLifecycle.IDLE
                if stopped
                else TransportLifecycle.FAULTED
            )
            raise TransportError(self.terminal_error)
        self.mode = mode
        self.lifecycle = TransportLifecycle.RUNNING
        return True

    def render_audio(
        self,
        frames: int,
        status: object,
        destination: np.ndarray,
    ) -> TransportRenderState:
        """Render playback and publish reusable capture state in place."""
        state = self._render_state
        state.playback_active = False
        state.capture_active = False
        state.frames = frames
        state.armed_tracks = _NO_ARMED_TRACKS
        if not self.running:
            destination.fill(0)
            return state
        mode = self.mode
        start_position = self.position_frames
        state.position = start_position
        state.playback_active = True
        if mode.shuttling:
            self._process_shuttle_audio(
                frames, mode, start_position, destination
            )
            return state
        record_armed = self.record_armed
        armed_tracks = self._armed_tracks
        recording = record_armed and any(armed_tracks)
        playback = self._prepare_destination(frames, destination)
        copied = self._consume_playback(playback)
        reached_end = False
        if copied < frames:
            if start_position + copied < self._playback_end_frames:
                self._fail_realtime(RealtimeFault.PLAYBACK_UNDERRUN)
            elif not record_armed:
                self.end_requested = True
                reached_end = True
        elif (
            not record_armed
            and start_position + copied >= self._playback_end_frames
        ):
            self.end_requested = True
            reached_end = True

        if recording:
            for track_index, armed in enumerate(armed_tracks):
                if armed:
                    playback[:, track_index].fill(0)
            state.capture_active = True
            state.armed_tracks = armed_tracks
            if status:
                self._fail_realtime(RealtimeFault.RECORDING_DROPOUT)

        if reached_end:
            self.position_frames = self._known_frames
        else:
            self.position_frames += frames
        return state

    def prepare_audio(
        self,
        frames: int,
        status: object,
        destination: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, CaptureContext | None]:
        """Compatibility wrapper for callers using the original interface."""
        playback = self._prepare_destination(frames, destination)
        state = self.render_audio(frames, status, playback)
        if not state.playback_active:
            return None, None
        context = (
            CaptureContext(state.position, state.frames, state.armed_tracks)
            if state.capture_active
            else None
        )
        return playback, context

    def enqueue_capture(
        self,
        state: TransportRenderState,
        input_data: np.ndarray,
        stereo_bus: np.ndarray,
    ) -> None:
        """Route one recorded callback block into preallocated ring storage."""
        if (
            not state.capture_active
            or input_data.shape[0] != state.frames
            or stereo_bus.shape
            != (state.frames, STEREO_BUS_CHANNEL_COUNT)
        ):
            self._fail_realtime(RealtimeFault.INVALID_CAPTURE)
            return
        try:
            self._capture_ring.put_routed_nowait(
                state.position,
                input_data,
                stereo_bus,
                self._track_inputs,
                state.armed_tracks,
            )
            self._known_frames = max(
                self._known_frames, state.position + state.frames
            )
        except Full:
            self._fail_realtime(RealtimeFault.CAPTURE_OVERFLOW)

    def submit_capture(
        self,
        context: object,
        input_data: np.ndarray,
        stereo_bus: np.ndarray,
    ) -> None:
        """Enqueue device inputs and the rendered bus without blocking."""
        if not isinstance(context, CaptureContext):
            self._fail("Recording received an invalid capture context.")
            return
        if (
            input_data.shape[0] != context.frames
            or stereo_bus.shape
            != (context.frames, STEREO_BUS_CHANNEL_COUNT)
        ):
            self._fail("Recording received an invalid audio block.")
            return
        state = TransportRenderState(
            playback_active=True,
            capture_active=True,
            position=context.position,
            frames=context.frames,
            armed_tracks=context.armed_tracks,
        )
        self.enqueue_capture(state, input_data, stereo_bus)

    def process_audio(
        self, indata: np.ndarray, frames: int, status: object
    ) -> np.ndarray | None:
        """Compatibility helper for callers without an internal bus renderer."""
        playback, context = self.prepare_audio(frames, status)
        if context is not None:
            self.submit_capture(
                context,
                indata,
                np.zeros(
                    (frames, STEREO_BUS_CHANNEL_COUNT), dtype=np.float32
                ),
            )
        return playback

    def _process_shuttle_audio(
        self,
        frames: int,
        mode: TransportMode,
        start_position: int,
        destination: np.ndarray | None = None,
    ) -> np.ndarray:
        playback = self._prepare_destination(frames, destination)
        copied = self._consume_playback(playback)
        source_advance = copied * SHUTTLE_SPEED
        if mode is TransportMode.FAST_FORWARD:
            self.position_frames = min(
                self._playback_end_frames, start_position + source_advance
            )
            reached_boundary = self.position_frames >= self._playback_end_frames
        else:
            self.position_frames = max(0, start_position - source_advance)
            reached_boundary = self.position_frames <= 0

        if copied < frames and not reached_boundary:
            self._fail_realtime(RealtimeFault.SHUTTLE_UNDERRUN)
        if reached_boundary:
            self.end_requested = True
        return playback

    @staticmethod
    def _prepare_destination(
        frames: int, destination: np.ndarray | None
    ) -> np.ndarray:
        if destination is None:
            return np.zeros(
                (frames, PROJECT_TRACK_COUNT), dtype=np.float32
            )
        if destination.shape != (frames, PROJECT_TRACK_COUNT):
            raise TransportError(
                "Playback destination must have one column per project track."
            )
        destination.fill(0)
        return destination

    def stop(self) -> None:
        """Stop, drain recorded blocks, flush the project, and retain position."""
        self.record_armed = False
        self.mode = TransportMode.STOPPED
        self.lifecycle = TransportLifecycle.STOPPING
        if self._finish_workers(timeout=5.0):
            self._finalize_stopped_workers()
        else:
            self._fail("Timed out while finishing the recording.")
            self.lifecycle = TransportLifecycle.FAULTED

    def poll_lifecycle(self) -> None:
        """Finalize workers that exited after a bounded shutdown timed out."""
        if self.lifecycle not in {
            TransportLifecycle.STOPPING,
            TransportLifecycle.FAULTED,
        }:
            return
        if self._finish_workers(timeout=0.0):
            self._finalize_stopped_workers()

    def _finalize_stopped_workers(self) -> None:
        self._playback_ring.clear()
        if self.terminal_error is not None:
            self._capture_ring.clear()
        try:
            self.project.flush_audio()
        except ProjectError as exc:
            self._fail(str(exc))
        self._worker_mode = TransportMode.STOPPED
        self.lifecycle = TransportLifecycle.IDLE

    def return_to_zero(self) -> None:
        """Return the stopped transport to the first project frame."""
        if not self.running:
            self.position_frames = 0
            self.end_requested = False

    def _consume_playback(self, destination: np.ndarray) -> int:
        return self._playback_ring.read_into(destination)

    def _finish_workers(self, timeout: float) -> bool:
        """Signal both disk workers and close their playback handle."""
        self._stop_event.set()
        workers = (
            self._capture_worker_thread,
            self._playback_worker_thread,
        )
        deadline = monotonic() + max(0.0, timeout)
        for worker in workers:
            if worker is not None:
                try:
                    worker.join(timeout=max(0.0, deadline - monotonic()))
                except RuntimeError:
                    # Thread.start() itself may have failed. An unstarted
                    # worker owns no resources and cannot be joined.
                    pass
        all_stopped = not any(
            worker is not None and worker.is_alive() for worker in workers
        )
        if (
            self._capture_worker_thread is not None
            and not self._capture_worker_thread.is_alive()
        ):
            self._capture_worker_thread = None
        playback_worker = self._playback_worker_thread
        if playback_worker is None or not playback_worker.is_alive():
            self._playback_worker_thread = None
            reader = self._playback_reader
            self._playback_reader = None
            if reader is not None:
                try:
                    reader.close()
                except Exception as exc:
                    self._fail(f"Unable to close project playback: {exc}")
        return all_stopped

    def _playback_worker_main(self, reader: soundfile.SoundFile) -> None:
        try:
            while not self._stop_event.is_set():
                if self._worker_mode.shuttling:
                    complete = self._fill_shuttle_queue(
                        self._worker_mode, reader
                    )
                else:
                    complete = self._fill_playback_queue(reader)
                if complete:
                    return
                self._stop_event.wait(timeout=0.001)
        except Exception as exc:
            self._fail(f"Project playback operation failed: {exc}")
        finally:
            self._ready_event.set()

    def _capture_worker_main(self) -> None:
        last_checkpoint = monotonic()
        dirty = False
        try:
            while not self._stop_event.is_set() or not self._capture_ring.empty():
                batch = self._capture_ring.wait_for_batch(
                    CAPTURE_BATCH_FRAMES,
                    CAPTURE_COALESCE_SECONDS,
                    self._stop_event,
                )
                if batch is not None:
                    self._write_capture_batch(batch)
                    self._capture_ring.release(batch)
                    dirty = True
                now = monotonic()
                if (
                    dirty
                    and self._capture_ring.empty()
                    and now - last_checkpoint >= RECORDING_CHECKPOINT_SECONDS
                ):
                    self.project.flush_audio()
                    dirty = False
                    last_checkpoint = now
        except Exception as exc:
            self._fail(f"Project recording operation failed: {exc}")

    def _write_capture_batch(self, batch: _CaptureRingBatch) -> None:
        position = batch.position
        armed_tracks = _ARM_MASK_TRACKS[batch.armed_mask]
        for view in self._capture_ring.views(batch):
            self.project.write_routed_recording_block(
                position, view, armed_tracks
            )
            position += len(view)
        self._known_frames = max(
            self._known_frames, batch.position + batch.frames
        )

    def _fill_playback_queue(self, reader: soundfile.SoundFile) -> bool:
        while self._playback_ring.free_frames > 0:
            if self._read_cursor >= self._playback_end_frames:
                self._ready_event.set()
                return True
            read_frames = min(
                DISK_BLOCK_FRAMES,
                self._playback_end_frames - self._read_cursor,
                self._playback_ring.free_frames,
            )
            block = self.project.read_audio_block(
                self._read_cursor, read_frames, audio_file=reader
            )
            self._playback_ring.write(block)
            self._read_cursor += len(block)
            self._mark_prebuffered(len(block))
        return False

    def _fill_shuttle_queue(
        self, mode: TransportMode, reader: soundfile.SoundFile
    ) -> bool:
        while self._playback_ring.free_frames > 0:
            if mode is TransportMode.FAST_FORWARD:
                remaining = self._playback_end_frames - self._read_cursor
                if remaining <= 0:
                    self._ready_event.set()
                    return True
                output_frames = min(
                    SHUTTLE_OUTPUT_BLOCK_FRAMES,
                    (remaining + SHUTTLE_SPEED - 1) // SHUTTLE_SPEED,
                    self._playback_ring.free_frames,
                )
                indices = self._read_cursor + np.arange(
                    output_frames, dtype=np.int64
                ) * SHUTTLE_SPEED
                block_position = self._read_cursor
                self._read_cursor = min(
                    self._playback_end_frames,
                    self._read_cursor + output_frames * SHUTTLE_SPEED,
                )
            else:
                remaining = self._read_cursor
                if remaining <= 0:
                    self._ready_event.set()
                    return True
                output_frames = min(
                    SHUTTLE_OUTPUT_BLOCK_FRAMES,
                    (remaining + SHUTTLE_SPEED - 1) // SHUTTLE_SPEED,
                    self._playback_ring.free_frames,
                )
                indices = self._read_cursor - 1 - np.arange(
                    output_frames, dtype=np.int64
                ) * SHUTTLE_SPEED
                block_position = self._read_cursor
                self._read_cursor = max(
                    0, self._read_cursor - output_frames * SHUTTLE_SPEED
                )

            block = self._filtered_shuttle_samples(indices, reader)
            block *= SHUTTLE_GAIN
            self._playback_ring.write(block)
            self._mark_prebuffered(len(block))
        return False

    def _mark_prebuffered(self, frames: int) -> None:
        self._prebuffered_frames += frames
        if self._prebuffered_frames >= self._prebuffer_target_frames:
            self._ready_event.set()

    def _filtered_shuttle_samples(
        self, indices: np.ndarray, reader: soundfile.SoundFile
    ) -> np.ndarray:
        half_taps = SHUTTLE_FILTER_TAPS // 2
        read_start = int(indices.min()) - half_taps
        read_end = int(indices.max()) + half_taps + 1
        source = np.zeros(
            (read_end - read_start, PROJECT_TRACK_COUNT), dtype=np.float32
        )
        available_start = max(0, read_start)
        available_end = min(self._playback_end_frames, read_end)
        if available_end > available_start:
            available = self.project.read_audio_block(
                available_start,
                available_end - available_start,
                audio_file=reader,
            )
            offset = available_start - read_start
            source[offset : offset + len(available)] = available

        windows = np.lib.stride_tricks.sliding_window_view(
            source, SHUTTLE_FILTER_TAPS, axis=0
        )
        window_indices = indices - read_start - half_taps
        selected = windows[window_indices]
        return np.einsum(
            "nct,t->nc", selected, _SHUTTLE_FILTER_KERNEL, optimize=True
        ).astype(np.float32, copy=False)

    def _fail(self, message: str) -> None:
        if self._terminal_error is None and self._realtime_fault is RealtimeFault.NONE:
            self._terminal_error = message
        self.end_requested = True

    def _fail_realtime(self, fault: RealtimeFault) -> None:
        if self._terminal_error is None and self._realtime_fault is RealtimeFault.NONE:
            self._realtime_fault = fault
        self.end_requested = True

def _shuttle_filter_kernel() -> np.ndarray:
    """Return a low-pass FIR for ten-to-one tape-speed decimation."""
    positions = np.arange(SHUTTLE_FILTER_TAPS, dtype=np.float64)
    positions -= (SHUTTLE_FILTER_TAPS - 1) / 2
    cutoff = 0.45 / SHUTTLE_SPEED
    kernel = 2 * cutoff * np.sinc(2 * cutoff * positions)
    kernel *= np.hanning(SHUTTLE_FILTER_TAPS)
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


_SHUTTLE_FILTER_KERNEL = _shuttle_filter_kernel()
