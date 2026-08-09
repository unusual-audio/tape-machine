"""Project playback and recording transport."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from math import ceil
from queue import Empty, Full, Queue
from threading import Condition, Event, Thread
from time import monotonic
from typing import Sequence

import numpy as np
import soundfile

from tape_machine.audio import (
    PROJECT_TRACK_COUNT,
    STEREO_BUS_CHANNEL_COUNT,
    TrackInputRoute,
)
from tape_machine.project import AudioProject, ProjectError


DISK_BLOCK_FRAMES = 8192
PLAYBACK_BUFFER_SECONDS = 1.0
PLAYBACK_PREBUFFER_SECONDS = 0.25
CAPTURE_BUFFER_SECONDS = 3.0
CAPTURE_BATCH_FRAMES = 8192
CAPTURE_COALESCE_SECONDS = 0.010
RECORDING_CHECKPOINT_SECONDS = 5.0
SHUTTLE_SPEED = 10
SHUTTLE_GAIN_DB = -9.0
SHUTTLE_GAIN = 10 ** (SHUTTLE_GAIN_DB / 20.0)
SHUTTLE_OUTPUT_BLOCK_FRAMES = 1024
SHUTTLE_FILTER_TAPS = 81


class TransportMode(Enum):
    """Current direction and speed of the project transport."""

    STOPPED = "stopped"
    PLAYING = "playing"
    FAST_FORWARD = "fast_forward"
    REWIND = "rewind"

    @property
    def shuttling(self) -> bool:
        return self in {self.FAST_FORWARD, self.REWIND}


class TransportError(RuntimeError):
    """Raised when playback or recording cannot continue safely."""


@dataclass(frozen=True, slots=True)
class CaptureBlock:
    position: int
    input_data: np.ndarray
    stereo_bus: np.ndarray
    armed_tracks: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class CaptureContext:
    """Callback-local recording state awaiting the rendered stereo bus."""

    position: int
    frames: int
    armed_tracks: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class PlaybackBlock:
    position: int
    data: np.ndarray


class _CaptureBuffer:
    """A callback-facing queue limited by audio duration, not block count."""

    def __init__(self, max_frames: int) -> None:
        self.max_frames = max_frames
        self._blocks: deque[CaptureBlock] = deque()
        self._buffered_frames = 0
        self._condition = Condition()

    @property
    def buffered_frames(self) -> int:
        with self._condition:
            return self._buffered_frames

    def empty(self) -> bool:
        with self._condition:
            return not self._blocks

    def put_nowait(self, block: CaptureBlock) -> None:
        frames = len(block.input_data)
        with self._condition:
            if self._buffered_frames + frames > self.max_frames:
                raise Full
            self._blocks.append(block)
            self._buffered_frames += frames
            self._condition.notify()

    def get_nowait(self) -> CaptureBlock:
        with self._condition:
            if not self._blocks:
                raise Empty
            block = self._blocks.popleft()
            self._buffered_frames -= len(block.input_data)
            return block

    def clear(self) -> None:
        with self._condition:
            self._blocks.clear()
            self._buffered_frames = 0

    def wake(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def wait_for_batch(
        self,
        target_frames: int,
        coalesce_seconds: float,
        stop_event: Event,
    ) -> tuple[CaptureBlock, ...]:
        """Wait briefly, then pop one contiguous arm-mask-compatible batch."""
        with self._condition:
            while not self._blocks:
                if stop_event.is_set():
                    return ()
                self._condition.wait(timeout=0.05)

            deadline = monotonic() + coalesce_seconds
            while self._buffered_frames < target_frames:
                remaining = deadline - monotonic()
                if remaining <= 0 or stop_event.is_set():
                    break
                self._condition.wait(timeout=remaining)

            first = self._blocks.popleft()
            batch = [first]
            batch_frames = len(first.input_data)
            next_position = first.position + batch_frames
            while self._blocks:
                candidate = self._blocks[0]
                candidate_frames = len(candidate.input_data)
                if (
                    candidate.armed_tracks != first.armed_tracks
                    or candidate.position != next_position
                    or batch_frames + candidate_frames > target_frames
                ):
                    break
                batch.append(self._blocks.popleft())
                batch_frames += candidate_frames
                next_position += candidate_frames
            self._buffered_frames -= batch_frames
            return tuple(batch)


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
        self.record_armed = False
        self.end_requested = False
        self.terminal_error: str | None = None

        self._track_inputs: tuple[TrackInputRoute, ...] = (
            None,
        ) * PROJECT_TRACK_COUNT
        self._armed_tracks: tuple[bool, ...] = (False,) * PROJECT_TRACK_COUNT
        self._known_frames = project.frames
        self._playback_end_frames = project.frames
        self._read_cursor = 0
        self._worker_mode = TransportMode.STOPPED
        self._playback_queue: Queue[PlaybackBlock] = Queue(maxsize=1)
        self._capture_queue = _CaptureBuffer(
            max(1, round(project.sample_rate * CAPTURE_BUFFER_SECONDS))
        )
        self._current_playback: PlaybackBlock | None = None
        self._current_offset = 0
        self._stop_event = Event()
        self._ready_event = Event()
        self._playback_space_event = Event()
        self._prebuffer_target_frames = 0
        self._prebuffered_frames = 0
        self._playback_reader: soundfile.SoundFile | None = None
        self._playback_worker_thread: Thread | None = None
        self._capture_worker_thread: Thread | None = None

    @property
    def running(self) -> bool:
        return self.mode is not TransportMode.STOPPED

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
        self._known_frames = self.project.frames
        self._playback_end_frames = self._known_frames
        self._read_cursor = self.position_frames
        self.end_requested = False
        self.terminal_error = None
        self._current_playback = None
        self._current_offset = 0
        playback_block_frames = (
            SHUTTLE_OUTPUT_BLOCK_FRAMES if mode.shuttling else DISK_BLOCK_FRAMES
        )
        playback_blocks = max(
            1,
            ceil(
                self.project.sample_rate
                * PLAYBACK_BUFFER_SECONDS
                / playback_block_frames
            ),
        )
        self._playback_queue = Queue(maxsize=playback_blocks)
        self._capture_queue.clear()
        self._stop_event.clear()
        self._ready_event.clear()
        self._playback_space_event.clear()
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
        self._playback_reader = self.project.open_playback_reader()
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
        if not self._ready_event.wait(timeout=2.0):
            self._finish_workers(timeout=2.0)
            raise TransportError("Timed out while prebuffering project audio.")
        if self.terminal_error is not None:
            self._finish_workers(timeout=2.0)
            raise TransportError(self.terminal_error)
        self.mode = mode
        return True

    def prepare_audio(
        self,
        frames: int,
        status: object,
        destination: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, CaptureContext | None]:
        """Consume playback and return capture state for the rendered bus."""
        if not self.running:
            return None, None
        mode = self.mode
        start_position = self.position_frames
        if mode.shuttling:
            return (
                self._process_shuttle_audio(
                    frames, mode, start_position, destination
                ),
                None,
            )
        record_armed = self.record_armed
        armed_tracks = self._armed_tracks
        recording = record_armed and any(armed_tracks)
        playback = self._prepare_destination(frames, destination)
        copied = self._consume_playback(playback)
        reached_end = False
        if copied < frames:
            if start_position + copied < self._playback_end_frames:
                self._fail("Project playback could not keep up with the audio stream.")
            elif not record_armed:
                self.end_requested = True
                reached_end = True
        elif (
            not record_armed
            and start_position + copied >= self._playback_end_frames
        ):
            self.end_requested = True
            reached_end = True

        capture_context = None
        if recording:
            playback[:, armed_tracks] = 0
            capture_context = CaptureContext(
                start_position, frames, armed_tracks
            )
            if status:
                self._fail(f"Audio dropout while recording: {status}")

        if reached_end:
            self.position_frames = self._known_frames
        else:
            self.position_frames += frames
        return playback, capture_context

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
        try:
            self._capture_queue.put_nowait(
                CaptureBlock(
                    context.position,
                    input_data.copy(),
                    stereo_bus.copy(),
                    context.armed_tracks,
                )
            )
            self._known_frames = max(
                self._known_frames, context.position + context.frames
            )
        except Full:
            self._fail(
                "Recording buffer overflowed after "
                f"{CAPTURE_BUFFER_SECONDS:g} seconds; "
                "storage could not keep up."
            )

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
            self._fail(
                "Project shuttle playback could not keep up with the audio stream."
            )
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
        if not self._finish_workers(timeout=5.0):
            self._fail("Timed out while finishing the recording.")
        self._current_playback = None
        self._current_offset = 0
        self._drain_queue(self._playback_queue)
        if self._capture_worker_thread is None:
            try:
                self.project.flush_audio()
            except ProjectError as exc:
                self._fail(str(exc))

    def return_to_zero(self) -> None:
        """Return the stopped transport to the first project frame."""
        if not self.running:
            self.position_frames = 0
            self.end_requested = False

    def _consume_playback(self, destination: np.ndarray) -> int:
        copied = 0
        while copied < len(destination):
            if self._current_playback is None:
                try:
                    self._current_playback = self._playback_queue.get_nowait()
                except Empty:
                    break
                self._current_offset = 0
                self._playback_space_event.set()
            available = len(self._current_playback.data) - self._current_offset
            amount = min(len(destination) - copied, available)
            destination[copied : copied + amount] = self._current_playback.data[
                self._current_offset : self._current_offset + amount
            ]
            copied += amount
            self._current_offset += amount
            if self._current_offset == len(self._current_playback.data):
                self._current_playback = None
        return copied

    def _finish_workers(self, timeout: float) -> bool:
        """Signal both disk workers and close their playback handle."""
        self._stop_event.set()
        self._capture_queue.wake()
        self._playback_space_event.set()
        workers = (
            self._capture_worker_thread,
            self._playback_worker_thread,
        )
        for worker in workers:
            if worker is not None:
                worker.join(timeout=timeout)
        all_stopped = not any(
            worker is not None and worker.is_alive() for worker in workers
        )
        if (
            self._capture_worker_thread is not None
            and not self._capture_worker_thread.is_alive()
        ):
            self._capture_worker_thread = None
        if (
            self._playback_worker_thread is not None
            and not self._playback_worker_thread.is_alive()
        ):
            self._playback_worker_thread = None
            reader = self._playback_reader
            self._playback_reader = None
            if reader is not None:
                try:
                    reader.close()
                except Exception as exc:
                    self._fail(f"Unable to close project playback: {exc}")
        if all_stopped:
            self._worker_mode = TransportMode.STOPPED
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
                self._playback_space_event.wait(timeout=0.05)
                self._playback_space_event.clear()
        except Exception as exc:
            self._fail(f"Project playback operation failed: {exc}")
        finally:
            self._ready_event.set()

    def _capture_worker_main(self) -> None:
        last_checkpoint = monotonic()
        dirty = False
        try:
            while not self._stop_event.is_set() or not self._capture_queue.empty():
                batch = self._capture_queue.wait_for_batch(
                    CAPTURE_BATCH_FRAMES,
                    CAPTURE_COALESCE_SECONDS,
                    self._stop_event,
                )
                if batch:
                    self._write_capture_batch(batch)
                    dirty = True
                now = monotonic()
                if (
                    dirty
                    and self._capture_queue.empty()
                    and now - last_checkpoint >= RECORDING_CHECKPOINT_SECONDS
                ):
                    self.project.flush_audio()
                    dirty = False
                    last_checkpoint = now
        except Exception as exc:
            self._fail(f"Project recording operation failed: {exc}")

    def _write_capture_batch(self, batch: Sequence[CaptureBlock]) -> None:
        first = batch[0]
        if len(batch) == 1:
            input_data = first.input_data
            stereo_bus = first.stereo_bus
        else:
            input_data = np.concatenate(
                tuple(block.input_data for block in batch), axis=0
            )
            stereo_bus = np.concatenate(
                tuple(block.stereo_bus for block in batch), axis=0
            )
        self.project.write_recording_block(
            first.position,
            input_data,
            stereo_bus,
            self._track_inputs,
            first.armed_tracks,
        )
        self._known_frames = max(
            self._known_frames, first.position + len(input_data)
        )

    def _fill_playback_queue(self, reader: soundfile.SoundFile) -> bool:
        while not self._playback_queue.full():
            if self._read_cursor >= self._playback_end_frames:
                self._ready_event.set()
                return True
            read_frames = min(
                DISK_BLOCK_FRAMES,
                self._playback_end_frames - self._read_cursor,
            )
            block = self.project.read_audio_block(
                self._read_cursor, read_frames, audio_file=reader
            )
            self._playback_queue.put_nowait(
                PlaybackBlock(self._read_cursor, block)
            )
            self._read_cursor += len(block)
            self._mark_prebuffered(len(block))
        return False

    def _fill_shuttle_queue(
        self, mode: TransportMode, reader: soundfile.SoundFile
    ) -> bool:
        while not self._playback_queue.full():
            if mode is TransportMode.FAST_FORWARD:
                remaining = self._playback_end_frames - self._read_cursor
                if remaining <= 0:
                    self._ready_event.set()
                    return True
                output_frames = min(
                    SHUTTLE_OUTPUT_BLOCK_FRAMES,
                    (remaining + SHUTTLE_SPEED - 1) // SHUTTLE_SPEED,
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
            self._playback_queue.put_nowait(
                PlaybackBlock(block_position, block)
            )
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
        if self.terminal_error is None:
            self.terminal_error = message
        self.end_requested = True

    @staticmethod
    def _drain_queue(queue: Queue[object]) -> None:
        while True:
            try:
                queue.get_nowait()
            except Empty:
                return


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
