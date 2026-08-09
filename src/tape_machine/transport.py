"""Project playback and recording transport."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from queue import Empty, Full, Queue
from threading import Event, Thread
from time import monotonic

import numpy as np

from tape_machine.audio import PROJECT_TRACK_COUNT
from tape_machine.project import AudioProject, ProjectError


DISK_BLOCK_FRAMES = 4096
PLAYBACK_QUEUE_BLOCKS = 8
CAPTURE_QUEUE_BLOCKS = 128
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
    data: np.ndarray
    armed_tracks: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class PlaybackBlock:
    position: int
    data: np.ndarray


def format_transport_time(position_frames: int, sample_rate: int) -> str:
    """Format a frame position as unbounded MM:SS.mmm."""
    total_ms = max(0, position_frames) * 1000 // sample_rate
    minutes, remainder = divmod(total_ms, 60_000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


class TransportController:
    """Coordinate an audio callback with a dedicated project disk worker."""

    def __init__(self, project: AudioProject) -> None:
        self.project = project
        self.position_frames = 0
        self.mode = TransportMode.STOPPED
        self.record_armed = False
        self.end_requested = False
        self.terminal_error: str | None = None

        self._track_inputs: tuple[int | None, ...] = (None,) * PROJECT_TRACK_COUNT
        self._armed_tracks: tuple[bool, ...] = (False,) * PROJECT_TRACK_COUNT
        self._known_frames = project.frames
        self._playback_end_frames = project.frames
        self._read_cursor = 0
        self._worker_mode = TransportMode.STOPPED
        self._playback_queue: Queue[PlaybackBlock] = Queue(
            maxsize=PLAYBACK_QUEUE_BLOCKS
        )
        self._capture_queue: Queue[CaptureBlock] = Queue(
            maxsize=CAPTURE_QUEUE_BLOCKS
        )
        self._current_playback: PlaybackBlock | None = None
        self._current_offset = 0
        self._stop_event = Event()
        self._ready_event = Event()
        self._worker: Thread | None = None

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
        track_inputs: tuple[int | None, ...],
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
        self._drain_queue(self._playback_queue)
        self._drain_queue(self._capture_queue)
        self._stop_event.clear()
        self._ready_event.clear()
        self._worker_mode = mode
        self._worker = Thread(
            target=self._disk_worker,
            name="TapeMachineTransport",
            daemon=True,
        )
        self._worker.start()
        if not self._ready_event.wait(timeout=2.0):
            self._stop_event.set()
            self._worker.join(timeout=2.0)
            self._worker = None
            self._worker_mode = TransportMode.STOPPED
            raise TransportError("Timed out while prebuffering project audio.")
        if self.terminal_error is not None:
            self._stop_event.set()
            self._worker.join(timeout=2.0)
            self._worker = None
            self._worker_mode = TransportMode.STOPPED
            raise TransportError(self.terminal_error)
        self.mode = mode
        return True

    def process_audio(
        self, indata: np.ndarray, frames: int, status: object
    ) -> np.ndarray | None:
        """Consume playback and enqueue capture without blocking the callback."""
        if not self.running:
            return None
        mode = self.mode
        start_position = self.position_frames
        if mode.shuttling:
            return self._process_shuttle_audio(frames, mode, start_position)
        record_armed = self.record_armed
        armed_tracks = self._armed_tracks
        recording = record_armed and any(armed_tracks)
        playback = np.zeros((frames, PROJECT_TRACK_COUNT), dtype=np.float32)
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

        if recording:
            playback[:, armed_tracks] = 0
            try:
                self._capture_queue.put_nowait(
                    CaptureBlock(start_position, indata.copy(), armed_tracks)
                )
                self._known_frames = max(
                    self._known_frames, start_position + frames
                )
            except Full:
                self._fail("Recording buffer overflowed; the disk is too slow.")
            if status:
                self._fail(f"Audio dropout while recording: {status}")

        if reached_end:
            self.position_frames = self._known_frames
        else:
            self.position_frames += frames
        return playback

    def _process_shuttle_audio(
        self,
        frames: int,
        mode: TransportMode,
        start_position: int,
    ) -> np.ndarray:
        playback = np.zeros((frames, PROJECT_TRACK_COUNT), dtype=np.float32)
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

    def stop(self) -> None:
        """Stop, drain recorded blocks, flush the project, and retain position."""
        self.record_armed = False
        self.mode = TransportMode.STOPPED
        self._stop_event.set()
        worker = self._worker
        if worker is not None:
            worker.join(timeout=5.0)
            if worker.is_alive():
                self._fail("Timed out while finishing the recording.")
        self._worker = None
        self._worker_mode = TransportMode.STOPPED
        self._current_playback = None
        self._current_offset = 0
        self._drain_queue(self._playback_queue)
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

    def _disk_worker(self) -> None:
        last_flush = monotonic()
        try:
            while not self._stop_event.is_set() or not self._capture_queue.empty():
                wrote_capture = self._write_next_capture()
                if not self._stop_event.is_set():
                    self._fill_playback_queue()
                if monotonic() - last_flush >= 1.0:
                    self.project.flush_audio()
                    last_flush = monotonic()
                if not wrote_capture:
                    self._stop_event.wait(0.002)
            self.project.flush_audio()
        except Exception as exc:
            self._fail(f"Project disk operation failed: {exc}")
        finally:
            self._ready_event.set()

    def _write_next_capture(self) -> bool:
        try:
            capture = self._capture_queue.get_nowait()
        except Empty:
            return False
        self.project.write_recording_block(
            capture.position,
            capture.data,
            self._track_inputs,
            capture.armed_tracks,
        )
        self._known_frames = max(
            self._known_frames, capture.position + len(capture.data)
        )
        return True

    def _fill_playback_queue(self) -> None:
        if self._worker_mode.shuttling:
            self._fill_shuttle_queue(self._worker_mode)
            return
        while not self._playback_queue.full():
            if self._read_cursor >= self._playback_end_frames:
                self._ready_event.set()
                return
            read_frames = min(
                DISK_BLOCK_FRAMES,
                self._playback_end_frames - self._read_cursor,
            )
            block = self.project.read_audio_block(self._read_cursor, read_frames)
            self._playback_queue.put_nowait(
                PlaybackBlock(self._read_cursor, block)
            )
            self._read_cursor += len(block)
            self._ready_event.set()

    def _fill_shuttle_queue(self, mode: TransportMode) -> None:
        while not self._playback_queue.full():
            if mode is TransportMode.FAST_FORWARD:
                remaining = self._playback_end_frames - self._read_cursor
                if remaining <= 0:
                    self._ready_event.set()
                    return
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
                    return
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

            block = self._filtered_shuttle_samples(indices)
            block *= SHUTTLE_GAIN
            self._playback_queue.put_nowait(
                PlaybackBlock(block_position, block)
            )
            self._ready_event.set()

    def _filtered_shuttle_samples(self, indices: np.ndarray) -> np.ndarray:
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
                available_start, available_end - available_start
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
