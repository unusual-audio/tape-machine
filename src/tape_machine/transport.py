"""Project playback and recording transport."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Thread
from time import monotonic

import numpy as np

from tape_machine.audio import PROJECT_TRACK_COUNT
from tape_machine.project import AudioProject, ProjectError


DISK_BLOCK_FRAMES = 4096
PLAYBACK_QUEUE_BLOCKS = 8
CAPTURE_QUEUE_BLOCKS = 128


class TransportError(RuntimeError):
    """Raised when playback or recording cannot continue safely."""


@dataclass(frozen=True, slots=True)
class CaptureBlock:
    position: int
    data: np.ndarray


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
        self.running = False
        self.record_armed = False
        self.end_requested = False
        self.terminal_error: str | None = None

        self._track_inputs: tuple[int | None, ...] = (None,) * PROJECT_TRACK_COUNT
        self._armed_tracks: tuple[bool, ...] = (False,) * PROJECT_TRACK_COUNT
        self._known_frames = project.frames
        self._playback_end_frames = project.frames
        self._read_cursor = 0
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
    def recording(self) -> bool:
        return self.running and self.record_armed and any(self._armed_tracks)

    @property
    def armed_tracks(self) -> tuple[bool, ...]:
        return self._armed_tracks

    def toggle_record(self) -> bool:
        """Toggle record mode for the next callback boundary."""
        self.record_armed = not self.record_armed
        return self.record_armed

    def play(
        self,
        track_inputs: tuple[int | None, ...],
        armed_tracks: tuple[bool, ...],
    ) -> bool:
        """Prebuffer and start transport at the current frame position."""
        if self.running:
            return True
        if len(track_inputs) != PROJECT_TRACK_COUNT or len(armed_tracks) != (
            PROJECT_TRACK_COUNT
        ):
            raise TransportError("Transport requires exactly eight project tracks.")
        if self.record_armed and not any(armed_tracks):
            raise TransportError("Record-enable at least one track before recording.")
        if self.record_armed and any(armed_tracks) and not self.project.writable:
            raise TransportError("This project file is read-only and cannot record.")
        if self.position_frames >= self.project.frames and not (
            self.record_armed and any(armed_tracks)
        ):
            return False

        self._track_inputs = track_inputs
        self._armed_tracks = armed_tracks
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
        self._worker = Thread(
            target=self._disk_worker,
            name="TapeMachineTransport",
            daemon=True,
        )
        self._worker.start()
        if not self._ready_event.wait(timeout=2.0):
            self._stop_event.set()
            self._worker.join(timeout=2.0)
            raise TransportError("Timed out while prebuffering project audio.")
        if self.terminal_error is not None:
            raise TransportError(self.terminal_error)
        self.running = True
        return True

    def process_audio(
        self, indata: np.ndarray, frames: int, status: object
    ) -> np.ndarray | None:
        """Consume playback and enqueue capture without blocking the callback."""
        if not self.running:
            return None
        start_position = self.position_frames
        playback = np.zeros((frames, PROJECT_TRACK_COUNT), dtype=np.float32)
        copied = self._consume_playback(playback)
        reached_end = False
        if copied < frames:
            if start_position + copied < self._playback_end_frames:
                self._fail("Project playback could not keep up with the audio stream.")
            elif not self.recording:
                self.end_requested = True
                reached_end = True
        elif (
            not self.recording
            and start_position + copied >= self._playback_end_frames
        ):
            self.end_requested = True
            reached_end = True

        if self.recording:
            playback[:, self._armed_tracks] = 0
            try:
                self._capture_queue.put_nowait(
                    CaptureBlock(start_position, indata.copy())
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

    def stop(self) -> None:
        """Stop, drain recorded blocks, flush the project, and retain position."""
        self.running = False
        self._stop_event.set()
        worker = self._worker
        if worker is not None:
            worker.join(timeout=5.0)
            if worker.is_alive():
                self._fail("Timed out while finishing the recording.")
        self._worker = None
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
            self._armed_tracks,
        )
        self._known_frames = max(
            self._known_frames, capture.position + len(capture.data)
        )
        return True

    def _fill_playback_queue(self) -> None:
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
