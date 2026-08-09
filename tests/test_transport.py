"""Tests for project playback, punch recording, and transport position."""

from pathlib import Path
from queue import Full
from threading import Event, Thread
from time import sleep

import numpy as np
import pytest
import soundfile

import tape_machine.transport as transport_module
from tape_machine.audio import StereoBusInput
from tape_machine.project import AudioProject, ProjectMetadata
from tape_machine.transport import (
    CAPTURE_BUFFER_SECONDS,
    CaptureContext,
    SHUTTLE_GAIN,
    SHUTTLE_SPEED,
    TransportController,
    TransportError,
    TransportLifecycle,
    TransportMode,
    _CaptureRing,
    _CaptureRingBatch,
    format_transport_time,
)


TRACK_INPUTS = (0, 1, None, None, None, None, None, None)


def project_with_audio(tmp_path: Path, audio: np.ndarray) -> AudioProject:
    project = AudioProject.create(
        tmp_path / "transport.wav", 48_000, ProjectMetadata()
    )
    project.audio_file.write(audio)
    project.flush_audio()
    return project


def test_transport_time_is_minutes_seconds_and_milliseconds() -> None:
    assert format_transport_time(0, 48_000) == "00:00.000"
    assert format_transport_time(48_000 * 61 + 12_000, 48_000) == "01:01.250"
    assert format_transport_time(48_000 * 6_001, 48_000) == "100:01.000"


def test_global_record_rolls_with_no_record_enabled_tracks(
    tmp_path: Path,
) -> None:
    original = np.full((1, 8), 0.1, np.float32)
    project = project_with_audio(tmp_path, original)
    transport = TransportController(project)
    transport.toggle_record()

    assert transport.play(TRACK_INPUTS, (False,) * 8) is True
    playback = transport.process_audio(np.zeros((2, 2), np.float32), 2, None)

    assert playback[0] == pytest.approx(original[0], abs=1e-6)
    assert not playback[1].any()
    assert transport.position_frames == 2
    assert transport.end_requested is False
    transport.stop()
    assert transport.record_armed is False
    assert transport.armed_tracks == (False,) * 8
    assert project.frames == 1

    project.close()


def test_stop_disarms_global_record_but_preserves_track_record_enables(
    tmp_path: Path,
) -> None:
    project = project_with_audio(tmp_path, np.zeros((1, 8), np.float32))
    transport = TransportController(project)
    armed_tracks = (True, False, True) + (False,) * 5
    transport.toggle_record()

    assert transport.play(TRACK_INPUTS, armed_tracks) is True
    transport.stop()

    assert transport.record_armed is False
    assert transport.armed_tracks == armed_tracks
    project.close()


def test_play_at_eof_without_global_record_does_not_roll(tmp_path: Path) -> None:
    project = project_with_audio(tmp_path, np.zeros((1, 8), np.float32))
    transport = TransportController(project)
    transport.position_frames = 1

    assert transport.play(TRACK_INPUTS, (False,) * 8) is False

    project.close()


def test_playback_reads_tracks_and_stops_at_exact_eof(tmp_path: Path) -> None:
    audio = np.arange(48, dtype=np.float32).reshape(6, 8) / 100
    project = project_with_audio(tmp_path, audio)
    transport = TransportController(project)

    assert transport.play(TRACK_INPUTS, (False,) * 8) is True
    first = transport.process_audio(np.zeros((4, 2), np.float32), 4, None)
    second = transport.process_audio(np.zeros((4, 2), np.float32), 4, None)

    assert first == pytest.approx(audio[:4], abs=1e-6)
    assert second[:2] == pytest.approx(audio[4:], abs=1e-6)
    assert not second[2:].any()
    assert transport.end_requested is True
    assert transport.position_frames == 6
    transport.stop()
    transport.return_to_zero()
    assert transport.position_frames == 0
    project.close()


def test_fast_forward_plays_filtered_audio_at_ten_times_speed(
    tmp_path: Path,
) -> None:
    ramp = np.arange(1000, dtype=np.float32) / 1000
    audio = np.repeat(ramp[:, np.newaxis], 8, axis=1)
    project = project_with_audio(tmp_path, audio)
    transport = TransportController(project)
    transport.position_frames = 100

    assert transport.fast_forward() is True
    playback = transport.process_audio(
        np.zeros((3, 1), np.float32), 3, None
    )

    assert transport.mode is TransportMode.FAST_FORWARD
    assert transport.position_frames == 100 + 3 * SHUTTLE_SPEED
    assert playback[:, 0] == pytest.approx(
        ramp[[100, 110, 120]] * SHUTTLE_GAIN, abs=2e-5
    )
    transport.stop()
    assert transport.position_frames == 130
    project.close()


def test_rewind_plays_filtered_audio_backwards_at_ten_times_speed(
    tmp_path: Path,
) -> None:
    ramp = np.arange(1000, dtype=np.float32) / 1000
    audio = np.repeat(ramp[:, np.newaxis], 8, axis=1)
    project = project_with_audio(tmp_path, audio)
    transport = TransportController(project)
    transport.position_frames = 200

    assert transport.rewind() is True
    playback = transport.process_audio(
        np.zeros((3, 1), np.float32), 3, None
    )

    assert transport.mode is TransportMode.REWIND
    assert transport.position_frames == 200 - 3 * SHUTTLE_SPEED
    assert playback[:, 0] == pytest.approx(
        ramp[[199, 189, 179]] * SHUTTLE_GAIN, abs=2e-5
    )
    transport.stop()
    assert transport.position_frames == 170
    project.close()


def test_shuttle_filter_attenuates_source_frequencies_above_new_nyquist(
    tmp_path: Path,
) -> None:
    alternating = np.tile(
        np.array([-0.5, 0.5], np.float32), 1000
    )
    audio = np.repeat(alternating[:, np.newaxis], 8, axis=1)
    project = project_with_audio(tmp_path, audio)
    transport = TransportController(project)
    transport.position_frames = 200

    transport.fast_forward()
    playback = transport.process_audio(
        np.zeros((8, 1), np.float32), 8, None
    )

    assert np.max(np.abs(playback)) < 0.005
    transport.stop()
    project.close()


def test_shuttle_stops_exactly_at_project_boundaries(tmp_path: Path) -> None:
    project = project_with_audio(tmp_path, np.ones((25, 8), np.float32) * 0.1)
    transport = TransportController(project)
    transport.position_frames = 23

    assert transport.fast_forward() is True
    transport.process_audio(np.zeros((4, 1), np.float32), 4, None)
    assert transport.position_frames == 25
    assert transport.end_requested is True
    transport.stop()
    assert transport.fast_forward() is False

    transport.position_frames = 5
    assert transport.rewind() is True
    transport.process_audio(np.zeros((4, 1), np.float32), 4, None)
    assert transport.position_frames == 0
    assert transport.end_requested is True
    transport.stop()
    assert transport.rewind() is False
    project.close()


def test_shuttle_disarms_global_record_without_changing_tracks_or_audio(
    tmp_path: Path,
) -> None:
    original = np.full((100, 8), 0.1, np.float32)
    project = project_with_audio(tmp_path, original)
    transport = TransportController(project)
    armed_tracks = (True, False, True) + (False,) * 5
    transport.set_armed_tracks(armed_tracks)
    transport.toggle_record()

    assert transport.fast_forward() is True
    assert transport.record_armed is False
    assert transport.armed_tracks == armed_tracks
    transport.process_audio(np.ones((2, 2), np.float32), 2, None)
    assert transport._capture_ring.empty()
    transport.stop()
    assert project.read_audio_block(0, 100) == pytest.approx(
        original, abs=1e-6
    )
    project.close()


def test_recording_replaces_only_armed_tracks_and_duplicates_inputs(
    tmp_path: Path,
) -> None:
    original = np.full((4, 8), 0.1, dtype=np.float32)
    project = project_with_audio(tmp_path, original)
    transport = TransportController(project)
    input_audio = np.array(
        [[0.2, -0.2], [0.3, -0.3], [0.4, -0.4], [0.5, -0.5]],
        dtype=np.float32,
    )
    routes = (0, 1, 0, None, None, None, None, None)
    armed = (True, False, True, True, False, False, False, False)
    transport.toggle_record()

    assert transport.play(routes, armed) is True
    playback = transport.process_audio(input_audio, 4, None)
    transport.stop()
    recorded = project.read_audio_block(0, 4)

    assert not playback[:, [0, 2, 3]].any()
    assert playback[:, 1] == pytest.approx(original[:, 1], abs=1e-6)
    assert recorded[:, 0] == pytest.approx(input_audio[:, 0], abs=1e-6)
    assert recorded[:, 1] == pytest.approx(original[:, 1], abs=1e-6)
    assert recorded[:, 2] == pytest.approx(input_audio[:, 0], abs=1e-6)
    assert not recorded[:, 3].any()
    assert recorded[:, 4:] == pytest.approx(original[:, 4:], abs=1e-6)
    project.close()


def test_stereo_bus_inputs_record_like_ordinary_sources(
    tmp_path: Path,
) -> None:
    original = np.full((4, 8), 0.1, dtype=np.float32)
    project = project_with_audio(tmp_path, original)
    transport = TransportController(project)
    routes = (
        StereoBusInput.LEFT,
        StereoBusInput.RIGHT,
    ) + (None,) * 6
    armed = (True, True) + (False,) * 6
    stereo_bus = np.column_stack(
        (
            np.full(4, 1.25, np.float32),
            np.full(4, -1.25, np.float32),
        )
    )
    transport.toggle_record()

    assert transport.play(routes, armed) is True
    playback, capture_context = transport.prepare_audio(4, None)
    assert capture_context is not None
    transport.submit_capture(
        capture_context,
        np.zeros((4, 1), dtype=np.float32),
        stereo_bus,
    )
    transport.stop()
    recorded = project.read_audio_block(0, 4)

    assert not playback[:, :2].any()
    assert playback[:, 2:] == pytest.approx(original[:, 2:], abs=1e-6)
    assert recorded[:, 0] == pytest.approx([1.0] * 4, abs=1e-6)
    assert recorded[:, 1] == pytest.approx([-1.0] * 4, abs=1e-6)
    assert recorded[:, 2:] == pytest.approx(original[:, 2:], abs=1e-6)
    project.close()


def test_recording_from_eof_extends_the_project(tmp_path: Path) -> None:
    project = project_with_audio(tmp_path, np.zeros((2, 8), np.float32))
    transport = TransportController(project)
    transport.position_frames = 2
    transport.toggle_record()
    input_audio = np.full((3, 1), 0.25, dtype=np.float32)

    assert transport.play((0,) + (None,) * 7, (True,) + (False,) * 7)
    transport.process_audio(input_audio, 3, None)
    sleep(0.02)
    transport.toggle_record()
    after_punch = transport.process_audio(
        np.full((3, 1), 0.75, np.float32), 3, None
    )
    transport.stop()

    assert project.frames == 5
    assert transport.position_frames == 5
    assert transport.end_requested is True
    assert not after_punch.any()
    assert project.read_audio_block(2, 3)[:, 0] == pytest.approx(
        input_audio[:, 0], abs=1e-6
    )
    project.close()


def test_record_toggle_punches_out_while_transport_keeps_rolling(
    tmp_path: Path,
) -> None:
    original = np.full((8, 8), 0.1, dtype=np.float32)
    project = project_with_audio(tmp_path, original)
    transport = TransportController(project)
    transport.toggle_record()
    armed = (True,) + (False,) * 7

    transport.play((0,) + (None,) * 7, armed)
    during_punch = transport.process_audio(
        np.full((4, 1), 0.5, np.float32), 4, None
    )
    transport.toggle_record()
    after_punch = transport.process_audio(
        np.full((4, 1), 0.75, np.float32), 4, None
    )
    transport.stop()
    recorded = project.read_audio_block(0, 8)

    assert not during_punch[:, 0].any()
    assert after_punch[:, 0] == pytest.approx(original[4:, 0], abs=1e-6)
    assert recorded[:4, 0] == pytest.approx([0.5] * 4, abs=1e-6)
    assert recorded[4:, 0] == pytest.approx(original[4:, 0], abs=1e-6)
    project.close()


def test_track_record_enable_can_punch_between_tracks_while_rolling(
    tmp_path: Path,
) -> None:
    original = np.full((12, 8), 0.1, dtype=np.float32)
    project = project_with_audio(tmp_path, original)
    transport = TransportController(project)
    transport.toggle_record()
    transport.play(TRACK_INPUTS, (False,) * 8)
    first_input = np.full((4, 2), 0.2, dtype=np.float32)
    second_input = np.column_stack(
        (np.full(4, 0.3, np.float32), np.full(4, -0.3, np.float32))
    )
    third_input = np.column_stack(
        (np.full(4, 0.4, np.float32), np.full(4, -0.4, np.float32))
    )

    before_punch = transport.process_audio(first_input, 4, None)
    transport.set_armed_tracks((True, False) + (False,) * 6)
    first_punch = transport.process_audio(second_input, 4, None)
    transport.set_armed_tracks((False, True) + (False,) * 6)
    second_punch = transport.process_audio(third_input, 4, None)
    transport.set_armed_tracks((False,) * 8)
    transport.stop()
    recorded = project.read_audio_block(0, 12)

    assert before_punch == pytest.approx(original[:4], abs=1e-6)
    assert not first_punch[:, 0].any()
    assert first_punch[:, 1] == pytest.approx(original[4:8, 1], abs=1e-6)
    assert second_punch[:, 0] == pytest.approx(original[8:, 0], abs=1e-6)
    assert not second_punch[:, 1].any()
    assert recorded[:4] == pytest.approx(original[:4], abs=1e-6)
    assert recorded[4:8, 0] == pytest.approx(second_input[:, 0], abs=1e-6)
    assert recorded[4:8, 1] == pytest.approx(original[4:8, 1], abs=1e-6)
    assert recorded[8:, 0] == pytest.approx(original[8:, 0], abs=1e-6)
    assert recorded[8:, 1] == pytest.approx(third_input[:, 1], abs=1e-6)
    project.close()


def test_capture_block_keeps_callback_arm_mask_after_ui_changes(
    tmp_path: Path,
) -> None:
    project = project_with_audio(tmp_path, np.zeros((1, 8), np.float32))
    transport = TransportController(project)
    callback_mask = (True, False) + (False,) * 6
    transport.running = True
    transport.record_armed = True
    transport.set_armed_tracks(callback_mask)

    transport.process_audio(np.ones((1, 2), np.float32), 1, None)
    transport.set_armed_tracks((False, True) + (False,) * 6)
    batch = transport._capture_ring.wait_for_batch(8, 0, Event())

    assert batch is not None
    assert batch.armed_mask == 1
    transport.running = False
    project.close()


def test_live_punch_after_rolling_through_silence_extends_project(
    tmp_path: Path,
) -> None:
    project = project_with_audio(tmp_path, np.zeros((2, 8), np.float32))
    transport = TransportController(project)
    transport.position_frames = 2
    transport.toggle_record()

    assert transport.play((0,) + (None,) * 7, (False,) * 8)
    transport.process_audio(np.zeros((3, 1), np.float32), 3, None)
    transport.set_armed_tracks((True,) + (False,) * 7)
    transport.process_audio(np.full((3, 1), 0.5, np.float32), 3, None)
    transport.stop()

    assert project.frames == 8
    recorded = project.read_audio_block(0, 8)
    assert not recorded[2:5].any()
    assert recorded[5:, 0] == pytest.approx([0.5] * 3, abs=1e-6)
    project.close()


def test_capture_capacity_is_three_seconds_at_project_sample_rate(
    tmp_path: Path,
) -> None:
    project = AudioProject.create(
        tmp_path / "capacity.wav", 44_100, ProjectMetadata()
    )
    transport = TransportController(project)

    assert transport._capture_ring.max_frames == round(
        44_100 * CAPTURE_BUFFER_SECONDS
    )
    project.close()


def test_capture_ring_uses_frame_budget_and_batches_compatible_blocks() -> None:
    capture_ring = _CaptureRing(max_frames=32)
    first_mask = (True,) + (False,) * 7
    second_mask = (False, True) + (False,) * 6
    routes = (0, 1) + (None,) * 6

    def put(position: int, frames: int, mask: tuple[bool, ...]) -> None:
        capture_ring.put_routed_nowait(
            position,
            np.zeros((frames, 1), dtype=np.float32),
            np.zeros((frames, 2), dtype=np.float32),
            routes,
            mask,
        )

    put(0, 2, first_mask)
    put(2, 3, first_mask)
    put(5, 2, second_mask)
    with pytest.raises(Full):
        put(7, 26, second_mask)

    batch = capture_ring.wait_for_batch(8, 0, Event())

    assert batch == _CaptureRingBatch(0, 5, 2, 1)
    capture_ring.release(batch)
    assert capture_ring.buffered_frames == 2
    next_batch = capture_ring.wait_for_batch(8, 0, Event())
    assert next_batch is not None
    assert next_batch.armed_mask == 2


def test_capture_overflow_reports_the_duration_of_the_safety_buffer(
    tmp_path: Path,
) -> None:
    project = AudioProject.create(
        tmp_path / "overflow.wav", 48_000, ProjectMetadata()
    )
    transport = TransportController(project)
    transport._capture_ring = _CaptureRing(max_frames=1)
    context = CaptureContext(0, 2, (True,) + (False,) * 7)

    transport.submit_capture(
        context,
        np.zeros((2, 1), dtype=np.float32),
        np.zeros((2, 2), dtype=np.float32),
    )

    assert transport.terminal_error == (
        "Recording buffer overflowed after 3 seconds; "
        "storage could not keep up."
    )
    project.close()


def test_playback_uses_an_independent_reader_and_supplied_destination(
    tmp_path: Path,
) -> None:
    audio = np.full((4, 8), 0.25, dtype=np.float32)
    project = project_with_audio(tmp_path, audio)
    transport = TransportController(project)

    assert transport.play(TRACK_INPUTS, (False,) * 8)
    reader = transport._playback_reader
    destination = np.full((4, 8), 99, dtype=np.float32)
    playback, _ = transport.prepare_audio(4, None, destination)

    assert reader is not None
    assert reader is not project.audio_file
    assert playback is destination
    assert playback == pytest.approx(audio, abs=1e-6)
    transport.stop()
    assert reader.closed
    project.close()


def test_recording_checkpoint_waits_until_capture_backlog_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mask = (True,) + (False,) * 7

    class FakeProject:
        frames = 0
        sample_rate = 48_000

        def __init__(self) -> None:
            self.writes: list[int] = []
            self.flushes = 0

        def write_routed_recording_block(
            self,
            position: int,
            track_data: np.ndarray,
            armed_tracks: tuple[bool, ...],
        ) -> None:
            self.writes.append(position)

        def flush_audio(self) -> None:
            self.flushes += 1

    class ScriptedBuffer:
        def __init__(self) -> None:
            self.batches = [
                _CaptureRingBatch(0, 2, 1, 1),
                _CaptureRingBatch(2, 2, 1, 1),
            ]

        def empty(self) -> bool:
            return not self.batches

        def wait_for_batch(
            self,
            target_frames: int,
            coalesce_seconds: float,
            stop_event: Event,
        ) -> _CaptureRingBatch | None:
            batch = self.batches.pop(0)
            if not self.batches:
                stop_event.set()
            return batch

        def views(self, batch: _CaptureRingBatch) -> tuple[np.ndarray, ...]:
            return (np.zeros((batch.frames, 8), dtype=np.float32),)

        def release(self, batch: _CaptureRingBatch) -> None:
            pass

    project = FakeProject()
    controller = TransportController(project)  # type: ignore[arg-type]
    controller._capture_ring = ScriptedBuffer()  # type: ignore[assignment]
    times = iter((0.0, 6.0, 6.0))
    monkeypatch.setattr(transport_module, "monotonic", lambda: next(times))

    controller._capture_worker_main()

    assert project.writes == [0, 2]
    assert project.flushes == 1


def test_worker_timeout_keeps_transport_faulted_until_late_exit(
    tmp_path: Path,
) -> None:
    project = project_with_audio(tmp_path, np.zeros((4, 8), np.float32))
    transport = TransportController(project)

    class LateWorker:
        alive = True

        def join(self, timeout: float = 0) -> None:
            pass

        def is_alive(self) -> bool:
            return self.alive

    worker = LateWorker()
    transport.mode = TransportMode.PLAYING
    transport.lifecycle = TransportLifecycle.RUNNING
    transport._capture_worker_thread = worker  # type: ignore[assignment]

    transport.stop()

    assert transport.mode is TransportMode.STOPPED
    assert transport.lifecycle is TransportLifecycle.FAULTED
    assert transport.busy is True
    with pytest.raises(TransportError, match="previous transport operation"):
        transport.play(TRACK_INPUTS, (False,) * 8)

    worker.alive = False
    transport.poll_lifecycle()

    assert transport.lifecycle is TransportLifecycle.IDLE
    assert transport.busy is False
    project.close()


def test_partial_worker_start_can_be_cleaned_up_without_joining_error(
    tmp_path: Path,
) -> None:
    project = project_with_audio(tmp_path, np.zeros((1, 8), np.float32))
    transport = TransportController(project)
    reader = project.open_playback_reader()
    transport._playback_reader = reader
    transport._playback_worker_thread = Thread(target=lambda: None)

    assert transport._finish_workers(timeout=0) is True
    assert transport._playback_worker_thread is None
    assert reader.closed
    project.close()


def test_record_start_aborts_if_project_becomes_read_only_during_handle_rotation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = project_with_audio(tmp_path, np.zeros((1, 8), np.float32))
    transport = TransportController(project)
    transport.toggle_record()
    opened_reader: soundfile.SoundFile | None = None

    def lose_write_access() -> soundfile.SoundFile:
        nonlocal opened_reader
        project.audio_file.close()
        project.audio_file = soundfile.SoundFile(project.path, mode="r")
        opened_reader = soundfile.SoundFile(project.path, mode="r")
        return opened_reader

    monkeypatch.setattr(project, "open_playback_reader", lose_write_access)

    with pytest.raises(TransportError, match="became read-only"):
        transport.play(TRACK_INPUTS, (True,) + (False,) * 7)

    assert transport.lifecycle is TransportLifecycle.IDLE
    assert opened_reader is not None and opened_reader.closed
    project.close()
