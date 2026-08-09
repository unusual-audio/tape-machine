"""Tests for project playback, punch recording, and transport position."""

from pathlib import Path
from time import sleep

import numpy as np
import pytest

from tape_machine.project import AudioProject, ProjectMetadata
from tape_machine.transport import (
    TransportController,
    TransportError,
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


def test_record_requires_at_least_one_record_enabled_track(
    tmp_path: Path,
) -> None:
    project = project_with_audio(tmp_path, np.zeros((1, 8), np.float32))
    transport = TransportController(project)
    transport.toggle_record()

    with pytest.raises(TransportError, match="Record-enable"):
        transport.play(TRACK_INPUTS, (False,) * 8)

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
