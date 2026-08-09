"""Tests for portable WAV project files."""

import json
from pathlib import Path

import pytest
import soundfile

from tape_machine.project import (
    PROJECT_COMMENT_PREFIX,
    AudioProject,
    MixerMetadata,
    ProjectError,
    ProjectMetadata,
    TrackMixMetadata,
)


def project_metadata() -> ProjectMetadata:
    mix = MixerMetadata(
        tracks=tuple(
            TrackMixMetadata(
                level_db=-float(index),
                pan=(index - 3.5) / 3.5,
                record_enabled=index % 2 == 0,
                input_monitoring=index % 3 == 0,
                muted=index == 4,
                soloed=index == 6,
            )
            for index in range(8)
        ),
        bus_level_db=-6.0,
    )
    return ProjectMetadata(mix=mix)


def test_create_and_reopen_rf64_project(tmp_path: Path) -> None:
    path = tmp_path / "session.wav"

    project = AudioProject.create(path, 48_000, project_metadata())

    assert project.format == "RF64"
    assert project.subtype == "PCM_24"
    assert project.channels == 8
    assert project.sample_rate == 48_000
    assert project.frames == 0
    assert project.dirty is False
    project.close()

    reopened = AudioProject.open(path)
    assert reopened.metadata == project_metadata()
    assert reopened.dirty is False
    reopened.close()


def test_import_untagged_eight_channel_wav_preserves_comment(
    tmp_path: Path,
) -> None:
    path = tmp_path / "import.wav"
    with soundfile.SoundFile(
        path,
        "w",
        samplerate=44_100,
        channels=8,
        subtype="PCM_16",
        format="WAV",
    ) as audio_file:
        audio_file.comment = "Location recording"
        audio_file.write([[0.0] * 8])

    project = AudioProject.open(path)

    assert project.sample_rate == 44_100
    assert project.dirty is True
    assert project.metadata.source_comment == "Location recording"
    project.save()
    project.close()

    with soundfile.SoundFile(path) as audio_file:
        assert audio_file.comment.startswith(PROJECT_COMMENT_PREFIX)
        assert audio_file.frames == 1


def test_open_rejects_wrong_channel_count(tmp_path: Path) -> None:
    path = tmp_path / "stereo.wav"
    soundfile.write(path, [[0.0, 0.0]], 48_000)

    with pytest.raises(ProjectError, match="exactly 8 channels"):
        AudioProject.open(path)


def test_open_rejects_malformed_tagged_metadata(tmp_path: Path) -> None:
    path = tmp_path / "broken.wav"
    with soundfile.SoundFile(
        path, "w", samplerate=48_000, channels=8, subtype="PCM_24"
    ) as audio_file:
        audio_file.comment = PROJECT_COMMENT_PREFIX + "{broken"

    with pytest.raises(ProjectError, match="not valid JSON"):
        AudioProject.open(path)


def test_project_metadata_contains_only_project_state() -> None:
    metadata = project_metadata()
    comment = metadata.to_comment()
    payload = json.loads(comment[len(PROJECT_COMMENT_PREFIX) :])

    assert payload["schema_version"] == 5
    assert set(payload) == {"application", "schema_version", "mix"}
    assert ProjectMetadata.from_comment(comment) == metadata


@pytest.mark.parametrize("schema_version", [1, 2, 3, 4])
def test_legacy_project_metadata_schemas_are_rejected(
    schema_version: int,
) -> None:
    payload = json.loads(
        project_metadata().to_comment()[len(PROJECT_COMMENT_PREFIX) :]
    )
    payload["schema_version"] = schema_version

    with pytest.raises(ProjectError, match="unsupported metadata schema"):
        ProjectMetadata.from_comment(
            PROJECT_COMMENT_PREFIX + json.dumps(payload)
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda mix: mix["tracks"].pop(), "8 tracks"),
        (
            lambda mix: mix["tracks"][0].update(level_db=7),
            "track level",
        ),
        (
            lambda mix: mix["tracks"][0].update(pan=float("nan")),
            "track pan",
        ),
        (
            lambda mix: mix["tracks"][0].update(muted="yes"),
            "muted must be boolean",
        ),
    ],
)
def test_project_metadata_rejects_invalid_mix(mutate, message: str) -> None:
    payload = json.loads(
        project_metadata().to_comment()[len(PROJECT_COMMENT_PREFIX) :]
    )
    mutate(payload["mix"])

    with pytest.raises(ProjectError, match=message):
        ProjectMetadata.from_comment(
            PROJECT_COMMENT_PREFIX + json.dumps(payload)
        )


def test_staging_mix_marks_project_dirty_until_saved(tmp_path: Path) -> None:
    path = tmp_path / "staged.wav"
    project = AudioProject.create(path, 48_000, ProjectMetadata())
    original_comment = project.audio_file.comment
    staged = project.metadata.with_mix(
        MixerMetadata(bus_level_db=-9.0)
    )

    project.stage_metadata(staged)

    assert project.metadata == staged
    assert project.dirty is True
    assert project.audio_file.comment == original_comment

    project.save()

    assert project.dirty is False
    assert ProjectMetadata.from_comment(project.audio_file.comment) == staged
    project.close()


def test_save_updates_state_only_after_flush_succeeds() -> None:
    class FailingFile:
        comment = "old"
        mode = "r+"

        def flush(self) -> None:
            raise OSError("disk full")

    original = ProjectMetadata()
    project = AudioProject(
        Path("project.wav"), FailingFile(), original, dirty=True
    )
    candidate = project_metadata()

    with pytest.raises(ProjectError, match="disk full"):
        project.save(candidate)

    assert project.metadata is original
    assert project.dirty is True
