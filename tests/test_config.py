"""Tests for persistent application-local configuration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tape_machine.audio import (
    AudioDevice,
    AudioSettings,
    DeviceReference,
    StereoBusInput,
)
from tape_machine.config import (
    MAX_RECENT_FILES,
    AppConfig,
    AppConfigStore,
    StoredAudioSettings,
    WindowPosition,
)


def stored_audio_settings() -> StoredAudioSettings:
    return StoredAudioSettings(
        input_device=DeviceReference(
            "Studio Input",
            "Core Audio",
            max_input_channels=8,
            max_output_channels=0,
            default_sample_rate=48_000,
            channel_name_signature=("Mic 1", "Mic 2"),
        ),
        output_device=DeviceReference(
            "Studio Output",
            "Core Audio",
            max_input_channels=0,
            max_output_channels=4,
            default_sample_rate=48_000,
            channel_name_signature=("Monitor L", "Monitor R"),
        ),
        sample_rate=96_000,
        track_inputs=(7, 6, 5, 4, 3, 2, 1, 0),
        bus_outputs=(2, 3),
        buffer_size=256,
    )


def test_config_round_trips_all_sections(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    config = AppConfig(
        recent_files=(tmp_path / "one.wav", tmp_path / "two.wav"),
        audio_settings=stored_audio_settings(),
        main_window_position=WindowPosition(120, 80),
        audio_settings_window_position=WindowPosition(-800, 140),
    )

    AppConfigStore(path).save(config)

    assert AppConfigStore(path).load() == config
    assert not list(tmp_path.glob(".config.json.*.tmp"))


def test_config_round_trips_stereo_bus_inputs(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    stored = stored_audio_settings()
    stored = StoredAudioSettings(
        stored.input_device,
        stored.output_device,
        stored.sample_rate,
        (StereoBusInput.LEFT, StereoBusInput.RIGHT) + (None,) * 6,
        stored.bus_outputs,
    )

    AppConfigStore(path).save(AppConfig(audio_settings=stored))

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 5
    assert payload["audio_settings"]["track_inputs"][:2] == [
        "stereo_bus_l",
        "stereo_bus_r",
    ]
    assert payload["audio_settings"]["buffer_size"] == 0
    assert AppConfigStore(path).load().audio_settings == stored


def test_invalid_stored_buffer_size_discards_audio_settings(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.json"
    AppConfigStore(path).save(
        AppConfig(audio_settings=stored_audio_settings())
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["audio_settings"]["buffer_size"] = 4096
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert AppConfigStore(path).load().audio_settings is None


def test_invalid_section_resets_only_that_section_and_preserves_backup(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.json"
    config = AppConfig(
        recent_files=(tmp_path / "session.wav",),
        audio_settings=stored_audio_settings(),
        main_window_position=WindowPosition(100, 200),
    )
    AppConfigStore(path).save(config)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["window_positions"]["main"] = {"x": "broken", "y": 200}
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = AppConfigStore(path).load_with_recovery()

    assert result.config.recent_files == config.recent_files
    assert result.config.audio_settings == config.audio_settings
    assert result.config.main_window_position is None
    assert result.recovered_sections == ("recent projects", "audio settings")
    assert result.reset_sections == ("window positions",)
    assert result.notice_required is True
    assert result.backup_path is not None and result.backup_path.exists()


def test_recent_files_are_normalized_deduplicated_and_limited(
    tmp_path: Path,
) -> None:
    config = AppConfig()
    paths = [tmp_path / f"{index}.wav" for index in range(MAX_RECENT_FILES + 2)]
    for path in paths:
        config = config.with_recent_file(path)
    config = config.with_recent_file(paths[5])

    assert config.recent_files[0] == paths[5].resolve()
    assert len(config.recent_files) == MAX_RECENT_FILES
    assert len(set(config.recent_files)) == MAX_RECENT_FILES

    config = config.without_recent_file(paths[5])
    assert paths[5].resolve() not in config.recent_files
    assert config.clear_recent_files().recent_files == ()


@pytest.mark.parametrize("schema_version", [1, 2, 3, 4, 99])
def test_unsupported_config_salvages_known_sections(
    tmp_path: Path, schema_version: int
) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": schema_version,
                "recent_files": [str(tmp_path / "session.wav")],
                "audio_settings": None,
                "window_positions": {"main": {"x": 10, "y": 20}},
            }
        ),
        encoding="utf-8",
    )

    result = AppConfigStore(path).load_with_recovery()

    assert result.recovered is True
    assert result.config.recent_files == ((tmp_path / "session.wav").resolve(),)
    assert result.config.main_window_position == WindowPosition(10, 20)
    assert result.backup_path is not None and result.backup_path.exists()
    assert result.notice_required is False
    assert json.loads(path.read_text())["schema_version"] == 5


@pytest.mark.parametrize(
    "contents",
    [
        "not json",
        "[]",
        '{"schema_version": true}',
        '{"schema_version": 99}',
    ],
)
def test_invalid_top_level_config_is_backed_up_and_reset(
    tmp_path: Path, contents: str
) -> None:
    path = tmp_path / "config.json"
    path.write_text(contents, encoding="utf-8")

    result = AppConfigStore(path).load_with_recovery()

    assert result.config == AppConfig()
    assert result.reset_sections
    assert result.backup_path is not None and result.backup_path.exists()
    assert json.loads(path.read_text())["schema_version"] == 5


def test_stored_audio_uses_stable_device_references() -> None:
    settings = AudioSettings(
        7, 9, 48_000, (0,) + (None,) * 7, (0, 1), 512
    )
    input_device = AudioDevice(7, "Input", "Core Audio", 8, 0, 48_000)
    output_device = AudioDevice(9, "Output", "Core Audio", 0, 2, 48_000)

    stored = StoredAudioSettings.from_settings(
        settings, input_device, output_device
    )

    assert stored.input_device == input_device.reference
    assert stored.output_device == output_device.reference
    assert stored.sample_rate == 48_000
    assert stored.track_inputs == settings.track_inputs
    assert stored.bus_outputs == settings.bus_outputs
    assert stored.buffer_size == 512


def test_stored_audio_resolves_current_runtime_device_ids() -> None:
    stored = stored_audio_settings()
    input_device = AudioDevice(12, "Studio Input", "Core Audio", 8, 0, 48_000)
    output_device = AudioDevice(19, "Studio Output", "Core Audio", 0, 4, 48_000)
    service = type(
        "FakeService",
        (),
        {
            "resolve_device": lambda self, reference, direction: (
                input_device if direction == "input" else output_device
            )
        },
    )()

    resolved = stored.resolve(service)

    assert resolved == AudioSettings(
        12,
        19,
        96_000,
        stored.track_inputs,
        stored.bus_outputs,
        stored.buffer_size,
    )
