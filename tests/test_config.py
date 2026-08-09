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
    AppConfigError,
    AppConfigStore,
    StoredAudioSettings,
    WindowPosition,
)


def stored_audio_settings() -> StoredAudioSettings:
    return StoredAudioSettings(
        input_device=DeviceReference("Studio Input", "Core Audio"),
        output_device=DeviceReference("Studio Output", "Core Audio"),
        sample_rate=96_000,
        track_inputs=(7, 6, 5, 4, 3, 2, 1, 0),
        bus_outputs=(2, 3),
    )


def test_config_round_trips_all_sections(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    config = AppConfig(
        recent_files=(tmp_path / "one.wav", tmp_path / "two.wav"),
        default_audio_settings=stored_audio_settings(),
        main_window_position=WindowPosition(120, 80),
        audio_settings_window_position=WindowPosition(-800, 140),
    )

    AppConfigStore(path).save(config)

    assert AppConfigStore(path).load() == config
    assert not list(tmp_path.glob(".config.json.*.tmp"))


def test_config_round_trips_stereo_bus_input_defaults(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    stored = stored_audio_settings()
    stored = StoredAudioSettings(
        stored.input_device,
        stored.output_device,
        stored.sample_rate,
        (StereoBusInput.LEFT, StereoBusInput.RIGHT) + (None,) * 6,
        stored.bus_outputs,
    )

    AppConfigStore(path).save(AppConfig(default_audio_settings=stored))

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["default_audio_settings"]["track_inputs"][:2] == [
        "stereo_bus_l",
        "stereo_bus_r",
    ]
    assert AppConfigStore(path).load().default_audio_settings == stored


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


def test_invalid_sections_are_discarded_without_losing_valid_sections(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recent_files": [str(tmp_path / "valid.wav"), 42],
                "default_audio_settings": {"sample_rate": "fast"},
                "window_positions": {
                    "main": {"x": 20, "y": 40},
                    "audio_settings": {"x": True, "y": 10},
                },
                "future_field": "ignored",
            }
        ),
        encoding="utf-8",
    )

    config = AppConfigStore(path).load()

    assert config.recent_files == ((tmp_path / "valid.wav").resolve(),)
    assert config.default_audio_settings is None
    assert config.main_window_position == WindowPosition(20, 40)
    assert config.audio_settings_window_position is None


def test_schema_one_config_with_numeric_routes_still_loads(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.json"
    stored = stored_audio_settings()
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "recent_files": [],
                "default_audio_settings": {
                    "input_device": {
                        "name": stored.input_device.name,
                        "host_api": stored.input_device.host_api,
                    },
                    "output_device": {
                        "name": stored.output_device.name,
                        "host_api": stored.output_device.host_api,
                    },
                    "sample_rate": stored.sample_rate,
                    "track_inputs": list(stored.track_inputs),
                    "bus_outputs": list(stored.bus_outputs),
                },
                "window_positions": {},
            }
        ),
        encoding="utf-8",
    )

    assert AppConfigStore(path).load().default_audio_settings == stored


@pytest.mark.parametrize(
    "contents",
    [
        "not json",
        "[]",
        '{"schema_version": true}',
        '{"schema_version": 99}',
    ],
)
def test_invalid_top_level_config_raises_a_readable_error(
    tmp_path: Path, contents: str
) -> None:
    path = tmp_path / "config.json"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(AppConfigError, match="configuration"):
        AppConfigStore(path).load()


def test_stored_audio_uses_stable_device_references() -> None:
    settings = AudioSettings(7, 9, 48_000, (0,) + (None,) * 7, (0, 1))
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
    )
