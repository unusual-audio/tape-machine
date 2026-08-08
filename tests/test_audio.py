"""Tests for audio device discovery and configuration."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import soundfile

from tape_machine.audio import (
    SAMPLE_RATES,
    AudioConfigurationError,
    AudioDeviceService,
    AudioSettings,
)


def test_soundfile_runtime_is_available() -> None:
    """Keep the file-writing dependency in the tested application environment."""
    assert callable(soundfile.SoundFile)


class UnsupportedSettingsError(RuntimeError):
    pass


@dataclass
class FakeDefaults:
    device: tuple[int | None, int | None] = (0, 1)


class FakeSoundDevice:
    def __init__(self) -> None:
        self.default = FakeDefaults()
        self.devices = [
            {
                "name": "Studio Input",
                "index": 0,
                "hostapi": 0,
                "max_input_channels": 8,
                "max_output_channels": 0,
                "default_samplerate": 48_000.0,
            },
            {
                "name": "Studio Output",
                "index": 1,
                "hostapi": 0,
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48_000.0,
            },
            {
                "name": "USB Interface",
                "index": 2,
                "hostapi": 0,
                "max_input_channels": 4,
                "max_output_channels": 4,
                "default_samplerate": 44_100.0,
            },
        ]
        self.host_apis = ({"name": "Core Audio"},)
        self.input_rates = {0: {44_100, 48_000, 96_000}, 2: set(SAMPLE_RATES)}
        self.output_rates = {1: {44_100, 48_000}, 2: set(SAMPLE_RATES)}
        self.fail_query = False
        self.check_calls: list[tuple[str, dict[str, object]]] = []

    def query_devices(self):
        if self.fail_query:
            raise RuntimeError("PortAudio is unavailable")
        return list(self.devices)

    def query_hostapis(self):
        return self.host_apis

    def check_input_settings(self, **kwargs):
        self.check_calls.append(("input", kwargs))
        if kwargs["samplerate"] not in self.input_rates.get(kwargs["device"], set()):
            raise UnsupportedSettingsError("unsupported input rate")

    def check_output_settings(self, **kwargs):
        self.check_calls.append(("output", kwargs))
        if kwargs["samplerate"] not in self.output_rates.get(
            kwargs["device"], set()
        ):
            raise UnsupportedSettingsError("unsupported output rate")


@pytest.fixture
def backend() -> FakeSoundDevice:
    return FakeSoundDevice()


def test_initialize_filters_devices_and_uses_defaults(backend: FakeSoundDevice) -> None:
    service = AudioDeviceService(backend)

    settings = service.initialize()

    assert [device.index for device in service.input_devices] == [0, 2]
    assert [device.index for device in service.output_devices] == [1, 2]
    assert settings == AudioSettings(0, 1, 48_000)
    assert "Core Audio" in service.input_devices[0].label("input")
    assert "8 channels" in service.input_devices[0].label("input")


def test_missing_system_defaults_fall_back_to_first_devices(
    backend: FakeSoundDevice,
) -> None:
    backend.default.device = (-1, 99)
    service = AudioDeviceService(backend)

    service.refresh_devices()

    assert service.default_input_device_id == 0
    assert service.default_output_device_id == 1


def test_supported_rates_are_intersection_and_use_probe_format(
    backend: FakeSoundDevice,
) -> None:
    service = AudioDeviceService(backend)
    service.refresh_devices()

    rates = service.supported_sample_rates(0, 1)

    assert rates == (44_100, 48_000)
    assert backend.check_calls[0] == (
        "input",
        {
            "device": 0,
            "channels": 1,
            "dtype": "float32",
            "samplerate": 44_100,
        },
    )


def test_suggestion_preserves_supported_preference(backend: FakeSoundDevice) -> None:
    service = AudioDeviceService(backend)
    service.refresh_devices()

    suggested = service.suggest_settings(AudioSettings(2, 2, 96_000))

    assert suggested == AudioSettings(2, 2, 96_000)


def test_suggestion_falls_back_to_48khz_then_lowest(
    backend: FakeSoundDevice,
) -> None:
    service = AudioDeviceService(backend)
    service.refresh_devices()

    assert service.suggest_settings(AudioSettings(0, 1, 96_000)) == AudioSettings(
        0, 1, 48_000
    )

    backend.input_rates[0] = {44_100}
    backend.output_rates[1] = {44_100}
    assert service.suggest_settings() == AudioSettings(0, 1, 44_100)


def test_apply_revalidates_after_device_disappears(
    backend: FakeSoundDevice,
) -> None:
    service = AudioDeviceService(backend)
    service.initialize()
    original = service.current_settings
    backend.devices = [device for device in backend.devices if device["index"] != 1]

    with pytest.raises(AudioConfigurationError, match="output device 1"):
        service.apply(AudioSettings(0, 1, 48_000))

    assert service.current_settings == original


def test_apply_keeps_previous_settings_when_rate_is_rejected(
    backend: FakeSoundDevice,
) -> None:
    service = AudioDeviceService(backend)
    service.initialize()
    original = service.current_settings

    with pytest.raises(AudioConfigurationError, match="no longer support"):
        service.apply(AudioSettings(0, 1, 96_000))

    assert service.current_settings == original


def test_empty_or_incompatible_inventory_has_no_suggestion(
    backend: FakeSoundDevice,
) -> None:
    service = AudioDeviceService(backend)
    backend.output_rates[1] = set()

    assert service.initialize() is None

    backend.devices = [backend.devices[0]]
    service.refresh_devices()
    assert service.suggest_settings() is None


def test_enumeration_failure_is_descriptive(backend: FakeSoundDevice) -> None:
    service = AudioDeviceService(backend)
    backend.fail_query = True

    with pytest.raises(AudioConfigurationError, match="PortAudio is unavailable"):
        service.refresh_devices()
