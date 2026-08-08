"""Tests for audio device discovery and configuration."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import soundfile

from tape_machine.audio import (
    PROJECT_TRACK_COUNT,
    SAMPLE_RATES,
    UNASSIGNED_BUS_OUTPUTS,
    UNASSIGNED_TRACK_INPUTS,
    AudioConfigurationError,
    DeviceReference,
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
    assert settings.track_inputs == UNASSIGNED_TRACK_INPUTS
    assert settings.bus_outputs == UNASSIGNED_BUS_OUTPUTS
    assert settings.required_input_channels == 0
    assert settings.required_output_channels == 0
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


def test_device_reference_resolves_exact_match_then_default(
    backend: FakeSoundDevice,
) -> None:
    service = AudioDeviceService(backend)
    service.refresh_devices()

    assert service.resolve_device(
        DeviceReference("USB Interface", "Core Audio"), "input"
    ).index == 2
    assert service.resolve_device(
        DeviceReference("Missing", "Core Audio"), "input"
    ).index == 0


def test_supported_rates_are_intersection_and_use_probe_format(
    backend: FakeSoundDevice,
) -> None:
    service = AudioDeviceService(backend)
    service.refresh_devices()

    track_inputs = (0,) + (None,) * (PROJECT_TRACK_COUNT - 1)
    rates = service.supported_sample_rates(0, 1, track_inputs)

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
    track_inputs = (3, 3, None, None, None, None, None, None)

    suggested = service.suggest_settings(
        AudioSettings(2, 2, 96_000, track_inputs, (2, 3))
    )

    assert suggested == AudioSettings(2, 2, 96_000, track_inputs, (2, 3))


def test_suggestion_preserves_routing_when_input_device_changes(
    backend: FakeSoundDevice,
) -> None:
    service = AudioDeviceService(backend)
    service.refresh_devices()
    backend.default.device = (0, 1)
    preferred = AudioSettings(
        99,
        1,
        48_000,
        (1, 1, None, None, None, None, None, None),
    )

    suggested = service.suggest_settings(preferred)

    assert suggested == AudioSettings(
        0,
        1,
        48_000,
        (1, 1, None, None, None, None, None, None),
    )


def test_suggestion_preserves_output_routing_when_output_device_changes(
    backend: FakeSoundDevice,
) -> None:
    service = AudioDeviceService(backend)
    service.refresh_devices()
    preferred = AudioSettings(
        0,
        99,
        48_000,
        UNASSIGNED_TRACK_INPUTS,
        (0, 1),
    )

    suggested = service.suggest_settings(preferred)

    assert suggested == AudioSettings(
        0, 1, 48_000, UNASSIGNED_TRACK_INPUTS, (0, 1)
    )


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


def test_project_sample_rate_is_not_limited_to_studio_rate_list(
    backend: FakeSoundDevice,
) -> None:
    backend.input_rates[0].add(32_000)
    backend.output_rates[1].add(32_000)
    service = AudioDeviceService(backend)
    service.refresh_devices()

    service.validate(AudioSettings(0, 1, 32_000))


def test_repeated_input_can_feed_multiple_tracks(backend: FakeSoundDevice) -> None:
    service = AudioDeviceService(backend)
    service.refresh_devices()
    settings = AudioSettings(
        0,
        1,
        48_000,
        (2, 2, None, None, None, None, None, None),
    )

    service.validate(settings)

    assert settings.required_input_channels == 3
    input_call = next(call for call in backend.check_calls if call[0] == "input")
    assert input_call[1]["channels"] == 3


def test_unassigned_routing_skips_input_capability_probe(
    backend: FakeSoundDevice,
) -> None:
    service = AudioDeviceService(backend)
    service.refresh_devices()

    rates = service.supported_sample_rates(
        0, 1, UNASSIGNED_TRACK_INPUTS
    )

    assert rates == (44_100, 48_000)
    assert {kind for kind, kwargs in backend.check_calls} == {"output"}
    assert all(
        kwargs["channels"] == 1
        for kind, kwargs in backend.check_calls
        if kind == "output"
    )


def test_output_probe_reaches_highest_mapped_channel(
    backend: FakeSoundDevice,
) -> None:
    service = AudioDeviceService(backend)
    service.refresh_devices()

    rates = service.supported_sample_rates(
        2,
        2,
        UNASSIGNED_TRACK_INPUTS,
        (1, 3),
    )

    assert rates == SAMPLE_RATES
    assert all(
        kwargs["channels"] == 4
        for kind, kwargs in backend.check_calls
        if kind == "output"
    )


def test_partial_stereo_output_routing_is_valid(
    backend: FakeSoundDevice,
) -> None:
    service = AudioDeviceService(backend)
    service.refresh_devices()
    settings = AudioSettings(
        0,
        1,
        48_000,
        UNASSIGNED_TRACK_INPUTS,
        (1, None),
    )

    service.validate(settings)

    assert settings.required_output_channels == 2
    output_call = next(call for call in backend.check_calls if call[0] == "output")
    assert output_call[1]["channels"] == 2


@pytest.mark.parametrize(
    ("track_inputs", "message"),
    [
        ((None,) * 7, "exactly 8 tracks"),
        ((-1,) + (None,) * 7, "invalid input channel"),
        ((True,) + (None,) * 7, "invalid input channel"),
    ],
)
def test_invalid_track_routing_is_rejected(
    backend: FakeSoundDevice,
    track_inputs: tuple[int | None, ...],
    message: str,
) -> None:
    service = AudioDeviceService(backend)
    service.refresh_devices()

    with pytest.raises(AudioConfigurationError, match=message):
        service.validate(AudioSettings(0, 1, 48_000, track_inputs))


def test_unavailable_input_mapping_routes_into_the_void(
    backend: FakeSoundDevice,
) -> None:
    service = AudioDeviceService(backend)
    service.refresh_devices()
    settings = AudioSettings(
        0,
        1,
        48_000,
        (8, None, None, None, None, None, None, None),
    )

    service.validate(settings)

    assert {kind for kind, kwargs in backend.check_calls} == {"output"}


@pytest.mark.parametrize(
    ("bus_outputs", "message"),
    [
        ((None,), "exactly 2 bus channels"),
        ((-1, None), "invalid output channel"),
        ((True, None), "invalid output channel"),
        ((0, 0), "distinct output channels"),
    ],
)
def test_invalid_stereo_output_routing_is_rejected(
    backend: FakeSoundDevice,
    bus_outputs: tuple[int | None, ...],
    message: str,
) -> None:
    service = AudioDeviceService(backend)
    service.refresh_devices()

    with pytest.raises(AudioConfigurationError, match=message):
        service.validate(
            AudioSettings(
                0,
                1,
                48_000,
                UNASSIGNED_TRACK_INPUTS,
                bus_outputs,
            )
        )


def test_unavailable_output_mapping_uses_minimal_device_probe(
    backend: FakeSoundDevice,
) -> None:
    service = AudioDeviceService(backend)
    service.refresh_devices()

    service.validate(
        AudioSettings(
            0,
            1,
            48_000,
            UNASSIGNED_TRACK_INPUTS,
            (2, None),
        )
    )

    output_call = next(call for call in backend.check_calls if call[0] == "output")
    assert output_call[1]["channels"] == 1


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
