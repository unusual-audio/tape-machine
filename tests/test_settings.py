"""Unit tests for routing-matrix interaction logic."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from tape_machine.audio import (
    AUDIO_BUFFER_SIZES,
    UNASSIGNED_BUS_OUTPUTS,
    UNASSIGNED_TRACK_INPUTS,
    AudioDevice,
    AudioSettings,
    StereoBusInput,
    TrackInputRoute,
)
from tape_machine.settings import (
    AudioSettingsDraft,
    AudioSettingsWindow,
    BufferSizeChoice,
    DeviceChoice,
    device_channel_labels,
    input_source_rows,
)


@dataclass
class FakeSwitch:
    value: bool


def routing_window(
    track_inputs: tuple[TrackInputRoute, ...],
) -> tuple[AudioSettingsWindow, list[int | None]]:
    window = AudioSettingsWindow.__new__(AudioSettingsWindow)
    window._updating = False
    window.track_inputs = track_inputs
    window.route_switches = {}
    rate_updates: list[int | None] = []
    window._selected_rate = lambda: 48_000
    window._refresh_save_state_after_routing_change = lambda: (
        rate_updates.append(window._selected_rate())
    )
    return window, rate_updates


def output_routing_window(
    bus_outputs: tuple[int | None, ...],
) -> tuple[AudioSettingsWindow, list[int | None]]:
    window = AudioSettingsWindow.__new__(AudioSettingsWindow)
    window._updating = False
    window.bus_outputs = bus_outputs
    window.output_route_switches = {}
    rate_updates: list[int | None] = []
    window._selected_rate = lambda: 48_000
    window._refresh_save_state_after_routing_change = lambda: (
        rate_updates.append(window._selected_rate())
    )
    return window, rate_updates


def test_selecting_another_input_replaces_the_track_source() -> None:
    window, rate_updates = routing_window(
        (0, None, None, None, None, None, None, None)
    )
    previous = FakeSwitch(True)
    selected = FakeSwitch(True)
    window.route_switches = {(0, 0): previous, (1, 0): selected}

    window._on_route_changed(selected, input_channel=1, track_index=0)

    assert window.track_inputs == (1, None, None, None, None, None, None, None)
    assert previous.value is False
    assert rate_updates == [48_000]


def test_one_input_can_feed_multiple_tracks() -> None:
    window, _ = routing_window(UNASSIGNED_TRACK_INPUTS)
    first_track = FakeSwitch(True)
    second_track = FakeSwitch(True)
    window.route_switches = {(0, 0): first_track, (0, 1): second_track}

    window._on_route_changed(first_track, input_channel=0, track_index=0)
    window._on_route_changed(second_track, input_channel=0, track_index=1)

    assert window.track_inputs == (0, 0, None, None, None, None, None, None)


def test_unchecking_a_route_leaves_the_track_unassigned() -> None:
    window, _ = routing_window(
        (0, None, None, None, None, None, None, None)
    )
    route = FakeSwitch(False)
    window.route_switches = {(0, 0): route}

    window._on_route_changed(route, input_channel=0, track_index=0)

    assert window.track_inputs == UNASSIGNED_TRACK_INPUTS


def test_stereo_bus_source_behaves_like_an_ordinary_route() -> None:
    window, rate_updates = routing_window(
        (0, None, None, None, None, None, None, None)
    )
    physical = FakeSwitch(True)
    loopback = FakeSwitch(True)
    window.route_switches = {
        (0, 0): physical,
        (StereoBusInput.LEFT, 0): loopback,
    }

    window._on_route_changed(
        loopback, input_channel=StereoBusInput.LEFT, track_index=0
    )

    assert window.track_inputs == (StereoBusInput.LEFT,) + (None,) * 7
    assert physical.value is False
    assert rate_updates == [48_000]


def test_input_matrix_rows_end_with_stereo_bus_loopbacks() -> None:
    input_device = AudioDevice(1, "Input", "Core Audio", 2, 0, 48_000)

    rows = input_source_rows(input_device, UNASSIGNED_TRACK_INPUTS)

    assert rows == (
        (0, "Input 1"),
        (1, "Input 2"),
        (StereoBusInput.LEFT, "Stereo bus L"),
        (StereoBusInput.RIGHT, "Stereo bus R"),
    )


def test_channel_labels_use_names_and_disambiguate_duplicates() -> None:
    device = AudioDevice(
        1,
        "Interface",
        "Core Audio",
        4,
        3,
        48_000,
        input_channel_names=(" Mic ", "mic", "Instrument", None),
        output_channel_names=("Monitor", None, "Phones"),
    )

    assert device_channel_labels(device, "input") == (
        "Mic (Input 1)",
        "mic (Input 2)",
        "Instrument",
        "Input 4",
    )
    assert device_channel_labels(device, "output") == (
        "Monitor",
        "Output 2",
        "Phones",
    )


def test_channel_labels_mark_saved_routes_beyond_device_as_unavailable() -> None:
    device = AudioDevice(
        1,
        "Interface",
        "Core Audio",
        1,
        1,
        48_000,
        input_channel_names=("Mic",),
        output_channel_names=("Monitor",),
    )

    assert device_channel_labels(device, "input", 3) == (
        "Mic",
        "Input 2 (unavailable)",
        "Input 3 (unavailable)",
    )
    assert device_channel_labels(device, "output", 2) == (
        "Monitor",
        "Output 2 (unavailable)",
    )


def test_input_rows_use_native_names_before_stereo_bus_loopbacks() -> None:
    input_device = AudioDevice(
        1,
        "Input",
        "Core Audio",
        2,
        0,
        48_000,
        input_channel_names=("Mic", "Instrument"),
    )

    rows = input_source_rows(input_device, UNASSIGNED_TRACK_INPUTS)

    assert rows == (
        (0, "Mic"),
        (1, "Instrument"),
        (StereoBusInput.LEFT, "Stereo bus L"),
        (StereoBusInput.RIGHT, "Stereo bus R"),
    )


def test_selecting_another_output_replaces_the_bus_side() -> None:
    window, rate_updates = output_routing_window((0, None))
    previous = FakeSwitch(True)
    selected = FakeSwitch(True)
    window.output_route_switches = {(0, 0): previous, (2, 0): selected}

    window._on_output_route_changed(selected, output_channel=2, bus_index=0)

    assert window.bus_outputs == (2, None)
    assert previous.value is False
    assert rate_updates == [48_000]


def test_physical_output_moves_between_stereo_sides() -> None:
    window, _ = output_routing_window((0, None))
    left = FakeSwitch(True)
    right = FakeSwitch(True)
    window.output_route_switches = {(0, 0): left, (0, 1): right}

    window._on_output_route_changed(right, output_channel=0, bus_index=1)

    assert window.bus_outputs == (None, 0)
    assert left.value is False


def test_unchecking_output_leaves_bus_side_unassigned() -> None:
    window, _ = output_routing_window((0, 1))
    route = FakeSwitch(False)
    window.output_route_switches = {(1, 1): route}

    window._on_output_route_changed(route, output_channel=1, bus_index=1)

    assert window.bus_outputs == (0, None)


def test_stereo_output_can_start_fully_unassigned() -> None:
    window, _ = output_routing_window(UNASSIGNED_BUS_OUTPUTS)

    assert window.bus_outputs == (None, None)


def test_changing_output_device_preserves_portable_stereo_routing() -> None:
    window = AudioSettingsWindow.__new__(AudioSettingsWindow)
    window._updating = False
    window.input_selection = object()
    window.output_selection = object()
    window.track_inputs = (0, None, None, None, None, None, None, None)
    window.bus_outputs = (0, 1)
    input_device = object()
    output_device = object()
    window._selected_input_device = lambda: input_device
    window._selected_output_device = lambda: output_device
    window._selected_rate = lambda: 48_000
    rebuilt: list[tuple[object, ...]] = []
    window._rebuild_routing_matrices = lambda *args: rebuilt.append(args)
    rate_updates: list[int | None] = []
    window._update_sample_rates = rate_updates.append

    window._on_device_changed(window.output_selection)

    assert rebuilt == [
        (
            input_device,
            output_device,
            window.track_inputs,
            window.bus_outputs,
        )
    ]
    assert rate_updates == [48_000]


def test_locked_rate_does_not_probe_an_edited_active_configuration() -> None:
    input_device = AudioDevice(1, "Interface", "Core Audio", 8, 2, 48_000)
    output_device = AudioDevice(1, "Interface", "Core Audio", 8, 2, 48_000)
    settings = AudioSettings(
        1, 1, 48_000, (0,) + (None,) * 7, (0, 1), 256
    )
    compatibility_calls: list[AudioSettings] = []
    window = AudioSettingsWindow.__new__(AudioSettingsWindow)
    window._updating = False
    window.locked_sample_rate = 48_000
    window.track_inputs = (1,) + (None,) * 7
    window.bus_outputs = settings.bus_outputs
    window.input_selection = SimpleNamespace(
        value=DeviceChoice(input_device, "input")
    )
    window.output_selection = SimpleNamespace(
        value=DeviceChoice(output_device, "output")
    )
    window.sample_rate_selection = SimpleNamespace(
        items=[], value=None, enabled=True
    )
    window.buffer_size_selection = SimpleNamespace(
        value=BufferSizeChoice(256)
    )
    window.save_button = SimpleNamespace(enabled=False)
    window.status_label = SimpleNamespace(text="")
    window.service = SimpleNamespace(
        compatibility_error=lambda candidate: compatibility_calls.append(
            candidate
        )
    )

    window._update_sample_rates()

    assert compatibility_calls == []
    assert window.save_button.enabled is True
    assert window.status_label.text == (
        "Project sample rate is fixed by the WAV file."
    )


@pytest.mark.parametrize(
    ("locked_rate", "message"),
    [
        (None, ""),
        (48_000, "Project sample rate is fixed by the WAV file."),
    ],
)
def test_route_click_only_updates_draft_save_state(
    locked_rate: int | None,
    message: str,
) -> None:
    input_device = AudioDevice(1, "Input", "Core Audio", 8, 0, 48_000)
    output_device = AudioDevice(2, "Output", "Core Audio", 0, 2, 48_000)
    window = AudioSettingsWindow.__new__(AudioSettingsWindow)
    window.locked_sample_rate = locked_rate
    window.input_selection = SimpleNamespace(
        value=DeviceChoice(input_device, "input")
    )
    window.output_selection = SimpleNamespace(
        value=DeviceChoice(output_device, "output")
    )
    window.track_inputs = (0,) + (None,) * 7
    window.bus_outputs = (0, 1)
    window._selected_rate = lambda: 48_000
    window.buffer_size_selection = SimpleNamespace(
        value=BufferSizeChoice(256)
    )
    window.save_button = SimpleNamespace(enabled=False)
    window.status_label = SimpleNamespace(text="old status")
    window.service = SimpleNamespace(
        supported_sample_rates=lambda *args: pytest.fail(
            "route click must not probe Core Audio"
        ),
        compatibility_error=lambda *args: pytest.fail(
            "route click must not validate Core Audio"
        ),
    )

    window._refresh_save_state_after_routing_change()

    assert window.save_button.enabled is True
    assert window.status_label.text == message


def test_save_applies_settings_and_closes() -> None:
    input_device = AudioDevice(1, "Input", "Core Audio", 8, 0, 48_000)
    output_device = AudioDevice(2, "Output", "Core Audio", 0, 2, 48_000)
    settings = AudioSettings(
        1, 2, 48_000, (0,) + (None,) * 7, (0, 1), 512
    )
    events: list[AudioSettings] = []
    hides: list[None] = []
    window = AudioSettingsWindow.__new__(AudioSettingsWindow)
    window.input_selection = SimpleNamespace(
        value=DeviceChoice(input_device, "input")
    )
    window.output_selection = SimpleNamespace(
        value=DeviceChoice(output_device, "output")
    )
    window.track_inputs = settings.track_inputs
    window.bus_outputs = settings.bus_outputs
    window._selected_rate = lambda: settings.sample_rate
    window.buffer_size_selection = SimpleNamespace(
        value=BufferSizeChoice(512)
    )
    window.save_button = SimpleNamespace(enabled=True)
    window.status_label = SimpleNamespace(text="")
    window.on_applied = events.append
    window.window = SimpleNamespace(hide=lambda: hides.append(None))

    window._save(SimpleNamespace())

    assert events == [settings]
    assert hides == [None]


def test_buffer_size_choices_show_samples_and_automatic() -> None:
    assert str(BufferSizeChoice(0)) == "Automatic"
    assert str(BufferSizeChoice(128)) == "128 samples"


@pytest.mark.parametrize("buffer_size", AUDIO_BUFFER_SIZES)
def test_buffer_size_selection_round_trips_standard_choices(
    buffer_size: int,
) -> None:
    window = AudioSettingsWindow.__new__(AudioSettingsWindow)
    window.buffer_size_selection = SimpleNamespace(
        value=BufferSizeChoice(buffer_size)
    )

    assert window._selected_buffer_size() == buffer_size
    assert AudioSettingsDraft.from_settings(
        AudioSettings(1, 2, 48_000, buffer_size=buffer_size)
    ).buffer_size == buffer_size


def test_open_leaves_an_already_visible_draft_intact() -> None:
    window = AudioSettingsWindow.__new__(AudioSettingsWindow)
    window.window = SimpleNamespace(visible=True)

    window.open()

    assert window.window.visible is True
