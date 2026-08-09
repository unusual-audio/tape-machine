"""Unit tests for routing-matrix interaction logic."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from tape_machine.audio import (
    UNASSIGNED_BUS_OUTPUTS,
    UNASSIGNED_TRACK_INPUTS,
    AudioDevice,
    AudioSettings,
)
from tape_machine.settings import AudioSettingsWindow, DeviceChoice


@dataclass
class FakeSwitch:
    value: bool


def routing_window(
    track_inputs: tuple[int | None, ...],
) -> tuple[AudioSettingsWindow, list[int | None]]:
    window = AudioSettingsWindow.__new__(AudioSettingsWindow)
    window._updating = False
    window.track_inputs = track_inputs
    window.route_switches = {}
    rate_updates: list[int | None] = []
    window._selected_rate = lambda: 48_000
    window._update_sample_rates = rate_updates.append
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
    window._update_sample_rates = rate_updates.append
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


def test_locked_rate_does_not_reprobe_the_active_configuration() -> None:
    input_device = AudioDevice(1, "Interface", "Core Audio", 8, 2, 48_000)
    output_device = AudioDevice(1, "Interface", "Core Audio", 8, 2, 48_000)
    settings = AudioSettings(1, 1, 48_000, (0,) + (None,) * 7, (0, 1))
    compatibility_calls: list[AudioSettings] = []
    window = AudioSettingsWindow.__new__(AudioSettingsWindow)
    window._updating = False
    window.locked_sample_rate = 48_000
    window.trusted_settings = settings
    window.track_inputs = settings.track_inputs
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
