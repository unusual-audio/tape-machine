"""Audio settings window for Tape Machine."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable

import toga
from toga.style.pack import CENTER, COLUMN, END, ROW

from tape_machine.audio import (
    AUDIO_BUFFER_SIZES,
    PROJECT_TRACK_COUNT,
    STEREO_BUS_CHANNEL_COUNT,
    UNASSIGNED_BUS_OUTPUTS,
    UNASSIGNED_TRACK_INPUTS,
    AudioConfigurationError,
    AudioDevice,
    AudioDeviceService,
    AudioSettings,
    StereoBusInput,
    TrackInputRoute,
    is_physical_input,
)


_ROUTING_LABEL_WIDTH = 240


@dataclass(frozen=True, slots=True)
class DeviceChoice:
    """Display wrapper used by Toga's selection widget."""

    device: AudioDevice
    direction: str

    def __str__(self) -> str:
        return self.device.label(self.direction)


@dataclass(frozen=True, slots=True)
class SampleRateChoice:
    """Human-readable sample-rate selection value."""

    value: int

    def __str__(self) -> str:
        if self.value % 1_000 == 0:
            return f"{self.value // 1_000} kHz"
        return f"{self.value / 1_000:g} kHz"


@dataclass(frozen=True, slots=True)
class BufferSizeChoice:
    """Human-readable audio callback buffer size."""

    value: int

    def __str__(self) -> str:
        if self.value == 0:
            return "Automatic"
        return f"{self.value} samples"


def device_channel_labels(
    device: AudioDevice,
    direction: str,
    row_count: int | None = None,
) -> tuple[str, ...]:
    """Build readable, unambiguous labels for physical device channels."""
    if direction == "input":
        channel_count = device.max_input_channels
        channel_names = device.input_channel_names
        prefix = "Input"
    else:
        channel_count = device.max_output_channels
        channel_names = device.output_channel_names
        prefix = "Output"

    normalized_names = tuple(
        name.strip() if isinstance(name, str) and name.strip() else None
        for name in channel_names[:channel_count]
    )
    duplicate_counts = Counter(
        name.casefold() for name in normalized_names if name is not None
    )
    labels: list[str] = []
    for channel_index in range(
        channel_count if row_count is None else row_count
    ):
        if channel_index >= channel_count:
            labels.append(f"{prefix} {channel_index + 1} (unavailable)")
            continue
        name = (
            normalized_names[channel_index]
            if channel_index < len(normalized_names)
            else None
        )
        if name is None:
            labels.append(f"{prefix} {channel_index + 1}")
        elif duplicate_counts[name.casefold()] > 1:
            labels.append(f"{name} ({prefix} {channel_index + 1})")
        else:
            labels.append(name)
    return tuple(labels)


def input_source_rows(
    input_device: AudioDevice,
    track_inputs: tuple[TrackInputRoute, ...],
) -> tuple[tuple[TrackInputRoute, str], ...]:
    """Return physical and virtual rows shown by the input matrix."""
    stored_input_count = max(
        (
            channel
            for channel in track_inputs
            if is_physical_input(channel)
        ),
        default=-1,
    ) + 1
    row_count = max(input_device.max_input_channels, stored_input_count)
    physical_labels = device_channel_labels(
        input_device, "input", row_count
    )
    rows: list[tuple[TrackInputRoute, str]] = [
        (input_channel, physical_labels[input_channel])
        for input_channel in range(row_count)
    ]
    rows.extend(
        (
            (StereoBusInput.LEFT, "Stereo bus L"),
            (StereoBusInput.RIGHT, "Stereo bus R"),
        )
    )
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class AudioSettingsDraft:
    """Editable audio state, including projects without resolved devices."""

    input_device_id: int | None
    output_device_id: int | None
    sample_rate: int | None
    track_inputs: tuple[TrackInputRoute, ...] = UNASSIGNED_TRACK_INPUTS
    bus_outputs: tuple[int | None, ...] = UNASSIGNED_BUS_OUTPUTS
    buffer_size: int = 0

    @classmethod
    def from_settings(cls, settings: AudioSettings | None) -> AudioSettingsDraft:
        if settings is None:
            return cls(None, None, None)
        return cls(
            settings.input_device_id,
            settings.output_device_id,
            settings.sample_rate,
            settings.track_inputs,
            settings.bus_outputs,
            settings.buffer_size,
        )


class AudioSettingsWindow:
    """A reusable settings window that edits a draft audio configuration."""

    def __init__(
        self,
        service: AudioDeviceService,
        on_applied: Callable[[AudioSettings], None],
        *,
        position: tuple[int, int] | None = None,
    ) -> None:
        self.service = service
        self.default_on_applied = on_applied
        self.on_applied = on_applied
        self.draft = AudioSettingsDraft.from_settings(None)
        self.locked_sample_rate: int | None = None
        self.trusted_settings: AudioSettings | None = None
        self._updating = False
        self.track_inputs = UNASSIGNED_TRACK_INPUTS
        self.bus_outputs = UNASSIGNED_BUS_OUTPUTS
        self.route_switches: dict[
            tuple[TrackInputRoute, int], toga.Switch
        ] = {}
        self.output_route_switches: dict[tuple[int, int], toga.Switch] = {}

        self.input_selection = toga.Selection(
            items=[], on_change=self._on_device_changed, flex=1
        )
        self.output_selection = toga.Selection(
            items=[], on_change=self._on_device_changed, flex=1
        )
        self.sample_rate_selection = toga.Selection(items=[], flex=1)
        self.buffer_size_selection = toga.Selection(
            items=[BufferSizeChoice(size) for size in AUDIO_BUFFER_SIZES],
            flex=1,
        )
        self.buffer_size_selection.value = BufferSizeChoice(0)
        self.status_label = toga.Label("", margin_top=12)
        self.routing_scroll = toga.ScrollContainer(
            horizontal=False,
            vertical=True,
            flex=1,
            margin_top=8,
        )
        self.save_button = toga.Button(
            "Save", on_press=self._save, enabled=False, margin_left=8
        )
        self.button_row = toga.Box(
            children=[
                toga.Button("Cancel", on_press=self._cancel),
                self.save_button,
            ],
            direction=ROW,
            justify_content=END,
            margin_top=20,
        )

        content = toga.Box(
            children=[
                toga.Label(
                    "Audio",
                    font_size=20,
                    font_weight="bold",
                    margin_bottom=16,
                ),
                self._setting_row("Input device", self.input_selection),
                self._setting_row("Output device", self.output_selection),
                self._setting_row("Sample rate", self.sample_rate_selection),
                self._setting_row("Buffer size", self.buffer_size_selection),
                toga.Divider(margin_top=8, margin_bottom=16),
                self.routing_scroll,
                self.status_label,
                self.button_row,
            ],
            direction=COLUMN,
            margin=24,
        )

        self.window = toga.Window(
            title="Audio Settings",
            position=position,
            size=(900, 640),
            resizable=True,
            minimizable=False,
            content=content,
            on_close=self._on_close,
        )

    @staticmethod
    def _setting_row(label: str, control: toga.Widget) -> toga.Box:
        return toga.Box(
            children=[
                toga.Label(label, width=110, margin_top=5),
                control,
            ],
            direction=ROW,
            margin_bottom=12,
        )

    def open(
        self,
        draft: AudioSettingsDraft | None = None,
        *,
        locked_sample_rate: int | None = None,
        trusted_settings: AudioSettings | None = None,
        on_applied: Callable[[AudioSettings], None] | None = None,
    ) -> None:
        """Refresh and show the window, or leave an already-visible draft intact."""
        if self.window.visible:
            return
        self.draft = draft or AudioSettingsDraft.from_settings(
            self.service.current_settings
        )
        self.locked_sample_rate = locked_sample_rate
        self.trusted_settings = trusted_settings
        self.on_applied = on_applied or self.default_on_applied
        self._load_draft()
        self.window.show()

    def _set_save_enabled(self, enabled: bool) -> None:
        self.save_button.enabled = enabled

    def _load_draft(self) -> None:
        self._updating = True
        self.status_label.text = ""
        self._set_save_enabled(False)
        try:
            inputs, outputs = self.service.refresh_devices()
            self.input_selection.items = [
                DeviceChoice(device, "input") for device in inputs
            ]
            self.output_selection.items = [
                DeviceChoice(device, "output") for device in outputs
            ]
            self.input_selection.enabled = bool(inputs)
            self.output_selection.enabled = bool(outputs)

            loaded_draft = self.draft
            if (
                loaded_draft.sample_rate is None
                and loaded_draft.input_device_id is None
                and loaded_draft.output_device_id is None
            ):
                suggested = self.service.suggest_settings()
                loaded_draft = AudioSettingsDraft.from_settings(suggested)

            self.buffer_size_selection.value = BufferSizeChoice(
                loaded_draft.buffer_size
                if loaded_draft.buffer_size in AUDIO_BUFFER_SIZES
                else 0
            )

            if inputs and outputs:
                self._select_device(
                    self.input_selection,
                    loaded_draft.input_device_id
                    if loaded_draft.input_device_id is not None
                    else self.service.default_input_device_id,
                )
                self._select_device(
                    self.output_selection,
                    loaded_draft.output_device_id
                    if loaded_draft.output_device_id is not None
                    else self.service.default_output_device_id,
                )
                self._rebuild_routing_matrices(
                    self._selected_input_device(),
                    self._selected_output_device(),
                    loaded_draft.track_inputs,
                    loaded_draft.bus_outputs,
                )
                self._update_sample_rates(
                    self.locked_sample_rate or loaded_draft.sample_rate
                )
            else:
                self._rebuild_routing_matrices(
                    self._selected_input_device(),
                    self._selected_output_device(),
                    loaded_draft.track_inputs,
                    loaded_draft.bus_outputs,
                )
                self.sample_rate_selection.items = []
                self.sample_rate_selection.enabled = False
                self._show_inventory_problem()
        except AudioConfigurationError as exc:
            self.input_selection.items = []
            self.output_selection.items = []
            self.sample_rate_selection.items = []
            self._rebuild_routing_matrices(
                None,
                None,
                UNASSIGNED_TRACK_INPUTS,
                UNASSIGNED_BUS_OUTPUTS,
            )
            self.input_selection.enabled = False
            self.output_selection.enabled = False
            self.sample_rate_selection.enabled = False
            self.status_label.text = str(exc)
        finally:
            self._updating = False

    @staticmethod
    def _select_device(
        selection: toga.Selection, device_id: int | None
    ) -> None:
        if device_id is None:
            return
        for item in selection.items:
            choice = item.value
            if isinstance(choice, DeviceChoice) and choice.device.index == device_id:
                selection.value = choice
                return

    def _on_device_changed(self, widget: toga.Widget, **kwargs: object) -> None:
        if self._updating:
            return
        current_rate = self._selected_rate()
        if widget is self.input_selection:
            self._rebuild_routing_matrices(
                self._selected_input_device(),
                self._selected_output_device(),
                self.track_inputs,
                self.bus_outputs,
            )
        elif widget is self.output_selection:
            self._rebuild_routing_matrices(
                self._selected_input_device(),
                self._selected_output_device(),
                self.track_inputs,
                self.bus_outputs,
            )
        self._update_sample_rates(current_rate)

    def _update_sample_rates(self, preferred_rate: int | None = None) -> None:
        previous_updating = self._updating
        self._updating = True
        try:
            input_choice = self.input_selection.value
            output_choice = self.output_selection.value
            if not isinstance(input_choice, DeviceChoice) or not isinstance(
                output_choice, DeviceChoice
            ):
                self.sample_rate_selection.items = []
                self.sample_rate_selection.enabled = False
                self._set_save_enabled(False)
                self._show_inventory_problem()
                return

            if self.locked_sample_rate is not None:
                locked_choice = SampleRateChoice(self.locked_sample_rate)
                self.sample_rate_selection.items = [locked_choice]
                self.sample_rate_selection.value = locked_choice
                self.sample_rate_selection.enabled = False
                candidate = AudioSettings(
                    input_choice.device.index,
                    output_choice.device.index,
                    self.locked_sample_rate,
                    self.track_inputs,
                    self.bus_outputs,
                    self._selected_buffer_size(),
                )
                compatibility_error = (
                    None
                    if candidate == self.trusted_settings
                    else self.service.compatibility_error(candidate)
                )
                self._set_save_enabled(compatibility_error is None)
                self.status_label.text = compatibility_error or (
                    "Project sample rate is fixed by the WAV file."
                )
                return

            rates = self.service.supported_sample_rates(
                input_choice.device.index,
                output_choice.device.index,
                self.track_inputs,
                self.bus_outputs,
            )
            choices = [SampleRateChoice(rate) for rate in rates]
            self.sample_rate_selection.items = choices
            self.sample_rate_selection.enabled = bool(choices)
            if not choices:
                self._set_save_enabled(False)
                self.status_label.text = (
                    "The selected devices have no supported studio "
                    "sample rate in common."
                )
                return

            selected_rate = (
                preferred_rate
                if preferred_rate in rates
                else 48_000
                if 48_000 in rates
                else rates[0]
            )
            self.sample_rate_selection.value = SampleRateChoice(selected_rate)
            self.status_label.text = ""
            self._set_save_enabled(True)
        except AudioConfigurationError as exc:
            self.sample_rate_selection.items = []
            self.sample_rate_selection.enabled = False
            self._set_save_enabled(False)
            self.status_label.text = str(exc)
        finally:
            self._updating = previous_updating

    def _selected_input_device(self) -> AudioDevice | None:
        choice = self.input_selection.value
        return choice.device if isinstance(choice, DeviceChoice) else None

    def _selected_output_device(self) -> AudioDevice | None:
        choice = self.output_selection.value
        return choice.device if isinstance(choice, DeviceChoice) else None

    def _rebuild_routing_matrices(
        self,
        input_device: AudioDevice | None,
        output_device: AudioDevice | None,
        track_inputs: tuple[TrackInputRoute, ...],
        bus_outputs: tuple[int | None, ...],
    ) -> None:
        self.track_inputs = track_inputs
        self.bus_outputs = bus_outputs
        self.route_switches = {}
        self.output_route_switches = {}

        input_matrix = self._build_input_routing_matrix(
            input_device, track_inputs
        )
        output_matrix = self._build_output_routing_matrix(
            output_device, bus_outputs
        )
        self.routing_scroll.content = toga.Box(
            children=[
                toga.Label(
                    "Input routing",
                    font_size=16,
                    font_weight="bold",
                ),
                toga.Label(
                    "Each track accepts one input. "
                    "An input can feed multiple tracks.",
                    margin_top=4,
                ),
                input_matrix,
                toga.Divider(margin_top=12, margin_bottom=16),
                toga.Label(
                    "Output routing",
                    font_size=16,
                    font_weight="bold",
                ),
                toga.Label(
                    "Each stereo side accepts one distinct device output.",
                    margin_top=4,
                ),
                output_matrix,
            ],
            direction=COLUMN,
            margin=8,
        )

    def _build_input_routing_matrix(
        self,
        input_device: AudioDevice | None,
        track_inputs: tuple[TrackInputRoute, ...],
    ) -> toga.Widget:
        if input_device is None:
            return toga.Label(
                "Select an input device to configure track routing.",
                margin_top=12,
            )

        label_width = _ROUTING_LABEL_WIDTH
        track_width = 44
        project_tracks_label = toga.Box(
            children=[
                toga.Box(width=label_width),
                toga.Label(
                    "Project tracks",
                    width=track_width * PROJECT_TRACK_COUNT,
                    text_align=CENTER,
                    font_weight="bold",
                ),
            ],
            direction=ROW,
            margin_bottom=6,
        )
        header = toga.Box(
            children=[
                toga.Label("Input source", width=label_width),
                *[
                    toga.Label(
                        str(track_index + 1),
                        width=track_width,
                        text_align=CENTER,
                    )
                    for track_index in range(PROJECT_TRACK_COUNT)
                ],
            ],
            direction=ROW,
            margin_bottom=8,
        )

        rows: list[toga.Box] = []
        for input_channel, input_label in input_source_rows(
            input_device, track_inputs
        ):
            cells: list[toga.Widget] = [
                toga.Label(
                    input_label,
                    width=label_width,
                    margin_top=4,
                )
            ]
            for track_index in range(PROJECT_TRACK_COUNT):
                route_switch = toga.Switch(
                    "\u200b",
                    id=(
                        f"input-{input_channel + 1}-track-{track_index + 1}"
                        if is_physical_input(input_channel)
                        else f"{input_channel.value}-track-{track_index + 1}"
                    ),
                    value=track_inputs[track_index] == input_channel,
                    on_change=self._route_handler(input_channel, track_index),
                )
                self.route_switches[(input_channel, track_index)] = route_switch
                cells.append(
                    toga.Box(
                        children=[route_switch],
                        width=track_width,
                        justify_content=CENTER,
                    )
                )
            rows.append(
                toga.Box(children=cells, direction=ROW, margin_bottom=6)
            )

        return toga.Box(
            children=[project_tracks_label, header, *rows],
            direction=COLUMN,
            width=label_width + track_width * PROJECT_TRACK_COUNT,
            margin_top=12,
        )

    def _build_output_routing_matrix(
        self,
        output_device: AudioDevice | None,
        bus_outputs: tuple[int | None, ...],
    ) -> toga.Widget:
        if output_device is None:
            return toga.Label(
                "Select an output device to configure stereo routing.",
                margin_top=12,
            )

        label_width = _ROUTING_LABEL_WIDTH
        bus_width = 52
        stereo_bus_label = toga.Box(
            children=[
                toga.Box(width=label_width),
                toga.Label(
                    "Stereo bus",
                    width=bus_width * STEREO_BUS_CHANNEL_COUNT,
                    text_align=CENTER,
                    font_weight="bold",
                ),
            ],
            direction=ROW,
            margin_bottom=6,
        )
        header = toga.Box(
            children=[
                toga.Label("Device output", width=label_width),
                *[
                    toga.Label(side, width=bus_width, text_align=CENTER)
                    for side in ("L", "R")
                ],
            ],
            direction=ROW,
            margin_bottom=8,
        )

        stored_output_count = max(
            (channel for channel in bus_outputs if channel is not None),
            default=-1,
        ) + 1
        row_count = max(output_device.max_output_channels, stored_output_count)
        output_labels = device_channel_labels(
            output_device, "output", row_count
        )
        rows: list[toga.Box] = []
        for output_channel in range(row_count):
            cells: list[toga.Widget] = [
                toga.Label(
                    output_labels[output_channel],
                    width=label_width,
                    margin_top=4,
                )
            ]
            for bus_index, bus_side in enumerate(("l", "r")):
                route_switch = toga.Switch(
                    "\u200b",
                    id=f"output-{output_channel + 1}-bus-{bus_side}",
                    value=bus_outputs[bus_index] == output_channel,
                    on_change=self._output_route_handler(
                        output_channel, bus_index
                    ),
                )
                self.output_route_switches[(output_channel, bus_index)] = (
                    route_switch
                )
                cells.append(
                    toga.Box(
                        children=[route_switch],
                        width=bus_width,
                        justify_content=CENTER,
                    )
                )
            rows.append(
                toga.Box(children=cells, direction=ROW, margin_bottom=6)
            )

        return toga.Box(
            children=[stereo_bus_label, header, *rows],
            direction=COLUMN,
            width=label_width + bus_width * STEREO_BUS_CHANNEL_COUNT,
            margin_top=12,
        )

    def _route_handler(
        self, input_channel: TrackInputRoute, track_index: int
    ):
        def handler(widget: toga.Switch, **kwargs: object) -> None:
            self._on_route_changed(widget, input_channel, track_index)

        return handler

    def _on_route_changed(
        self,
        widget: toga.Switch,
        input_channel: TrackInputRoute,
        track_index: int,
    ) -> None:
        if self._updating:
            return

        track_inputs = list(self.track_inputs)
        previous_input = track_inputs[track_index]
        if widget.value:
            track_inputs[track_index] = input_channel
            if previous_input is not None and previous_input != input_channel:
                previous_switch = self.route_switches.get(
                    (previous_input, track_index)
                )
                if previous_switch is not None:
                    self._updating = True
                    try:
                        previous_switch.value = False
                    finally:
                        self._updating = False
        elif previous_input == input_channel:
            track_inputs[track_index] = None

        self.track_inputs = tuple(track_inputs)
        self._update_sample_rates(self._selected_rate())

    def _output_route_handler(self, output_channel: int, bus_index: int):
        def handler(widget: toga.Switch, **kwargs: object) -> None:
            self._on_output_route_changed(widget, output_channel, bus_index)

        return handler

    def _on_output_route_changed(
        self,
        widget: toga.Switch,
        output_channel: int,
        bus_index: int,
    ) -> None:
        if self._updating:
            return

        bus_outputs = list(self.bus_outputs)
        previous_output = bus_outputs[bus_index]
        if widget.value:
            bus_outputs[bus_index] = output_channel
            switches_to_clear: list[toga.Switch] = []
            if previous_output is not None and previous_output != output_channel:
                previous_switch = self.output_route_switches.get(
                    (previous_output, bus_index)
                )
                if previous_switch is not None:
                    switches_to_clear.append(previous_switch)

            other_bus_index = 1 - bus_index
            if bus_outputs[other_bus_index] == output_channel:
                bus_outputs[other_bus_index] = None
                other_switch = self.output_route_switches.get(
                    (output_channel, other_bus_index)
                )
                if other_switch is not None:
                    switches_to_clear.append(other_switch)

            if switches_to_clear:
                self._updating = True
                try:
                    for route_switch in switches_to_clear:
                        route_switch.value = False
                finally:
                    self._updating = False
        elif previous_output == output_channel:
            bus_outputs[bus_index] = None

        self.bus_outputs = tuple(bus_outputs)
        self._update_sample_rates(self._selected_rate())

    def _show_inventory_problem(self) -> None:
        if not self.service.input_devices and not self.service.output_devices:
            message = "No audio input or output devices are available."
        elif not self.service.input_devices:
            message = "No audio input devices are available."
        elif not self.service.output_devices:
            message = "No audio output devices are available."
        else:
            message = (
                "The available devices do not share a supported studio sample rate."
            )
        self.status_label.text = message

    def _selected_rate(self) -> int | None:
        choice = self.sample_rate_selection.value
        return choice.value if isinstance(choice, SampleRateChoice) else None

    def _selected_buffer_size(self) -> int:
        choice = self.buffer_size_selection.value
        return choice.value if isinstance(choice, BufferSizeChoice) else 0

    def _selected_settings(self) -> AudioSettings | None:
        input_choice = self.input_selection.value
        output_choice = self.output_selection.value
        sample_rate = self._selected_rate()
        if (
            not isinstance(input_choice, DeviceChoice)
            or not isinstance(output_choice, DeviceChoice)
            or sample_rate is None
        ):
            self._set_save_enabled(False)
            self.status_label.text = "Choose an input, output, and sample rate."
            return None

        return AudioSettings(
            input_device_id=input_choice.device.index,
            output_device_id=output_choice.device.index,
            sample_rate=sample_rate,
            track_inputs=self.track_inputs,
            bus_outputs=self.bus_outputs,
            buffer_size=self._selected_buffer_size(),
        )

    def _save(self, widget: toga.Widget, **kwargs: object) -> None:
        settings = self._selected_settings()
        if settings is None:
            return
        try:
            self.on_applied(settings)
        except (AudioConfigurationError, RuntimeError) as exc:
            self.status_label.text = str(exc)
            return

        self.window.hide()

    def _cancel(self, widget: toga.Widget | None = None, **kwargs: object) -> None:
        self.window.hide()

    def _on_close(self, window: toga.Window, **kwargs: object) -> bool:
        self.window.hide()
        return False
