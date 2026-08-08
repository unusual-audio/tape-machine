"""Audio settings window for Tape Machine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import toga
from toga.style.pack import COLUMN, END, ROW

from tape_machine.audio import (
    AudioConfigurationError,
    AudioDevice,
    AudioDeviceService,
    AudioSettings,
)


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


class AudioSettingsWindow:
    """A reusable settings window that edits a draft audio configuration."""

    def __init__(
        self,
        service: AudioDeviceService,
        on_applied: Callable[[AudioSettings], None],
    ) -> None:
        self.service = service
        self.on_applied = on_applied
        self._updating = False

        self.input_selection = toga.Selection(
            items=[], on_change=self._on_device_changed, flex=1
        )
        self.output_selection = toga.Selection(
            items=[], on_change=self._on_device_changed, flex=1
        )
        self.sample_rate_selection = toga.Selection(items=[], flex=1)
        self.status_label = toga.Label("", margin_top=12)
        self.save_button = toga.Button(
            "Save", on_press=self._save, enabled=False, margin_left=8
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
                self.status_label,
                toga.Box(
                    children=[
                        toga.Button("Cancel", on_press=self._cancel),
                        self.save_button,
                    ],
                    direction=ROW,
                    justify_content=END,
                    margin_top=20,
                ),
            ],
            direction=COLUMN,
            margin=24,
        )

        self.window = toga.Window(
            title="Audio Settings",
            size=(600, 300),
            resizable=False,
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

    def open(self) -> None:
        """Refresh and show the window, or leave an already-visible draft intact."""
        if self.window.visible:
            return
        self._load_draft()
        self.window.show()

    def _load_draft(self) -> None:
        self._updating = True
        self.status_label.text = ""
        self.save_button.enabled = False
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

            suggested = self.service.suggest_settings(
                self.service.current_settings
            )
            if suggested is not None:
                self._select_device(
                    self.input_selection, suggested.input_device_id
                )
                self._select_device(
                    self.output_selection, suggested.output_device_id
                )
                self._update_sample_rates(suggested.sample_rate)
            else:
                self.sample_rate_selection.items = []
                self.sample_rate_selection.enabled = False
                self._show_inventory_problem()
        except AudioConfigurationError as exc:
            self.input_selection.items = []
            self.output_selection.items = []
            self.sample_rate_selection.items = []
            self.input_selection.enabled = False
            self.output_selection.enabled = False
            self.sample_rate_selection.enabled = False
            self.status_label.text = str(exc)
        finally:
            self._updating = False

    @staticmethod
    def _select_device(selection: toga.Selection, device_id: int) -> None:
        for item in selection.items:
            choice = item.value
            if isinstance(choice, DeviceChoice) and choice.device.index == device_id:
                selection.value = choice
                return

    def _on_device_changed(self, widget: toga.Widget, **kwargs: object) -> None:
        if self._updating:
            return
        current_rate = self._selected_rate()
        self._update_sample_rates(current_rate)

    def _update_sample_rates(self, preferred_rate: int | None = None) -> None:
        self._updating = True
        try:
            input_choice = self.input_selection.value
            output_choice = self.output_selection.value
            if not isinstance(input_choice, DeviceChoice) or not isinstance(
                output_choice, DeviceChoice
            ):
                self.sample_rate_selection.items = []
                self.sample_rate_selection.enabled = False
                self.save_button.enabled = False
                self._show_inventory_problem()
                return

            rates = self.service.supported_sample_rates(
                input_choice.device.index, output_choice.device.index
            )
            choices = [SampleRateChoice(rate) for rate in rates]
            self.sample_rate_selection.items = choices
            self.sample_rate_selection.enabled = bool(choices)
            if not choices:
                self.save_button.enabled = False
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
            self.save_button.enabled = True
        except AudioConfigurationError as exc:
            self.sample_rate_selection.items = []
            self.sample_rate_selection.enabled = False
            self.save_button.enabled = False
            self.status_label.text = str(exc)
        finally:
            self._updating = False

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

    def _save(self, widget: toga.Widget, **kwargs: object) -> None:
        input_choice = self.input_selection.value
        output_choice = self.output_selection.value
        sample_rate = self._selected_rate()
        if (
            not isinstance(input_choice, DeviceChoice)
            or not isinstance(output_choice, DeviceChoice)
            or sample_rate is None
        ):
            self.save_button.enabled = False
            self.status_label.text = "Choose an input, output, and sample rate."
            return

        settings = AudioSettings(
            input_device_id=input_choice.device.index,
            output_device_id=output_choice.device.index,
            sample_rate=sample_rate,
        )
        try:
            self.service.apply(settings)
        except AudioConfigurationError as exc:
            self.status_label.text = str(exc)
            return

        self.on_applied(settings)
        self.window.hide()

    def _cancel(self, widget: toga.Widget | None = None, **kwargs: object) -> None:
        self.window.hide()

    def _on_close(self, window: toga.Window, **kwargs: object) -> bool:
        self.window.hide()
        return False
