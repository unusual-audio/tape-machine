"""Interactive mixer controls for the project screen."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, pi, sin
from typing import Callable

import toga
from toga.style.pack import CENTER, COLUMN, ROW

from tape_machine.audio import PROJECT_TRACK_COUNT


MIN_LEVEL_DB = -60.0
MAX_LEVEL_DB = 6.0
UNITY_LEVEL_DB = 0.0

_TRACK_STRIP_WIDTH = 88
_FADER_HEIGHT = 230
_FADER_TOP = 12
_FADER_BOTTOM = 214

_CONTROL_COLORS = {
    "record_enabled": "#d94b4b",
    "input_monitoring": "#4b8fd9",
    "muted": "#d98b3e",
    "soloed": "#e0bd35",
}


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a numeric control value to its supported range."""
    return max(minimum, min(maximum, float(value)))


def format_level_db(value: float) -> str:
    """Format a mixer level for its compact readout."""
    if value <= MIN_LEVEL_DB:
        return "−∞ dB"
    return f"{value:+.1f} dB"


def clear_canvas(canvas: toga.Canvas) -> None:
    """Clear drawing actions without invoking Widget.clear on Toga 0.5."""
    canvas.root_state.drawing_actions.clear()
    canvas.redraw()


@dataclass(slots=True)
class TrackMixerState:
    """Session-only state for one project track strip."""

    input_assigned: bool = False
    level_db: float = UNITY_LEVEL_DB
    pan: float = 0.0
    record_enabled: bool = False
    input_monitoring: bool = False
    muted: bool = False
    soloed: bool = False


@dataclass(slots=True)
class MixerState:
    """Session-only state for the eight tracks and stereo bus."""

    tracks: list[TrackMixerState] = field(
        default_factory=lambda: [
            TrackMixerState() for _ in range(PROJECT_TRACK_COUNT)
        ]
    )
    bus_level_db: float = UNITY_LEVEL_DB

    @classmethod
    def from_track_inputs(
        cls, track_inputs: tuple[int | None, ...]
    ) -> MixerState:
        state = cls()
        state.update_input_routes(track_inputs)
        return state

    def update_input_routes(
        self, track_inputs: tuple[int | None, ...]
    ) -> None:
        """Update input availability and clear invalid input-only states."""
        if len(track_inputs) != PROJECT_TRACK_COUNT:
            raise ValueError(
                f"Mixer routing must contain {PROJECT_TRACK_COUNT} tracks."
            )
        for track, channel in zip(self.tracks, track_inputs, strict=True):
            track.input_assigned = channel is not None
            if not track.input_assigned:
                track.record_enabled = False
                track.input_monitoring = False

    def set_track_level(self, track_index: int, value: float) -> float:
        value = clamp(value, MIN_LEVEL_DB, MAX_LEVEL_DB)
        self.tracks[track_index].level_db = value
        return value

    def set_bus_level(self, value: float) -> float:
        self.bus_level_db = clamp(value, MIN_LEVEL_DB, MAX_LEVEL_DB)
        return self.bus_level_db

    def set_pan(self, track_index: int, value: float) -> float:
        value = clamp(value, -1.0, 1.0)
        self.tracks[track_index].pan = value
        return value

    def toggle(self, track_index: int, control: str) -> bool:
        """Toggle a track button, respecting input-route availability."""
        if control not in _CONTROL_COLORS:
            raise ValueError(f"Unknown mixer control {control!r}.")
        track = self.tracks[track_index]
        if control in {"record_enabled", "input_monitoring"} and not (
            track.input_assigned
        ):
            return False
        value = not getattr(track, control)
        setattr(track, control, value)
        return value


class VerticalFader:
    """A compact Canvas-based vertical audio fader."""

    def __init__(
        self,
        value: float,
        on_change: Callable[[float], None],
    ) -> None:
        self.value = clamp(value, MIN_LEVEL_DB, MAX_LEVEL_DB)
        self.on_change = on_change
        self.canvas = toga.Canvas(
            width=48,
            height=_FADER_HEIGHT,
            on_resize=self._draw,
            on_press=self._move,
            on_drag=self._move,
            on_activate=self._reset,
        )
        self.readout = toga.Label(
            format_level_db(self.value),
            font_size=10,
            text_align=CENTER,
            margin_top=3,
        )
        self.widget = toga.Box(
            children=[self.canvas, self.readout],
            direction=COLUMN,
            align_items=CENTER,
        )
        self._draw(self.canvas)

    @staticmethod
    def value_from_y(y: float) -> float:
        ratio = clamp(
            (_FADER_BOTTOM - y) / (_FADER_BOTTOM - _FADER_TOP), 0.0, 1.0
        )
        return MIN_LEVEL_DB + ratio * (MAX_LEVEL_DB - MIN_LEVEL_DB)

    def set_value(self, value: float, *, notify: bool = True) -> None:
        self.value = clamp(value, MIN_LEVEL_DB, MAX_LEVEL_DB)
        self.readout.text = format_level_db(self.value)
        self._draw(self.canvas)
        if notify:
            self.on_change(self.value)

    def _move(
        self, widget: toga.Canvas, x: int, y: int, **kwargs: object
    ) -> None:
        self.set_value(self.value_from_y(y))

    def _reset(
        self, widget: toga.Canvas, x: int, y: int, **kwargs: object
    ) -> None:
        self.set_value(UNITY_LEVEL_DB)

    def _draw(self, widget: toga.Canvas, **kwargs: object) -> None:
        clear_canvas(widget)
        center_x = 24
        with widget.stroke(color="#777777", line_width=3):
            widget.begin_path()
            widget.move_to(center_x, _FADER_TOP)
            widget.line_to(center_x, _FADER_BOTTOM)

        for tick_db in (6, 0, -12, -24, -36, -48, -60):
            ratio = (tick_db - MIN_LEVEL_DB) / (MAX_LEVEL_DB - MIN_LEVEL_DB)
            tick_y = _FADER_BOTTOM - ratio * (
                _FADER_BOTTOM - _FADER_TOP
            )
            with widget.stroke(color="#999999", line_width=1):
                widget.begin_path()
                widget.move_to(7, tick_y)
                widget.line_to(14, tick_y)
                widget.move_to(34, tick_y)
                widget.line_to(41, tick_y)

        ratio = (self.value - MIN_LEVEL_DB) / (MAX_LEVEL_DB - MIN_LEVEL_DB)
        handle_y = _FADER_BOTTOM - ratio * (_FADER_BOTTOM - _FADER_TOP)
        with widget.fill(color="#d6d6d6"):
            widget.rect(5, handle_y - 8, 38, 16)
        with widget.stroke(color="#404040", line_width=2):
            widget.begin_path()
            widget.move_to(6, handle_y)
            widget.line_to(42, handle_y)


class PanKnob:
    """A Canvas-based rotary pan control adjusted with a vertical drag."""

    def __init__(
        self,
        value: float,
        on_change: Callable[[float], None],
    ) -> None:
        self.value = clamp(value, -1.0, 1.0)
        self.on_change = on_change
        self._drag_start_y = 0
        self._drag_start_value = self.value
        self.canvas = toga.Canvas(
            width=56,
            height=52,
            on_resize=self._draw,
            on_press=self._start_drag,
            on_drag=self._drag,
            on_activate=self._reset,
        )
        markings = toga.Box(
            children=[
                toga.Label("L", font_size=9, flex=1),
                toga.Label("C", font_size=9, text_align=CENTER, flex=1),
                toga.Label("R", font_size=9, text_align="right", flex=1),
            ],
            direction=ROW,
            width=56,
        )
        self.widget = toga.Box(
            children=[self.canvas, markings],
            direction=COLUMN,
            align_items=CENTER,
        )
        self._draw(self.canvas)

    @staticmethod
    def value_from_drag(start_value: float, start_y: float, y: float) -> float:
        return clamp(start_value + (start_y - y) / 75, -1.0, 1.0)

    def set_value(self, value: float, *, notify: bool = True) -> None:
        self.value = clamp(value, -1.0, 1.0)
        self._draw(self.canvas)
        if notify:
            self.on_change(self.value)

    def _start_drag(
        self, widget: toga.Canvas, x: int, y: int, **kwargs: object
    ) -> None:
        self._drag_start_y = y
        self._drag_start_value = self.value

    def _drag(
        self, widget: toga.Canvas, x: int, y: int, **kwargs: object
    ) -> None:
        self.set_value(
            self.value_from_drag(self._drag_start_value, self._drag_start_y, y)
        )

    def _reset(
        self, widget: toga.Canvas, x: int, y: int, **kwargs: object
    ) -> None:
        self.set_value(0.0)

    def _draw(self, widget: toga.Canvas, **kwargs: object) -> None:
        clear_canvas(widget)
        center_x = 28
        center_y = 27
        radius = 18
        with widget.fill(color="#f0f0f0"):
            widget.arc(center_x, center_y, radius)
        with widget.stroke(color="#a9a9a9", line_width=2):
            widget.arc(center_x, center_y, radius)

        angle = (-pi / 2) + self.value * (3 * pi / 4)
        with widget.stroke(color="#303030", line_width=3):
            widget.begin_path()
            widget.move_to(center_x, center_y)
            widget.line_to(
                center_x + cos(angle) * 13,
                center_y + sin(angle) * 13,
            )


class TrackStrip:
    """UI for one project-track mixer strip."""

    def __init__(self, index: int, mixer_state: MixerState) -> None:
        self.index = index
        self.mixer_state = mixer_state
        self.state = mixer_state.tracks[index]
        self.pan = PanKnob(
            self.state.pan,
            lambda value: self.mixer_state.set_pan(self.index, value),
        )
        self.fader = VerticalFader(
            self.state.level_db,
            lambda value: self.mixer_state.set_track_level(self.index, value),
        )
        self.buttons: dict[str, toga.Button] = {}
        button_specs = (
            ("record_enabled", "R"),
            ("input_monitoring", "I"),
            ("muted", "M"),
            ("soloed", "S"),
        )
        for control, label in button_specs:
            self.buttons[control] = toga.Button(
                label,
                on_press=self._toggle_handler(control),
                width=32,
                height=26,
                font_size=10,
            )
        button_grid = toga.Box(
            children=[
                toga.Box(
                    children=[
                        self.buttons["record_enabled"],
                        self.buttons["input_monitoring"],
                    ],
                    direction=ROW,
                    gap=4,
                ),
                toga.Box(
                    children=[self.buttons["muted"], self.buttons["soloed"]],
                    direction=ROW,
                    gap=4,
                    margin_top=4,
                ),
            ],
            direction=COLUMN,
            align_items=CENTER,
            margin_top=8,
            margin_bottom=8,
        )
        self.widget = toga.Box(
            children=[
                toga.Label(
                    str(index + 1),
                    font_weight="bold",
                    text_align=CENTER,
                    margin_bottom=6,
                ),
                toga.Label("PAN", font_size=9, text_align=CENTER),
                self.pan.widget,
                button_grid,
                self.fader.widget,
            ],
            direction=COLUMN,
            align_items=CENTER,
            width=_TRACK_STRIP_WIDTH,
            margin_left=4,
            margin_right=4,
        )
        self.sync_controls()

    def _toggle_handler(
        self, control: str
    ) -> Callable[[toga.Button], None]:
        def handler(widget: toga.Button, **kwargs: object) -> None:
            self.mixer_state.toggle(self.index, control)
            self.sync_controls()

        return handler

    def sync_controls(self) -> None:
        for control, button in self.buttons.items():
            button.enabled = not (
                control in {"record_enabled", "input_monitoring"}
                and not self.state.input_assigned
            )
            if getattr(self.state, control):
                button.style.background_color = _CONTROL_COLORS[control]
            else:
                del button.style.background_color


class StereoBusStrip:
    """Level-only stereo-bus strip aligned with the track strips."""

    def __init__(self, mixer_state: MixerState) -> None:
        self.mixer_state = mixer_state
        self.fader = VerticalFader(
            mixer_state.bus_level_db, mixer_state.set_bus_level
        )
        self.widget = toga.Box(
            children=[
                toga.Label(
                    "Stereo Bus",
                    font_size=10,
                    font_weight="bold",
                    text_align=CENTER,
                    margin_bottom=6,
                ),
                toga.Box(height=152),
                self.fader.widget,
            ],
            direction=COLUMN,
            align_items=CENTER,
            width=100,
            margin_left=12,
            margin_right=4,
        )


class MixerView:
    """Nine-strip project mixer view."""

    def __init__(
        self,
        track_inputs: tuple[int | None, ...],
        state: MixerState | None = None,
    ) -> None:
        self.state = state or MixerState.from_track_inputs(track_inputs)
        if state is not None:
            self.state.update_input_routes(track_inputs)
        self.track_strips = [
            TrackStrip(index, self.state) for index in range(PROJECT_TRACK_COUNT)
        ]
        self.bus_strip = StereoBusStrip(self.state)
        mixer_row = toga.Box(
            children=[
                *(strip.widget for strip in self.track_strips),
                toga.Divider(
                    direction=toga.Divider.VERTICAL,
                    margin_left=4,
                    margin_right=4,
                ),
                self.bus_strip.widget,
            ],
            direction=ROW,
            align_items="start",
            width=880,
            margin_top=10,
            margin_bottom=10,
        )
        self.widget = toga.ScrollContainer(
            content=mixer_row,
            horizontal=True,
            vertical=False,
            flex=1,
        )

    def update_track_routes(
        self, track_inputs: tuple[int | None, ...]
    ) -> None:
        self.state.update_input_routes(track_inputs)
        for strip in self.track_strips:
            strip.sync_controls()
