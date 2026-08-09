"""Interactive mixer controls for the project screen."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, pi, sin
from typing import Callable

import toga
from toga.style.pack import CENTER, COLUMN, ROW

from tape_machine.audio import (
    PROJECT_TRACK_COUNT,
    TrackInputRoute,
    is_physical_input,
)
from tape_machine.project import (
    MIX_MAX_LEVEL_DB,
    MIX_MIN_LEVEL_DB,
    MixerMetadata,
    TrackMixMetadata,
)
from tape_machine.theme import (
    ACCENT_BLUE,
    ACCENT_ORANGE,
    ACCENT_RED,
    ACCENT_YELLOW,
    FADER_HANDLE,
    FADER_INDICATOR,
    FADER_RAIL,
    FADER_TICK,
    KNOB_BACKGROUND,
    KNOB_BORDER,
    KNOB_INDICATOR,
    METER_GREEN,
    METER_OFF,
    METER_RED,
    METER_YELLOW,
)


MIN_LEVEL_DB = MIX_MIN_LEVEL_DB
MAX_LEVEL_DB = MIX_MAX_LEVEL_DB
UNITY_LEVEL_DB = 0.0

_TRACK_STRIP_WIDTH = 88
_FADER_WIDTH = 58
_FADER_HEIGHT = 230
_FADER_TOP = 12
_FADER_BOTTOM = 214
_METER_CEILING_DB = 0.0
_METER_YELLOW_DB = -12.0
_METER_RED_DB = -3.0

_CONTROL_COLORS = {
    "record_enabled": ACCENT_RED,
    "input_monitoring": ACCENT_BLUE,
    "muted": ACCENT_ORANGE,
    "soloed": ACCENT_YELLOW,
}


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a numeric control value to its supported range."""
    return max(minimum, min(maximum, float(value)))


def format_level_db(value: float) -> str:
    """Format a mixer level for its compact readout."""
    if value <= MIN_LEVEL_DB:
        return "−∞ dB"
    return f"{value:+.1f} dB"


def level_y(value: float) -> float:
    """Map a fader or meter dB value onto the shared vertical scale."""
    ratio = (clamp(value, MIN_LEVEL_DB, MAX_LEVEL_DB) - MIN_LEVEL_DB) / (
        MAX_LEVEL_DB - MIN_LEVEL_DB
    )
    return _FADER_BOTTOM - ratio * (_FADER_BOTTOM - _FADER_TOP)


def clear_canvas(canvas: toga.Canvas) -> None:
    """Clear drawing actions without invoking Widget.clear on Toga 0.5."""
    canvas.root_state.drawing_actions.clear()
    canvas.redraw()


@dataclass(slots=True)
class TrackMixerState:
    """Live state for one project track strip."""

    input_assigned: bool = False
    input_monitorable: bool = False
    level_db: float = UNITY_LEVEL_DB
    pan: float = 0.0
    record_enabled: bool = False
    input_monitoring: bool = False
    muted: bool = False
    soloed: bool = False


@dataclass(slots=True)
class MixerState:
    """Live state for the eight tracks and stereo bus."""

    tracks: list[TrackMixerState] = field(
        default_factory=lambda: [
            TrackMixerState() for _ in range(PROJECT_TRACK_COUNT)
        ]
    )
    bus_level_db: float = UNITY_LEVEL_DB
    on_change: Callable[[], None] | None = field(
        default=None, repr=False, compare=False
    )

    @classmethod
    def from_track_inputs(
        cls, track_inputs: tuple[TrackInputRoute, ...]
    ) -> MixerState:
        return cls.from_metadata(track_inputs, MixerMetadata())

    @classmethod
    def from_metadata(
        cls,
        track_inputs: tuple[TrackInputRoute, ...],
        mix: MixerMetadata,
    ) -> MixerState:
        """Restore persisted controls and derive input availability from routes."""
        if len(track_inputs) != PROJECT_TRACK_COUNT:
            raise ValueError(
                f"Mixer routing must contain {PROJECT_TRACK_COUNT} tracks."
            )
        tracks = []
        for channel, saved in zip(track_inputs, mix.tracks, strict=True):
            input_assigned = channel is not None
            input_monitorable = is_physical_input(channel)
            tracks.append(
                TrackMixerState(
                    input_assigned=input_assigned,
                    input_monitorable=input_monitorable,
                    level_db=saved.level_db,
                    pan=saved.pan,
                    record_enabled=(
                        saved.record_enabled if input_assigned else False
                    ),
                    input_monitoring=(
                        saved.input_monitoring if input_monitorable else False
                    ),
                    muted=saved.muted,
                    soloed=saved.soloed,
                )
            )
        return cls(tracks=tracks, bus_level_db=mix.bus_level_db)

    def to_metadata(
        self, track_inputs: tuple[TrackInputRoute, ...] | None = None
    ) -> MixerMetadata:
        """Snapshot controls, normalizing input-only states for absent routes."""
        if track_inputs is not None and len(track_inputs) != PROJECT_TRACK_COUNT:
            raise ValueError(
                f"Mixer routing must contain {PROJECT_TRACK_COUNT} tracks."
            )
        saved_tracks = []
        for index, track in enumerate(self.tracks):
            input_assigned = (
                track.input_assigned
                if track_inputs is None
                else track_inputs[index] is not None
            )
            input_monitorable = (
                track.input_monitorable
                if track_inputs is None
                else is_physical_input(track_inputs[index])
            )
            saved_tracks.append(
                TrackMixMetadata(
                    level_db=track.level_db,
                    pan=track.pan,
                    record_enabled=(
                        track.record_enabled if input_assigned else False
                    ),
                    input_monitoring=(
                        track.input_monitoring if input_monitorable else False
                    ),
                    muted=track.muted,
                    soloed=track.soloed,
                )
            )
        return MixerMetadata(
            tracks=tuple(saved_tracks), bus_level_db=self.bus_level_db
        )

    def update_input_routes(
        self, track_inputs: tuple[TrackInputRoute, ...]
    ) -> None:
        """Update input availability and clear invalid input-only states."""
        if len(track_inputs) != PROJECT_TRACK_COUNT:
            raise ValueError(
                f"Mixer routing must contain {PROJECT_TRACK_COUNT} tracks."
            )
        for track, channel in zip(self.tracks, track_inputs, strict=True):
            track.input_assigned = channel is not None
            track.input_monitorable = is_physical_input(channel)
            if not track.input_assigned:
                track.record_enabled = False
            if not track.input_monitorable:
                track.input_monitoring = False
        self._notify()

    def set_track_level(self, track_index: int, value: float) -> float:
        value = clamp(value, MIN_LEVEL_DB, MAX_LEVEL_DB)
        self.tracks[track_index].level_db = value
        self._notify()
        return value

    def set_bus_level(self, value: float) -> float:
        self.bus_level_db = clamp(value, MIN_LEVEL_DB, MAX_LEVEL_DB)
        self._notify()
        return self.bus_level_db

    def set_pan(self, track_index: int, value: float) -> float:
        value = clamp(value, -1.0, 1.0)
        self.tracks[track_index].pan = value
        self._notify()
        return value

    def toggle(self, track_index: int, control: str) -> bool:
        """Toggle a track button, respecting input-route availability."""
        if control not in _CONTROL_COLORS:
            raise ValueError(f"Unknown mixer control {control!r}.")
        track = self.tracks[track_index]
        if control == "record_enabled" and not track.input_assigned:
            return False
        if control == "input_monitoring" and not track.input_monitorable:
            return False
        value = not getattr(track, control)
        setattr(track, control, value)
        self._notify()
        return value

    def clear_monitoring(self) -> None:
        """Disable input monitoring on every track."""
        changed = False
        for track in self.tracks:
            if track.input_monitoring:
                track.input_monitoring = False
                changed = True
        if changed:
            self._notify()

    def _notify(self) -> None:
        if self.on_change is not None:
            self.on_change()


class VerticalFader:
    """A compact Canvas-based vertical audio fader."""

    def __init__(
        self,
        value: float,
        on_change: Callable[[float], None],
        *,
        meter_channels: int = 1,
    ) -> None:
        if meter_channels not in (1, 2):
            raise ValueError("A fader meter must have one or two channels.")
        self.value = clamp(value, MIN_LEVEL_DB, MAX_LEVEL_DB)
        self.on_change = on_change
        self.meter_channels = meter_channels
        self.meter_levels = (MIN_LEVEL_DB,) * meter_channels
        self.canvas = toga.Canvas(
            width=_FADER_WIDTH,
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

    def set_meter_levels(self, levels: tuple[float, ...]) -> None:
        """Update one mono or stereo meter without changing fader state."""
        if len(levels) != self.meter_channels:
            raise ValueError(
                f"This fader requires {self.meter_channels} meter channels."
            )
        normalized = tuple(
            clamp(level, MIN_LEVEL_DB, _METER_CEILING_DB)
            for level in levels
        )
        if normalized == self.meter_levels:
            return
        self.meter_levels = normalized
        self._draw(self.canvas)

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
        with widget.stroke(color=FADER_RAIL, line_width=3):
            widget.begin_path()
            widget.move_to(center_x, _FADER_TOP)
            widget.line_to(center_x, _FADER_BOTTOM)

        for tick_db in (6, 0, -12, -24, -36, -48, -60):
            tick_y = level_y(tick_db)
            with widget.stroke(color=FADER_TICK, line_width=1):
                widget.begin_path()
                widget.move_to(7, tick_y)
                widget.line_to(14, tick_y)
                widget.move_to(34, tick_y)
                widget.line_to(41, tick_y)

        meter_geometry = (
            ((51, 5),)
            if self.meter_channels == 1
            else ((49, 3), (54, 3))
        )
        for (meter_x, meter_width), meter_level in zip(
            meter_geometry, self.meter_levels, strict=True
        ):
            self._draw_meter(widget, meter_x, meter_width, meter_level)

        handle_y = level_y(self.value)
        with widget.fill(color=FADER_HANDLE):
            widget.rect(5, handle_y - 8, 38, 16)
        with widget.stroke(color=FADER_INDICATOR, line_width=2):
            widget.begin_path()
            widget.move_to(6, handle_y)
            widget.line_to(42, handle_y)

    @staticmethod
    def _draw_meter(
        widget: toga.Canvas,
        x: int,
        width: int,
        level: float,
    ) -> None:
        meter_top = level_y(_METER_CEILING_DB)
        with widget.fill(color=METER_OFF):
            widget.rect(
                x, meter_top, width, _FADER_BOTTOM - meter_top
            )
        segments = (
            (MIN_LEVEL_DB, _METER_YELLOW_DB, METER_GREEN),
            (_METER_YELLOW_DB, _METER_RED_DB, METER_YELLOW),
            (_METER_RED_DB, _METER_CEILING_DB, METER_RED),
        )
        for segment_floor, segment_ceiling, color in segments:
            lit_ceiling = min(level, segment_ceiling)
            if lit_ceiling <= segment_floor:
                continue
            top = level_y(lit_ceiling)
            bottom = level_y(segment_floor)
            with widget.fill(color=color):
                widget.rect(x, top, width, bottom - top)


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
        with widget.fill(color=KNOB_BACKGROUND):
            widget.arc(center_x, center_y, radius)
        with widget.stroke(color=KNOB_BORDER, line_width=2):
            widget.arc(center_x, center_y, radius)

        angle = (-pi / 2) + self.value * (3 * pi / 4)
        with widget.stroke(color=KNOB_INDICATOR, line_width=3):
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
        self.monitoring_available = False
        self.record_enable_locked = False
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
            if control == "record_enabled":
                button.enabled = (
                    self.state.input_assigned and not self.record_enable_locked
                )
            elif control == "input_monitoring":
                button.enabled = (
                    self.state.input_monitorable and self.monitoring_available
                )
            else:
                button.enabled = True
            if getattr(self.state, control):
                button.style.background_color = _CONTROL_COLORS[control]
            else:
                del button.style.background_color


class StereoBusStrip:
    """Level-only stereo-bus strip aligned with the track strips."""

    def __init__(self, mixer_state: MixerState) -> None:
        self.mixer_state = mixer_state
        self.fader = VerticalFader(
            mixer_state.bus_level_db,
            mixer_state.set_bus_level,
            meter_channels=2,
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
        track_inputs: tuple[TrackInputRoute, ...],
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
        self, track_inputs: tuple[TrackInputRoute, ...]
    ) -> None:
        self.state.update_input_routes(track_inputs)
        for strip in self.track_strips:
            strip.sync_controls()

    def set_monitoring_available(self, available: bool) -> None:
        """Enable monitor controls only while an audio stream is running."""
        for strip in self.track_strips:
            strip.monitoring_available = available
            strip.sync_controls()

    def set_record_enable_locked(self, locked: bool) -> None:
        """Lock record-enable controls during transport state transitions."""
        for strip in self.track_strips:
            strip.record_enable_locked = locked
            strip.sync_controls()

    def set_meter_levels(
        self,
        track_db: tuple[float, ...],
        bus_db: tuple[float, float],
    ) -> None:
        """Distribute one engine meter snapshot across the mixer strips."""
        if len(track_db) != PROJECT_TRACK_COUNT:
            raise ValueError(
                f"Mixer metering requires {PROJECT_TRACK_COUNT} tracks."
            )
        if len(bus_db) != 2:
            raise ValueError("Stereo-bus metering requires two channels.")
        for strip, level in zip(
            self.track_strips, track_db, strict=True
        ):
            strip.fader.set_meter_levels((level,))
        self.bus_strip.fader.set_meter_levels(bus_db)
