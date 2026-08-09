"""Tests for the session-only project mixer."""

from types import SimpleNamespace

import pytest

from tape_machine.audio import StereoBusInput
from tape_machine.mixer import (
    MAX_LEVEL_DB,
    MIN_LEVEL_DB,
    MixerState,
    MixerView,
    PanKnob,
    TrackStrip,
    VerticalFader,
    format_level_db,
    level_y,
)
from tape_machine.project import MixerMetadata, TrackMixMetadata
from tape_machine.theme import METER_GREEN, METER_OFF, METER_RED, METER_YELLOW


def test_mixer_defaults_to_unity_centered_and_inactive() -> None:
    state = MixerState.from_track_inputs(
        (0, 1, None, None, None, None, None, None)
    )

    assert len(state.tracks) == 8
    assert state.bus_level_db == 0
    assert [track.input_assigned for track in state.tracks] == [
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert all(track.level_db == 0 for track in state.tracks)
    assert all(track.pan == 0 for track in state.tracks)
    assert not any(track.record_enabled for track in state.tracks)
    assert not any(track.input_monitoring for track in state.tracks)
    assert not any(track.muted for track in state.tracks)
    assert not any(track.soloed for track in state.tracks)
    assert [track.name for track in state.tracks] == [
        f"Track {index}" for index in range(1, 9)
    ]


def test_mixer_restores_and_snapshots_all_persisted_controls() -> None:
    saved_track = TrackMixMetadata(
        name="Lead Vocal",
        level_db=-12.5,
        pan=0.75,
        record_enabled=True,
        input_monitoring=True,
        muted=True,
        soloed=True,
    )
    mix = MixerMetadata(
        tracks=(saved_track,)
        + tuple(
            TrackMixMetadata(name=f"Track {index}")
            for index in range(2, 9)
        ),
        bus_level_db=-4.5,
    )
    routes = (99,) + (None,) * 7

    state = MixerState.from_metadata(routes, mix)

    assert state.tracks[0].input_assigned is True
    assert state.tracks[0].level_db == -12.5
    assert state.tracks[0].pan == 0.75
    assert state.tracks[0].record_enabled is True
    assert state.tracks[0].input_monitoring is True
    assert state.tracks[0].muted is True
    assert state.tracks[0].soloed is True
    assert state.tracks[0].name == "Lead Vocal"
    assert state.bus_level_db == -4.5
    assert state.to_metadata(routes) == mix


def test_restoring_unassigned_track_clears_only_input_controls() -> None:
    saved_track = TrackMixMetadata(
        name="Track 1",
        record_enabled=True,
        input_monitoring=True,
        muted=True,
        soloed=True,
    )
    mix = MixerMetadata(
        tracks=(saved_track,) + (TrackMixMetadata(),) * 7
    )

    state = MixerState.from_metadata((None,) * 8, mix)
    track = state.tracks[0]

    assert track.record_enabled is False
    assert track.input_monitoring is False
    assert track.muted is True
    assert track.soloed is True
    assert state.to_metadata().tracks[0] == TrackMixMetadata(
        muted=True, soloed=True, name="Track 1"
    )


def test_input_only_controls_require_a_route() -> None:
    state = MixerState.from_track_inputs(
        (0, None, None, None, None, None, None, None)
    )

    assert state.toggle(0, "record_enabled") is True
    assert state.toggle(0, "input_monitoring") is True
    assert state.toggle(1, "record_enabled") is False
    assert state.toggle(1, "input_monitoring") is False
    assert state.toggle(1, "muted") is True
    assert state.toggle(1, "soloed") is True


def test_route_removal_clears_record_and_monitor_only() -> None:
    state = MixerState.from_track_inputs(
        (12, None, None, None, None, None, None, None)
    )
    state.toggle(0, "record_enabled")
    state.toggle(0, "input_monitoring")
    state.toggle(0, "muted")
    state.toggle(0, "soloed")

    state.update_input_routes((None,) * 8)

    track = state.tracks[0]
    assert track.input_assigned is False
    assert track.record_enabled is False
    assert track.input_monitoring is False
    assert track.muted is True
    assert track.soloed is True


def test_portable_channel_number_counts_as_assigned() -> None:
    state = MixerState.from_track_inputs(
        (99, None, None, None, None, None, None, None)
    )

    assert state.tracks[0].input_assigned is True
    assert state.toggle(0, "record_enabled") is True


def test_stereo_bus_route_can_record_but_cannot_input_monitor() -> None:
    saved_track = TrackMixMetadata(
        record_enabled=True, input_monitoring=True
    )
    state = MixerState.from_metadata(
        (StereoBusInput.LEFT,) + (None,) * 7,
        MixerMetadata(
            tracks=(saved_track,) + (TrackMixMetadata(),) * 7
        ),
    )
    track = state.tracks[0]

    assert track.input_assigned is True
    assert track.input_monitorable is False
    assert track.record_enabled is True
    assert track.input_monitoring is False
    assert state.toggle(0, "record_enabled") is False
    assert state.toggle(0, "record_enabled") is True
    assert state.toggle(0, "input_monitoring") is False
    assert state.to_metadata().tracks[0].input_monitoring is False


def test_multiple_tracks_can_solo_independently() -> None:
    state = MixerState()

    state.toggle(0, "soloed")
    state.toggle(1, "soloed")

    assert state.tracks[0].soloed is True
    assert state.tracks[1].soloed is True


def test_mixer_values_are_clamped() -> None:
    state = MixerState()

    assert state.set_track_level(0, -100) == MIN_LEVEL_DB
    assert state.set_track_level(1, 20) == MAX_LEVEL_DB
    assert state.set_bus_level(20) == MAX_LEVEL_DB
    assert state.set_pan(0, -2) == -1
    assert state.set_pan(1, 2) == 1


def test_fader_coordinates_cover_the_level_range() -> None:
    assert VerticalFader.value_from_y(12) == pytest.approx(MAX_LEVEL_DB)
    assert VerticalFader.value_from_y(214) == pytest.approx(MIN_LEVEL_DB)
    assert VerticalFader.value_from_y(-20) == pytest.approx(MAX_LEVEL_DB)
    assert VerticalFader.value_from_y(300) == pytest.approx(MIN_LEVEL_DB)
    assert level_y(MIN_LEVEL_DB) == pytest.approx(214)
    assert level_y(0) == pytest.approx(30.363636)
    assert level_y(MAX_LEVEL_DB) == pytest.approx(12)


def test_fader_meter_clamps_levels_and_redraws_only_for_changes() -> None:
    redraws: list[object] = []
    fader = VerticalFader.__new__(VerticalFader)
    fader.meter_channels = 2
    fader.meter_levels = (MIN_LEVEL_DB, MIN_LEVEL_DB)
    fader.canvas = object()
    fader._draw = lambda canvas: redraws.append(canvas)

    fader.set_meter_levels((-80.0, 4.0))
    fader.set_meter_levels((-80.0, 4.0))

    assert fader.meter_levels == (MIN_LEVEL_DB, 0.0)
    assert redraws == [fader.canvas]
    with pytest.raises(ValueError, match="requires 2"):
        fader.set_meter_levels((-6.0,))


def test_meter_draws_green_yellow_and_red_threshold_segments() -> None:
    class FillContext:
        def __init__(self, canvas: "FakeCanvas", color: str) -> None:
            self.canvas = canvas
            self.color = color

        def __enter__(self) -> None:
            self.canvas.color = self.color

        def __exit__(self, *args: object) -> None:
            self.canvas.color = None

    class FakeCanvas:
        def __init__(self) -> None:
            self.color: str | None = None
            self.rectangles: list[tuple[str | None, float]] = []

        def fill(self, *, color: str) -> FillContext:
            return FillContext(self, color)

        def rect(
            self, x: float, y: float, width: float, height: float
        ) -> None:
            self.rectangles.append((self.color, height))

    yellow_canvas = FakeCanvas()
    VerticalFader._draw_meter(yellow_canvas, 0, 5, -6)
    assert [color for color, _ in yellow_canvas.rectangles] == [
        METER_OFF,
        METER_GREEN,
        METER_YELLOW,
    ]

    red_canvas = FakeCanvas()
    VerticalFader._draw_meter(red_canvas, 0, 5, -1)
    assert [color for color, _ in red_canvas.rectangles] == [
        METER_OFF,
        METER_GREEN,
        METER_YELLOW,
        METER_RED,
    ]
    assert all(height > 0 for _, height in red_canvas.rectangles)


def test_pan_vertical_drag_and_level_readouts() -> None:
    assert PanKnob.value_from_drag(0, 100, 25) == 1
    assert PanKnob.value_from_drag(0, 100, 175) == -1
    assert format_level_db(MIN_LEVEL_DB) == "−∞ dB"
    assert format_level_db(0) == "+0.0 dB"


def test_invalid_mixer_route_or_control_is_rejected() -> None:
    state = MixerState()

    with pytest.raises(ValueError, match="8 tracks"):
        state.update_input_routes((0,))
    with pytest.raises(ValueError, match="Unknown mixer control"):
        state.toggle(0, "phase")


def test_mixer_changes_notify_the_audio_engine() -> None:
    state = MixerState.from_track_inputs(
        (0, None, None, None, None, None, None, None)
    )
    notifications: list[None] = []
    state.on_change = lambda: notifications.append(None)

    state.set_track_level(0, -3)
    state.set_pan(0, 0.5)
    state.set_bus_level(-6)
    state.toggle(0, "input_monitoring")
    state.toggle(0, "muted")

    assert len(notifications) == 5


def test_track_names_are_normalized_and_notify_only_when_changed() -> None:
    state = MixerState()
    notifications: list[None] = []
    state.on_change = lambda: notifications.append(None)

    assert state.set_track_name(0, "  Lead Vocal  ") == "Lead Vocal"
    assert state.set_track_name(0, "Lead Vocal") == "Lead Vocal"
    assert state.set_track_name(0, "   ") == "Track 1"
    assert state.set_track_name(1, "x" * 20) == "x" * 16

    assert state.tracks[0].name == "Track 1"
    assert state.tracks[1].name == "x" * 16
    assert len(notifications) == 3


def test_scribble_strip_limits_and_commits_inline_edits() -> None:
    state = MixerState()
    notifications: list[None] = []
    state.on_change = lambda: notifications.append(None)
    strip = TrackStrip.__new__(TrackStrip)
    strip.index = 0
    strip.mixer_state = state
    strip._normalizing_name = False
    interactions: list[str] = []
    strip.on_interaction = lambda: interactions.append("defocus")
    widget = SimpleNamespace(value="🎤" * 17)

    strip._limit_name(widget)
    assert widget.value == "🎤" * 16
    assert notifications == []

    widget.value = "  Vocals  "
    strip._commit_name(widget)
    assert widget.value == "Vocals"
    assert state.tracks[0].name == "Vocals"

    widget.value = ""
    strip._commit_name(widget)
    assert widget.value == "Track 1"
    assert state.tracks[0].name == "Track 1"
    assert interactions == []

    widget.value = "Guitar"
    strip._confirm_name(widget)
    assert state.tracks[0].name == "Guitar"
    assert interactions == ["defocus"]
    assert len(notifications) == 3


def test_mixer_controls_request_defocus_before_their_actions() -> None:
    events: list[object] = []
    fader = VerticalFader.__new__(VerticalFader)
    fader.on_interaction = lambda: events.append("defocus")
    fader.set_value = lambda value: events.append(("fader", value))

    fader._start_move(SimpleNamespace(), 0, 214)

    assert events == ["defocus", ("fader", MIN_LEVEL_DB)]

    events.clear()
    pan = PanKnob.__new__(PanKnob)
    pan.on_interaction = lambda: events.append("defocus")
    pan.value = 0.25

    pan._start_drag(SimpleNamespace(), 0, 100)

    assert events == ["defocus"]
    assert pan._drag_start_y == 100
    assert pan._drag_start_value == 0.25

    events.clear()
    strip = TrackStrip.__new__(TrackStrip)
    strip.index = 0
    strip.on_interaction = lambda: events.append("defocus")
    strip.mixer_state = SimpleNamespace(
        toggle=lambda index, control: events.append((index, control))
    )
    strip.sync_controls = lambda: events.append("sync")

    strip._toggle_handler("muted")(SimpleNamespace())

    assert events == ["defocus", (0, "muted"), "sync"]


def test_mixer_view_clears_only_active_scribble_strip_focus() -> None:
    calls: list[object] = []
    native_window = SimpleNamespace(
        makeFirstResponder=lambda responder: calls.append(responder)
    )
    window = SimpleNamespace(_impl=SimpleNamespace(native=native_window))
    inactive_input = SimpleNamespace(
        _impl=SimpleNamespace(has_focus=False), window=window
    )
    active_input = SimpleNamespace(
        _impl=SimpleNamespace(has_focus=True), window=window
    )
    view = MixerView.__new__(MixerView)
    view.track_strips = [
        SimpleNamespace(name_input=inactive_input),
        SimpleNamespace(name_input=active_input),
    ]

    view._end_name_editing()

    assert calls == [None]

    active_input._impl.has_focus = False
    view._end_name_editing()
    assert calls == [None]


def test_scribble_strip_tab_order_cycles_through_all_tracks() -> None:
    native_inputs = [
        SimpleNamespace(nextKeyView=None) for _ in range(8)
    ]
    view = MixerView.__new__(MixerView)
    view.track_strips = [
        SimpleNamespace(
            name_input=SimpleNamespace(
                _impl=SimpleNamespace(native=native_input)
            )
        )
        for native_input in native_inputs
    ]

    view._configure_name_tab_order()

    assert [native.nextKeyView for native in native_inputs] == [
        *native_inputs[1:],
        native_inputs[0],
    ]


def test_clearing_monitoring_notifies_only_when_state_changes() -> None:
    state = MixerState.from_track_inputs(
        (0, None, None, None, None, None, None, None)
    )
    state.tracks[0].input_monitoring = True
    notifications: list[None] = []
    state.on_change = lambda: notifications.append(None)

    state.clear_monitoring()
    state.clear_monitoring()

    assert state.tracks[0].input_monitoring is False
    assert len(notifications) == 1


def test_record_enable_lock_is_only_a_temporary_transition_state() -> None:
    sync_calls: list[int] = []
    strips = [
        type(
            "FakeStrip",
            (),
            {
                "record_enable_locked": False,
                "sync_controls": lambda self: sync_calls.append(1),
            },
        )()
        for _ in range(8)
    ]
    view = MixerView.__new__(MixerView)
    view.track_strips = strips

    view.set_record_enable_locked(True)
    view.set_record_enable_locked(False)

    assert all(strip.record_enable_locked is False for strip in strips)
    assert len(sync_calls) == 16


def test_unavailable_monitoring_preserves_saved_monitor_state() -> None:
    state = MixerState.from_track_inputs((0,) + (None,) * 7)
    state.tracks[0].input_monitoring = True
    strips = [
        type(
            "FakeStrip",
            (),
            {
                "monitoring_available": True,
                "sync_controls": lambda self: None,
            },
        )()
        for _ in range(8)
    ]
    view = MixerView.__new__(MixerView)
    view.state = state
    view.track_strips = strips

    view.set_monitoring_available(False)

    assert all(strip.monitoring_available is False for strip in strips)
    assert state.tracks[0].input_monitoring is True


def test_mixer_view_distributes_track_and_stereo_bus_meter_levels() -> None:
    received: list[tuple[float, ...]] = []

    def fader() -> SimpleNamespace:
        return SimpleNamespace(
            set_meter_levels=lambda levels: received.append(levels)
        )

    view = MixerView.__new__(MixerView)
    view.track_strips = [
        SimpleNamespace(fader=fader()) for _ in range(8)
    ]
    view.bus_strip = SimpleNamespace(fader=fader())
    track_levels = tuple(-float(index) for index in range(8))

    view.set_meter_levels(track_levels, (-9.0, -12.0))

    assert received == [
        *((level,) for level in track_levels),
        (-9.0, -12.0),
    ]
    with pytest.raises(ValueError, match="8 tracks"):
        view.set_meter_levels((-1.0,), (-2.0, -3.0))
