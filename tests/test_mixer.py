"""Tests for the session-only project mixer."""

import pytest

from tape_machine.mixer import (
    MAX_LEVEL_DB,
    MIN_LEVEL_DB,
    MixerState,
    PanKnob,
    VerticalFader,
    format_level_db,
)


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
