"""Tests for the main-window audio summary."""

import asyncio
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from tape_machine.app import (
    ShuttleButton,
    TapeMachine,
    _LOGO_DISPLAY_HEIGHT,
    _LOGO_DISPLAY_WIDTH,
    _LOGO_RESOURCE,
    _fit_window_position,
    _tabular_number_font,
)
from tape_machine.audio import (
    AudioDevice,
    AudioSettings,
    DeviceReference,
    StereoBusInput,
)
from tape_machine.config import AppConfig, StoredAudioSettings, WindowPosition
from tape_machine.engine import AudioEngineError, MeterSnapshot
from tape_machine.mixer import MixerState
from tape_machine.project import ProjectMetadata
from tape_machine.transport import TransportMode


class FakeService:
    def __init__(
        self,
        settings: AudioSettings | None,
        input_name: str = "Interface",
        output_name: str = "Interface",
    ) -> None:
        self.current_settings = settings
        self.refresh_calls = 0
        self.validate_calls = 0
        self._input_device = AudioDevice(
            index=1,
            name=input_name,
            host_api="Test API",
            max_input_channels=8,
            max_output_channels=2,
            default_sample_rate=48_000,
        )
        self._output_device = AudioDevice(
            index=1,
            name=output_name,
            host_api="Test API",
            max_input_channels=8,
            max_output_channels=2,
            default_sample_rate=48_000,
        )

    def device(self, device_id: int, direction: str) -> AudioDevice | None:
        if device_id != 1:
            return None
        return self._input_device if direction == "input" else self._output_device

    def refresh_devices(self) -> None:
        self.refresh_calls += 1

    def validate(self, settings: AudioSettings) -> None:
        self.validate_calls += 1


class FakeLifecycleEngine:
    def __init__(self, *, fail_start: bool = False) -> None:
        self.running = True
        self.fail_start = fail_start
        self.stop_calls = 0
        self.starts: list[AudioSettings] = []

    def stop(self) -> None:
        self.stop_calls += 1
        self.running = False

    def start(
        self,
        settings: AudioSettings,
        mixer_state: MixerState,
        input_device: AudioDevice,
        output_device: AudioDevice,
    ) -> None:
        self.stop()
        self.starts.append(settings)
        if self.fail_start:
            raise AudioEngineError("Unable to start input monitoring: busy")
        self.running = True


class FakeProject:
    def __init__(self) -> None:
        self.metadata = ProjectMetadata()
        self.saved: list[ProjectMetadata] = []
        self.staged: list[ProjectMetadata] = []
        self.dirty = False

    def save(self, metadata: ProjectMetadata) -> None:
        self.saved.append(metadata)
        self.metadata = metadata
        self.dirty = False

    def stage_metadata(self, metadata: ProjectMetadata) -> None:
        self.staged.append(metadata)
        if metadata != self.metadata:
            self.metadata = metadata
            self.dirty = True


def summary_app(settings: AudioSettings | None) -> SimpleNamespace:
    app = SimpleNamespace(
        audio_service=FakeService(settings),
        status_line_label=SimpleNamespace(text=""),
        routing_status_link_visible=False,
        status_line_suffix_label=SimpleNamespace(text=""),
        project=None,
        audio_engine_error=None,
    )
    app._set_routing_status_link_visible = lambda visible: setattr(
        app, "routing_status_link_visible", visible
    )
    return app


def test_time_counter_uses_tabular_figures_in_a_proportional_font() -> None:
    from toga_cocoa.libs import (
        NSMutableDictionary,
        NSAttributedString,
        NSFont,
        NSFontAttributeName,
    )

    font = _tabular_number_font(NSFont.boldSystemFontOfSize(18))
    attributes = NSMutableDictionary.alloc().init()
    attributes[NSFontAttributeName] = font

    def width(text: str) -> float:
        attributed = NSAttributedString.alloc().initWithString(
            text, attributes=attributes
        )
        return attributed.size().width

    assert font.isFixedPitch() is False
    assert width("11:11.111") == pytest.approx(width("88:88.888"))
    assert width("00:00.000") == pytest.approx(width("12:34.567"))


def test_logo_resource_has_two_x_dimensions_and_alpha() -> None:
    data = _LOGO_RESOURCE.read_bytes()

    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack(">II", data[16:24])
    assert (width, height) == (
        _LOGO_DISPLAY_WIDTH * 2,
        _LOGO_DISPLAY_HEIGHT * 2,
    )
    assert data[25] == 6


def test_status_line_marks_incomplete_routing() -> None:
    app = summary_app(AudioSettings(1, 1, 48_000))

    TapeMachine._update_audio_summary(app)

    assert app.status_line_label.text == "Interface  •  48 kHz"
    assert app.routing_status_link_visible is True


def test_status_line_omits_message_when_all_routing_is_complete() -> None:
    app = summary_app(
        AudioSettings(1, 1, 48_000, tuple(range(8)), (0, 1))
    )

    TapeMachine._update_audio_summary(app)

    assert app.status_line_label.text == "Interface  •  48 kHz"
    assert app.routing_status_link_visible is False


def test_status_line_accepts_partial_input_and_output_routing() -> None:
    app = summary_app(
        AudioSettings(
            1,
            1,
            48_000,
            track_inputs=(None, 0, None, None, None, None, None, None),
            bus_outputs=(1, None),
        )
    )

    TapeMachine._update_audio_summary(app)

    assert app.status_line_label.text == "Interface  •  48 kHz"
    assert app.routing_status_link_visible is False


def test_status_line_counts_stereo_bus_loopback_as_an_input_route() -> None:
    app = summary_app(
        AudioSettings(
            1,
            1,
            48_000,
            (StereoBusInput.LEFT,) + (None,) * 7,
            (0, 1),
        )
    )

    TapeMachine._update_audio_summary(app)

    assert app.status_line_label.text == "Interface  •  48 kHz"
    assert app.routing_status_link_visible is False


def test_status_line_marks_unrouted_stereo_bus_as_incomplete() -> None:
    app = summary_app(
        AudioSettings(
            1,
            1,
            48_000,
            track_inputs=(0, None, None, None, None, None, None, None),
        )
    )

    TapeMachine._update_audio_summary(app)

    assert app.status_line_label.text == "Interface  •  48 kHz"
    assert app.routing_status_link_visible is True


def test_status_line_separates_different_device_names() -> None:
    app = summary_app(
        AudioSettings(1, 1, 48_000, tuple(range(8)), (0, 1))
    )
    app.audio_service = FakeService(
        app.audio_service.current_settings,
        input_name="Studio Input",
        output_name="Monitor Output",
    )

    TapeMachine._update_audio_summary(app)

    assert app.status_line_label.text == (
        "Studio Input / Monitor Output  •  48 kHz"
    )


def test_status_line_handles_missing_audio_configuration() -> None:
    app = summary_app(None)

    TapeMachine._update_audio_summary(app)

    assert app.status_line_label.text == (
        "No audio device  •  Sample rate unavailable"
    )
    assert app.routing_status_link_visible is True


def test_status_line_accepts_mappings_beyond_available_device_channels() -> None:
    app = summary_app(
        AudioSettings(
            1,
            1,
            48_000,
            track_inputs=(8, 1, 2, 3, 4, 5, 6, 7),
            bus_outputs=(0, 1),
        )
    )

    TapeMachine._update_audio_summary(app)

    assert app.status_line_label.text == "Interface  •  48 kHz"
    assert app.routing_status_link_visible is False


def test_status_line_marks_project_audio_engine_failure() -> None:
    app = summary_app(
        AudioSettings(1, 1, 48_000, tuple(range(8)), (0, 1))
    )
    app.project = object()
    app.audio_engine_error = "Device busy"

    TapeMachine._update_audio_summary(app)

    assert app.status_line_label.text == "Interface  •  48 kHz"
    assert app.routing_status_link_visible is False
    assert app.status_line_suffix_label.text == "  •  Audio unavailable"


def test_status_link_hides_after_entering_a_fully_routed_project() -> None:
    app = summary_app(AudioSettings(1, 1, 48_000))
    TapeMachine._update_audio_summary(app)
    assert app.routing_status_link_visible is True

    app.project = SimpleNamespace(
        sample_rate=48_000,
        metadata=ProjectMetadata(),
    )
    app.audio_service.current_settings = AudioSettings(
        1, 1, 48_000, tuple(range(8)), (0, 1)
    )

    TapeMachine._update_audio_summary(app)

    assert app.routing_status_link_visible is False


def test_routing_status_link_opens_audio_settings() -> None:
    calls: list[None] = []
    app = SimpleNamespace(preferences=lambda: calls.append(None))

    TapeMachine._open_routing_settings(app)

    assert calls == [None]


def test_project_preferences_use_global_settings_at_project_rate() -> None:
    opened: list[tuple[object, dict[str, object]]] = []
    global_settings = AudioSettings(
        1, 2, 96_000, tuple(range(8)), (0, 1), 512
    )
    active_settings = AudioSettings(
        1, 2, 48_000, tuple(range(8)), (0, 1), 512
    )
    app = SimpleNamespace(
        project=SimpleNamespace(sample_rate=48_000),
        audio_service=SimpleNamespace(current_settings=active_settings),
        global_audio_settings=global_settings,
        audio_engine=SimpleNamespace(running=True),
        settings_window=SimpleNamespace(
            open=lambda draft, **kwargs: opened.append((draft, kwargs))
        ),
    )

    TapeMachine.preferences(app)

    draft, kwargs = opened[0]
    assert draft.buffer_size == 512
    assert draft.track_inputs == tuple(range(8))
    assert draft.sample_rate == 48_000
    assert kwargs["locked_sample_rate"] == 48_000


def test_project_uses_wav_rate_without_changing_global_sample_rate() -> None:
    global_settings = AudioSettings(
        1, 1, 96_000, tuple(range(8)), (0, 1), 256
    )
    compatibility_checks: list[AudioSettings] = []
    service = SimpleNamespace(
        current_settings=global_settings,
        refresh_devices=lambda: None,
        suggest_settings=lambda preferred: preferred,
        compatibility_error=lambda settings: compatibility_checks.append(
            settings
        ),
    )
    app = SimpleNamespace(
        project=SimpleNamespace(sample_rate=44_100),
        audio_service=service,
        global_audio_settings=global_settings,
    )

    error = TapeMachine._resolve_project_audio(app)

    assert error is None
    assert app.global_audio_settings == global_settings
    assert service.current_settings.sample_rate == 44_100
    assert service.current_settings.track_inputs == global_settings.track_inputs
    assert compatibility_checks == [service.current_settings]


def test_routing_status_link_is_inserted_and_removed_for_state_changes() -> None:
    link = object()
    inserted: list[tuple[int, object]] = []
    removed: list[object] = []
    app = SimpleNamespace(
        _routing_status_link_visible=False,
        routing_status_link_group=link,
        status_line_content=SimpleNamespace(
            insert=lambda index, child: inserted.append((index, child)),
            remove=removed.append,
        ),
    )

    TapeMachine._set_routing_status_link_visible(app, True)
    TapeMachine._set_routing_status_link_visible(app, True)
    TapeMachine._set_routing_status_link_visible(app, False)

    assert inserted == [(1, link)]
    assert removed == [link]


def test_mixer_change_rebuilds_running_engine_matrix() -> None:
    calls: list[object] = []
    arm_calls: list[tuple[bool, ...]] = []
    routes = (None, None, 2, None, None, None, None, None)
    state = MixerState.from_track_inputs(routes)
    state.tracks[2].record_enabled = True
    state.tracks[2].name = "Harmony"
    project = FakeProject()
    app = SimpleNamespace(
        mixer_state=state,
        project=project,
        transport=SimpleNamespace(
            set_armed_tracks=lambda armed: arm_calls.append(armed)
        ),
        audio_engine=SimpleNamespace(
            running=True,
            update_mix=lambda mixer_state: calls.append(mixer_state),
        ),
        audio_service=SimpleNamespace(
            current_settings=AudioSettings(
                1, 1, 48_000, routes, (0, 1)
            )
        ),
    )

    TapeMachine._mixer_changed(app)

    assert calls == [state]
    assert arm_calls == [(False, False, True, False, False, False, False, False)]
    assert project.dirty is True
    assert project.staged[-1].mix.tracks[2].record_enabled is True
    assert project.staged[-1].mix.tracks[2].name == "Harmony"


def test_meter_sync_distributes_the_latest_engine_snapshot() -> None:
    calls: list[tuple[tuple[float, ...], tuple[float, float]]] = []
    snapshot = MeterSnapshot(
        tuple(-float(index) for index in range(8)),
        (-9.0, -12.0),
    )
    app = SimpleNamespace(
        audio_engine=SimpleNamespace(meter_snapshot=snapshot),
        mixer_view=SimpleNamespace(
            set_meter_levels=lambda tracks, bus: calls.append(
                (tracks, bus)
            )
        ),
    )

    TapeMachine._sync_meters(app)

    assert calls == [(snapshot.track_db, snapshot.bus_db)]


def test_audio_failure_stops_engine_and_disables_monitoring() -> None:
    events: list[object] = []
    app = SimpleNamespace(
        audio_engine=SimpleNamespace(stop=lambda: events.append("stop")),
        audio_engine_error=None,
        mixer_view=SimpleNamespace(
            set_monitoring_available=lambda available: events.append(available)
        ),
        _update_audio_summary=lambda: events.append("summary"),
        _schedule_audio_engine_dialog=lambda message: events.append(message),
        _sync_transport_controls=lambda: None,
    )

    TapeMachine._set_audio_engine_failure(app, "Device busy")

    assert app.audio_engine_error == "Device busy"
    assert events == ["stop", False, "summary", "Device busy"]


def test_project_audio_settings_persist_globally_after_stream_starts() -> None:
    old_settings = AudioSettings(1, 1, 48_000)
    global_settings = AudioSettings(1, 1, 96_000)
    new_settings = AudioSettings(
        1,
        1,
        48_000,
        (0, None, None, None, None, None, None, None),
        (0, 1),
        256,
    )
    service = FakeService(old_settings)
    engine = FakeLifecycleEngine()
    project = FakeProject()
    mixer_state = MixerState()
    mixer_state.tracks[0].level_db = -8.0
    mixer_state.tracks[0].muted = True
    mixer_state.bus_level_db = -3.0
    view_events: list[object] = []
    saved_configs: list[AppConfig] = []
    app = object.__new__(TapeMachine)
    app.project = project
    app.audio_service = service
    app.audio_engine = engine
    app.mixer_state = mixer_state
    app.mixer_view = SimpleNamespace(
        update_track_routes=lambda routes: view_events.append(routes),
        set_monitoring_available=lambda available: view_events.append(
            available
        ),
    )
    app.global_audio_settings = global_settings
    app.app_config = AppConfig()
    app.config_store = SimpleNamespace(save=saved_configs.append)
    app.project_audio_error = "old error"
    app.audio_engine_error = "old error"
    app._update_audio_summary = lambda: view_events.append("summary")

    TapeMachine._project_audio_settings_applied(app, new_settings)

    assert engine.stop_calls == 1
    assert engine.starts == [new_settings]
    assert project.saved == []
    assert project.staged[-1].mix.tracks[0].level_db == -8.0
    assert project.staged[-1].mix.tracks[0].muted is True
    assert project.staged[-1].mix.bus_level_db == -3.0
    assert saved_configs == [app.app_config]
    assert app.app_config.audio_settings is not None
    assert app.app_config.audio_settings.sample_rate == 96_000
    assert app.app_config.audio_settings.buffer_size == 256
    assert app.global_audio_settings.sample_rate == 96_000
    assert app.global_audio_settings.track_inputs == new_settings.track_inputs
    assert service.current_settings == new_settings
    assert app.project_audio_error is None
    assert app.audio_engine_error is None
    assert view_events == [new_settings.track_inputs, True, "summary"]


def test_project_audio_settings_clear_input_states_for_removed_route() -> None:
    old_settings = AudioSettings(
        1,
        1,
        48_000,
        (0,) + (None,) * 7,
        (0, 1),
    )
    new_settings = AudioSettings(1, 1, 48_000, (None,) * 8, (0, 1))
    service = FakeService(old_settings)
    engine = FakeLifecycleEngine()
    project = FakeProject()
    state = MixerState.from_track_inputs(old_settings.track_inputs)
    state.tracks[0].record_enabled = True
    state.tracks[0].input_monitoring = True
    state.tracks[0].muted = True
    app = object.__new__(TapeMachine)
    app.project = project
    app.audio_service = service
    app.audio_engine = engine
    app.mixer_state = state
    app.mixer_view = SimpleNamespace(
        update_track_routes=lambda routes: None,
        set_monitoring_available=lambda available: None,
    )
    app.global_audio_settings = old_settings
    app.app_config = AppConfig()
    app.config_store = SimpleNamespace(save=lambda config: None)
    app.project_audio_error = None
    app.audio_engine_error = None
    app._update_audio_summary = lambda: None

    TapeMachine._project_audio_settings_applied(app, new_settings)

    saved_track = project.staged[-1].mix.tracks[0]
    assert saved_track.record_enabled is False
    assert saved_track.input_monitoring is False
    assert saved_track.muted is True


def test_saving_unchanged_project_settings_keeps_active_stream() -> None:
    current_settings = AudioSettings(1, 1, 48_000)
    service = FakeService(current_settings)
    engine = FakeLifecycleEngine()
    project = FakeProject()
    saved_configs: list[AppConfig] = []
    app = object.__new__(TapeMachine)
    app.project = project
    app.audio_service = service
    app.audio_engine = engine
    app.mixer_state = MixerState()
    app.mixer_view = None
    app.global_audio_settings = current_settings
    app.app_config = AppConfig()
    app.config_store = SimpleNamespace(save=saved_configs.append)
    app.project_audio_error = None
    app.audio_engine_error = None
    app._update_audio_summary = lambda: None

    TapeMachine._project_audio_settings_applied(app, current_settings)

    assert service.refresh_calls == 0
    assert service.validate_calls == 0
    assert engine.stop_calls == 0
    assert engine.starts == []
    assert saved_configs == [app.app_config]


def test_failed_candidate_stream_keeps_project_metadata_and_old_settings() -> None:
    old_settings = AudioSettings(1, 1, 48_000)
    new_settings = AudioSettings(1, 1, 48_000, buffer_size=256)
    service = FakeService(old_settings)
    engine = FakeLifecycleEngine(fail_start=True)
    project = FakeProject()
    dialogs: list[str] = []
    saved_configs: list[AppConfig] = []
    app = object.__new__(TapeMachine)
    app.project = project
    app.audio_service = service
    app.audio_engine = engine
    app.mixer_state = MixerState()
    app.mixer_view = None
    app.global_audio_settings = old_settings
    app.app_config = AppConfig()
    app.config_store = SimpleNamespace(save=saved_configs.append)
    app._restore_audio_engine = lambda settings, was_running: True
    app._set_audio_engine_failure = lambda message, show_dialog=False: None
    app._schedule_audio_engine_dialog = lambda message: dialogs.append(message)

    with pytest.raises(AudioEngineError, match="busy"):
        TapeMachine._project_audio_settings_applied(app, new_settings)

    assert project.saved == []
    assert saved_configs == []
    assert service.current_settings == old_settings
    assert dialogs == ["Unable to start input monitoring: busy"]


def test_failed_global_config_save_restores_project_stream_and_settings() -> None:
    old_settings = AudioSettings(1, 1, 48_000)
    global_settings = AudioSettings(1, 1, 96_000)
    new_settings = AudioSettings(1, 1, 48_000, buffer_size=256)
    service = FakeService(old_settings)
    engine = FakeLifecycleEngine()
    project = FakeProject()
    original_config = AppConfig()
    app = object.__new__(TapeMachine)
    app.project = project
    app.audio_service = service
    app.audio_engine = engine
    app.mixer_state = MixerState()
    app.mixer_view = None
    app.global_audio_settings = global_settings
    app.app_config = original_config
    app.config_store = SimpleNamespace(
        save=lambda config: (_ for _ in ()).throw(RuntimeError("disk full"))
    )
    app.audio_engine_error = None
    app._update_audio_summary = lambda: None
    app._set_audio_engine_failure = lambda message: None

    with pytest.raises(RuntimeError, match="disk full"):
        TapeMachine._project_audio_settings_applied(app, new_settings)

    assert engine.starts == [new_settings, old_settings]
    assert engine.running is True
    assert service.current_settings == old_settings
    assert app.global_audio_settings == global_settings
    assert app.app_config is original_config
    assert project.staged == []


def test_project_commands_follow_open_project_state() -> None:
    recent_commands = [SimpleNamespace(enabled=False) for _ in range(2)]
    app = SimpleNamespace(
        project=None,
        new_project_command=SimpleNamespace(enabled=False),
        open_project_command=SimpleNamespace(enabled=False),
        save_project_command=SimpleNamespace(enabled=True),
        close_project_command=SimpleNamespace(enabled=True),
        settings_command=SimpleNamespace(enabled=True),
        transport=None,
        transport_starting=False,
        transport_stopping=False,
        momentary_shuttle_task=None,
        recent_file_commands=recent_commands,
    )

    TapeMachine._update_command_state(app)

    assert app.new_project_command.enabled is True
    assert app.open_project_command.enabled is True
    assert app.save_project_command.enabled is False
    assert app.close_project_command.enabled is False
    assert app.settings_command.enabled is True
    assert all(command.enabled is True for command in recent_commands)

    app.project = object()
    TapeMachine._update_command_state(app)

    assert app.new_project_command.enabled is False
    assert app.open_project_command.enabled is False
    assert app.save_project_command.enabled is True
    assert app.close_project_command.enabled is True
    assert all(command.enabled is False for command in recent_commands)

    app.transport = SimpleNamespace(running=True)
    TapeMachine._update_command_state(app)

    assert app.save_project_command.enabled is False
    assert app.close_project_command.enabled is False
    assert app.settings_command.enabled is False


def test_window_position_is_clamped_to_an_attached_screen() -> None:
    screens = [
        SimpleNamespace(
            origin=SimpleNamespace(x=0, y=0),
            size=SimpleNamespace(width=1920, height=1080),
        ),
        SimpleNamespace(
            origin=SimpleNamespace(x=-1280, y=0),
            size=SimpleNamespace(width=1280, height=1024),
        ),
    ]

    assert _fit_window_position(WindowPosition(-100, 900), (900, 640), screens) == (
        -900,
        384,
    )
    assert _fit_window_position(WindowPosition(3000, 2000), (912, 560), screens) == (
        1008,
        520,
    )
    assert _fit_window_position(None, (912, 560), screens) is None


def test_save_applies_and_persists_all_global_audio_settings() -> None:
    current = AudioSettings(1, 1, 48_000)
    candidate = AudioSettings(
        1, 1, 96_000, tuple(range(8)), (0, 1), 512
    )
    service = FakeService(current)
    saved: list[AppConfig] = []
    app = object.__new__(TapeMachine)
    app.audio_service = service
    app.app_config = AppConfig()
    app.config_store = SimpleNamespace(save=saved.append)
    app.project = None
    app.global_audio_settings = current
    summaries: list[None] = []
    app._update_audio_summary = lambda: summaries.append(None)

    TapeMachine._audio_settings_applied(app, candidate)

    assert service.refresh_calls == 1
    assert service.validate_calls == 1
    assert service.current_settings == candidate
    assert app.global_audio_settings == candidate
    assert saved == [app.app_config]
    assert app.app_config.audio_settings == StoredAudioSettings(
        input_device=service._input_device.reference,
        output_device=service._output_device.reference,
        sample_rate=96_000,
        track_inputs=tuple(range(8)),
        bus_outputs=(0, 1),
        buffer_size=512,
    )
    assert summaries == [None]


def test_failed_global_config_save_does_not_apply_audio_settings() -> None:
    current = AudioSettings(1, 1, 48_000)
    candidate = AudioSettings(1, 1, 96_000, buffer_size=512)
    service = FakeService(current)
    original_config = AppConfig()
    app = object.__new__(TapeMachine)
    app.audio_service = service
    app.app_config = original_config
    app.config_store = SimpleNamespace(
        save=lambda config: (_ for _ in ()).throw(RuntimeError("disk full"))
    )
    app.project = None
    app.global_audio_settings = current
    app._update_audio_summary = lambda: None

    with pytest.raises(RuntimeError, match="disk full"):
        TapeMachine._audio_settings_applied(app, candidate)

    assert service.current_settings == current
    assert app.global_audio_settings == current
    assert app.app_config is original_config


def test_recent_files_are_persisted_and_menu_is_rebuilt(tmp_path: Path) -> None:
    saved: list[AppConfig] = []
    rebuilds: list[None] = []
    app = object.__new__(TapeMachine)
    app.app_config = AppConfig()
    app.config_store = SimpleNamespace(save=saved.append)
    app._rebuild_recent_files_menu = lambda: rebuilds.append(None)
    project_path = tmp_path / "project.wav"

    TapeMachine._remember_recent_file(app, project_path)
    TapeMachine._remember_recent_file(app, project_path)
    TapeMachine.clear_recent_files(app)

    assert saved[0].recent_files == (project_path.resolve(),)
    assert saved[-1].recent_files == ()
    assert len(saved) == 2
    assert len(rebuilds) == 2


def test_window_positions_merge_with_existing_config() -> None:
    stored = StoredAudioSettings(
        DeviceReference("Input", "Core Audio"),
        DeviceReference("Output", "Core Audio"),
        48_000,
        (None,) * 8,
        (0, 1),
    )
    saved: list[AppConfig] = []
    app = object.__new__(TapeMachine)
    app.app_config = AppConfig(audio_settings=stored)
    app.config_store = SimpleNamespace(save=saved.append)
    app._main_window = SimpleNamespace(
        position=SimpleNamespace(x=120, y=80)
    )
    app.settings_window = SimpleNamespace(
        window=SimpleNamespace(position=SimpleNamespace(x=-700, y=140))
    )

    TapeMachine._save_window_positions(app)

    assert app.app_config.audio_settings is stored
    assert app.app_config.main_window_position == WindowPosition(120, 80)
    assert app.app_config.audio_settings_window_position == WindowPosition(-700, 140)
    assert saved == [app.app_config]


def test_transport_controls_show_toggle_stop_and_time_state() -> None:
    app = SimpleNamespace(
        transport=SimpleNamespace(
            running=True,
            mode=TransportMode.PLAYING,
            record_armed=True,
            position_frames=60_000,
        ),
        transport_starting=False,
        transport_stopping=False,
        momentary_shuttle_task=None,
        audio_engine=SimpleNamespace(running=True),
        project=SimpleNamespace(writable=True, sample_rate=48_000),
        transport_record_button=SimpleNamespace(
            enabled=False,
            text="",
            style=SimpleNamespace(background_color=None),
        ),
        transport_play_button=SimpleNamespace(enabled=True),
        transport_rewind_button=SimpleNamespace(
            enabled=False, active=False
        ),
        transport_fast_forward_button=SimpleNamespace(
            enabled=False, active=False
        ),
        transport_stop_rtz_button=SimpleNamespace(text="", enabled=False),
        transport_time_label=SimpleNamespace(text=""),
    )

    TapeMachine._sync_transport_controls(app)

    assert app.transport_record_button.enabled is True
    assert app.transport_record_button.text == "● REC"
    assert app.transport_play_button.enabled is False
    assert app.transport_rewind_button.enabled is False
    assert app.transport_fast_forward_button.enabled is False
    assert app.transport_stop_rtz_button.text == "■ STOP"
    assert app.transport_stop_rtz_button.enabled is True
    assert app.transport_time_label.text == "00:01.250"


def test_transport_controls_show_active_fast_forward_state() -> None:
    app = SimpleNamespace(
        transport=SimpleNamespace(
            running=True,
            mode=TransportMode.FAST_FORWARD,
            record_armed=False,
            position_frames=60_000,
        ),
        transport_starting=False,
        transport_stopping=False,
        momentary_shuttle_task=None,
        audio_engine=SimpleNamespace(running=True),
        project=SimpleNamespace(
            writable=True, sample_rate=48_000, frames=96_000
        ),
        transport_record_button=SimpleNamespace(
            enabled=True,
            text="",
            style=SimpleNamespace(background_color=None),
        ),
        transport_play_button=SimpleNamespace(enabled=True),
        transport_rewind_button=SimpleNamespace(
            enabled=False, active=False
        ),
        transport_fast_forward_button=SimpleNamespace(
            enabled=False, active=False
        ),
        transport_stop_rtz_button=SimpleNamespace(text="", enabled=False),
        transport_time_label=SimpleNamespace(text=""),
    )

    TapeMachine._sync_transport_controls(app)

    assert app.transport_record_button.enabled is False
    assert app.transport_play_button.enabled is False
    assert app.transport_rewind_button.enabled is True
    assert app.transport_fast_forward_button.enabled is True
    assert app.transport_fast_forward_button.active is True
    assert app.transport_rewind_button.active is False
    assert app.transport_stop_rtz_button.text == "■ STOP"
    assert app.transport_time_label.text == "00:01.250"


def test_switching_shuttle_direction_stops_before_restarting() -> None:
    events: list[str] = []

    class FakeTransport:
        mode = TransportMode.FAST_FORWARD

        @property
        def running(self) -> bool:
            return self.mode is not TransportMode.STOPPED

        def rewind(self) -> bool:
            events.append("rewind")
            self.mode = TransportMode.REWIND
            return True

        def fast_forward(self) -> bool:
            events.append("fast_forward")
            self.mode = TransportMode.FAST_FORWARD
            return True

    transport = FakeTransport()

    async def stop_transport() -> None:
        events.append("stop")
        transport.mode = TransportMode.STOPPED

    app = SimpleNamespace(
        transport=transport,
        audio_engine=SimpleNamespace(running=True),
        transport_starting=False,
        transport_stopping=False,
        mixer_view=None,
        _stop_transport=stop_transport,
        _update_command_state=lambda: None,
        _sync_transport_controls=lambda: None,
        _schedule_transport_dialog=lambda message: events.append(message),
    )

    asyncio.run(TapeMachine._toggle_shuttle(app, TransportMode.REWIND))

    assert events == ["stop", "rewind"]
    assert transport.mode is TransportMode.REWIND


def test_clicking_active_shuttle_button_stops_without_restarting() -> None:
    events: list[str] = []
    transport = SimpleNamespace(
        mode=TransportMode.FAST_FORWARD,
        running=True,
    )

    async def stop_transport() -> None:
        events.append("stop")
        transport.mode = TransportMode.STOPPED
        transport.running = False

    app = SimpleNamespace(
        transport=transport,
        audio_engine=SimpleNamespace(running=True),
        transport_starting=False,
        transport_stopping=False,
        mixer_view=None,
        _stop_transport=stop_transport,
    )

    asyncio.run(TapeMachine._toggle_shuttle(app, TransportMode.FAST_FORWARD))

    assert events == ["stop"]


def test_shuttle_controls_are_momentary_only_during_safe_playback() -> None:
    def control_state(record_armed: bool) -> tuple[bool, bool]:
        app = SimpleNamespace(
            transport=SimpleNamespace(
                running=True,
                mode=TransportMode.PLAYING,
                record_armed=record_armed,
                position_frames=48_000,
            ),
            transport_starting=False,
            transport_stopping=False,
            momentary_shuttle_task=None,
            audio_engine=SimpleNamespace(running=True),
            project=SimpleNamespace(
                writable=True, sample_rate=48_000, frames=96_000
            ),
            transport_record_button=SimpleNamespace(
                enabled=True,
                text="",
                style=SimpleNamespace(background_color=None),
            ),
            transport_play_button=SimpleNamespace(enabled=True),
            transport_rewind_button=SimpleNamespace(
                enabled=False, active=False
            ),
            transport_fast_forward_button=SimpleNamespace(
                enabled=False, active=False
            ),
            transport_stop_rtz_button=SimpleNamespace(
                text="", enabled=False
            ),
            transport_time_label=SimpleNamespace(text=""),
        )
        TapeMachine._sync_transport_controls(app)
        return (
            app.transport_rewind_button.enabled,
            app.transport_fast_forward_button.enabled,
        )

    assert control_state(record_armed=False) == (True, True)
    assert control_state(record_armed=True) == (False, False)


def test_momentary_shuttle_resumes_playback_on_release() -> None:
    events: list[object] = []
    transport = SimpleNamespace(
        mode=TransportMode.PLAYING,
        running=True,
        record_armed=False,
        terminal_error=None,
    )
    release = asyncio.Event()

    async def stop_transport() -> None:
        events.append("stop")
        transport.mode = TransportMode.STOPPED
        transport.running = False

    async def toggle_shuttle(mode: TransportMode) -> None:
        events.append(mode)
        transport.mode = mode
        transport.running = True

    async def play_transport() -> None:
        events.append("play")
        transport.mode = TransportMode.PLAYING
        transport.running = True

    app = SimpleNamespace(
        transport=transport,
        momentary_shuttle_task=None,
        momentary_shuttle_mode=TransportMode.FAST_FORWARD,
        momentary_shuttle_release=release,
        momentary_shuttle_resume=True,
        _stop_transport=stop_transport,
        _toggle_shuttle=toggle_shuttle,
        _play_transport=play_transport,
        _sync_transport_controls=lambda: None,
    )

    async def exercise() -> None:
        task = asyncio.create_task(
            TapeMachine._run_momentary_shuttle(
                app, TransportMode.FAST_FORWARD, release
            )
        )
        app.momentary_shuttle_task = task
        while transport.mode is not TransportMode.FAST_FORWARD:
            await asyncio.sleep(0)
        release.set()
        await task

    asyncio.run(exercise())

    assert events == [
        "stop",
        TransportMode.FAST_FORWARD,
        "stop",
        "play",
    ]
    assert transport.mode is TransportMode.PLAYING


def test_shuttle_button_uses_native_button_mouse_events(monkeypatch) -> None:
    events: list[str] = []
    current_event = SimpleNamespace(type=ShuttleButton._MOUSE_DOWN_EVENT)
    native = SimpleNamespace(
        window=SimpleNamespace(currentEvent=lambda: current_event),
        sendActionOn=lambda mask: events.append(f"mask:{mask}"),
    )

    class FakeButton:
        def __init__(self, text, on_press, width, height) -> None:
            self.text = text
            self.on_press = on_press
            self.style = SimpleNamespace(background_color=None)
            self.enabled = True
            self._impl = SimpleNamespace(native=native)

    monkeypatch.setattr("tape_machine.app.toga.Button", FakeButton)
    button = ShuttleButton(
        "Rewind",
        lambda: events.append("press"),
        lambda: events.append("release"),
        width=72,
    )

    button.widget.on_press(button.widget)
    current_event.type = ShuttleButton._MOUSE_UP_EVENT
    button.widget.on_press(button.widget)

    assert events == [f"mask:{ShuttleButton._MOUSE_EVENT_MASK}", "press", "release"]
    assert button.widget.text == "Rewind"


def test_shuttle_button_delivers_release_after_becoming_disabled() -> None:
    events: list[str] = []
    button = ShuttleButton.__new__(ShuttleButton)
    button.on_press = lambda: events.append("press")
    button.on_release = lambda: events.append("release")
    button._enabled = True
    button._pointer_down = False
    button.widget = SimpleNamespace(
        enabled=True,
        style=SimpleNamespace(background_color=None),
    )

    button._press(button.widget)
    button.enabled = False
    button._release(button.widget)

    assert events == ["press", "release"]
    assert button.widget.enabled is False
