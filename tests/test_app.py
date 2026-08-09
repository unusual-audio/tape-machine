"""Tests for the main-window audio summary."""

import asyncio
import struct
from pathlib import Path
from threading import get_ident
from types import SimpleNamespace

import pytest

import tape_machine.app as app_module
from tape_machine.app import (
    ShuttleButton,
    TapeMachine,
    _LOGO_DISPLAY_HEIGHT,
    _LOGO_DISPLAY_WIDTH,
    _LOGO_RESOURCE,
    _display_parent_path,
    _fit_window_position,
    _tabular_number_font,
)
from tape_machine.audio import (
    AudioConfigurationError,
    AudioDevice,
    AudioSettings,
    DeviceReference,
    RoutingStatus,
    StereoBusInput,
)
from tape_machine.config import (
    MAX_RECENT_FILES,
    AppConfig,
    StoredAudioSettings,
    WindowPosition,
)
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


def test_recent_project_parent_paths_abbreviate_the_home_directory() -> None:
    home = Path.home()

    assert _display_parent_path(home / "project.wav") == "~"
    assert (
        _display_parent_path(home / "Music" / "Sessions" / "project.wav")
        == "~/Music/Sessions"
    )


def test_initial_screen_uses_left_aligned_logo_and_quiet_recent_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeWidget:
        def __init__(
            self,
            kind: str,
            *args: object,
            children: list[object] | None = None,
            **kwargs: object,
        ) -> None:
            self.kind = kind
            self.args = args
            self.children = list(children or ())
            self.options = kwargs

    monkeypatch.setattr(
        app_module.toga,
        "Box",
        lambda *args, **kwargs: FakeWidget("box", *args, **kwargs),
    )
    monkeypatch.setattr(
        app_module.toga,
        "Label",
        lambda *args, **kwargs: FakeWidget("label", *args, **kwargs),
    )
    monkeypatch.setattr(
        app_module.toga,
        "ImageView",
        lambda *args, **kwargs: FakeWidget("image", *args, **kwargs),
    )
    monkeypatch.setattr(
        app_module,
        "_link_button",
        lambda *args, **kwargs: FakeWidget("link", *args, **kwargs),
    )

    recent_files = tuple(
        Path.home() / "Music" / f"Session {index:02d}.wav"
        for index in range(MAX_RECENT_FILES + 2)
    )
    footer = FakeWidget("footer")
    app = SimpleNamespace(
        app_config=AppConfig(recent_files=recent_files),
        new_project=object(),
        open_project=object(),
        _recent_file_action=lambda path: path,
        _build_status_footer=lambda: footer,
    )
    app._initial_recent_file_widgets = (
        lambda: TapeMachine._initial_recent_file_widgets(app)
    )

    screen = TapeMachine._build_initial_screen(app)

    central_content = screen.children[0]
    logo, actions, recent_section = central_content.children
    assert logo.args == (_LOGO_RESOURCE,)
    assert logo.options["width"] == _LOGO_DISPLAY_WIDTH
    assert logo.options["height"] == _LOGO_DISPLAY_HEIGHT
    assert central_content.options["align_items"] == app_module.START
    assert central_content.options["justify_content"] == app_module.START
    assert central_content.options["margin_top"] == 16
    assert central_content.options["margin_left"] == 16
    assert central_content.options["margin_right"] == 16
    assert central_content.options["margin_bottom"] == 12
    assert [link.args[0] for link in actions.children] == [
        "New Project",
        "Open Project…",
    ]
    assert all(link.options["font_size"] == 11 for link in actions.children)
    assert recent_section.children[0].args == ("Recent Projects",)
    assert screen.children[1] is footer

    rows = app.initial_recent_files_box.children
    assert len(rows) == MAX_RECENT_FILES
    assert all(row.options["width"] == 640 for row in rows)
    assert all(row.children[0].options["font_size"] == 11 for row in rows)
    assert all(row.children[1].options["font_size"] == 10 for row in rows)

    app.app_config = AppConfig()
    empty_state = TapeMachine._initial_recent_file_widgets(app)
    assert empty_state[0].args == ("No recent projects",)
    assert empty_state[0].options["font_size"] == 11


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


def test_status_line_warns_about_mappings_beyond_available_device_channels() -> None:
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
    assert app.routing_status_link_visible is True


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
    async def preferences() -> None:
        calls.append(None)

    app = SimpleNamespace(preferences=preferences)

    asyncio.run(TapeMachine._open_routing_settings(app))

    assert calls == [None]


def test_project_preferences_use_global_settings_at_project_rate() -> None:
    opened: list[tuple[object, dict[str, object]]] = []
    global_settings = AudioSettings(
        1, 2, 96_000, tuple(range(8)), (0, 1), 512
    )
    active_settings = AudioSettings(
        1, 2, 48_000, tuple(range(8)), (0, 1), 512
    )
    async def open_settings(draft, **kwargs) -> None:
        opened.append((draft, kwargs))

    app = SimpleNamespace(
        project=SimpleNamespace(sample_rate=48_000),
        audio_service=SimpleNamespace(current_settings=active_settings),
        global_audio_settings=global_settings,
        audio_engine=SimpleNamespace(running=True),
        settings_window=SimpleNamespace(open=open_settings),
    )

    asyncio.run(TapeMachine.preferences(app))

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


def test_routing_status_uses_distinct_degraded_and_incomplete_links(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    styled: list[str] = []
    visible: list[bool] = []
    link = SimpleNamespace(text="", style=SimpleNamespace(color=""))
    app = SimpleNamespace(
        _routing_status=RoutingStatus.COMPLETE,
        routing_status_link=link,
        _set_routing_status_link_visible=visible.append,
    )
    monkeypatch.setattr(
        app_module,
        "_style_status_link",
        lambda button, color: styled.append(color),
    )

    TapeMachine._set_routing_status(app, RoutingStatus.DEGRADED)
    assert link.text == "Some routes unavailable"
    assert link.style.color == app_module.ACCENT_ORANGE

    TapeMachine._set_routing_status(app, RoutingStatus.INCOMPLETE)
    assert link.text == "Routing incomplete"
    assert link.style.color == app_module.ACCENT_RED
    assert styled == [app_module.ACCENT_ORANGE, app_module.ACCENT_RED]
    assert visible == [True, True]


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


def test_audio_failure_disables_monitoring_after_coordinated_stop() -> None:
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
    assert events == [False, "summary", "Device busy"]


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
    probe_states: list[tuple[str, bool]] = []

    def refresh_devices() -> None:
        service.refresh_calls += 1
        probe_states.append(("refresh", engine.running))

    def validate(settings: AudioSettings) -> None:
        service.validate_calls += 1
        probe_states.append(("validate", engine.running))

    service.refresh_devices = refresh_devices
    service.validate = validate
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

    asyncio.run(TapeMachine._project_audio_settings_applied(app, new_settings))

    assert engine.stop_calls == 2
    assert engine.starts == [new_settings]
    assert probe_states == [("refresh", False), ("validate", False)]
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

    asyncio.run(TapeMachine._project_audio_settings_applied(app, new_settings))

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

    asyncio.run(
        TapeMachine._project_audio_settings_applied(app, current_settings)
    )

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
    async def restored(settings, was_running) -> bool:
        return True

    app._restore_audio_engine = restored
    app._set_audio_engine_failure = lambda message, show_dialog=False: None
    app._schedule_audio_engine_dialog = lambda message: dialogs.append(message)

    with pytest.raises(AudioEngineError, match="busy"):
        asyncio.run(
            TapeMachine._project_audio_settings_applied(app, new_settings)
        )

    assert project.saved == []
    assert saved_configs == []
    assert service.current_settings == old_settings
    assert dialogs == ["Unable to start input monitoring: busy"]


def test_failed_audio_probe_restores_stream_after_stopping_it() -> None:
    old_settings = AudioSettings(1, 1, 48_000)
    new_settings = AudioSettings(1, 1, 48_000, buffer_size=256)
    service = FakeService(old_settings)
    service.validate = lambda settings: (_ for _ in ()).throw(
        AudioConfigurationError("Core Audio rejected the route")
    )
    engine = FakeLifecycleEngine()
    restores: list[tuple[AudioSettings | None, bool]] = []
    app = object.__new__(TapeMachine)
    app.project = FakeProject()
    app.audio_service = service
    app.audio_engine = engine
    app.mixer_state = MixerState()
    app.mixer_view = None
    app.global_audio_settings = old_settings
    app.app_config = AppConfig()

    async def restore(
        settings: AudioSettings | None, was_running: bool
    ) -> bool:
        restores.append((settings, was_running))
        engine.running = True
        return True

    app._restore_audio_engine = restore
    app._set_audio_engine_failure = lambda *args, **kwargs: None

    with pytest.raises(
        AudioConfigurationError, match="Core Audio rejected the route"
    ):
        asyncio.run(
            TapeMachine._project_audio_settings_applied(app, new_settings)
        )

    assert engine.stop_calls == 1
    assert engine.running is True
    assert restores == [(old_settings, True)]


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
        asyncio.run(
            TapeMachine._project_audio_settings_applied(app, new_settings)
        )

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

    app.project_saving = True
    TapeMachine._update_command_state(app)

    assert app.new_project_command.enabled is False
    assert app.open_project_command.enabled is False
    assert app.settings_command.enabled is False
    assert all(command.enabled is False for command in recent_commands)

    app.project_saving = False
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


def test_startup_audio_discovery_runs_off_the_ui_thread() -> None:
    main_thread = get_ident()
    discovery_threads: list[int] = []
    settings = AudioSettings(1, 1, 48_000)
    updates: list[None] = []
    shown_notices: list[list[str]] = []

    def discover() -> tuple[AudioSettings, tuple[str, ...]]:
        discovery_threads.append(get_ident())
        return settings, ("Confirm the fallback device.",)

    async def show_notices(notices: list[str]) -> None:
        shown_notices.append(list(notices))

    app = SimpleNamespace(
        audio_service=SimpleNamespace(current_settings=None),
        global_audio_settings=None,
        audio_transitioning=True,
        audio_initialization_task=object(),
        _discover_startup_audio=discover,
        _update_command_state=lambda: updates.append(None),
        _update_audio_summary=lambda: updates.append(None),
        _show_startup_notices=show_notices,
    )

    asyncio.run(
        TapeMachine._initialize_startup_audio(app, ["Recovered configuration."])
    )

    assert discovery_threads and discovery_threads[0] != main_thread
    assert app.audio_service.current_settings == settings
    assert app.global_audio_settings == settings
    assert app.audio_transitioning is False
    assert app.audio_initialization_task is None
    assert updates == [None, None]
    assert shown_notices == [
        ["Recovered configuration.", "Confirm the fallback device."]
    ]


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
    app.audio_transition_lock = asyncio.Lock()
    app.audio_transitioning = False
    app._update_command_state = lambda: None
    summaries: list[None] = []
    app._update_audio_summary = lambda: summaries.append(None)

    asyncio.run(TapeMachine._audio_settings_applied(app, candidate))

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
    app.audio_transition_lock = asyncio.Lock()
    app.audio_transitioning = False
    app._update_command_state = lambda: None
    app._update_audio_summary = lambda: None

    with pytest.raises(RuntimeError, match="disk full"):
        asyncio.run(TapeMachine._audio_settings_applied(app, candidate))

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


def test_recent_file_action_opens_its_project_path(tmp_path: Path) -> None:
    opened: list[Path] = []
    project_path = tmp_path / "project.wav"

    async def open_project(path: Path) -> None:
        opened.append(path)

    app = SimpleNamespace(_open_project_path=open_project)

    asyncio.run(TapeMachine._recent_file_action(app, project_path)())

    assert opened == [project_path]


def test_initial_recent_files_refresh_replaces_visible_rows() -> None:
    old_rows = [object(), object()]
    new_rows = [object(), object(), object()]

    class FakeBox:
        def __init__(self) -> None:
            self.children = list(old_rows)

        def remove(self, child: object) -> None:
            self.children.remove(child)

        def add(self, *children: object) -> None:
            self.children.extend(children)

    recent_box = FakeBox()
    app = SimpleNamespace(
        project=None,
        initial_recent_files_box=recent_box,
        _initial_recent_file_widgets=lambda: new_rows,
    )

    TapeMachine._refresh_initial_recent_files(app)

    assert recent_box.children == new_rows

    app.project = object()
    app._initial_recent_file_widgets = lambda: [object()]
    TapeMachine._refresh_initial_recent_files(app)

    assert recent_box.children == new_rows


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
