"""Tests for the main-window audio summary."""

from types import SimpleNamespace

import pytest

from tape_machine.app import TapeMachine
from tape_machine.audio import AudioDevice, AudioSettings, DeviceReference
from tape_machine.engine import AudioEngineError
from tape_machine.mixer import MixerState
from tape_machine.project import ProjectMetadata


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
        reference = DeviceReference("Interface", "Test API")
        self.metadata = ProjectMetadata(reference, reference)
        self.saved: list[ProjectMetadata] = []

    def save(self, metadata: ProjectMetadata) -> None:
        self.saved.append(metadata)
        self.metadata = metadata


def summary_app(settings: AudioSettings | None) -> SimpleNamespace:
    return SimpleNamespace(
        audio_service=FakeService(settings),
        status_line_label=SimpleNamespace(text=""),
        project=None,
    )


def test_status_line_marks_incomplete_routing() -> None:
    app = summary_app(AudioSettings(1, 1, 48_000))

    TapeMachine._update_audio_summary(app)

    assert app.status_line_label.text == (
        "Interface  •  48 kHz  •  Routing incomplete"
    )


def test_status_line_omits_message_when_all_routing_is_complete() -> None:
    app = summary_app(
        AudioSettings(1, 1, 48_000, tuple(range(8)), (0, 1))
    )

    TapeMachine._update_audio_summary(app)

    assert app.status_line_label.text == "Interface  •  48 kHz"


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

    assert app.status_line_label.text == (
        "Interface  •  48 kHz  •  Routing incomplete"
    )


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
        "No audio device  •  Sample rate unavailable  •  Routing incomplete"
    )


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


def test_status_line_marks_project_audio_engine_failure() -> None:
    app = summary_app(
        AudioSettings(1, 1, 48_000, tuple(range(8)), (0, 1))
    )
    app.project = object()
    app.audio_engine_error = "Device busy"

    TapeMachine._update_audio_summary(app)

    assert app.status_line_label.text == (
        "Interface  •  48 kHz  •  Audio unavailable"
    )


def test_mixer_change_rebuilds_running_engine_matrix() -> None:
    calls: list[object] = []
    state = object()
    app = SimpleNamespace(
        mixer_state=state,
        audio_engine=SimpleNamespace(
            running=True,
            update_mix=lambda mixer_state: calls.append(mixer_state),
        ),
    )

    TapeMachine._mixer_changed(app)

    assert calls == [state]


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


def test_project_audio_settings_commit_after_stream_starts() -> None:
    old_settings = AudioSettings(1, 1, 48_000)
    new_settings = AudioSettings(
        1,
        1,
        48_000,
        (0, None, None, None, None, None, None, None),
        (0, 1),
    )
    service = FakeService(old_settings)
    engine = FakeLifecycleEngine()
    project = FakeProject()
    view_events: list[object] = []
    app = SimpleNamespace(
        project=project,
        audio_service=service,
        audio_engine=engine,
        mixer_state=MixerState(),
        mixer_view=SimpleNamespace(
            update_track_routes=lambda routes: view_events.append(routes),
            set_monitoring_available=lambda available: view_events.append(
                available
            ),
        ),
        project_audio_error="old error",
        audio_engine_error="old error",
        _update_audio_summary=lambda: view_events.append("summary"),
    )

    TapeMachine._project_audio_settings_applied(app, new_settings)

    assert engine.stop_calls == 1
    assert engine.starts == [new_settings]
    assert project.saved[0].track_inputs == new_settings.track_inputs
    assert service.current_settings == new_settings
    assert app.project_audio_error is None
    assert app.audio_engine_error is None
    assert view_events == [new_settings.track_inputs, True, "summary"]


def test_saving_unchanged_project_settings_keeps_active_stream() -> None:
    current_settings = AudioSettings(1, 1, 48_000)
    service = FakeService(current_settings)
    engine = FakeLifecycleEngine()
    project = FakeProject()
    app = SimpleNamespace(
        project=project,
        audio_service=service,
        audio_engine=engine,
        mixer_state=MixerState(),
        mixer_view=None,
        project_audio_error=None,
        audio_engine_error=None,
        _update_audio_summary=lambda: None,
    )

    TapeMachine._project_audio_settings_applied(app, current_settings)

    assert service.refresh_calls == 0
    assert service.validate_calls == 0
    assert engine.stop_calls == 0
    assert engine.starts == []
    assert project.saved


def test_failed_candidate_stream_keeps_project_metadata_and_old_settings() -> None:
    old_settings = AudioSettings(1, 1, 48_000)
    new_settings = AudioSettings(
        1,
        1,
        48_000,
        (0, None, None, None, None, None, None, None),
        (0, 1),
    )
    service = FakeService(old_settings)
    engine = FakeLifecycleEngine(fail_start=True)
    project = FakeProject()
    dialogs: list[str] = []
    app = SimpleNamespace(
        project=project,
        audio_service=service,
        audio_engine=engine,
        mixer_state=MixerState(),
        mixer_view=None,
        _restore_audio_engine=lambda settings, was_running: True,
        _set_audio_engine_failure=lambda message, show_dialog=False: None,
        _schedule_audio_engine_dialog=lambda message: dialogs.append(message),
    )

    with pytest.raises(AudioEngineError, match="busy"):
        TapeMachine._project_audio_settings_applied(app, new_settings)

    assert project.saved == []
    assert service.current_settings == old_settings
    assert dialogs == ["Unable to start input monitoring: busy"]


def test_project_commands_follow_open_project_state() -> None:
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
    )

    TapeMachine._update_command_state(app)

    assert app.new_project_command.enabled is True
    assert app.open_project_command.enabled is True
    assert app.save_project_command.enabled is False
    assert app.close_project_command.enabled is False
    assert app.settings_command.enabled is True

    app.project = object()
    TapeMachine._update_command_state(app)

    assert app.new_project_command.enabled is False
    assert app.open_project_command.enabled is False
    assert app.save_project_command.enabled is True
    assert app.close_project_command.enabled is True

    app.transport = SimpleNamespace(running=True)
    TapeMachine._update_command_state(app)

    assert app.save_project_command.enabled is False
    assert app.close_project_command.enabled is False
    assert app.settings_command.enabled is False


def test_transport_controls_show_toggle_stop_and_time_state() -> None:
    app = SimpleNamespace(
        transport=SimpleNamespace(
            running=True,
            record_armed=True,
            position_frames=60_000,
        ),
        transport_starting=False,
        transport_stopping=False,
        audio_engine=SimpleNamespace(running=True),
        project=SimpleNamespace(writable=True, sample_rate=48_000),
        transport_record_button=SimpleNamespace(
            enabled=False,
            text="",
            style=SimpleNamespace(background_color=None),
        ),
        transport_play_button=SimpleNamespace(enabled=True),
        transport_stop_rtz_button=SimpleNamespace(text="", enabled=False),
        transport_time_label=SimpleNamespace(text=""),
    )

    TapeMachine._sync_transport_controls(app)

    assert app.transport_record_button.enabled is True
    assert app.transport_record_button.text == "● Record"
    assert app.transport_play_button.enabled is False
    assert app.transport_stop_rtz_button.text == "Stop"
    assert app.transport_stop_rtz_button.enabled is True
    assert app.transport_time_label.text == "00:01.250"
