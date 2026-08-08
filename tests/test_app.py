"""Tests for the main-window audio summary."""

from types import SimpleNamespace

from tape_machine.app import TapeMachine
from tape_machine.audio import AudioDevice, AudioSettings


class FakeService:
    def __init__(
        self,
        settings: AudioSettings | None,
        input_name: str = "Interface",
        output_name: str = "Interface",
    ) -> None:
        self.current_settings = settings
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


def test_project_commands_follow_open_project_state() -> None:
    app = SimpleNamespace(
        project=None,
        new_project_command=SimpleNamespace(enabled=False),
        open_project_command=SimpleNamespace(enabled=False),
        save_project_command=SimpleNamespace(enabled=True),
        close_project_command=SimpleNamespace(enabled=True),
    )

    TapeMachine._update_command_state(app)

    assert app.new_project_command.enabled is True
    assert app.open_project_command.enabled is True
    assert app.save_project_command.enabled is False
    assert app.close_project_command.enabled is False

    app.project = object()
    TapeMachine._update_command_state(app)

    assert app.new_project_command.enabled is False
    assert app.open_project_command.enabled is False
    assert app.save_project_command.enabled is True
    assert app.close_project_command.enabled is True
