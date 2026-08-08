"""The Tape Machine Toga application."""

import toga
from toga.style.pack import COLUMN

from tape_machine.audio import (
    AudioConfigurationError,
    AudioDeviceService,
    AudioSettings,
)
from tape_machine.settings import AudioSettingsWindow


class TapeMachine(toga.App):
    """The main Tape Machine application."""

    def startup(self) -> None:
        """Create and show the main application window."""
        self.audio_service = AudioDeviceService()
        try:
            self.audio_service.initialize()
            startup_error = None
        except AudioConfigurationError as exc:
            startup_error = str(exc)

        self.audio_summary = toga.Label("", margin_top=12)
        self.audio_detail = toga.Label("", margin_top=4)

        content = toga.Box(
            children=[
                toga.Label(
                    "Tape Machine",
                    font_size=28,
                    font_weight="bold",
                ),
                toga.Label(
                    "Multitrack recording for macOS.",
                    margin_top=4,
                ),
                toga.Label(
                    "Audio configuration",
                    font_size=16,
                    font_weight="bold",
                    margin_top=28,
                ),
                self.audio_summary,
                self.audio_detail,
                toga.Button(
                    "Audio Settings…",
                    on_press=self.preferences,
                    margin_top=20,
                ),
            ],
            direction=COLUMN,
            margin=24,
        )

        self.main_window = toga.MainWindow(
            title=self.formal_name,
            size=(640, 420),
        )
        self.main_window.content = content

        self.settings_window = AudioSettingsWindow(
            self.audio_service, self._audio_settings_applied
        )
        self.commands.add(toga.Command.standard(self, toga.Command.PREFERENCES))
        self._update_audio_summary(startup_error)
        self.main_window.show()

    def preferences(
        self, widget: toga.Widget | toga.Command | None = None, **kwargs: object
    ) -> None:
        """Open the session audio settings window."""
        self.settings_window.open()

    def _audio_settings_applied(self, settings: AudioSettings) -> None:
        self._update_audio_summary()

    def _update_audio_summary(self, error: str | None = None) -> None:
        settings = self.audio_service.current_settings
        if settings is None:
            self.audio_summary.text = "Audio is not configured."
            self.audio_detail.text = error or "Open Audio Settings to choose devices."
            return

        input_device = self.audio_service.device(settings.input_device_id, "input")
        output_device = self.audio_service.device(settings.output_device_id, "output")
        input_name = input_device.name if input_device else "Unavailable input"
        output_name = output_device.name if output_device else "Unavailable output"
        self.audio_summary.text = f"{settings.sample_rate / 1_000:g} kHz"
        self.audio_detail.text = f"Input: {input_name}  •  Output: {output_name}"


def main() -> TapeMachine:
    """Create the application instance used by Briefcase."""
    return TapeMachine(
        formal_name="Tape Machine",
        app_id="pkg.unusualaudio.tape-machine",
    )
