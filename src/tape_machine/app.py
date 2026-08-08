"""The Tape Machine Toga application."""

import toga
from toga.style.pack import COLUMN


class TapeMachine(toga.App):
    """The main Tape Machine application."""

    def startup(self) -> None:
        """Create and show the main application window."""
        self.status_label = toga.Label(
            "Ready",
            font_size=14,
            margin_top=8,
        )

        content = toga.Box(
            children=[
                toga.Label(
                    "Tape Machine",
                    font_size=28,
                    font_weight="bold",
                ),
                toga.Label(
                    "A native macOS app built with BeeWare and Toga.",
                    margin_top=4,
                ),
                toga.Button(
                    "Test the app",
                    on_press=self.test_app,
                    margin_top=20,
                ),
                self.status_label,
            ],
            direction=COLUMN,
            margin=24,
        )

        self.main_window = toga.MainWindow(
            title=self.formal_name,
            size=(640, 420),
        )
        self.main_window.content = content
        self.main_window.show()

    def test_app(self, widget: toga.Widget, **kwargs: object) -> None:
        """Confirm that event handling is working."""
        self.status_label.text = "Tape Machine is running."


def main() -> TapeMachine:
    """Create the application instance used by Briefcase."""
    return TapeMachine(
        formal_name="Tape Machine",
        app_id="pkg.unusualaudio.tape-machine",
    )
