"""Application-wide dark theme constants and native appearance setup."""

from __future__ import annotations

from typing import Any


DARK_AQUA_APPEARANCE = "NSAppearanceNameDarkAqua"

ACCENT_BLUE = "#4b8fd9"
ACCENT_RED = "#d94b4b"
ACCENT_ORANGE = "#d98b3e"
ACCENT_YELLOW = "#e0bd35"

FADER_RAIL = "#666b72"
FADER_TICK = "#7d838a"
FADER_HANDLE = "#555a60"
FADER_INDICATOR = "#e3e6e8"
FADER_READOUT = "#747980"

METER_OFF = "#24272b"
METER_GREEN = "#45ad5f"
METER_YELLOW = "#d6ad36"
METER_RED = "#dc4b4b"

KNOB_BACKGROUND = "#3b3f45"
KNOB_BORDER = "#737980"
KNOB_INDICATOR = "#e3e6e8"


def force_dark_appearance(
    native_app: Any, appearance_class: Any | None = None
) -> None:
    """Force macOS Dark Aqua instead of inheriting the system appearance."""
    if appearance_class is None:
        from rubicon.objc import ObjCClass

        appearance_class = ObjCClass("NSAppearance")

    native_app.appearance = appearance_class.appearanceNamed(
        DARK_AQUA_APPEARANCE
    )
