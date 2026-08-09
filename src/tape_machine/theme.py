"""Application-wide dark theme constants and native appearance setup."""

from __future__ import annotations

from typing import Any


DARK_AQUA_APPEARANCE = "NSAppearanceNameDarkAqua"

ACCENT_BLUE = "#4b8fd9"
ACCENT_RED = "#d94b4b"
ACCENT_ORANGE = "#d98b3e"
ACCENT_YELLOW = "#e0bd35"
SECONDARY_TEXT = "#92979e"

FADER_RAIL = "#44484d"
FADER_TICK = "#565b61"
FADER_HANDLE = "#3d4146"
FADER_INDICATOR = "#a7acb1"
FADER_READOUT = "#60656b"

METER_OFF = "#24272b"
METER_GREEN = "#45ad5f"
METER_YELLOW = "#d6ad36"
METER_RED = "#dc4b4b"

KNOB_BACKGROUND = "#2d3034"
KNOB_BORDER = "#50555b"
KNOB_INDICATOR = "#a7acb1"
KNOB_MARKING = "#60656b"


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
