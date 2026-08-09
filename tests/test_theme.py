"""Tests for the application-wide appearance override."""

from types import SimpleNamespace

from tape_machine.theme import DARK_AQUA_APPEARANCE, force_dark_appearance


def test_dark_aqua_is_forced_independently_of_system_appearance() -> None:
    requested_appearances: list[str] = []
    dark_appearance = object()

    class FakeAppearance:
        @staticmethod
        def appearanceNamed(name: str) -> object:
            requested_appearances.append(name)
            return dark_appearance

    native_app = SimpleNamespace(appearance=None)

    force_dark_appearance(native_app, FakeAppearance)

    assert requested_appearances == [DARK_AQUA_APPEARANCE]
    assert native_app.appearance is dark_appearance
