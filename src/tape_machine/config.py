"""Persistent, application-local Tape Machine configuration."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from tape_machine.audio import (
    PROJECT_TRACK_COUNT,
    STEREO_BUS_CHANNEL_COUNT,
    AudioDevice,
    AudioDeviceService,
    AudioSettings,
    DeviceReference,
    StereoBusInput,
    TrackInputRoute,
)


CONFIG_SCHEMA_VERSION = 2
MAX_RECENT_FILES = 10


class AppConfigError(RuntimeError):
    """Raised when application configuration cannot be loaded or saved."""


@dataclass(frozen=True, slots=True)
class WindowPosition:
    """Persisted top-left window coordinates in Toga screen space."""

    x: int
    y: int


@dataclass(frozen=True, slots=True)
class StoredAudioSettings:
    """Portable application audio defaults independent of PortAudio IDs."""

    input_device: DeviceReference
    output_device: DeviceReference
    sample_rate: int
    track_inputs: tuple[TrackInputRoute, ...]
    bus_outputs: tuple[int | None, ...]

    @classmethod
    def from_settings(
        cls,
        settings: AudioSettings,
        input_device: AudioDevice,
        output_device: AudioDevice,
    ) -> StoredAudioSettings:
        return cls(
            input_device=input_device.reference,
            output_device=output_device.reference,
            sample_rate=settings.sample_rate,
            track_inputs=settings.track_inputs,
            bus_outputs=settings.bus_outputs,
        )

    def resolve(self, service: AudioDeviceService) -> AudioSettings | None:
        """Resolve portable references to the current runtime device IDs."""
        input_device = service.resolve_device(self.input_device, "input")
        output_device = service.resolve_device(self.output_device, "output")
        if input_device is None or output_device is None:
            return None
        return AudioSettings(
            input_device_id=input_device.index,
            output_device_id=output_device.index,
            sample_rate=self.sample_rate,
            track_inputs=self.track_inputs,
            bus_outputs=self.bus_outputs,
        )


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Versioned application state that is not stored in project WAV files."""

    recent_files: tuple[Path, ...] = ()
    default_audio_settings: StoredAudioSettings | None = None
    main_window_position: WindowPosition | None = None
    audio_settings_window_position: WindowPosition | None = None

    def with_recent_file(self, path: Path) -> AppConfig:
        normalized = _normalize_path(path)
        recent = (normalized,) + tuple(
            candidate
            for candidate in self.recent_files
            if candidate != normalized
        )
        return replace(self, recent_files=recent[:MAX_RECENT_FILES])

    def without_recent_file(self, path: Path) -> AppConfig:
        normalized = _normalize_path(path)
        return replace(
            self,
            recent_files=tuple(
                candidate
                for candidate in self.recent_files
                if candidate != normalized
            ),
        )

    def clear_recent_files(self) -> AppConfig:
        return replace(self, recent_files=())

    def with_default_audio_settings(
        self, settings: StoredAudioSettings
    ) -> AppConfig:
        return replace(self, default_audio_settings=settings)

    def with_window_positions(
        self,
        main: WindowPosition,
        audio_settings: WindowPosition,
    ) -> AppConfig:
        return replace(
            self,
            main_window_position=main,
            audio_settings_window_position=audio_settings,
        )


class AppConfigStore:
    """Load and atomically save one application configuration file."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> AppConfig:
        if not self.path.exists():
            return AppConfig()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise AppConfigError(
                f"Unable to load application configuration: {exc}"
            ) from exc
        if not isinstance(raw, dict):
            raise AppConfigError("Application configuration must be a JSON object.")
        schema_version = raw.get("schema_version")
        if (
            not isinstance(schema_version, int)
            or isinstance(schema_version, bool)
            or schema_version not in {1, CONFIG_SCHEMA_VERSION}
        ):
            raise AppConfigError(
                "Application configuration has an unsupported version."
            )

        recent_files = _parse_recent_files(raw.get("recent_files"))
        default_audio = _parse_audio_settings(
            raw.get("default_audio_settings"),
            allow_loopback=schema_version >= 2,
        )
        positions = raw.get("window_positions")
        if not isinstance(positions, dict):
            positions = {}
        return AppConfig(
            recent_files=recent_files,
            default_audio_settings=default_audio,
            main_window_position=_parse_position(positions.get("main")),
            audio_settings_window_position=_parse_position(
                positions.get("audio_settings")
            ),
        )

    def save(self, config: AppConfig) -> None:
        payload = {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "recent_files": [str(path) for path in config.recent_files],
            "default_audio_settings": _audio_settings_payload(
                config.default_audio_settings
            ),
            "window_positions": {
                "main": _position_payload(config.main_window_position),
                "audio_settings": _position_payload(
                    config.audio_settings_window_position
                ),
            },
        }
        temporary_path: Path | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                json.dump(payload, temporary, indent=2, sort_keys=True)
                temporary.write("\n")
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, self.path)
        except OSError as exc:
            raise AppConfigError(
                f"Unable to save application configuration: {exc}"
            ) from exc
        finally:
            if temporary_path is not None and temporary_path.exists():
                try:
                    temporary_path.unlink()
                except OSError:
                    pass


def _normalize_path(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _parse_recent_files(value: Any) -> tuple[Path, ...]:
    if not isinstance(value, list):
        return ()
    recent: list[Path] = []
    for item in value:
        if not isinstance(item, str) or not item:
            continue
        path = _normalize_path(Path(item))
        if path not in recent:
            recent.append(path)
        if len(recent) == MAX_RECENT_FILES:
            break
    return tuple(recent)


def _parse_device_reference(value: Any) -> DeviceReference | None:
    if not isinstance(value, dict):
        return None
    name = value.get("name")
    host_api = value.get("host_api")
    if not isinstance(name, str) or not name:
        return None
    if not isinstance(host_api, str) or not host_api:
        return None
    return DeviceReference(name=name, host_api=host_api)


def _parse_routes(
    value: Any, expected_length: int, *, unique: bool = False
) -> tuple[int | None, ...] | None:
    if not isinstance(value, list) or len(value) != expected_length:
        return None
    routes: list[int | None] = []
    assigned: set[int] = set()
    for channel in value:
        if channel is None:
            routes.append(None)
            continue
        if (
            not isinstance(channel, int)
            or isinstance(channel, bool)
            or channel < 0
            or unique
            and channel in assigned
        ):
            return None
        routes.append(channel)
        assigned.add(channel)
    return tuple(routes)


def _parse_track_inputs(
    value: Any, *, allow_loopback: bool
) -> tuple[TrackInputRoute, ...] | None:
    if not isinstance(value, list) or len(value) != PROJECT_TRACK_COUNT:
        return None
    routes: list[TrackInputRoute] = []
    for route in value:
        if route is None:
            routes.append(None)
        elif (
            isinstance(route, int)
            and not isinstance(route, bool)
            and route >= 0
        ):
            routes.append(route)
        elif isinstance(route, str) and allow_loopback:
            try:
                routes.append(StereoBusInput(route))
            except ValueError:
                return None
        else:
            return None
    return tuple(routes)


def _parse_audio_settings(
    value: Any, *, allow_loopback: bool
) -> StoredAudioSettings | None:
    if not isinstance(value, dict):
        return None
    input_device = _parse_device_reference(value.get("input_device"))
    output_device = _parse_device_reference(value.get("output_device"))
    sample_rate = value.get("sample_rate")
    track_inputs = _parse_track_inputs(
        value.get("track_inputs"), allow_loopback=allow_loopback
    )
    bus_outputs = _parse_routes(
        value.get("bus_outputs"), STEREO_BUS_CHANNEL_COUNT, unique=True
    )
    if (
        input_device is None
        or output_device is None
        or not isinstance(sample_rate, int)
        or isinstance(sample_rate, bool)
        or sample_rate <= 0
        or track_inputs is None
        or bus_outputs is None
    ):
        return None
    return StoredAudioSettings(
        input_device=input_device,
        output_device=output_device,
        sample_rate=sample_rate,
        track_inputs=track_inputs,
        bus_outputs=bus_outputs,
    )


def _parse_position(value: Any) -> WindowPosition | None:
    if not isinstance(value, dict):
        return None
    x = value.get("x")
    y = value.get("y")
    if (
        not isinstance(x, int)
        or isinstance(x, bool)
        or not isinstance(y, int)
        or isinstance(y, bool)
    ):
        return None
    return WindowPosition(x=x, y=y)


def _device_reference_payload(reference: DeviceReference) -> dict[str, str]:
    return {"name": reference.name, "host_api": reference.host_api}


def _audio_settings_payload(
    settings: StoredAudioSettings | None,
) -> dict[str, Any] | None:
    if settings is None:
        return None
    return {
        "input_device": _device_reference_payload(settings.input_device),
        "output_device": _device_reference_payload(settings.output_device),
        "sample_rate": settings.sample_rate,
        "track_inputs": [
            route.value if isinstance(route, StereoBusInput) else route
            for route in settings.track_inputs
        ],
        "bus_outputs": list(settings.bus_outputs),
    }


def _position_payload(position: WindowPosition | None) -> dict[str, int] | None:
    if position is None:
        return None
    return {"x": position.x, "y": position.y}
