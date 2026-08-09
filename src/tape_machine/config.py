"""Persistent, application-local Tape Machine configuration."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from tape_machine.audio import (
    AUDIO_BUFFER_SIZES,
    PROJECT_TRACK_COUNT,
    STEREO_BUS_CHANNEL_COUNT,
    AudioDevice,
    AudioDeviceService,
    AudioSettings,
    DeviceReference,
    StereoBusInput,
    TrackInputRoute,
)


CONFIG_SCHEMA_VERSION = 5
# Fourteen compact launcher rows fit above the fixed main-window footer.
MAX_RECENT_FILES = 14


class AppConfigError(RuntimeError):
    """Raised when application configuration cannot be loaded or saved."""


@dataclass(frozen=True, slots=True)
class ConfigLoadResult:
    """Recovered application configuration and user-facing diagnostics."""

    config: AppConfig
    recovered_sections: tuple[str, ...] = ()
    reset_sections: tuple[str, ...] = ()
    backup_path: Path | None = None

    @property
    def recovered(self) -> bool:
        return bool(self.backup_path or self.reset_sections)

    @property
    def notice_required(self) -> bool:
        return any(
            section != "configuration version" for section in self.reset_sections
        )


@dataclass(frozen=True, slots=True)
class WindowPosition:
    """Persisted top-left window coordinates in Toga screen space."""

    x: int
    y: int


@dataclass(frozen=True, slots=True)
class StoredAudioSettings:
    """Portable global audio settings independent of PortAudio IDs."""

    input_device: DeviceReference
    output_device: DeviceReference
    sample_rate: int
    track_inputs: tuple[TrackInputRoute, ...]
    bus_outputs: tuple[int | None, ...]
    buffer_size: int = 0

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
            buffer_size=settings.buffer_size,
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
            buffer_size=self.buffer_size,
        )


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Versioned application state that is not stored in project WAV files."""

    recent_files: tuple[Path, ...] = ()
    audio_settings: StoredAudioSettings | None = None
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

    def with_audio_settings(self, settings: StoredAudioSettings) -> AppConfig:
        return replace(self, audio_settings=settings)

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
        """Load configuration, recovering valid sections when necessary."""
        return self.load_with_recovery().config

    def load_with_recovery(self) -> ConfigLoadResult:
        if not self.path.exists():
            return ConfigLoadResult(AppConfig())
        try:
            text = self.path.read_text(encoding="utf-8")
        except UnicodeError:
            return self._recover_config(
                AppConfig(), recovered_sections=(), reset_sections=("all settings",)
            )
        except OSError as exc:
            raise AppConfigError(
                f"Unable to load application configuration: {exc}"
            ) from exc
        try:
            raw = json.loads(text)
        except (UnicodeError, json.JSONDecodeError):
            return self._recover_config(
                AppConfig(), recovered_sections=(), reset_sections=("all settings",)
            )
        if not isinstance(raw, dict):
            return self._recover_config(
                AppConfig(), recovered_sections=(), reset_sections=("all settings",)
            )
        schema_version = raw.get("schema_version")
        schema_supported = (
            isinstance(schema_version, int)
            and not isinstance(schema_version, bool)
            and schema_version == CONFIG_SCHEMA_VERSION
        )

        recovered_sections: list[str] = []
        reset_sections: list[str] = []
        raw_recent_files = raw.get("recent_files")
        recent_files_valid = isinstance(raw_recent_files, list) and all(
            isinstance(item, str) for item in raw_recent_files
        )
        recent_files = (
            _parse_recent_files(raw_recent_files)
            if recent_files_valid
            else ()
        )
        if recent_files_valid:
            recovered_sections.append("recent projects")
        else:
            reset_sections.append("recent projects")
        audio_settings = _parse_audio_settings(raw.get("audio_settings"))
        if raw.get("audio_settings") is None or audio_settings is not None:
            recovered_sections.append("audio settings")
        else:
            reset_sections.append("audio settings")
        positions = raw.get("window_positions")
        positions_valid = isinstance(positions, dict) and all(
            value is None or _parse_position(value) is not None
            for value in (
                positions.get("main") if isinstance(positions, dict) else None,
                positions.get("audio_settings")
                if isinstance(positions, dict)
                else None,
            )
        )
        if not positions_valid:
            positions = {}
            reset_sections.append("window positions")
        else:
            recovered_sections.append("window positions")
        config = AppConfig(
            recent_files=recent_files,
            audio_settings=audio_settings,
            main_window_position=_parse_position(positions.get("main")),
            audio_settings_window_position=_parse_position(
                positions.get("audio_settings")
            ),
        )
        if schema_supported and not reset_sections:
            return ConfigLoadResult(config)
        if not schema_supported:
            reset_sections.insert(0, "configuration version")
        return self._recover_config(
            config,
            recovered_sections=tuple(recovered_sections),
            reset_sections=tuple(reset_sections),
        )

    def _recover_config(
        self,
        config: AppConfig,
        *,
        recovered_sections: tuple[str, ...],
        reset_sections: tuple[str, ...],
    ) -> ConfigLoadResult:
        backup_path = self._backup_path()
        try:
            shutil.copy2(self.path, backup_path)
            self.save(config)
        except OSError as exc:
            raise AppConfigError(
                f"Unable to recover application configuration: {exc}"
            ) from exc
        return ConfigLoadResult(
            config,
            recovered_sections,
            reset_sections,
            backup_path,
        )

    def _backup_path(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        candidate = self.path.with_name(
            f"{self.path.stem}.invalid-{timestamp}{self.path.suffix}"
        )
        counter = 2
        while candidate.exists():
            candidate = self.path.with_name(
                f"{self.path.stem}.invalid-{timestamp}-{counter}{self.path.suffix}"
            )
            counter += 1
        return candidate

    def save(self, config: AppConfig) -> None:
        payload = {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "recent_files": [str(path) for path in config.recent_files],
            "audio_settings": _audio_settings_payload(config.audio_settings),
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
            _fsync_directory(self.path.parent)
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
    max_input_channels = _optional_nonnegative_int(value.get("max_input_channels"))
    max_output_channels = _optional_nonnegative_int(value.get("max_output_channels"))
    default_sample_rate = _optional_positive_int(value.get("default_sample_rate"))
    raw_signature = value.get("channel_name_signature")
    signature = (
        tuple(raw_signature)
        if isinstance(raw_signature, list)
        and all(isinstance(item, str) for item in raw_signature)
        else ()
    )
    return DeviceReference(
        name=name,
        host_api=host_api,
        max_input_channels=max_input_channels,
        max_output_channels=max_output_channels,
        default_sample_rate=default_sample_rate,
        channel_name_signature=signature,
    )


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


def _parse_track_inputs(value: Any) -> tuple[TrackInputRoute, ...] | None:
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
        elif isinstance(route, str):
            try:
                routes.append(StereoBusInput(route))
            except ValueError:
                return None
        else:
            return None
    return tuple(routes)


def _parse_audio_settings(value: Any) -> StoredAudioSettings | None:
    if not isinstance(value, dict):
        return None
    input_device = _parse_device_reference(value.get("input_device"))
    output_device = _parse_device_reference(value.get("output_device"))
    sample_rate = value.get("sample_rate")
    buffer_size = value.get("buffer_size")
    track_inputs = _parse_track_inputs(value.get("track_inputs"))
    bus_outputs = _parse_routes(
        value.get("bus_outputs"), STEREO_BUS_CHANNEL_COUNT, unique=True
    )
    if (
        input_device is None
        or output_device is None
        or not isinstance(sample_rate, int)
        or isinstance(sample_rate, bool)
        or sample_rate <= 0
        or not isinstance(buffer_size, int)
        or isinstance(buffer_size, bool)
        or buffer_size not in AUDIO_BUFFER_SIZES
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
        buffer_size=buffer_size,
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


def _device_reference_payload(reference: DeviceReference) -> dict[str, Any]:
    return {
        "name": reference.name,
        "host_api": reference.host_api,
        "max_input_channels": reference.max_input_channels,
        "max_output_channels": reference.max_output_channels,
        "default_sample_rate": reference.default_sample_rate,
        "channel_name_signature": list(reference.channel_name_signature),
    }


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
        "buffer_size": settings.buffer_size,
    }


def _position_payload(position: WindowPosition | None) -> dict[str, int] | None:
    if position is None:
        return None
    return {"x": position.x, "y": position.y}


def _optional_nonnegative_int(value: Any) -> int | None:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else None
    )


def _optional_positive_int(value: Any) -> int | None:
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
        else None
    )


def _fsync_directory(path: Path) -> None:
    try:
        directory_fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        pass
