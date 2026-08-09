"""Portable Tape Machine projects stored in multichannel WAV files."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field, replace
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
import soundfile

from tape_machine.audio import (
    AUDIO_BUFFER_SIZES,
    PROJECT_TRACK_COUNT,
    STEREO_BUS_CHANNEL_COUNT,
    UNASSIGNED_BUS_OUTPUTS,
    UNASSIGNED_TRACK_INPUTS,
    DeviceReference,
    StereoBusInput,
    TrackInputRoute,
    is_physical_input,
)


PROJECT_APPLICATION_ID = "pkg.unusualaudio.tape-machine"
PROJECT_SCHEMA_VERSION = 4
PROJECT_COMMENT_PREFIX = "TAPE_MACHINE_PROJECT:"
WAV_FORMATS = {"WAV", "WAVEX", "RF64"}
MIX_MIN_LEVEL_DB = -60.0
MIX_MAX_LEVEL_DB = 6.0


class ProjectError(RuntimeError):
    """Raised when a project file cannot be created, opened, or saved."""


@dataclass(frozen=True, slots=True)
class TrackMixMetadata:
    """Persisted controls for one project track."""

    level_db: float = 0.0
    pan: float = 0.0
    record_enabled: bool = False
    input_monitoring: bool = False
    muted: bool = False
    soloed: bool = False

    def __post_init__(self) -> None:
        _validate_mix_number(
            "track level", self.level_db, MIX_MIN_LEVEL_DB, MIX_MAX_LEVEL_DB
        )
        _validate_mix_number("track pan", self.pan, -1.0, 1.0)
        for name, value in (
            ("record_enabled", self.record_enabled),
            ("input_monitoring", self.input_monitoring),
            ("muted", self.muted),
            ("soloed", self.soloed),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"Track mix {name} must be boolean.")


@dataclass(frozen=True, slots=True)
class MixerMetadata:
    """Persisted controls for all project tracks and the stereo bus."""

    tracks: tuple[TrackMixMetadata, ...] = field(
        default_factory=lambda: tuple(
            TrackMixMetadata() for _ in range(PROJECT_TRACK_COUNT)
        )
    )
    bus_level_db: float = 0.0

    def __post_init__(self) -> None:
        if len(self.tracks) != PROJECT_TRACK_COUNT:
            raise ValueError(
                f"Project mix must contain {PROJECT_TRACK_COUNT} tracks."
            )
        if not all(isinstance(track, TrackMixMetadata) for track in self.tracks):
            raise ValueError("Project mix tracks must be track mix objects.")
        _validate_mix_number(
            "bus level", self.bus_level_db, MIX_MIN_LEVEL_DB, MIX_MAX_LEVEL_DB
        )


@dataclass(frozen=True, slots=True)
class ProjectMetadata:
    """Versioned project configuration embedded in WAV comment metadata."""

    input_device: DeviceReference | None = None
    output_device: DeviceReference | None = None
    track_inputs: tuple[TrackInputRoute, ...] = UNASSIGNED_TRACK_INPUTS
    bus_outputs: tuple[int | None, ...] = UNASSIGNED_BUS_OUTPUTS
    buffer_size: int = 0
    source_comment: str | None = None
    mix: MixerMetadata = field(default_factory=MixerMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.mix, MixerMetadata):
            raise ValueError("Project mix must be a mixer metadata object.")
        if (
            not isinstance(self.buffer_size, int)
            or isinstance(self.buffer_size, bool)
            or self.buffer_size not in AUDIO_BUFFER_SIZES
        ):
            raise ValueError(
                f"Invalid project audio buffer size {self.buffer_size!r}."
            )
        if len(self.track_inputs) != PROJECT_TRACK_COUNT:
            raise ValueError(
                f"Project input routing must contain {PROJECT_TRACK_COUNT} tracks."
            )
        for channel in self.track_inputs:
            if channel is not None and (
                not isinstance(channel, StereoBusInput)
                and (not is_physical_input(channel) or channel < 0)
            ):
                raise ValueError(f"Invalid project input channel {channel!r}.")

        if len(self.bus_outputs) != STEREO_BUS_CHANNEL_COUNT:
            raise ValueError(
                "Project output routing must contain two stereo bus channels."
            )
        assigned_outputs: set[int] = set()
        for channel in self.bus_outputs:
            if channel is None:
                continue
            if (
                not isinstance(channel, int)
                or isinstance(channel, bool)
                or channel < 0
            ):
                raise ValueError(f"Invalid project output channel {channel!r}.")
            if channel in assigned_outputs:
                raise ValueError("Stereo bus outputs must be distinct.")
            assigned_outputs.add(channel)

    def with_audio(
        self,
        input_device: DeviceReference,
        output_device: DeviceReference,
        track_inputs: tuple[TrackInputRoute, ...],
        bus_outputs: tuple[int | None, ...],
        buffer_size: int,
    ) -> ProjectMetadata:
        """Return metadata updated from saved Audio Settings."""
        return replace(
            self,
            input_device=input_device,
            output_device=output_device,
            track_inputs=track_inputs,
            bus_outputs=bus_outputs,
            buffer_size=buffer_size,
        )

    def with_mix(self, mix: MixerMetadata) -> ProjectMetadata:
        """Return metadata updated from the project mixer."""
        return replace(self, mix=mix)

    def to_comment(self) -> str:
        """Serialize metadata to the tagged WAV comment representation."""
        payload: dict[str, Any] = {
            "application": PROJECT_APPLICATION_ID,
            "schema_version": PROJECT_SCHEMA_VERSION,
            "input_device": _reference_payload(self.input_device),
            "output_device": _reference_payload(self.output_device),
            "track_inputs": [
                route.value if isinstance(route, StereoBusInput) else route
                for route in self.track_inputs
            ],
            "bus_outputs": list(self.bus_outputs),
            "buffer_size": self.buffer_size,
            "mix": _mixer_payload(self.mix),
        }
        if self.source_comment:
            payload["source_comment"] = self.source_comment
        return PROJECT_COMMENT_PREFIX + json.dumps(
            payload, separators=(",", ":"), sort_keys=True
        )

    @classmethod
    def from_comment(cls, comment: str) -> ProjectMetadata | None:
        """Parse tagged project metadata, or return None for an untagged WAV."""
        if not comment.startswith(PROJECT_COMMENT_PREFIX):
            return None
        try:
            payload = json.loads(comment[len(PROJECT_COMMENT_PREFIX) :])
        except json.JSONDecodeError as exc:
            raise ProjectError(f"Project metadata is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ProjectError("Project metadata must be a JSON object.")
        if payload.get("application") != PROJECT_APPLICATION_ID:
            raise ProjectError("Project metadata has an unknown application ID.")
        schema_version = payload.get("schema_version")
        if (
            not isinstance(schema_version, int)
            or isinstance(schema_version, bool)
            or schema_version not in {1, 2, 3, PROJECT_SCHEMA_VERSION}
        ):
            raise ProjectError(
                "This project uses an unsupported metadata schema version."
            )

        try:
            track_inputs = _track_inputs_from_payload(
                payload["track_inputs"], allow_loopback=schema_version >= 3
            )
            bus_outputs = tuple(payload["bus_outputs"])
            source_comment = payload.get("source_comment")
            if source_comment is not None and not isinstance(source_comment, str):
                raise ValueError("source_comment must be text")
            mix = (
                MixerMetadata()
                if schema_version == 1
                else _mixer_from_payload(payload["mix"])
            )
            buffer_size = (
                payload["buffer_size"] if schema_version >= 4 else 0
            )
            return cls(
                input_device=_reference_from_payload(payload.get("input_device")),
                output_device=_reference_from_payload(payload.get("output_device")),
                track_inputs=track_inputs,
                bus_outputs=bus_outputs,
                buffer_size=buffer_size,
                source_comment=source_comment,
                mix=mix,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProjectError(f"Project metadata is invalid: {exc}") from exc


class AudioProject:
    """An open eight-track project backed by one writable WAV file."""

    def __init__(
        self,
        path: Path,
        audio_file: soundfile.SoundFile,
        metadata: ProjectMetadata,
        *,
        dirty: bool,
    ) -> None:
        self.path = path
        self.audio_file = audio_file
        self.metadata = metadata
        self.dirty = dirty

    @classmethod
    def create(
        cls,
        path: Path,
        sample_rate: int,
        metadata: ProjectMetadata,
    ) -> AudioProject:
        """Create and keep open an empty eight-channel RF64 project."""
        path = Path(path)
        audio_file: soundfile.SoundFile | None = None
        try:
            audio_file = soundfile.SoundFile(
                path,
                mode="w+",
                samplerate=sample_rate,
                channels=PROJECT_TRACK_COUNT,
                subtype="PCM_24",
                format="RF64",
            )
            project = cls(path, audio_file, metadata, dirty=True)
            project.save()
            return project
        except Exception as exc:
            if audio_file is not None:
                try:
                    audio_file.close()
                except Exception:
                    pass
            raise ProjectError(f"Unable to create project: {exc}") from exc

    @classmethod
    def open(
        cls,
        path: Path,
        import_metadata: ProjectMetadata,
    ) -> AudioProject:
        """Open a tagged project or import an untagged eight-channel WAV."""
        path = Path(path)
        try:
            inspection_file = soundfile.SoundFile(path, mode="r")
        except Exception as exc:
            raise ProjectError(f"Unable to inspect project: {exc}") from exc

        try:
            if inspection_file.format not in WAV_FORMATS:
                raise ProjectError("The selected file is not a WAV-family file.")
            if inspection_file.channels != PROJECT_TRACK_COUNT:
                raise ProjectError(
                    f"Projects must contain exactly {PROJECT_TRACK_COUNT} channels."
                )

            comment = inspection_file.comment
            metadata = ProjectMetadata.from_comment(comment)
            dirty = metadata is None
            if metadata is None:
                metadata = replace(
                    import_metadata,
                    source_comment=comment or None,
                )
        finally:
            inspection_file.close()

        try:
            audio_file = soundfile.SoundFile(path, mode="r+")
        except Exception:
            try:
                audio_file = soundfile.SoundFile(path, mode="r")
            except Exception as exc:
                raise ProjectError(f"Unable to open project: {exc}") from exc
        try:
            return cls(path, audio_file, metadata, dirty=dirty)
        except Exception:
            audio_file.close()
            raise

    @property
    def sample_rate(self) -> int:
        return self.audio_file.samplerate

    @property
    def channels(self) -> int:
        return self.audio_file.channels

    @property
    def frames(self) -> int:
        return self.audio_file.frames

    @property
    def writable(self) -> bool:
        """Whether the open project handle can modify audio data."""
        return "+" in self.audio_file.mode or "w" in self.audio_file.mode

    @property
    def format(self) -> str:
        return self.audio_file.format

    @property
    def subtype(self) -> str:
        return self.audio_file.subtype

    def save(self, metadata: ProjectMetadata | None = None) -> None:
        """Write project metadata and flush audio without changing state on failure."""
        candidate = metadata or self.metadata
        if "+" not in self.audio_file.mode and "w" not in self.audio_file.mode:
            self._rewrite_for_metadata(candidate)
            self.metadata = candidate
            self.dirty = False
            return

        previous_comment = self.audio_file.comment
        try:
            self.audio_file.comment = candidate.to_comment()
            self.audio_file.flush()
        except Exception as exc:
            try:
                self.audio_file.comment = previous_comment
                self.audio_file.flush()
            except Exception:
                pass
            raise ProjectError(f"Unable to save project metadata: {exc}") from exc
        self.metadata = candidate
        self.dirty = False

    def stage_metadata(self, metadata: ProjectMetadata) -> None:
        """Update project metadata in memory and mark it for an explicit save."""
        if metadata != self.metadata:
            self.metadata = metadata
            self.dirty = True

    def read_audio_block(self, position: int, frames: int) -> np.ndarray:
        """Read an exact float32 block, padding beyond EOF with silence."""
        try:
            self.audio_file.seek(position)
            return self.audio_file.read(
                frames,
                dtype="float32",
                always_2d=True,
                fill_value=0.0,
            )
        except Exception as exc:
            raise ProjectError(f"Unable to read project audio: {exc}") from exc

    def write_recording_block(
        self,
        position: int,
        input_data: np.ndarray,
        stereo_bus: np.ndarray,
        track_inputs: tuple[TrackInputRoute, ...],
        armed_tracks: tuple[bool, ...],
    ) -> None:
        """Replace armed tracks while preserving every unarmed channel."""
        if not self.writable:
            raise ProjectError("This project file is read-only and cannot record.")
        if len(track_inputs) != PROJECT_TRACK_COUNT or len(armed_tracks) != (
            PROJECT_TRACK_COUNT
        ):
            raise ProjectError("Recording requires exactly eight project tracks.")
        if stereo_bus.shape != (len(input_data), STEREO_BUS_CHANNEL_COUNT):
            raise ProjectError("Recording requires a two-channel stereo bus block.")
        block = self.read_audio_block(position, len(input_data))
        for track_index, (input_channel, armed) in enumerate(
            zip(track_inputs, armed_tracks, strict=True)
        ):
            if not armed:
                continue
            if isinstance(input_channel, StereoBusInput):
                bus_channel = 0 if input_channel is StereoBusInput.LEFT else 1
                block[:, track_index] = np.clip(
                    stereo_bus[:, bus_channel], -1.0, 1.0
                )
            elif (
                input_channel is None
                or not is_physical_input(input_channel)
                or input_channel >= input_data.shape[1]
            ):
                block[:, track_index] = 0
            else:
                block[:, track_index] = input_data[:, input_channel]
        try:
            self.audio_file.seek(position)
            self.audio_file.write(block)
        except Exception as exc:
            raise ProjectError(f"Unable to write recorded audio: {exc}") from exc

    def flush_audio(self) -> None:
        """Flush recorded audio and its updated RF64 header to disk."""
        try:
            self.audio_file.flush()
        except Exception as exc:
            raise ProjectError(f"Unable to flush recorded audio: {exc}") from exc

    def _rewrite_for_metadata(self, metadata: ProjectMetadata) -> None:
        """Atomically convert a read-only-open import to writable RF64."""
        temp_file = tempfile.NamedTemporaryFile(
            prefix=f".{self.path.name}.",
            suffix=".tmp.wav",
            dir=self.path.parent,
            delete=False,
        )
        temp_path = Path(temp_file.name)
        temp_file.close()
        source = self.audio_file
        source_metadata = source.copy_metadata()
        dtype = _transfer_dtype(source.subtype)
        try:
            with soundfile.SoundFile(
                temp_path,
                mode="w",
                samplerate=source.samplerate,
                channels=source.channels,
                subtype=source.subtype,
                format="RF64",
            ) as target:
                source.seek(0)
                for block in source.blocks(
                    blocksize=65_536,
                    dtype=dtype,
                    always_2d=True,
                ):
                    target.write(block)
                for key, value in source_metadata.items():
                    if key != "comment":
                        setattr(target, key, value)
                target.comment = metadata.to_comment()
                target.flush()

            source.close()
            os.replace(temp_path, self.path)
            self.audio_file = soundfile.SoundFile(self.path, mode="r+")
        except Exception as exc:
            if not source.closed:
                source.seek(0)
            else:
                try:
                    self.audio_file = soundfile.SoundFile(self.path, mode="r")
                except Exception:
                    pass
            temp_path.unlink(missing_ok=True)
            raise ProjectError(f"Unable to save project metadata: {exc}") from exc

    def close(self) -> None:
        """Close the underlying SoundFile handle."""
        self.audio_file.close()


def _reference_payload(reference: DeviceReference | None) -> dict[str, str] | None:
    if reference is None:
        return None
    return {"name": reference.name, "host_api": reference.host_api}


def _track_inputs_from_payload(
    payload: object, *, allow_loopback: bool
) -> tuple[TrackInputRoute, ...]:
    if not isinstance(payload, list):
        raise ValueError("track_inputs must be an array")
    routes: list[TrackInputRoute] = []
    for route in payload:
        if isinstance(route, str) and allow_loopback:
            try:
                routes.append(StereoBusInput(route))
            except ValueError as exc:
                raise ValueError(f"unknown input source {route!r}") from exc
        else:
            routes.append(route)
    return tuple(routes)


def _reference_from_payload(payload: object) -> DeviceReference | None:
    if payload is None:
        return None
    if not isinstance(payload, dict):
        raise ValueError("device reference must be an object or null")
    name = payload.get("name")
    host_api = payload.get("host_api")
    if not isinstance(name, str) or not name:
        raise ValueError("device name must be non-empty text")
    if not isinstance(host_api, str) or not host_api:
        raise ValueError("device host_api must be non-empty text")
    return DeviceReference(name=name, host_api=host_api)


def _mixer_payload(mix: MixerMetadata) -> dict[str, Any]:
    return {
        "tracks": [
            {
                "level_db": track.level_db,
                "pan": track.pan,
                "record_enabled": track.record_enabled,
                "input_monitoring": track.input_monitoring,
                "muted": track.muted,
                "soloed": track.soloed,
            }
            for track in mix.tracks
        ],
        "bus_level_db": mix.bus_level_db,
    }


def _mixer_from_payload(payload: object) -> MixerMetadata:
    if not isinstance(payload, dict):
        raise ValueError("mix must be an object")
    tracks_payload = payload["tracks"]
    if not isinstance(tracks_payload, list):
        raise ValueError("mix tracks must be an array")
    tracks: list[TrackMixMetadata] = []
    for track_payload in tracks_payload:
        if not isinstance(track_payload, dict):
            raise ValueError("mix tracks must be objects")
        tracks.append(
            TrackMixMetadata(
                level_db=track_payload["level_db"],
                pan=track_payload["pan"],
                record_enabled=track_payload["record_enabled"],
                input_monitoring=track_payload["input_monitoring"],
                muted=track_payload["muted"],
                soloed=track_payload["soloed"],
            )
        )
    return MixerMetadata(
        tracks=tuple(tracks),
        bus_level_db=payload["bus_level_db"],
    )


def _validate_mix_number(
    name: str, value: object, minimum: float, maximum: float
) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ValueError(
            f"Project mix {name} must be between {minimum:g} and {maximum:g}."
        )


def _transfer_dtype(subtype: str) -> str:
    if subtype == "DOUBLE":
        return "float64"
    if subtype == "FLOAT":
        return "float32"
    return "int32"
