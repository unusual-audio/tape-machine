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
    PROJECT_TRACK_COUNT,
    STEREO_BUS_CHANNEL_COUNT,
    StereoBusInput,
    TrackInputRoute,
    is_physical_input,
)


PROJECT_APPLICATION_ID = "pkg.unusualaudio.tape-machine"
PROJECT_SCHEMA_VERSION = 5
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

    source_comment: str | None = None
    mix: MixerMetadata = field(default_factory=MixerMetadata)

    def __post_init__(self) -> None:
        if not isinstance(self.mix, MixerMetadata):
            raise ValueError("Project mix must be a mixer metadata object.")
        if self.source_comment is not None and not isinstance(
            self.source_comment, str
        ):
            raise ValueError("Project source comment must be text.")

    def with_mix(self, mix: MixerMetadata) -> ProjectMetadata:
        """Return metadata updated from the project mixer."""
        return replace(self, mix=mix)

    def to_comment(self) -> str:
        """Serialize metadata to the tagged WAV comment representation."""
        payload: dict[str, Any] = {
            "application": PROJECT_APPLICATION_ID,
            "schema_version": PROJECT_SCHEMA_VERSION,
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
            or schema_version != PROJECT_SCHEMA_VERSION
        ):
            raise ProjectError(
                "This project uses an unsupported metadata schema version."
            )

        try:
            source_comment = payload.get("source_comment")
            if source_comment is not None and not isinstance(source_comment, str):
                raise ValueError("source_comment must be text")
            return cls(
                source_comment=source_comment,
                mix=_mixer_from_payload(payload["mix"]),
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
                metadata = ProjectMetadata(source_comment=comment or None)
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

    def open_playback_reader(self) -> soundfile.SoundFile:
        """Open an independent read handle for transport playback."""
        writable = self.writable
        try:
            # libsndfile finalizes the RF64 length header when the writer is
            # closed. Rotate it before opening a concurrent playback reader.
            self.audio_file.close()
            self.audio_file = soundfile.SoundFile(
                self.path, mode="r+" if writable else "r"
            )
            return soundfile.SoundFile(self.path, mode="r")
        except Exception as exc:
            raise ProjectError(
                f"Unable to open project audio for playback: {exc}"
            ) from exc

    def read_audio_block(
        self,
        position: int,
        frames: int,
        *,
        audio_file: soundfile.SoundFile | None = None,
    ) -> np.ndarray:
        """Read an exact float32 block, padding beyond EOF with silence."""
        reader = self.audio_file if audio_file is None else audio_file
        try:
            reader.seek(position)
            return reader.read(
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
        if position > self.frames:
            self._extend_with_silence(position)
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

    def _extend_with_silence(self, target_frames: int) -> None:
        """Materialize a silent gap before recording beyond the current EOF."""
        try:
            self.audio_file.seek(self.frames)
            remaining = target_frames - self.frames
            silence = np.zeros((8192, PROJECT_TRACK_COUNT), dtype=np.float32)
            while remaining > 0:
                frames = min(remaining, len(silence))
                self.audio_file.write(silence[:frames])
                remaining -= frames
        except Exception as exc:
            raise ProjectError(f"Unable to extend project audio: {exc}") from exc

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
