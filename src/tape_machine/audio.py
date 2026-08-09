"""Audio device discovery and session-scoped configuration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

import sounddevice


SAMPLE_RATES = (44_100, 48_000, 88_200, 96_000, 176_400, 192_000)
AUDIO_BUFFER_SIZES = (0, 16, 32, 64, 128, 256, 512, 1024, 2048)
PROJECT_TRACK_COUNT = 8
STEREO_BUS_CHANNEL_COUNT = 2
UNASSIGNED_BUS_OUTPUTS: tuple[int | None, ...] = (None,) * STEREO_BUS_CHANNEL_COUNT


class StereoBusInput(StrEnum):
    """Virtual input sources exposing the rendered stereo bus."""

    LEFT = "stereo_bus_l"
    RIGHT = "stereo_bus_r"


type TrackInputRoute = int | StereoBusInput | None

UNASSIGNED_TRACK_INPUTS: tuple[TrackInputRoute, ...] = (
    None,
) * PROJECT_TRACK_COUNT


def is_physical_input(route: TrackInputRoute) -> bool:
    """Return whether a route identifies a device input channel."""
    return isinstance(route, int) and not isinstance(route, bool)


class AudioConfigurationError(RuntimeError):
    """Raised when audio hardware cannot satisfy a requested configuration."""


class SoundDeviceBackend(Protocol):
    """Subset of sounddevice used by :class:`AudioDeviceService`."""

    default: Any

    def query_devices(self) -> Any: ...

    def query_hostapis(self) -> Any: ...

    def check_input_settings(self, **kwargs: Any) -> None: ...

    def check_output_settings(self, **kwargs: Any) -> None: ...


@dataclass(frozen=True, slots=True)
class AudioDevice:
    """A PortAudio device available during the current application session."""

    index: int
    name: str
    host_api: str
    max_input_channels: int
    max_output_channels: int
    default_sample_rate: int
    input_channel_names: tuple[str | None, ...] = ()
    output_channel_names: tuple[str | None, ...] = ()

    def label(self, direction: str) -> str:
        """Return an unambiguous label for an input or output selector."""
        channels = (
            self.max_input_channels
            if direction == "input"
            else self.max_output_channels
        )
        channel_word = "channel" if channels == 1 else "channels"
        return (
            f"{self.name} — {self.host_api} "
            f"({channels} {channel_word}, device {self.index})"
        )

    @property
    def reference(self) -> DeviceReference:
        """Return the stable descriptor stored in application configuration."""
        return DeviceReference(name=self.name, host_api=self.host_api)


@dataclass(frozen=True, slots=True)
class DeviceReference:
    """Portable audio-device identity independent of PortAudio indices."""

    name: str
    host_api: str


@dataclass(frozen=True, slots=True)
class AudioSettings:
    """The audio configuration used by the running application."""

    input_device_id: int
    output_device_id: int
    sample_rate: int
    track_inputs: tuple[TrackInputRoute, ...] = UNASSIGNED_TRACK_INPUTS
    bus_outputs: tuple[int | None, ...] = UNASSIGNED_BUS_OUTPUTS
    buffer_size: int = 0

    @property
    def required_input_channels(self) -> int:
        """Number of leading device channels needed to satisfy the routing."""
        assigned = [
            channel
            for channel in self.track_inputs
            if is_physical_input(channel)
        ]
        return max(assigned, default=-1) + 1

    @property
    def required_output_channels(self) -> int:
        """Number of leading device channels needed to satisfy the routing."""
        assigned = [channel for channel in self.bus_outputs if channel is not None]
        return max(assigned, default=-1) + 1


class AudioDeviceService:
    """Discover audio devices and manage the active in-memory configuration."""

    def __init__(self, backend: SoundDeviceBackend = sounddevice) -> None:
        self._backend = backend
        self.input_devices: tuple[AudioDevice, ...] = ()
        self.output_devices: tuple[AudioDevice, ...] = ()
        self.default_input_device_id: int | None = None
        self.default_output_device_id: int | None = None
        self.current_settings: AudioSettings | None = None

    def initialize(self) -> AudioSettings | None:
        """Discover hardware and select a usable session default, if possible."""
        self.refresh_devices()
        self.current_settings = self.suggest_settings()
        return self.current_settings

    def refresh_devices(
        self,
    ) -> tuple[tuple[AudioDevice, ...], tuple[AudioDevice, ...]]:
        """Refresh the input and output device inventories."""
        try:
            raw_devices = self._backend.query_devices()
        except Exception as exc:
            raise AudioConfigurationError(
                f"Unable to enumerate audio devices: {exc}"
            ) from exc

        try:
            raw_host_apis = self._backend.query_hostapis()
        except Exception:
            raw_host_apis = ()

        devices: list[AudioDevice] = []
        for fallback_index, raw_device in enumerate(raw_devices):
            index = int(raw_device.get("index", fallback_index))
            host_api_index = int(raw_device.get("hostapi", -1))
            try:
                host_api = str(raw_host_apis[host_api_index]["name"])
            except (IndexError, KeyError, TypeError):
                host_api = f"Host API {host_api_index}"

            max_input_channels = int(
                raw_device.get("max_input_channels", 0)
            )
            max_output_channels = int(
                raw_device.get("max_output_channels", 0)
            )

            devices.append(
                AudioDevice(
                    index=index,
                    name=str(raw_device.get("name", f"Device {index}")),
                    host_api=host_api,
                    max_input_channels=max_input_channels,
                    max_output_channels=max_output_channels,
                    default_sample_rate=int(
                        round(float(raw_device.get("default_samplerate", 48_000)))
                    ),
                    input_channel_names=self._channel_names(
                        index,
                        max_input_channels,
                        is_input=True,
                        host_api=host_api,
                    ),
                    output_channel_names=self._channel_names(
                        index,
                        max_output_channels,
                        is_input=False,
                        host_api=host_api,
                    ),
                )
            )

        devices.sort(key=lambda device: device.index)
        self.input_devices = tuple(
            device for device in devices if device.max_input_channels > 0
        )
        self.output_devices = tuple(
            device for device in devices if device.max_output_channels > 0
        )

        input_default, output_default = self._read_backend_defaults()
        self.default_input_device_id = self._available_or_first(
            input_default, self.input_devices
        )
        self.default_output_device_id = self._available_or_first(
            output_default, self.output_devices
        )
        return self.input_devices, self.output_devices

    def _channel_names(
        self,
        device_index: int,
        channel_count: int,
        *,
        is_input: bool,
        host_api: str,
    ) -> tuple[str | None, ...]:
        """Copy optional Core Audio channel names from PortAudio."""
        missing = (None,) * channel_count
        if host_api != "Core Audio" or channel_count == 0:
            return missing

        library = getattr(self._backend, "_lib", None)
        ffi = getattr(self._backend, "_ffi", None)
        get_channel_name = getattr(
            library, "PaMacCore_GetChannelName", None
        )
        if ffi is None or not callable(get_channel_name):
            return missing

        names: list[str | None] = []
        for channel_index in range(channel_count):
            try:
                pointer = get_channel_name(
                    device_index, channel_index, is_input
                )
                if pointer == ffi.NULL:
                    names.append(None)
                    continue
                name = ffi.string(pointer).decode("utf-8").strip()
                names.append(name or None)
            except Exception:
                names.append(None)
        return tuple(names)

    def supported_sample_rates(
        self,
        input_device_id: int,
        output_device_id: int,
        track_inputs: tuple[TrackInputRoute, ...] = UNASSIGNED_TRACK_INPUTS,
        bus_outputs: tuple[int | None, ...] = UNASSIGNED_BUS_OUTPUTS,
    ) -> tuple[int, ...]:
        """Return standard rates accepted by both selected devices."""
        input_device = self._require_device(
            input_device_id, self.input_devices, "input"
        )
        output_device = self._require_device(
            output_device_id, self.output_devices, "output"
        )
        self._validate_track_inputs(track_inputs, input_device)
        self._validate_bus_outputs(bus_outputs, output_device)
        required_input_channels = self._available_input_channels(
            track_inputs, input_device.max_input_channels
        )
        required_output_channels = self._available_output_channels(
            bus_outputs, output_device.max_output_channels
        )

        supported: list[int] = []
        for sample_rate in SAMPLE_RATES:
            try:
                if required_input_channels:
                    self._backend.check_input_settings(
                        device=input_device_id,
                        channels=required_input_channels,
                        dtype="float32",
                        samplerate=sample_rate,
                    )
                self._backend.check_output_settings(
                    device=output_device_id,
                    channels=required_output_channels,
                    dtype="float32",
                    samplerate=sample_rate,
                )
            except Exception:
                continue
            supported.append(sample_rate)
        return tuple(supported)

    def suggest_settings(
        self, preferred: AudioSettings | None = None
    ) -> AudioSettings | None:
        """Build a valid draft from active settings or current system defaults."""
        input_ids = {device.index for device in self.input_devices}
        output_ids = {device.index for device in self.output_devices}

        input_device_id = (
            preferred.input_device_id
            if preferred and preferred.input_device_id in input_ids
            else self.default_input_device_id
        )
        output_device_id = (
            preferred.output_device_id
            if preferred and preferred.output_device_id in output_ids
            else self.default_output_device_id
        )
        if input_device_id is None or output_device_id is None:
            return None

        input_device = self.device(input_device_id, "input")
        output_device = self.device(output_device_id, "output")
        if input_device is None or output_device is None:
            return None

        track_inputs = self._suggest_track_inputs(
            preferred, input_device_id, input_device.max_input_channels
        )
        bus_outputs = self._suggest_bus_outputs(
            preferred, output_device_id, output_device.max_output_channels
        )
        rates = self.supported_sample_rates(
            input_device_id, output_device_id, track_inputs, bus_outputs
        )
        if not rates:
            return None

        if preferred and preferred.sample_rate in rates:
            sample_rate = preferred.sample_rate
        elif 48_000 in rates:
            sample_rate = 48_000
        else:
            sample_rate = rates[0]

        buffer_size = (
            preferred.buffer_size
            if preferred and preferred.buffer_size in AUDIO_BUFFER_SIZES
            else 0
        )

        return AudioSettings(
            input_device_id,
            output_device_id,
            sample_rate,
            track_inputs,
            bus_outputs,
            buffer_size,
        )

    def apply(self, settings: AudioSettings) -> None:
        """Validate and activate settings for the remainder of this session."""
        self.refresh_devices()
        self.validate(settings)
        self.current_settings = settings

    def validate(self, settings: AudioSettings) -> None:
        """Validate a complete configuration against the current hardware."""
        input_device = self._require_device(
            settings.input_device_id, self.input_devices, "input"
        )
        output_device = self._require_device(
            settings.output_device_id, self.output_devices, "output"
        )
        if (
            not isinstance(settings.sample_rate, int)
            or isinstance(settings.sample_rate, bool)
            or settings.sample_rate <= 0
        ):
            raise AudioConfigurationError(
                f"{settings.sample_rate!r} is not a valid sample rate."
            )
        if (
            not isinstance(settings.buffer_size, int)
            or isinstance(settings.buffer_size, bool)
            or settings.buffer_size not in AUDIO_BUFFER_SIZES
        ):
            raise AudioConfigurationError(
                f"{settings.buffer_size!r} is not a supported audio buffer size."
            )

        self._validate_track_inputs(settings.track_inputs, input_device)
        self._validate_bus_outputs(settings.bus_outputs, output_device)

        try:
            required_input_channels = self._available_input_channels(
                settings.track_inputs, input_device.max_input_channels
            )
            if required_input_channels:
                self._backend.check_input_settings(
                    device=settings.input_device_id,
                    channels=required_input_channels,
                    dtype="float32",
                    samplerate=settings.sample_rate,
                )
            self._backend.check_output_settings(
                device=settings.output_device_id,
                channels=self._available_output_channels(
                    settings.bus_outputs, output_device.max_output_channels
                ),
                dtype="float32",
                samplerate=settings.sample_rate,
            )
        except Exception as exc:
            raise AudioConfigurationError(
                "The selected devices no longer support "
                f"{settings.sample_rate:,} Hz: {exc}"
            ) from exc

    def device(self, device_id: int, direction: str) -> AudioDevice | None:
        """Look up a device in the current directional inventory."""
        inventory = self.input_devices if direction == "input" else self.output_devices
        return next(
            (device for device in inventory if device.index == device_id), None
        )

    def resolve_device(
        self, reference: DeviceReference | None, direction: str
    ) -> AudioDevice | None:
        """Resolve a stored descriptor, falling back to the system default."""
        inventory = self.input_devices if direction == "input" else self.output_devices
        if reference is not None:
            match = next(
                (
                    device
                    for device in inventory
                    if device.name == reference.name
                    and device.host_api == reference.host_api
                ),
                None,
            )
            if match is not None:
                return match

        default_id = (
            self.default_input_device_id
            if direction == "input"
            else self.default_output_device_id
        )
        return self.device(default_id, direction) if default_id is not None else None

    def compatibility_error(self, settings: AudioSettings) -> str | None:
        """Return a hardware compatibility message without activating settings."""
        try:
            self.validate(settings)
        except AudioConfigurationError as exc:
            return str(exc)
        return None

    def _read_backend_defaults(self) -> tuple[int | None, int | None]:
        try:
            input_id, output_id = self._backend.default.device
        except (AttributeError, TypeError, ValueError):
            return None, None
        return self._normalise_id(input_id), self._normalise_id(output_id)

    @staticmethod
    def _normalise_id(value: Any) -> int | None:
        if value is None:
            return None
        try:
            value = int(value)
        except (TypeError, ValueError):
            return None
        return value if value >= 0 else None

    @staticmethod
    def _available_or_first(
        preferred_id: int | None, devices: tuple[AudioDevice, ...]
    ) -> int | None:
        if preferred_id is not None and any(
            device.index == preferred_id for device in devices
        ):
            return preferred_id
        return devices[0].index if devices else None

    @staticmethod
    def _suggest_track_inputs(
        preferred: AudioSettings | None,
        input_device_id: int,
        max_input_channels: int,
    ) -> tuple[TrackInputRoute, ...]:
        if (
            preferred is None
            or len(preferred.track_inputs) != PROJECT_TRACK_COUNT
        ):
            return UNASSIGNED_TRACK_INPUTS

        return tuple(
            channel
            if isinstance(channel, StereoBusInput)
            or is_physical_input(channel)
            and channel >= 0
            else None
            for channel in preferred.track_inputs
        )

    @staticmethod
    def _required_input_channels(
        track_inputs: tuple[TrackInputRoute, ...],
    ) -> int:
        assigned = [
            channel for channel in track_inputs if is_physical_input(channel)
        ]
        return max(assigned, default=-1) + 1

    @staticmethod
    def _suggest_bus_outputs(
        preferred: AudioSettings | None,
        output_device_id: int,
        max_output_channels: int,
    ) -> tuple[int | None, ...]:
        if (
            preferred is None
            or len(preferred.bus_outputs) != STEREO_BUS_CHANNEL_COUNT
        ):
            return UNASSIGNED_BUS_OUTPUTS

        outputs: list[int | None] = []
        used_outputs: set[int] = set()
        for channel in preferred.bus_outputs:
            if (
                isinstance(channel, int)
                and not isinstance(channel, bool)
                and channel >= 0
                and channel not in used_outputs
            ):
                outputs.append(channel)
                used_outputs.add(channel)
            else:
                outputs.append(None)
        return tuple(outputs)

    @staticmethod
    def _required_output_channels(bus_outputs: tuple[int | None, ...]) -> int:
        assigned = [channel for channel in bus_outputs if channel is not None]
        return max(assigned, default=-1) + 1

    @staticmethod
    def _available_input_channels(
        track_inputs: tuple[TrackInputRoute, ...], max_input_channels: int
    ) -> int:
        assigned = [
            channel
            for channel in track_inputs
            if is_physical_input(channel) and channel < max_input_channels
        ]
        return max(assigned, default=-1) + 1

    @staticmethod
    def _available_output_channels(
        bus_outputs: tuple[int | None, ...], max_output_channels: int
    ) -> int:
        assigned = [
            channel
            for channel in bus_outputs
            if channel is not None and channel < max_output_channels
        ]
        return max(1, max(assigned, default=-1) + 1)

    @staticmethod
    def _validate_track_inputs(
        track_inputs: tuple[TrackInputRoute, ...], input_device: AudioDevice
    ) -> None:
        if len(track_inputs) != PROJECT_TRACK_COUNT:
            raise AudioConfigurationError(
                f"Input routing must contain exactly {PROJECT_TRACK_COUNT} tracks."
            )

        for track_index, channel in enumerate(track_inputs):
            if channel is None or isinstance(channel, StereoBusInput):
                continue
            if (
                not is_physical_input(channel)
                or channel < 0
            ):
                raise AudioConfigurationError(
                    f"Track {track_index + 1} has invalid input channel "
                    f"{channel!r}."
                )

    @staticmethod
    def _validate_bus_outputs(
        bus_outputs: tuple[int | None, ...], output_device: AudioDevice
    ) -> None:
        if len(bus_outputs) != STEREO_BUS_CHANNEL_COUNT:
            raise AudioConfigurationError(
                "Stereo output routing must contain exactly 2 bus channels."
            )

        used_outputs: set[int] = set()
        for bus_index, channel in enumerate(bus_outputs):
            if channel is None:
                continue
            bus_side = "L" if bus_index == 0 else "R"
            if (
                not isinstance(channel, int)
                or isinstance(channel, bool)
                or channel < 0
            ):
                raise AudioConfigurationError(
                    f"Stereo bus {bus_side} has invalid output channel "
                    f"{channel!r}."
                )
            if channel in used_outputs:
                raise AudioConfigurationError(
                    "Stereo bus L and R must use distinct output channels."
                )
            used_outputs.add(channel)

    @staticmethod
    def _require_device(
        device_id: int, devices: tuple[AudioDevice, ...], direction: str
    ) -> AudioDevice:
        device = next(
            (device for device in devices if device.index == device_id), None
        )
        if device is None:
            raise AudioConfigurationError(
                f"Audio {direction} device {device_id} is no longer available."
            )
        return device
