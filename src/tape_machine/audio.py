"""Audio device discovery and session-scoped configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import sounddevice


SAMPLE_RATES = (44_100, 48_000, 88_200, 96_000, 176_400, 192_000)


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


@dataclass(frozen=True, slots=True)
class AudioSettings:
    """The audio configuration used by the running application."""

    input_device_id: int
    output_device_id: int
    sample_rate: int


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

            devices.append(
                AudioDevice(
                    index=index,
                    name=str(raw_device.get("name", f"Device {index}")),
                    host_api=host_api,
                    max_input_channels=int(raw_device.get("max_input_channels", 0)),
                    max_output_channels=int(raw_device.get("max_output_channels", 0)),
                    default_sample_rate=int(
                        round(float(raw_device.get("default_samplerate", 48_000)))
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

    def supported_sample_rates(
        self, input_device_id: int, output_device_id: int
    ) -> tuple[int, ...]:
        """Return standard rates accepted by both selected devices."""
        self._require_device(input_device_id, self.input_devices, "input")
        self._require_device(output_device_id, self.output_devices, "output")

        supported: list[int] = []
        for sample_rate in SAMPLE_RATES:
            try:
                self._backend.check_input_settings(
                    device=input_device_id,
                    channels=1,
                    dtype="float32",
                    samplerate=sample_rate,
                )
                self._backend.check_output_settings(
                    device=output_device_id,
                    channels=1,
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

        rates = self.supported_sample_rates(input_device_id, output_device_id)
        if not rates:
            return None

        if preferred and preferred.sample_rate in rates:
            sample_rate = preferred.sample_rate
        elif 48_000 in rates:
            sample_rate = 48_000
        else:
            sample_rate = rates[0]

        return AudioSettings(input_device_id, output_device_id, sample_rate)

    def apply(self, settings: AudioSettings) -> None:
        """Validate and activate settings for the remainder of this session."""
        self.refresh_devices()
        self.validate(settings)
        self.current_settings = settings

    def validate(self, settings: AudioSettings) -> None:
        """Validate a complete configuration against the current hardware."""
        self._require_device(
            settings.input_device_id, self.input_devices, "input"
        )
        self._require_device(
            settings.output_device_id, self.output_devices, "output"
        )
        if settings.sample_rate not in SAMPLE_RATES:
            raise AudioConfigurationError(
                f"{settings.sample_rate} Hz is not an available studio sample rate."
            )

        try:
            self._backend.check_input_settings(
                device=settings.input_device_id,
                channels=1,
                dtype="float32",
                samplerate=settings.sample_rate,
            )
            self._backend.check_output_settings(
                device=settings.output_device_id,
                channels=1,
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
    def _require_device(
        device_id: int, devices: tuple[AudioDevice, ...], direction: str
    ) -> None:
        if not any(device.index == device_id for device in devices):
            raise AudioConfigurationError(
                f"Audio {direction} device {device_id} is no longer available."
            )
