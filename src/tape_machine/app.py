"""The Tape Machine Toga application."""

import asyncio
from pathlib import Path

import toga
from toga.style.pack import CENTER, COLUMN, ROW

from tape_machine.audio import (
    AudioConfigurationError,
    AudioDeviceService,
    AudioSettings,
)
from tape_machine.engine import AudioEngine, AudioEngineError
from tape_machine.mixer import MixerState, MixerView
from tape_machine.project import AudioProject, ProjectError, ProjectMetadata
from tape_machine.settings import AudioSettingsDraft, AudioSettingsWindow
from tape_machine.transport import (
    TransportController,
    TransportError,
    format_transport_time,
)


class TapeMachine(toga.App):
    """The main Tape Machine application."""

    def startup(self) -> None:
        """Create and show the main application window."""
        self.audio_service = AudioDeviceService()
        self.audio_engine = AudioEngine()
        try:
            self.audio_service.initialize()
            startup_error = None
        except AudioConfigurationError as exc:
            startup_error = str(exc)

        self.project: AudioProject | None = None
        self.mixer_state: MixerState | None = None
        self.mixer_view: MixerView | None = None
        self.transport: TransportController | None = None
        self.transport_task: asyncio.Task[None] | None = None
        self.transport_starting = False
        self.transport_stopping = False
        self.session_audio_settings = self.audio_service.current_settings
        self.project_audio_error: str | None = None
        self.audio_engine_error: str | None = None

        self.status_line_label = toga.Label("", font_size=11)

        self.main_window = toga.MainWindow(
            title=self.formal_name,
            size=(912, 560),
            resizable=False,
        )
        self.main_window.content = self._build_initial_screen()

        self.settings_window = AudioSettingsWindow(
            self.audio_service, self._session_audio_settings_applied
        )
        self.settings_command = toga.Command.standard(
            self, toga.Command.PREFERENCES
        )
        self.new_project_command = toga.Command.standard(
            self,
            toga.Command.NEW,
            action=self.new_project,
            text="New Project",
        )
        self.open_project_command = toga.Command.standard(
            self,
            toga.Command.OPEN,
            action=self.open_project,
            text="Open Project…",
        )
        self.save_project_command = toga.Command.standard(
            self,
            toga.Command.SAVE,
            action=self.save_project,
            text="Save Project",
        )
        self.close_project_command = toga.Command(
            self.close_project,
            "Close Project",
            shortcut=toga.Key.MOD_1 + "w",
            group=toga.Group.FILE,
            section=20,
        )
        self.commands.add(
            self.settings_command,
            self.new_project_command,
            self.open_project_command,
            self.close_project_command,
            self.save_project_command,
        )
        self._update_command_state()
        self._update_audio_summary(startup_error)
        self.main_window.show()

    def _build_initial_screen(self) -> toga.Box:
        central_content = toga.Box(
            children=[
                toga.Label(
                    "Tape Machine",
                    font_size=28,
                    font_weight="bold",
                    text_align=CENTER,
                ),
                toga.Label(
                    "Create a new eight-track project or open an existing "
                    "WAV project.",
                    text_align=CENTER,
                    margin_top=12,
                ),
                toga.Label(
                    "Choose File → New Project or File → Open Project… "
                    "to get started.",
                    text_align=CENTER,
                    margin_top=6,
                ),
            ],
            direction=COLUMN,
            align_items=CENTER,
            justify_content=CENTER,
            flex=1,
            margin=24,
        )
        return toga.Box(
            children=[
                central_content,
                self._build_status_footer(),
            ],
            direction=COLUMN,
        )

    def _build_project_screen(self) -> toga.Box:
        assert self.project is not None
        project = self.project
        self.mixer_state = MixerState.from_track_inputs(
            project.metadata.track_inputs
        )
        self.mixer_view = MixerView(
            project.metadata.track_inputs, self.mixer_state
        )
        self.mixer_state.on_change = self._mixer_changed
        self.transport = TransportController(project)
        self.audio_engine.set_transport(self.transport)
        self.transport_record_button = toga.Button(
            "Record",
            on_press=self._toggle_transport_record,
            width=80,
            height=28,
        )
        self.transport_play_button = toga.Button(
            "Play",
            on_press=self._play_transport,
            width=60,
            height=28,
        )
        self.transport_stop_rtz_button = toga.Button(
            "RTZ",
            on_press=self._stop_or_rtz,
            width=60,
            height=28,
        )
        self.transport_time_label = toga.Label(
            "00:00.000",
            width=104,
            font_size=14,
            font_weight="bold",
            text_align=CENTER,
            margin_left=10,
            margin_top=4,
        )
        transport_bar = toga.Box(
            children=[
                self.transport_record_button,
                self.transport_play_button,
                self.transport_stop_rtz_button,
                self.transport_time_label,
            ],
            direction=ROW,
            align_items=CENTER,
            justify_content=CENTER,
            gap=6,
            margin_bottom=2,
        )
        central_content = toga.Box(
            children=[transport_bar, self.mixer_view.widget],
            direction=COLUMN,
            align_items=CENTER,
            flex=1,
            margin_top=12,
            margin_left=16,
            margin_right=16,
        )
        return toga.Box(
            children=[central_content, self._build_status_footer()],
            direction=COLUMN,
        )

    def _build_status_footer(self) -> toga.Box:
        return toga.Box(
            children=[
                toga.Divider(margin_bottom=8),
                self.status_line_label,
            ],
            direction=COLUMN,
            margin_left=16,
            margin_right=16,
            margin_bottom=12,
        )

    def _update_command_state(self) -> None:
        project_is_open = self.project is not None
        transport_busy = bool(
            self.transport
            and (
                self.transport.running
                or self.transport_starting
                or self.transport_stopping
            )
        )
        self.new_project_command.enabled = not project_is_open
        self.open_project_command.enabled = not project_is_open
        self.save_project_command.enabled = project_is_open and not transport_busy
        self.close_project_command.enabled = project_is_open and not transport_busy
        self.settings_command.enabled = not transport_busy

    def preferences(
        self, widget: toga.Widget | toga.Command | None = None, **kwargs: object
    ) -> None:
        """Open session or project-specific Audio Settings."""
        if self.project is None:
            self.settings_window.open()
            return

        current = self.audio_service.current_settings
        draft = AudioSettingsDraft(
            input_device_id=(current.input_device_id if current else None),
            output_device_id=(current.output_device_id if current else None),
            sample_rate=self.project.sample_rate,
            track_inputs=self.project.metadata.track_inputs,
            bus_outputs=self.project.metadata.bus_outputs,
        )
        self.settings_window.open(
            draft,
            locked_sample_rate=self.project.sample_rate,
            trusted_settings=(current if self.audio_engine.running else None),
            on_applied=self._project_audio_settings_applied,
        )

    def _session_audio_settings_applied(self, settings: AudioSettings) -> None:
        self.audio_service.apply(settings)
        self.session_audio_settings = settings
        self._update_audio_summary()

    def _project_audio_settings_applied(self, settings: AudioSettings) -> None:
        if self.project is None:
            raise ProjectError("No project is open.")
        if self.mixer_state is None:
            raise ProjectError("The project mixer is not available.")

        old_settings = self.audio_service.current_settings
        reuse_stream = self.audio_engine.running and settings == old_settings
        if not reuse_stream:
            self.audio_service.refresh_devices()
            self.audio_service.validate(settings)
        input_device = self.audio_service.device(settings.input_device_id, "input")
        output_device = self.audio_service.device(settings.output_device_id, "output")
        if input_device is None or output_device is None:
            raise AudioConfigurationError("The selected audio devices disappeared.")

        old_engine_running = self.audio_engine.running
        if not reuse_stream:
            try:
                self.audio_engine.start(
                    settings,
                    self.mixer_state,
                    input_device,
                    output_device,
                )
            except AudioEngineError as exc:
                restored = self._restore_audio_engine(
                    old_settings, old_engine_running
                )
                if not restored:
                    self._set_audio_engine_failure(str(exc), show_dialog=False)
                self._schedule_audio_engine_dialog(str(exc))
                raise

        metadata = self.project.metadata.with_audio(
            input_device.reference,
            output_device.reference,
            settings.track_inputs,
            settings.bus_outputs,
        )
        try:
            self.project.save(metadata)
        except Exception:
            if not reuse_stream:
                self.audio_engine.stop()
                restored = self._restore_audio_engine(
                    old_settings, old_engine_running
                )
                if not restored:
                    self._set_audio_engine_failure(
                        "The previous audio stream could not be restored."
                    )
            raise
        self.audio_service.current_settings = settings
        self.project_audio_error = None
        if self.mixer_view is not None:
            self.mixer_view.update_track_routes(settings.track_inputs)
            self.mixer_view.set_monitoring_available(True)
        self.audio_engine_error = None
        self._update_audio_summary()

    async def new_project(
        self, widget: toga.Widget | None = None, **kwargs: object
    ) -> None:
        settings = self.audio_service.current_settings
        if settings is None:
            await self.main_window.dialog(
                toga.ErrorDialog(
                    "Audio Settings Required",
                    "Configure valid audio devices and a sample rate before "
                    "creating a project.",
                )
            )
            self.preferences()
            return

        path = await self.main_window.dialog(
            toga.SaveFileDialog(
                "New Project",
                suggested_filename="Untitled.wav",
                file_types=["wav"],
            )
        )
        if path is None:
            return
        path = Path(path)
        if path.suffix.lower() != ".wav":
            path = Path(f"{path}.wav")

        input_device = self.audio_service.device(settings.input_device_id, "input")
        output_device = self.audio_service.device(settings.output_device_id, "output")
        if input_device is None or output_device is None:
            await self._show_project_error("The selected audio devices disappeared.")
            return

        metadata = ProjectMetadata(
            input_device=input_device.reference,
            output_device=output_device.reference,
            track_inputs=settings.track_inputs,
            bus_outputs=settings.bus_outputs,
        )
        try:
            project = AudioProject.create(path, settings.sample_rate, metadata)
        except ProjectError as exc:
            await self._show_project_error(str(exc))
            return
        self._enter_project(project)

    async def open_project(
        self, widget: toga.Widget | None = None, **kwargs: object
    ) -> None:
        path = await self.main_window.dialog(
            toga.OpenFileDialog(
                "Open Project",
                file_types=["wav"],
                multiple_select=False,
            )
        )
        if path is None:
            return

        try:
            self.audio_service.refresh_devices()
            import_metadata = ProjectMetadata(
                input_device=self._default_device_reference("input"),
                output_device=self._default_device_reference("output"),
            )
            project = AudioProject.open(Path(path), import_metadata)
        except (AudioConfigurationError, ProjectError) as exc:
            await self._show_project_error(str(exc))
            return
        self._enter_project(project)

    def _default_device_reference(self, direction: str):
        device_id = (
            self.audio_service.default_input_device_id
            if direction == "input"
            else self.audio_service.default_output_device_id
        )
        if device_id is None:
            return None
        device = self.audio_service.device(device_id, direction)
        return device.reference if device else None

    def _enter_project(self, project: AudioProject) -> None:
        self.session_audio_settings = self.audio_service.current_settings
        self.project = project
        self.project_audio_error = self._resolve_project_audio()
        self._update_command_state()
        self.main_window.content = self._build_project_screen()
        self.main_window.title = f"{self.formal_name} — {project.path.name}"
        self._start_project_audio()
        self._update_command_state()
        self._sync_transport_controls()
        self.transport_task = asyncio.create_task(self._transport_clock())
        self._update_audio_summary(self.project_audio_error)

    def _toggle_transport_record(
        self, widget: toga.Widget | None = None, **kwargs: object
    ) -> None:
        if self.transport is None or self.project is None:
            return
        if not self.project.writable and not self.transport.record_armed:
            self._schedule_transport_dialog(
                "This project file is read-only and cannot record."
            )
            return
        self.transport.toggle_record()
        self._sync_transport_controls()

    async def _play_transport(
        self, widget: toga.Widget | None = None, **kwargs: object
    ) -> None:
        if (
            self.transport is None
            or self.mixer_state is None
            or self.audio_service.current_settings is None
            or not self.audio_engine.running
            or self.transport.running
            or self.transport_starting
        ):
            return
        armed_tracks = tuple(
            track.record_enabled for track in self.mixer_state.tracks
        )
        self.transport_starting = True
        if self.mixer_view is not None:
            self.mixer_view.set_transport_running(True)
        self._update_command_state()
        self._sync_transport_controls()
        try:
            started = await asyncio.to_thread(
                self.transport.play,
                self.audio_service.current_settings.track_inputs,
                armed_tracks,
            )
        except TransportError as exc:
            self._schedule_transport_dialog(str(exc))
            started = False
        finally:
            self.transport_starting = False
        if not started and self.mixer_view is not None:
            self.mixer_view.set_transport_running(False)
        self._update_command_state()
        self._sync_transport_controls()

    async def _stop_or_rtz(
        self, widget: toga.Widget | None = None, **kwargs: object
    ) -> None:
        if self.transport is None:
            return
        if self.transport.running:
            await self._stop_transport()
        else:
            self.transport.return_to_zero()
            self._sync_transport_controls()

    async def _stop_transport(self) -> None:
        if (
            self.transport is None
            or self.transport_stopping
            or not self.transport.running
        ):
            return
        self.transport_stopping = True
        self._update_command_state()
        self._sync_transport_controls()
        error: str | None = None
        try:
            await asyncio.to_thread(self.transport.stop)
            error = self.transport.terminal_error
        except Exception as exc:
            error = f"Unable to stop transport cleanly: {exc}"
        finally:
            self.transport_stopping = False
            if self.mixer_view is not None:
                self.mixer_view.set_transport_running(False)
        self._update_command_state()
        self._sync_transport_controls()
        if error is not None:
            self._schedule_transport_dialog(error)

    async def _transport_clock(self) -> None:
        transport = self.transport
        try:
            while self.transport is transport and transport is not None:
                self._sync_transport_controls()
                if transport.running and transport.end_requested:
                    await self._stop_transport()
                await asyncio.sleep(1 / 30)
        except asyncio.CancelledError:
            pass

    def _sync_transport_controls(self) -> None:
        if self.transport is None or not hasattr(
            self, "transport_record_button"
        ):
            return
        busy = self.transport_starting or self.transport_stopping
        rolling = self.transport.running
        engine_available = self.audio_engine.running
        self.transport_record_button.enabled = (
            engine_available
            and not busy
            and bool(self.project and self.project.writable)
        )
        if self.transport.record_armed:
            self.transport_record_button.text = "● Record"
            self.transport_record_button.style.background_color = "#d94b4b"
        else:
            self.transport_record_button.text = "Record"
            del self.transport_record_button.style.background_color
        self.transport_play_button.enabled = (
            engine_available and not rolling and not busy
        )
        self.transport_stop_rtz_button.text = "Stop" if rolling else "RTZ"
        self.transport_stop_rtz_button.enabled = (
            not busy and (rolling or self.transport.position_frames > 0)
        )
        if self.project is not None:
            self.transport_time_label.text = format_transport_time(
                self.transport.position_frames, self.project.sample_rate
            )

    def _schedule_transport_dialog(self, message: str) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(
            self.main_window.dialog(toga.ErrorDialog("Transport Error", message))
        )

    def _start_project_audio(self) -> None:
        settings = self.audio_service.current_settings
        if settings is None or self.project_audio_error is not None:
            self._set_audio_engine_failure(
                self.project_audio_error or "No usable audio devices are available."
            )
            return
        input_device = self.audio_service.device(settings.input_device_id, "input")
        output_device = self.audio_service.device(
            settings.output_device_id, "output"
        )
        if input_device is None or output_device is None:
            self._set_audio_engine_failure(
                "The selected audio devices are no longer available."
            )
            return
        assert self.mixer_state is not None
        try:
            self.audio_engine.start(
                settings,
                self.mixer_state,
                input_device,
                output_device,
            )
        except AudioEngineError as exc:
            self._set_audio_engine_failure(str(exc))
            return
        self.audio_engine_error = None
        if self.mixer_view is not None:
            self.mixer_view.set_monitoring_available(True)
        self._sync_transport_controls()

    def _restore_audio_engine(
        self,
        settings: AudioSettings | None,
        was_running: bool,
    ) -> bool:
        if not was_running or settings is None or self.mixer_state is None:
            return False
        input_device = self.audio_service.device(settings.input_device_id, "input")
        output_device = self.audio_service.device(
            settings.output_device_id, "output"
        )
        if input_device is None or output_device is None:
            return False
        try:
            self.audio_engine.start(
                settings,
                self.mixer_state,
                input_device,
                output_device,
            )
        except AudioEngineError:
            return False
        self.audio_engine_error = None
        if self.mixer_view is not None:
            self.mixer_view.set_monitoring_available(True)
        self._update_audio_summary()
        return True

    def _mixer_changed(self) -> None:
        if self.mixer_state is not None and self.audio_engine.running:
            self.audio_engine.update_mix(self.mixer_state)

    def _set_audio_engine_failure(
        self, message: str, *, show_dialog: bool = True
    ) -> None:
        self.audio_engine.stop()
        self.audio_engine_error = message
        if self.mixer_view is not None:
            self.mixer_view.set_monitoring_available(False)
        self._sync_transport_controls()
        self._update_audio_summary()
        if show_dialog:
            self._schedule_audio_engine_dialog(message)

    def _schedule_audio_engine_dialog(self, message: str) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        loop.create_task(
            self.main_window.dialog(
                toga.ErrorDialog("Audio Engine Error", message)
            )
        )

    def _resolve_project_audio(self) -> str | None:
        assert self.project is not None
        try:
            self.audio_service.refresh_devices()
        except AudioConfigurationError as exc:
            self.audio_service.current_settings = None
            return str(exc)

        metadata = self.project.metadata
        input_device = self.audio_service.resolve_device(
            metadata.input_device, "input"
        )
        output_device = self.audio_service.resolve_device(
            metadata.output_device, "output"
        )
        if input_device is None or output_device is None:
            self.audio_service.current_settings = None
            return "The project has no available input and output device pair."

        settings = AudioSettings(
            input_device.index,
            output_device.index,
            self.project.sample_rate,
            metadata.track_inputs,
            metadata.bus_outputs,
        )
        self.audio_service.current_settings = settings
        return self.audio_service.compatibility_error(settings)

    async def save_project(
        self, widget: toga.Widget | None = None, **kwargs: object
    ) -> None:
        if self.project is None:
            return
        try:
            self.project.save()
        except ProjectError as exc:
            await self._show_project_error(str(exc))
            return
        self._update_audio_summary(self.project_audio_error)

    async def close_project(
        self, widget: toga.Widget | None = None, **kwargs: object
    ) -> None:
        if self.project is None:
            return
        if self.transport is not None and (
            self.transport.running
            or self.transport_starting
            or self.transport_stopping
        ):
            return
        if self.project.dirty:
            discard = await self.main_window.dialog(
                toga.ConfirmDialog(
                    "Discard Unsaved Project Metadata?",
                    "This imported WAV has unsaved Tape Machine metadata. "
                    "Choose Cancel to return and save it first.",
                )
            )
            if not discard:
                return
        try:
            self.project.close()
        except Exception as exc:
            await self._show_project_error(f"Unable to close project: {exc}")
            return
        self._leave_project()

    def _leave_project(self) -> None:
        self.audio_engine.stop()
        self.audio_engine.set_transport(None)
        if self.transport_task is not None:
            self.transport_task.cancel()
        self.transport_task = None
        self.transport_starting = False
        self.transport_stopping = False
        if self.mixer_state is not None:
            self.mixer_state.on_change = None
        self.project = None
        self.mixer_state = None
        self.mixer_view = None
        self.transport = None
        self.project_audio_error = None
        self.audio_engine_error = None
        self.settings_window.window.hide()
        try:
            self.audio_service.refresh_devices()
            self.audio_service.current_settings = self.audio_service.suggest_settings(
                self.session_audio_settings
            )
        except AudioConfigurationError:
            self.audio_service.current_settings = None
        self.session_audio_settings = self.audio_service.current_settings
        self._update_command_state()
        self.main_window.content = self._build_initial_screen()
        self.main_window.title = self.formal_name
        self._update_audio_summary()

    async def on_exit(self) -> bool:
        if self.project is None:
            return True
        if self.transport_starting or self.transport_stopping:
            return False
        if self.transport is not None and self.transport.running:
            await self._stop_transport()
        if self.project.dirty:
            discard = await self.main_window.dialog(
                toga.ConfirmDialog(
                    "Discard Unsaved Project Metadata?",
                    "Exit without saving Tape Machine project metadata?",
                )
            )
            if not discard:
                return False
        try:
            self.project.close()
        except Exception as exc:
            await self._show_project_error(f"Unable to close project: {exc}")
            return False
        self.audio_engine.stop()
        return True

    async def _show_project_error(self, message: str) -> None:
        await self.main_window.dialog(toga.ErrorDialog("Project Error", message))

    def _update_audio_summary(self, error: str | None = None) -> None:
        settings = self.audio_service.current_settings
        input_device = None
        output_device = None
        routing = (
            settings.track_inputs
            if settings is not None
            else self.project.metadata.track_inputs
            if self.project is not None
            else ()
        )
        bus_outputs = (
            settings.bus_outputs
            if settings is not None
            else self.project.metadata.bus_outputs
            if self.project is not None
            else ()
        )
        if settings is not None:
            input_device = self.audio_service.device(
                settings.input_device_id, "input"
            )
            output_device = self.audio_service.device(
                settings.output_device_id, "output"
            )
        input_name = input_device.name if input_device else None
        output_name = output_device.name if output_device else None
        if input_name and output_name:
            device_summary = (
                input_name
                if input_name == output_name
                else f"{input_name} / {output_name}"
            )
        else:
            device_summary = input_name or output_name or "No audio device"

        sample_rate = (
            settings.sample_rate
            if settings is not None
            else self.project.sample_rate
            if self.project is not None
            else None
        )
        sample_rate_summary = (
            f"{sample_rate / 1_000:g} kHz"
            if sample_rate is not None
            else "Sample rate unavailable"
        )

        routing_complete = any(
            channel is not None for channel in routing
        ) and any(channel is not None for channel in bus_outputs)
        parts = [device_summary, sample_rate_summary]
        if not routing_complete:
            parts.append("Routing incomplete")
        if self.project is not None and self.audio_engine_error is not None:
            parts.append("Audio unavailable")
        self.status_line_label.text = "  •  ".join(parts)


def main() -> TapeMachine:
    """Create the application instance used by Briefcase."""
    return TapeMachine(
        formal_name="Tape Machine",
        app_id="pkg.unusualaudio.tape-machine",
    )
