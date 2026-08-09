"""The Tape Machine Toga application."""

import asyncio
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable

import toga
from rubicon.objc import NSObject, ObjCClass, objc_const, objc_method
from rubicon.objc.runtime import load_library
from toga.style.pack import CENTER, COLUMN, ROW, START
from toga_cocoa.libs import NSCursor

from tape_machine.audio import (
    UNASSIGNED_BUS_OUTPUTS,
    UNASSIGNED_TRACK_INPUTS,
    AudioConfigurationError,
    AudioDeviceService,
    AudioSettings,
)
from tape_machine.config import (
    AppConfig,
    AppConfigError,
    AppConfigStore,
    MAX_RECENT_FILES,
    StoredAudioSettings,
    WindowPosition,
)
from tape_machine.engine import AudioEngine, AudioEngineError
from tape_machine.mixer import MixerState, MixerView
from tape_machine.project import AudioProject, ProjectError, ProjectMetadata
from tape_machine.settings import AudioSettingsDraft, AudioSettingsWindow
from tape_machine.theme import (
    ACCENT_BLUE,
    ACCENT_RED,
    SECONDARY_TEXT,
    force_dark_appearance,
)
from tape_machine.transport import (
    TransportController,
    TransportError,
    TransportMode,
    format_transport_time,
)


_TRANSPORT_BUTTON_WIDTH = 80
_TRANSPORT_BAR_WIDTH = 880
_LOGO_DISPLAY_WIDTH = 208
_LOGO_DISPLAY_HEIGHT = 20
_TRANSPORT_CONTROL_HEIGHT = 28
_MAIN_CONTENT_HORIZONTAL_MARGIN = 16
_PROJECT_CONTENT_TOP_MARGIN = 12
_INITIAL_CONTENT_TOP_MARGIN = _PROJECT_CONTENT_TOP_MARGIN + (
    _TRANSPORT_CONTROL_HEIGHT - _LOGO_DISPLAY_HEIGHT
) // 2
_LOGO_RESOURCE = Path(__file__).with_name("resources") / "logo@2x.png"
_MAIN_WINDOW_SIZE = (912, 584)
_AUDIO_SETTINGS_WINDOW_SIZE = (900, 640)
_RECORD_BUTTON_TEXT = "● REC"
_REWIND_BUTTON_TEXT = "◀◀ REW"
_PLAY_BUTTON_TEXT = "▶ PLAY"
_RTZ_BUTTON_TEXT = "⇤ RTZ"
_STOP_BUTTON_TEXT = "■ STOP"
_FAST_FORWARD_BUTTON_TEXT = "▶▶ FWD"
_NSTRACKING_IN_VISIBLE_RECT = 0x200
_NUMBER_SPACING_FEATURE_TYPE = 6
_TABULAR_NUMBERS_SELECTOR = 0


class LinkCursorOwner(NSObject):
    """Set the pointing-hand cursor while it is over a text link."""

    @objc_method
    def cursorUpdate_(self, event) -> None:
        NSCursor.pointingHandCursor.set()


def _tabular_number_font(font: object) -> object:
    """Enable tabular figures while preserving the font's family and traits."""
    from toga_cocoa.libs import NSMutableArray, NSMutableDictionary, NSFont

    appkit = load_library("AppKit")
    settings_key = objc_const(appkit, "NSFontFeatureSettingsAttribute")
    type_key = objc_const(appkit, "NSFontFeatureTypeIdentifierKey")
    selector_key = objc_const(
        appkit, "NSFontFeatureSelectorIdentifierKey"
    )

    feature = NSMutableDictionary.alloc().init()
    feature[type_key] = _NUMBER_SPACING_FEATURE_TYPE
    feature[selector_key] = _TABULAR_NUMBERS_SELECTOR
    features = NSMutableArray.alloc().init()
    features.addObject(feature)
    attributes = NSMutableDictionary.alloc().init()
    attributes[settings_key] = features
    descriptor = font.fontDescriptor.fontDescriptorByAddingAttributes(
        attributes
    )
    return NSFont.fontWithDescriptor(descriptor, size=font.pointSize)


def _style_time_counter(label: toga.Label) -> None:
    """Apply stable-width tabular numerals to the proportional time font."""
    native = label._impl.native
    native.font = _tabular_number_font(native.font)


def _fit_window_position(
    position: WindowPosition | None,
    window_size: tuple[int, int],
    screens: list[object],
) -> tuple[int, int] | None:
    """Keep a restored window fully visible on an attached screen."""
    if position is None or not screens:
        return None
    target_screen = next(
        (
            screen
            for screen in screens
            if screen.origin.x <= position.x < screen.origin.x + screen.size.width
            and screen.origin.y <= position.y < screen.origin.y + screen.size.height
        ),
        screens[0],
    )
    minimum_x = target_screen.origin.x
    minimum_y = target_screen.origin.y
    maximum_x = minimum_x + max(0, target_screen.size.width - window_size[0])
    maximum_y = minimum_y + max(0, target_screen.size.height - window_size[1])
    return (
        max(minimum_x, min(position.x, maximum_x)),
        max(minimum_y, min(position.y, maximum_y)),
    )


def _style_link(
    button: toga.Button,
    *,
    color: str,
    tooltip: str | None = None,
) -> None:
    """Give a native Cocoa button the appearance of an inline text link."""
    from toga.colors import Color
    from toga_cocoa.colors import native_color
    from toga_cocoa.libs import (
        NSMutableDictionary,
        NSAttributedString,
        NSFontAttributeName,
        NSForegroundColorAttributeName,
        NSMakeRect,
        NSTrackingActiveInActiveApp,
        NSTrackingCursorUpdate,
        NSUnderlineStyleAttributeName,
    )

    native = button._impl.native
    attributes = NSMutableDictionary.alloc().init()
    attributes[NSFontAttributeName] = native.font
    attributes[NSForegroundColorAttributeName] = native_color(
        Color.parse(color)
    )
    attributes[NSUnderlineStyleAttributeName] = 1
    native.bordered = False
    native.toolTip = tooltip
    native.attributedTitle = NSAttributedString.alloc().initWithString(
        button.text, attributes=attributes
    )
    cursor_owner = LinkCursorOwner.alloc().init()
    tracking_area = ObjCClass("NSTrackingArea").alloc().initWithRect(
        NSMakeRect(0, 0, 0, 0),
        options=(
            NSTrackingCursorUpdate
            | NSTrackingActiveInActiveApp
            | _NSTRACKING_IN_VISIBLE_RECT
        ),
        owner=cursor_owner,
        userInfo=None,
    )
    native.addTrackingArea(tracking_area)
    button._link_cursor_owner = cursor_owner
    button._link_tracking_area = tracking_area
    button._impl.rehint()


def _style_status_link(button: toga.Button) -> None:
    """Style the routing warning as a red settings link."""
    _style_link(
        button,
        color=ACCENT_RED,
        tooltip="Open Audio Settings",
    )


def _display_parent_path(path: Path) -> str:
    """Format a recent project's parent path compactly for the launcher."""
    parent = path.expanduser().resolve(strict=False).parent
    home = Path.home().resolve(strict=False)
    try:
        relative = parent.relative_to(home)
    except ValueError:
        return str(parent)
    return "~" if relative == Path(".") else str(Path("~") / relative)


def _link_button(
    text: str,
    action: Callable[..., object],
    *,
    tooltip: str,
    font_size: int = 12,
) -> toga.Button:
    """Create one blue, underlined Cocoa text link."""
    button = toga.Button(
        text,
        on_press=action,
        height=20,
        font_size=font_size,
        color=ACCENT_BLUE,
    )
    _style_link(button, color=ACCENT_BLUE, tooltip=tooltip)
    return button


class ShuttleButton:
    """Native transport button with distinct press and release events."""

    _MOUSE_DOWN_EVENT = 1
    _MOUSE_UP_EVENT = 2
    _MOUSE_EVENT_MASK = (1 << _MOUSE_DOWN_EVENT) | (1 << _MOUSE_UP_EVENT)

    def __init__(
        self,
        text: str,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
        *,
        width: int,
    ) -> None:
        self.text = text
        self.width = width
        self.height = _TRANSPORT_CONTROL_HEIGHT
        self.on_press = on_press
        self.on_release = on_release
        self._enabled = True
        self._active = False
        self._pointer_down = False
        self.widget = toga.Button(
            text,
            on_press=self._native_event,
            width=width,
            height=self.height,
        )
        self.widget._impl.native.sendActionOn(self._MOUSE_EVENT_MASK)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, enabled: bool) -> None:
        if self._enabled != enabled:
            self._enabled = enabled
            # Keep the native control tracking until mouse-up so a momentary
            # shuttle always receives its release, even if transport state
            # temporarily disables the button while it is held.
            if not self._pointer_down:
                self.widget.enabled = enabled

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, active: bool) -> None:
        if self._active != active:
            self._active = active
            if active:
                self.widget.style.background_color = ACCENT_BLUE
            else:
                del self.widget.style.background_color

    def _native_event(self, widget: toga.Button, **kwargs: object) -> None:
        event_type = int(widget._impl.native.window.currentEvent().type)
        if event_type == self._MOUSE_DOWN_EVENT:
            self._press(widget)
        elif event_type == self._MOUSE_UP_EVENT:
            self._release(widget)
        else:
            # Keyboard and accessibility activation behaves like a click.
            self._press(widget)
            self._release(widget)

    def _press(self, widget: toga.Button, **kwargs: object) -> None:
        if not self._enabled:
            return
        self._pointer_down = True
        self.on_press()

    def _release(self, widget: toga.Button, **kwargs: object) -> None:
        if not self._pointer_down:
            return
        self._pointer_down = False
        self.on_release()
        self.widget.enabled = self._enabled


class TapeMachine(toga.App):
    """The main Tape Machine application."""

    def startup(self) -> None:
        """Create and show the main application window."""
        force_dark_appearance(self._impl.native)
        self.config_store = AppConfigStore(self.paths.config / "config.json")
        try:
            self.app_config = self.config_store.load()
        except AppConfigError as exc:
            print(exc, file=sys.stderr)
            self.app_config = AppConfig()

        self.audio_service = AudioDeviceService()
        self.audio_engine = AudioEngine()
        try:
            self.audio_service.refresh_devices()
            stored_audio = self.app_config.audio_settings
            preferred_audio = (
                stored_audio.resolve(self.audio_service)
                if stored_audio is not None
                else None
            )
            self.audio_service.current_settings = self.audio_service.suggest_settings(
                preferred_audio
            )
            self.global_audio_settings = self.audio_service.current_settings
            startup_error = None
        except AudioConfigurationError as exc:
            self.audio_service.current_settings = None
            self.global_audio_settings = None
            startup_error = str(exc)

        self.project: AudioProject | None = None
        self.mixer_state: MixerState | None = None
        self.mixer_view: MixerView | None = None
        self.transport: TransportController | None = None
        self.transport_task: asyncio.Task[None] | None = None
        self.transport_starting = False
        self.transport_stopping = False
        self.momentary_shuttle_mode: TransportMode | None = None
        self.momentary_shuttle_release: asyncio.Event | None = None
        self.momentary_shuttle_task: asyncio.Task[None] | None = None
        self.momentary_shuttle_resume = False
        self.project_audio_error: str | None = None
        self.audio_engine_error: str | None = None

        self.status_line_label = toga.Label("", font_size=11)
        self.routing_status_link = toga.Button(
            "Routing incomplete",
            id="routing-status-link",
            on_press=self._open_routing_settings,
            height=18,
            font_size=11,
            color=ACCENT_RED,
        )
        _style_status_link(self.routing_status_link)
        self.routing_status_link_group = toga.Box(
            children=[
                toga.Label("  •  ", font_size=11),
                self.routing_status_link,
            ],
            direction=ROW,
            align_items=CENTER,
        )
        self._routing_status_link_visible = False
        self.status_line_suffix_label = toga.Label("", font_size=11)
        self.status_line_content = toga.Box(
            children=[
                self.status_line_label,
                self.status_line_suffix_label,
            ],
            direction=ROW,
            align_items=CENTER,
        )

        self.main_window = toga.MainWindow(
            title=self.formal_name,
            position=_fit_window_position(
                self.app_config.main_window_position,
                _MAIN_WINDOW_SIZE,
                self.screens,
            ),
            size=_MAIN_WINDOW_SIZE,
            resizable=False,
        )
        self.main_window.content = self._build_initial_screen()

        self.settings_window = AudioSettingsWindow(
            self.audio_service,
            self._audio_settings_applied,
            position=_fit_window_position(
                self.app_config.audio_settings_window_position,
                _AUDIO_SETTINGS_WINDOW_SIZE,
                self.screens,
            ),
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
        self.recent_files_group = toga.Group(
            "Open Recent",
            parent=toga.Group.FILE,
            section=0,
            order=20,
            id="open-recent",
        )
        self.recent_file_commands: list[toga.Command] = []
        self.recent_menu_commands: list[toga.Command] = []
        self.commands.add(
            self.settings_command,
            self.new_project_command,
            self.open_project_command,
            self.close_project_command,
            self.save_project_command,
        )
        self._rebuild_recent_files_menu()
        self._update_command_state()
        self._update_audio_summary(startup_error)
        self.main_window.show()

    def _build_initial_screen(self) -> toga.Box:
        self.initial_logo_view = toga.ImageView(
            _LOGO_RESOURCE,
            width=_LOGO_DISPLAY_WIDTH,
            height=_LOGO_DISPLAY_HEIGHT,
        )
        action_links = toga.Box(
            children=[
                _link_button(
                    "New Project",
                    self.new_project,
                    tooltip="Create a new project",
                    font_size=11,
                ),
                _link_button(
                    "Open Project…",
                    self.open_project,
                    tooltip="Open an existing WAV project",
                    font_size=11,
                ),
            ],
            direction=ROW,
            align_items=CENTER,
            gap=22,
            margin_top=24,
        )
        self.initial_recent_files_box = toga.Box(
            children=self._initial_recent_file_widgets(),
            direction=COLUMN,
            align_items=START,
            gap=5,
        )
        recent_section = toga.Box(
            children=[
                toga.Label(
                    "Recent Projects",
                    color=SECONDARY_TEXT,
                    font_size=11,
                    font_weight="bold",
                    margin_bottom=8,
                ),
                self.initial_recent_files_box,
            ],
            direction=COLUMN,
            align_items=START,
            width=640,
            margin_top=38,
        )
        central_content = toga.Box(
            children=[
                self.initial_logo_view,
                action_links,
                recent_section,
            ],
            direction=COLUMN,
            align_items=START,
            justify_content=START,
            flex=1,
            margin_top=_INITIAL_CONTENT_TOP_MARGIN,
            margin_left=_MAIN_CONTENT_HORIZONTAL_MARGIN,
            margin_right=_MAIN_CONTENT_HORIZONTAL_MARGIN,
            margin_bottom=12,
        )
        return toga.Box(
            children=[
                central_content,
                self._build_status_footer(),
            ],
            direction=COLUMN,
        )

    def _initial_recent_file_widgets(self) -> list[toga.Widget]:
        """Build the current recent-project rows for the initial screen."""
        if not self.app_config.recent_files:
            return [
                toga.Label(
                    "No recent projects",
                    color=SECONDARY_TEXT,
                    font_size=11,
                )
            ]

        rows: list[toga.Widget] = []
        for path in self.app_config.recent_files[:MAX_RECENT_FILES]:
            rows.append(
                toga.Box(
                    children=[
                        _link_button(
                            path.name,
                            self._recent_file_action(path),
                            tooltip=str(path),
                            font_size=11,
                        ),
                        toga.Label(
                            _display_parent_path(path),
                            color=SECONDARY_TEXT,
                            font_size=10,
                            flex=1,
                        ),
                    ],
                    direction=ROW,
                    align_items=CENTER,
                    gap=10,
                    width=640,
                )
            )
        return rows

    def _refresh_initial_recent_files(self) -> None:
        """Synchronize the visible launcher list with application config."""
        recent_box = getattr(self, "initial_recent_files_box", None)
        if recent_box is None or getattr(self, "project", None) is not None:
            return
        for child in list(recent_box.children):
            recent_box.remove(child)
        recent_box.add(*self._initial_recent_file_widgets())

    def _build_project_screen(self) -> toga.Box:
        assert self.project is not None
        project = self.project
        settings = self.audio_service.current_settings
        track_inputs = (
            settings.track_inputs
            if settings is not None
            else UNASSIGNED_TRACK_INPUTS
        )
        self.mixer_state = MixerState.from_metadata(
            track_inputs, project.metadata.mix
        )
        self.mixer_view = MixerView(track_inputs, self.mixer_state)
        self.mixer_state.on_change = self._mixer_changed
        self.transport = TransportController(project)
        self.audio_engine.set_transport(self.transport)
        self.transport_record_button = toga.Button(
            _RECORD_BUTTON_TEXT,
            on_press=self._toggle_transport_record,
            width=_TRANSPORT_BUTTON_WIDTH,
            height=_TRANSPORT_CONTROL_HEIGHT,
        )
        self.transport_play_button = toga.Button(
            _PLAY_BUTTON_TEXT,
            on_press=self._play_transport,
            width=_TRANSPORT_BUTTON_WIDTH,
            height=_TRANSPORT_CONTROL_HEIGHT,
        )
        self.transport_rewind_button = ShuttleButton(
            _REWIND_BUTTON_TEXT,
            on_press=lambda: self._shuttle_pressed(TransportMode.REWIND),
            on_release=lambda: self._shuttle_released(TransportMode.REWIND),
            width=_TRANSPORT_BUTTON_WIDTH,
        )
        self.transport_fast_forward_button = ShuttleButton(
            _FAST_FORWARD_BUTTON_TEXT,
            on_press=lambda: self._shuttle_pressed(
                TransportMode.FAST_FORWARD
            ),
            on_release=lambda: self._shuttle_released(
                TransportMode.FAST_FORWARD
            ),
            width=_TRANSPORT_BUTTON_WIDTH,
        )
        self.transport_stop_rtz_button = toga.Button(
            _RTZ_BUTTON_TEXT,
            on_press=self._stop_or_rtz,
            width=_TRANSPORT_BUTTON_WIDTH,
            height=_TRANSPORT_CONTROL_HEIGHT,
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
        _style_time_counter(self.transport_time_label)
        self.logo_view = toga.ImageView(
            _LOGO_RESOURCE,
            width=_LOGO_DISPLAY_WIDTH,
            height=_LOGO_DISPLAY_HEIGHT,
        )
        transport_bar = toga.Box(
            children=[
                self.logo_view,
                toga.Box(flex=1),
                self.transport_rewind_button.widget,
                self.transport_fast_forward_button.widget,
                self.transport_stop_rtz_button,
                self.transport_play_button,
                self.transport_record_button,
                self.transport_time_label,
            ],
            direction=ROW,
            align_items=CENTER,
            width=_TRANSPORT_BAR_WIDTH,
            gap=6,
            margin_bottom=2,
        )
        central_content = toga.Box(
            children=[transport_bar, self.mixer_view.widget],
            direction=COLUMN,
            align_items=CENTER,
            flex=1,
            margin_top=_PROJECT_CONTENT_TOP_MARGIN,
            margin_left=_MAIN_CONTENT_HORIZONTAL_MARGIN,
            margin_right=_MAIN_CONTENT_HORIZONTAL_MARGIN,
        )
        return toga.Box(
            children=[central_content, self._build_status_footer()],
            direction=COLUMN,
        )

    def _build_status_footer(self) -> toga.Box:
        return toga.Box(
            children=[
                toga.Divider(margin_bottom=8),
                self.status_line_content,
            ],
            direction=COLUMN,
            margin_left=_MAIN_CONTENT_HORIZONTAL_MARGIN,
            margin_right=_MAIN_CONTENT_HORIZONTAL_MARGIN,
            margin_bottom=12,
        )

    def _recent_file_action(self, path: Path) -> Callable[..., object]:
        async def action(
            widget: toga.Widget | toga.Command | None = None,
            **kwargs: object,
        ) -> None:
            await self._open_project_path(path)

        return action

    def _rebuild_recent_files_menu(self) -> None:
        for command in self.recent_menu_commands:
            self.commands.discard(command)

        file_commands: list[toga.Command] = []
        menu_commands: list[toga.Command] = []
        if self.app_config.recent_files:
            for order, path in enumerate(self.app_config.recent_files):
                command = toga.Command(
                    self._recent_file_action(path),
                    path.name,
                    tooltip=str(path),
                    group=self.recent_files_group,
                    order=order,
                    enabled=self.project is None,
                    id=f"open-recent-{order}",
                )
                file_commands.append(command)
                menu_commands.append(command)
        else:
            menu_commands.append(
                toga.Command(
                    None,
                    "No Recent Files",
                    group=self.recent_files_group,
                    enabled=False,
                    id="open-recent-empty",
                )
            )

        menu_commands.append(
            toga.Command(
                self.clear_recent_files,
                "Clear Menu",
                group=self.recent_files_group,
                section=10,
                enabled=bool(self.app_config.recent_files),
                id="clear-recent-files",
            )
        )
        self.recent_file_commands = file_commands
        self.recent_menu_commands = menu_commands
        self.commands.add(*menu_commands)
        self._refresh_initial_recent_files()

    def _persist_noncritical_config(self) -> None:
        try:
            self.config_store.save(self.app_config)
        except AppConfigError as exc:
            print(exc, file=sys.stderr)

    def _remember_recent_file(self, path: Path) -> None:
        updated = self.app_config.with_recent_file(path)
        if updated == self.app_config:
            return
        self.app_config = updated
        self._persist_noncritical_config()
        self._rebuild_recent_files_menu()

    def _forget_recent_file(self, path: Path) -> None:
        updated = self.app_config.without_recent_file(path)
        if updated == self.app_config:
            return
        self.app_config = updated
        self._persist_noncritical_config()
        self._rebuild_recent_files_menu()

    def clear_recent_files(
        self,
        widget: toga.Widget | toga.Command | None = None,
        **kwargs: object,
    ) -> None:
        if not self.app_config.recent_files:
            return
        self.app_config = self.app_config.clear_recent_files()
        self._persist_noncritical_config()
        self._rebuild_recent_files_menu()

    def _update_command_state(self) -> None:
        project_is_open = self.project is not None
        transport_busy = bool(
            self.momentary_shuttle_task is not None
            or self.transport
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
        for command in getattr(self, "recent_file_commands", ()):
            command.enabled = not project_is_open

    def preferences(
        self, widget: toga.Widget | toga.Command | None = None, **kwargs: object
    ) -> None:
        """Open the global Audio Settings, using a project's fixed rate."""
        if self.project is None:
            self.settings_window.open()
            return

        current = self.audio_service.current_settings
        fallback = current or self.global_audio_settings
        draft = AudioSettingsDraft(
            input_device_id=(fallback.input_device_id if fallback else None),
            output_device_id=(fallback.output_device_id if fallback else None),
            sample_rate=self.project.sample_rate,
            track_inputs=(
                fallback.track_inputs if fallback else UNASSIGNED_TRACK_INPUTS
            ),
            bus_outputs=(
                fallback.bus_outputs if fallback else UNASSIGNED_BUS_OUTPUTS
            ),
            buffer_size=(fallback.buffer_size if fallback else 0),
        )
        self.settings_window.open(
            draft,
            locked_sample_rate=self.project.sample_rate,
        )

    def _open_routing_settings(
        self, widget: toga.Widget | None = None, **kwargs: object
    ) -> None:
        self.preferences()

    def _set_routing_status_link_visible(self, visible: bool) -> None:
        if visible == self._routing_status_link_visible:
            return
        self._routing_status_link_visible = visible
        if visible:
            self.status_line_content.insert(1, self.routing_status_link_group)
        else:
            self.status_line_content.remove(self.routing_status_link_group)

    def _updated_config_for_audio_settings(
        self, settings: AudioSettings
    ) -> AppConfig:
        input_device = self.audio_service.device(settings.input_device_id, "input")
        output_device = self.audio_service.device(settings.output_device_id, "output")
        if input_device is None or output_device is None:
            raise AudioConfigurationError("The selected audio devices disappeared.")
        stored = StoredAudioSettings.from_settings(
            settings, input_device, output_device
        )
        return self.app_config.with_audio_settings(stored)

    def _audio_settings_applied(self, settings: AudioSettings) -> None:
        if self.project is not None:
            self._project_audio_settings_applied(settings)
            return

        self.audio_service.refresh_devices()
        self.audio_service.validate(settings)
        updated = self._updated_config_for_audio_settings(settings)
        self.config_store.save(updated)
        self.app_config = updated
        self.global_audio_settings = settings
        self.audio_service.current_settings = settings
        self._update_audio_summary()

    def _project_audio_settings_applied(self, settings: AudioSettings) -> None:
        if self.project is None:
            raise ProjectError("No project is open.")
        if self.mixer_state is None:
            raise ProjectError("The project mixer is not available.")

        old_settings = self.audio_service.current_settings
        old_global_settings = self.global_audio_settings
        old_config = self.app_config
        old_engine_running = self.audio_engine.running
        reuse_stream = old_engine_running and settings == old_settings
        if not reuse_stream:
            if old_engine_running:
                self.audio_engine.stop()
            try:
                self.audio_service.refresh_devices()
                self.audio_service.validate(settings)
                input_device = self.audio_service.device(
                    settings.input_device_id, "input"
                )
                output_device = self.audio_service.device(
                    settings.output_device_id, "output"
                )
                if input_device is None or output_device is None:
                    raise AudioConfigurationError(
                        "The selected audio devices disappeared."
                    )
            except AudioConfigurationError:
                restored = self._restore_audio_engine(
                    old_settings, old_engine_running
                )
                if old_engine_running and not restored:
                    self._set_audio_engine_failure(
                        "The previous audio stream could not be restored.",
                        show_dialog=False,
                    )
                raise
        else:
            input_device = self.audio_service.device(
                settings.input_device_id, "input"
            )
            output_device = self.audio_service.device(
                settings.output_device_id, "output"
            )
            if input_device is None or output_device is None:
                raise AudioConfigurationError(
                    "The selected audio devices disappeared."
                )

        global_sample_rate = (
            old_global_settings.sample_rate
            if old_global_settings is not None
            else self.app_config.audio_settings.sample_rate
            if self.app_config.audio_settings is not None
            else settings.sample_rate
        )
        global_settings = replace(settings, sample_rate=global_sample_rate)
        updated_config = self._updated_config_for_audio_settings(global_settings)

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

        try:
            self.config_store.save(updated_config)
        except Exception:
            if not reuse_stream:
                self.audio_engine.stop()
                restored = self._restore_audio_engine(
                    old_settings, old_engine_running
                )
                if old_engine_running and not restored:
                    self._set_audio_engine_failure(
                        "The previous audio stream could not be restored."
                    )
            self.app_config = old_config
            self.global_audio_settings = old_global_settings
            raise
        self.app_config = updated_config
        self.global_audio_settings = global_settings
        self.audio_service.current_settings = settings
        self.project_audio_error = None
        if self.mixer_view is not None:
            self.mixer_view.update_track_routes(settings.track_inputs)
            self.mixer_view.set_monitoring_available(True)
        else:
            self.mixer_state.update_input_routes(settings.track_inputs)
        self.project.stage_metadata(
            self.project.metadata.with_mix(
                self.mixer_state.to_metadata(settings.track_inputs)
            )
        )
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

        try:
            project = AudioProject.create(
                path, settings.sample_rate, ProjectMetadata()
            )
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

        await self._open_project_path(Path(path))

    async def _open_project_path(self, path: Path) -> bool:
        """Open one known path and share handling with the recent-files menu."""

        try:
            project = AudioProject.open(path)
        except ProjectError as exc:
            if not path.exists():
                self._forget_recent_file(path)
            await self._show_project_error(str(exc))
            return False
        self._enter_project(project)
        return True

    def _enter_project(self, project: AudioProject) -> None:
        self.settings_window.window.hide()
        self._remember_recent_file(project.path)
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
        if self.momentary_shuttle_task is not None:
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
            self.mixer_view.set_record_enable_locked(True)
        self._update_command_state()
        self._sync_transport_controls()
        try:
            await asyncio.to_thread(
                self.transport.play,
                self.audio_service.current_settings.track_inputs,
                armed_tracks,
            )
        except TransportError as exc:
            self._schedule_transport_dialog(str(exc))
        finally:
            self.transport_starting = False
            if self.mixer_view is not None:
                self.mixer_view.set_record_enable_locked(False)
        self._update_command_state()
        self._sync_transport_controls()

    def _shuttle_pressed(self, requested_mode: TransportMode) -> None:
        transport = self.transport
        if (
            transport is None
            or not self.audio_engine.running
            or self.transport_starting
            or self.transport_stopping
            or self.momentary_shuttle_task is not None
        ):
            return
        if transport.mode is TransportMode.PLAYING:
            if transport.record_armed:
                return
            release = asyncio.Event()
            self.momentary_shuttle_mode = requested_mode
            self.momentary_shuttle_release = release
            self.momentary_shuttle_resume = True
            self.momentary_shuttle_task = asyncio.create_task(
                self._run_momentary_shuttle(requested_mode, release)
            )
            self._sync_transport_controls()
            return
        asyncio.create_task(self._toggle_shuttle(requested_mode))

    def _shuttle_released(self, requested_mode: TransportMode) -> None:
        if self.momentary_shuttle_mode is not requested_mode:
            return
        if self.momentary_shuttle_release is not None:
            self.momentary_shuttle_release.set()

    async def _run_momentary_shuttle(
        self, requested_mode: TransportMode, release: asyncio.Event
    ) -> None:
        transport = self.transport
        try:
            if (
                transport is None
                or transport.mode is not TransportMode.PLAYING
                or transport.record_armed
            ):
                return
            await self._stop_transport()
            if self.transport is not transport:
                return
            if transport.terminal_error is not None:
                return
            if not release.is_set():
                await self._toggle_shuttle(requested_mode)
            await release.wait()
            if self.transport is not transport:
                return
            if transport.mode.shuttling:
                await self._stop_transport()
            if (
                self.momentary_shuttle_resume
                and transport.terminal_error is None
            ):
                await self._play_transport()
        finally:
            if self.momentary_shuttle_task is asyncio.current_task():
                self.momentary_shuttle_mode = None
                self.momentary_shuttle_release = None
                self.momentary_shuttle_task = None
                self.momentary_shuttle_resume = False
                self._sync_transport_controls()

    async def _end_momentary_shuttle(self) -> None:
        task = self.momentary_shuttle_task
        if task is None or task is asyncio.current_task():
            return
        self.momentary_shuttle_resume = False
        if self.momentary_shuttle_release is not None:
            self.momentary_shuttle_release.set()
        await task

    async def _toggle_shuttle(self, requested_mode: TransportMode) -> None:
        if (
            self.transport is None
            or not self.audio_engine.running
            or self.transport_starting
            or self.transport_stopping
        ):
            return

        previous_mode = self.transport.mode
        if self.transport.running:
            if not previous_mode.shuttling:
                return
            await self._stop_transport()
            if previous_mode is requested_mode:
                return

        self.transport_starting = True
        if self.mixer_view is not None:
            self.mixer_view.set_record_enable_locked(True)
        self._update_command_state()
        self._sync_transport_controls()
        try:
            start = (
                self.transport.rewind
                if requested_mode is TransportMode.REWIND
                else self.transport.fast_forward
            )
            await asyncio.to_thread(start)
        except TransportError as exc:
            self._schedule_transport_dialog(str(exc))
        finally:
            self.transport_starting = False
            if self.mixer_view is not None:
                self.mixer_view.set_record_enable_locked(False)
        self._update_command_state()
        self._sync_transport_controls()

    async def _stop_or_rtz(
        self, widget: toga.Widget | None = None, **kwargs: object
    ) -> None:
        if self.transport is None:
            return
        if self.momentary_shuttle_task is not None:
            await self._end_momentary_shuttle()
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
        if self.mixer_view is not None:
            self.mixer_view.set_record_enable_locked(True)
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
                self.mixer_view.set_record_enable_locked(False)
        self._update_command_state()
        self._sync_transport_controls()
        if error is not None:
            self._schedule_transport_dialog(error)

    async def _transport_clock(self) -> None:
        transport = self.transport
        try:
            while self.transport is transport and transport is not None:
                self._sync_transport_controls()
                self._sync_meters()
                if transport.running and transport.end_requested:
                    await self._stop_transport()
                await asyncio.sleep(1 / 30)
        except asyncio.CancelledError:
            pass

    def _sync_meters(self) -> None:
        if self.mixer_view is None:
            return
        snapshot = self.audio_engine.meter_snapshot
        self.mixer_view.set_meter_levels(
            snapshot.track_db, snapshot.bus_db
        )

    def _sync_transport_controls(self) -> None:
        if self.transport is None or not hasattr(
            self, "transport_record_button"
        ):
            return
        busy = self.transport_starting or self.transport_stopping
        rolling = self.transport.running
        mode = self.transport.mode
        shuttling = mode.shuttling
        engine_available = self.audio_engine.running
        momentary_shuttle_active = self.momentary_shuttle_task is not None
        self.transport_record_button.enabled = (
            engine_available
            and not busy
            and not shuttling
            and not momentary_shuttle_active
            and bool(self.project and self.project.writable)
        )
        self.transport_record_button.text = _RECORD_BUTTON_TEXT
        if self.transport.record_armed:
            self.transport_record_button.style.background_color = ACCENT_RED
        else:
            del self.transport_record_button.style.background_color
        self.transport_play_button.enabled = (
            engine_available
            and not rolling
            and not busy
            and not momentary_shuttle_active
        )
        shuttle_controls_enabled = (
            engine_available
            and not busy
            and not momentary_shuttle_active
            and (
                mode is TransportMode.STOPPED
                or shuttling
                or (
                    mode is TransportMode.PLAYING
                    and not self.transport.record_armed
                )
            )
        )
        self.transport_rewind_button.enabled = (
            shuttle_controls_enabled
            and (
                shuttling
                or self.transport.position_frames > 0
            )
        )
        self.transport_fast_forward_button.enabled = (
            shuttle_controls_enabled
            and bool(
                shuttling
                or self.project
                and self.transport.position_frames < self.project.frames
            )
        )
        self.transport_rewind_button.active = mode is TransportMode.REWIND
        self.transport_fast_forward_button.active = (
            mode is TransportMode.FAST_FORWARD
        )
        self.transport_stop_rtz_button.text = (
            _STOP_BUTTON_TEXT
            if rolling or momentary_shuttle_active
            else _RTZ_BUTTON_TEXT
        )
        self.transport_stop_rtz_button.enabled = (
            not busy
            and (
                rolling
                or momentary_shuttle_active
                or self.transport.position_frames > 0
            )
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
        if self.mixer_state is not None and self.transport is not None:
            self.transport.set_armed_tracks(
                tuple(
                    track.record_enabled for track in self.mixer_state.tracks
                )
            )
        if self.mixer_state is not None and self.audio_engine.running:
            self.audio_engine.update_mix(self.mixer_state)
        if self.mixer_state is not None and self.project is not None:
            settings = self.audio_service.current_settings
            track_inputs = (
                settings.track_inputs
                if settings is not None
                else UNASSIGNED_TRACK_INPUTS
            )
            self.project.stage_metadata(
                self.project.metadata.with_mix(
                    self.mixer_state.to_metadata(track_inputs)
                )
            )

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

        global_settings = self.audio_service.suggest_settings(
            self.global_audio_settings
        )
        self.global_audio_settings = global_settings
        if global_settings is None:
            self.audio_service.current_settings = None
            return "No usable input and output device pair is available."

        settings = replace(
            global_settings, sample_rate=self.project.sample_rate
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
            or self.momentary_shuttle_task is not None
        ):
            return
        if self.project.dirty:
            discard = await self.main_window.dialog(
                toga.ConfirmDialog(
                    "Discard Unsaved Changes?",
                    "Close without saving changes to this project?",
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
        if self.momentary_shuttle_task is not None:
            self.momentary_shuttle_task.cancel()
        self.momentary_shuttle_mode = None
        self.momentary_shuttle_release = None
        self.momentary_shuttle_task = None
        self.momentary_shuttle_resume = False
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
                self.global_audio_settings
            )
            self.global_audio_settings = self.audio_service.current_settings
        except AudioConfigurationError:
            self.audio_service.current_settings = None
            self.global_audio_settings = None
        self._update_command_state()
        self.main_window.content = self._build_initial_screen()
        self.main_window.title = self.formal_name
        self._update_audio_summary()

    def _save_window_positions(self) -> None:
        try:
            main_position = self.main_window.position
            settings_position = self.settings_window.window.position
        except (AttributeError, RuntimeError) as exc:
            print(f"Unable to read window positions: {exc}", file=sys.stderr)
            return
        self.app_config = self.app_config.with_window_positions(
            WindowPosition(main_position.x, main_position.y),
            WindowPosition(settings_position.x, settings_position.y),
        )
        self._persist_noncritical_config()

    async def on_exit(self) -> bool:
        if self.project is None:
            self._save_window_positions()
            return True
        if self.transport_starting or self.transport_stopping:
            return False
        if self.momentary_shuttle_task is not None:
            await self._end_momentary_shuttle()
        if self.transport is not None and self.transport.running:
            await self._stop_transport()
        if self.project.dirty:
            discard = await self.main_window.dialog(
                toga.ConfirmDialog(
                    "Discard Unsaved Changes?",
                    "Exit without saving changes to this project?",
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
        self._save_window_positions()
        return True

    async def _show_project_error(self, message: str) -> None:
        await self.main_window.dialog(toga.ErrorDialog("Project Error", message))

    def _update_audio_summary(self, error: str | None = None) -> None:
        settings = self.audio_service.current_settings
        input_device = None
        output_device = None
        routing = settings.track_inputs if settings is not None else ()
        bus_outputs = settings.bus_outputs if settings is not None else ()
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
        self.status_line_label.text = (
            f"{device_summary}  •  {sample_rate_summary}"
        )
        self._set_routing_status_link_visible(not routing_complete)
        self.status_line_suffix_label.text = (
            "  •  Audio unavailable"
            if self.project is not None and self.audio_engine_error is not None
            else ""
        )


def main() -> TapeMachine:
    """Create the application instance used by Briefcase."""
    return TapeMachine(
        formal_name="Tape Machine",
        app_id="pkg.unusualaudio.tape-machine",
    )
