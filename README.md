# Tape Machine

A standalone macOS application built with [BeeWare Toga](https://toga.beeware.org/)
and packaged with [Briefcase](https://briefcase.beeware.org/). Audio devices are
discovered with [python-sounddevice](https://python-sounddevice.readthedocs.io/),
and multichannel project audio is stored with
[python-soundfile](https://python-soundfile.readthedocs.io/).

## Audio settings

Open **Tape Machine → Settings…** or press <kbd>⌘</kbd><kbd>,</kbd> to select
independent input and output devices, choose a supported studio sample rate and
audio buffer size, and route input sources to eight project tracks. Buffer choices
range from 16 to 2048 samples, plus **Automatic** for the device default. Smaller
buffers reduce monitoring latency but increase the risk of audio dropouts. In
addition to physical inputs, the
matrix provides **Stereo bus L** and **Stereo bus R** as ordinary loopback inputs.
Each track accepts one input, while the same input can feed multiple tracks. The
stereo project bus can route its left and right sides to distinct physical outputs;
either side may remain unassigned.
This configuration is global and applies to every project.
Routing matrices show Core Audio channel names when the device provides them,
falling back to numbered inputs and outputs for unnamed channels. Mappings remain
portable channel numbers internally: selecting an interface with fewer channels
does not erase them, and unavailable routes are shown as routing into the void.

**Save** applies the complete dialog configuration and stores it for the next app
launch. This includes both devices, the sample rate, all track-input routes, both
stereo-bus outputs, and the buffer size.

## Projects

Choose **File → New Project** or press <kbd>⌘N</kbd> to create an empty
eight-channel, 24-bit PCM RF64/WAV file at the currently selected sample rate.
**File → Open Project…** (<kbd>⌘O</kbd>) accepts eight-channel WAV, WAVEX, and
RF64 files. Tape Machine stores the complete mixer state as versioned project
configuration in the WAV comment metadata. Audio devices, routing, and buffer size
remain global and are never written to a project. Untagged eight-channel WAV files
can be imported and tagged when saved.

**File → Open Recent** keeps the fourteen most recently created or opened projects.
Selecting a missing file removes it from the menu, and **Clear Menu** removes the
entire history.

The WAV header owns the project sample rate. While a project is open, that rate is
used temporarily and is read-only in Audio Settings; the saved global sample rate
continues to be used for new projects. Use **File → Save Project** (<kbd>⌘S</kbd>)
and **File → Close Project** (<kbd>⌘W</kbd>) to manage the open file.

The project screen presents eight track strips and a stereo-bus strip. Each track
has a vertical −∞ to +6 dB fader, a pan knob, and record, input-monitor, mute,
and solo controls. Record and monitoring are available only when that track has
an assigned input. Stereo-bus loopback routes can be record-enabled but cannot be
input-monitored, preventing a direct feedback path. The stereo strip controls the
bus level. Input monitoring is
mixed in real time through the track level, constant-power pan, mute/solo state,
and stereo-bus level before being sent to the configured device outputs. Mixer
changes mark the project as modified and are written to metadata by **Save
Project**. Saved monitoring becomes active again when the project audio engine
starts successfully.

The transport provides a record toggle, rewind, play, fast-forward, a combined
stop/return-to-zero button, and a `MM:SS.mmm` position display. From Stop, rewind
and fast-forward are latched audible shuttle modes: clicking the active button
stops it, while clicking the opposite button changes direction. During ordinary
playback they are momentary controls; releasing the button resumes playback from
the new position. They are disabled whenever global Record is armed. Shuttle
audio runs backward or forward at 10× speed, is filtered to limit aliasing, is
reduced by 9 dB, and continues through the mixer. Shuttle stops automatically at
zero or the end of the project and cannot record.

With Record active, the transport can start and continue rolling without any
record-enabled tracks. A routed track's R button can be toggled while rolling to
punch that track in or out at the next audio block. Record can also be toggled
while rolling as the master punch control. Stopping the transport clears the
master Record toggle while leaving individual track record-enable buttons armed.

Armed tracks overwrite the corresponding project channels; physical sources are
recorded raw and pre-fader, while Stereo bus L/R records the actual post-level bus
signal. Unarmed tracks are preserved. During a punch, existing audio on armed
tracks is suppressed, while enabled physical-input monitoring remains audible.
Project playback passes through the live fader, pan, mute, solo, and stereo-bus
controls. Loopback recording uses this same bus and otherwise follows the ordinary
Record and punch workflow; it has no separate bounce mode.

Recording is deliberately destructive, like tape: captured audio is written to
the project as it is recorded and is not part of mixer-state undo or discard.
Close and quit confirmations therefore refer specifically to unsaved mixer
changes; discarding those changes never rolls recorded audio back.

## Application configuration

Recent projects, global Audio Settings, and the last positions of the main and
Audio Settings windows are stored separately from project WAV metadata. On macOS,
the versioned JSON file is located at
`~/Library/Preferences/pkg.unusualaudio.tape-machine/config.json`. If a saved
window is no longer on a connected display, it is moved onto the primary display.
If saved audio hardware is unavailable, Tape Machine uses a compatible session
fallback without overwriting the saved configuration. Duplicate device names are
matched with stored hardware characteristics when possible; ambiguous matches use
the system default for the session and ask the user to confirm Audio Settings.
The footer shows **Routing incomplete** when there is no usable input/output path,
and **Some routes unavailable** when the current hardware can run but one or more
saved channel assignments are out of range. Both warnings open Audio Settings.

If one configuration section is damaged, the app salvages the other sections,
resets only the invalid data, and keeps a timestamped `config.invalid-*.json`
backup beside the repaired file. Compatible data from an older configuration
schema is migrated the same way without interrupting startup.

## Development

Install the project and its development tools:

```sh
poetry install
```

Run the application in development mode:

```sh
poetry run briefcase dev
```

Run the automated tests:

```sh
poetry run pytest
```

Create and run the standalone macOS application bundle:

```sh
poetry run briefcase create macOS
poetry run briefcase build macOS
poetry run briefcase run macOS
```

Build a distributable disk image after creating and building the app:

```sh
poetry run briefcase package macOS
```
