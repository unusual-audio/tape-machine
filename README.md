# Tape Machine

<img alt="Screenshot" src="https://github.com/user-attachments/assets/41ff8f0d-c73b-4aa6-90d0-67954d055619" />

A standalone macOS application built with [BeeWare Toga](https://toga.beeware.org/)
and packaged with [Briefcase](https://briefcase.beeware.org/). Audio devices are
discovered with [python-sounddevice](https://python-sounddevice.readthedocs.io/),
and multichannel project audio is stored with
[python-soundfile](https://python-soundfile.readthedocs.io/).

Tape Machine always uses its dark appearance, independently of the macOS system
setting. The launcher provides direct **New Project** and **Open Project…** links
and shows recent projects for one-click access.

## Audio settings

<img alt="Screenshot" src="https://github.com/user-attachments/assets/9a0d9f26-5ecb-423c-9ec4-927daf6b35fd" />

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
does not erase them. Saved channels beyond the current device remain visible as
**unavailable**; unavailable inputs produce silence and unavailable outputs are
not sent to hardware.

**Save** applies the complete dialog configuration and stores it for the next app
launch. This includes both devices, the sample rate, all track-input routes, both
stereo-bus outputs, and the buffer size.

## Projects

Use the launcher links or choose **File → New Project** (<kbd>⌘N</kbd>) to create
an empty eight-channel, 24-bit PCM RF64/WAV file at the currently selected sample
rate. **Open Project…** (<kbd>⌘O</kbd>) accepts eight-channel WAV, WAVEX, and RF64
files. Tape Machine stores the complete mixer state as versioned project
configuration in the WAV comment metadata. Audio devices, routing, and buffer
size remain global and are never written to a project. Untagged eight-channel WAV
files can be imported and tagged when saved. A project opened without write access
can be played and mixed, but its record controls remain disabled.

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
bus level. Double-clicking a pan knob centers it. Double-clicking a fader moves it
to 0 dB; if it is already at 0 dB, double-clicking moves it to −∞.

The centered scribble strip below each track is an inline, optional channel name
of up to 16 characters. Return or moving focus commits an edit. Tab advances to
the next track and wraps from track 8 to track 1. Interacting with another mixer
control also ends scribble-strip editing.

Every fader includes a peak meter. While stopped, track meters show their assigned
inputs regardless of the track's R and I buttons. During playback they show tape;
during recording, armed tracks show the signal being recorded while unarmed tracks
continue to show tape. The stereo-bus fader has a two-channel bus meter.

Input monitoring is mixed in real time through the track level, constant-power
pan, mute/solo state, and stereo-bus level before being sent to the configured
device outputs. Mixer changes mark the project as modified and are written to
metadata by **Save Project**. Saved monitoring becomes active again when the
project audio engine starts successfully.

The right-aligned transport controls are ordered **REW**, **FWD**, **STOP/RTZ**,
**PLAY**, and **REC**, followed by a stable-width `MM:SS.mmm` position display.
The combined button shows **STOP** while rolling and **RTZ** while stopped. From
Stop, rewind and fast-forward are latched audible shuttle modes: clicking the
active button stops it, while clicking the opposite button changes direction.
During ordinary playback they are momentary controls; releasing the button resumes
playback from the new position. They are disabled whenever global Record is armed.
Shuttle audio runs backward or forward at 10× speed, is filtered to limit
aliasing, is reduced by 9 dB, and continues through the mixer. Shuttle stops
automatically at zero or the end of the project and cannot record.

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

## Runtime architecture

The audio callback performs bounded NumPy mixing and routing with preallocated
scratch buffers and single-producer/single-consumer rings; project disk I/O never
runs in the callback. Independent playback and capture workers provide one second
of playback buffering and three seconds of recording buffering. During steady
recording, callback blocks are coalesced into writes of up to 8192 frames to avoid
excessive seeks and small disk operations. Stopping drains captured audio and
flushes the project before file operations are re-enabled.

Device discovery, stream transitions, and project creation, opening, saving, and
closing run away from the UI thread. Unexpected stream termination and callback
failures are reported to the main window, while transport worker timeouts keep the
project unavailable until its file handles have actually been released.

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

Development currently requires macOS and Python 3.14.

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
