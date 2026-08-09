# Tape Machine

A standalone macOS application built with [BeeWare Toga](https://toga.beeware.org/)
and packaged with [Briefcase](https://briefcase.beeware.org/). Audio devices are
discovered with [python-sounddevice](https://python-sounddevice.readthedocs.io/),
and multichannel project audio is stored with
[python-soundfile](https://python-soundfile.readthedocs.io/).

## Audio settings

Open **Tape Machine → Settings…** or press <kbd>⌘</kbd><kbd>,</kbd> to select
independent input and output devices, choose a supported studio sample rate, and
route physical inputs to eight project tracks. Each track accepts one input, while
the same input can feed multiple tracks. The stereo project bus can route its left
and right sides to distinct physical outputs; either side may remain unassigned.
On the initial screen this configuration is the template for new projects.
Mappings are portable channel numbers: selecting an interface with fewer channels
does not erase them, and unavailable routes are shown as routing into the void.

## Projects

Choose **File → New Project** or press <kbd>⌘N</kbd> to create an empty
eight-channel, 24-bit PCM RF64/WAV file at the currently selected sample rate.
**File → Open Project…** (<kbd>⌘O</kbd>) accepts eight-channel WAV, WAVEX, and
RF64 files. Tape Machine stores versioned project configuration in the WAV comment
metadata, including preferred input and output devices, both routing matrices,
and the complete mixer state. Untagged eight-channel WAV files can be imported
and tagged when saved.

The WAV header owns the project sample rate. While a project is open the rate is
read-only, and Audio Settings are saved directly into the project file. Use
**File → Save Project** (<kbd>⌘S</kbd>) and **File → Close Project**
(<kbd>⌘W</kbd>) to manage the open file.

The project screen presents eight track strips and a stereo-bus strip. Each track
has a vertical −∞ to +6 dB fader, a pan knob, and record, input-monitor, mute,
and solo controls. Record and monitoring are available only when that track has
an assigned input. The stereo strip controls the bus level. Input monitoring is
mixed in real time through the track level, constant-power pan, mute/solo state,
and stereo-bus level before being sent to the configured device outputs. Mixer
changes mark the project as modified and are written to metadata by **Save
Project**. Saved monitoring becomes active again when the project audio engine
starts successfully.

The transport provides a record toggle, play, a combined stop/return-to-zero
button, and a `MM:SS.mmm` position display. With Record active, the transport can
start and continue rolling without any record-enabled tracks. A routed track's R
button can be toggled while rolling to punch that track in or out at the next
audio block. Record can also be toggled while rolling as the master punch control.
Stopping the transport clears the master Record toggle while leaving individual
track record-enable buttons armed.

Armed tracks overwrite the corresponding project channels with their raw,
pre-fader inputs; unarmed tracks are preserved. During a punch, existing audio on
armed tracks is suppressed, while enabled input monitoring remains audible.
Project playback passes through the live fader, pan, mute, solo, and stereo-bus
controls.

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
