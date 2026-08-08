# Tape Machine

A standalone macOS application built with [BeeWare Toga](https://toga.beeware.org/)
and packaged with [Briefcase](https://briefcase.beeware.org/). Audio devices are
discovered with [python-sounddevice](https://python-sounddevice.readthedocs.io/),
and recorded audio will be stored with
[python-soundfile](https://python-soundfile.readthedocs.io/).

## Audio settings

Open **Tape Machine → Settings…** or press <kbd>⌘</kbd><kbd>,</kbd> to select
independent input and output devices and a mutually supported studio sample rate.
The current version keeps this configuration for the running session only.

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
