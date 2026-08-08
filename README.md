# Tape Machine

A standalone macOS application built with [BeeWare Toga](https://toga.beeware.org/)
and packaged with [Briefcase](https://briefcase.beeware.org/).

## Development

Install the project and its development tools:

```sh
poetry install
```

Run the application in development mode:

```sh
poetry run briefcase dev
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
