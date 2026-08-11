#!/usr/bin/env bash
# ===========================================================================
#  Backgrounded - Linux launcher
#
#  The counterpart to Backgrounded.bat, and deliberately not a translation of
#  it. On Windows the program's output is the desktop wallpaper and its home is
#  the system tray, so the .bat relaunches under pythonw.exe to get rid of the
#  console. There is no wallpaper output and no tray here: the window IS the
#  program. So this runs in the foreground by default - you get the log, Ctrl+C
#  stops it cleanly (the world is saved on the way out), and closing the window
#  does the same. Pass --detach if you would rather have your prompt back.
#
#      ./backgrounded.sh                 run it
#      ./backgrounded.sh --detach        run it, return to the shell
#      ./backgrounded.sh --setup         install the dependencies and stop
#      ./backgrounded.sh --yes           don't ask before installing
#      ./backgrounded.sh -- --help       the app's own options
#
#  Anything this script does not recognise is forwarded to the app, so
#  `./backgrounded.sh --fresh --scene dawn` works. The bare `--` is only needed
#  for arguments that would otherwise be eaten here, i.e. --help.
#
#  Dependencies live in a virtualenv, because Debian/Ubuntu Python is
#  externally managed (PEP 668) and `pip install --user` is refused there. If
#  pygame, Pillow and numpy already import from the system interpreter, that is
#  used instead and no virtualenv is made.
# ===========================================================================
set -uo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE" || exit 1

# The virtualenv does NOT live beside this script on purpose. The repo is
# routinely kept on a CIFS share, and `python3 -m venv` there fails outright:
# the mount has no Unix extensions, so creating the bin/python symlink returns
# EIO. Somewhere under $HOME is local disk on every machine this has run on.
VENV="${BACKGROUNDED_VENV:-${XDG_DATA_HOME:-$HOME/.local/share}/Backgrounded/venv}"
DEPS=(pygame-ce pillow numpy)
IMPORTS='import pygame, PIL, numpy'

ASSUME_YES=0
SETUP_ONLY=0
DETACH=0
APP_ARGS=()

while (( $# )); do
    case "$1" in
        --yes|-y)     ASSUME_YES=1 ;;
        --setup)      SETUP_ONLY=1; ASSUME_YES=1 ;;
        --detach|-b)  DETACH=1 ;;
        --)           shift; APP_ARGS+=("$@"); break ;;
        -h|--help)
            # The header comment above, minus its rules, is the help text.
            sed -n '3,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0 ;;
        *)            APP_ARGS+=("$1") ;;
    esac
    shift
done

say() { printf '  %s\n' "$*"; }
die() { printf '\n'; say "$@"; printf '\n'; exit 1; }

# ------------------------------------------------------------ interpreter --
PY=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        # 3.11+ for the typing and tomllib-era syntax the package is written in.
        if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
            PY="$candidate"
            break
        fi
    fi
done

if [[ -z "$PY" ]]; then
    printf '\n'
    say "No Python 3.11 or newer was found on your PATH."
    say ""
    say "Debian/Ubuntu:  sudo apt install python3 python3-venv"
    say "Fedora:         sudo dnf install python3"
    say "Arch:           sudo pacman -S python"
    printf '\n'
    exit 1
fi

# ----------------------------------------------------------- dependencies --
# Order matters: an existing virtualenv wins over the system interpreter, so a
# machine that later grows a system pygame does not silently change which one
# the app runs against.
RUN=""
if [[ -x "$VENV/bin/python" ]] && "$VENV/bin/python" -c "$IMPORTS" >/dev/null 2>&1; then
    RUN="$VENV/bin/python"
elif "$PY" -c "$IMPORTS" >/dev/null 2>&1; then
    RUN="$PY"
fi

if [[ -z "$RUN" ]]; then
    printf '\n'
    say "Backgrounded needs three packages: ${DEPS[*]}"
    say "They will go in a virtualenv, not your system Python:"
    say "    $VENV"
    printf '\n'
    if (( ! ASSUME_YES )); then
        read -r -p "   Install them now? [Y/n] " reply
        case "${reply:-Y}" in
            [Nn]*) exit 1 ;;
        esac
        printf '\n'
    fi

    if [[ ! -x "$VENV/bin/python" ]]; then
        mkdir -p "$(dirname "$VENV")" || die "Could not create $(dirname "$VENV")"
        if ! "$PY" -m venv "$VENV"; then
            printf '\n'
            say "Could not create the virtualenv."
            say ""
            say "On Debian and Ubuntu the venv module ships separately:"
            say "    sudo apt install python3-venv"
            say ""
            say "Or point BACKGROUNDED_VENV at a virtualenv you made yourself:"
            say "    BACKGROUNDED_VENV=~/somewhere/venv ./backgrounded.sh"
            printf '\n'
            exit 1
        fi
    fi

    "$VENV/bin/python" -m pip install --upgrade pip >/dev/null 2>&1
    if ! "$VENV/bin/python" -m pip install "${DEPS[@]}"; then
        printf '\n'
        say "Could not install the packages. Try this yourself:"
        say "    $VENV/bin/python -m pip install ${DEPS[*]}"
        printf '\n'
        exit 1
    fi
    if ! "$VENV/bin/python" -c "$IMPORTS" >/dev/null 2>&1; then
        die "The packages installed but still will not import. Try deleting" \
            "$VENV and running this again."
    fi
    RUN="$VENV/bin/python"
fi

if (( SETUP_ONLY )); then
    printf '\n'
    say "Ready. Dependencies are in place for: $RUN"
    printf '\n'
    exit 0
fi

# ------------------------------------------------------------------ launch --
# A colony from a previous run is picked up automatically; --fresh starts over.
# The app holds an flock on its instance file, so a second copy declines to
# start rather than fighting this one for save.json.
if (( DETACH )); then
    LOG="${XDG_DATA_HOME:-$HOME/.local/share}/Backgrounded/backgrounded.log"
    nohup "$RUN" "$HERE/run.pyw" "${APP_ARGS[@]}" >/dev/null 2>&1 &
    printf '\n'
    say "Backgrounded started (pid $!). Close its window to stop it."
    say "Log: $LOG"
    printf '\n'
    exit 0
fi

exec "$RUN" "$HERE/run.pyw" "${APP_ARGS[@]}"
