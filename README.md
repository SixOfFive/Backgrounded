# Backgrounded

A Windows system-tray program that runs a small stickman civilisation and
paints it onto your desktop wallpaper.

It opens at night, in a storm. The world is almost black. One stickman carries
a candle, and the pool of warm light around them is the only thing you can
reliably see — until lightning cracks and the whole landscape is revealed for a
quarter of a second: every figure frozen mid-stride, the hut they are half way
through building, the tree line on the ridge. Then it's dark again.

They keep going whether or not you're looking. Close the program, come back
tomorrow, and they've still got the hut, the firepit, the names of everyone who
died getting there.

## Running it

```bash
python run.pyw
```

Double-click `run.pyw` to start it with no console window. It lives in the
system tray; the live view window is shown by default.

Useful flags while poking at it:

```bash
python run.pyw --fresh --scene night_storm --no-wallpaper
```

| Flag | Effect |
|---|---|
| `--fresh` | Ignore the saved world and generate a new one |
| `--scene <name>` | Opening scene (`night_storm`, `wildfire`, `blizzard`, …) |
| `--hide` | Start with the live view hidden |
| `--no-wallpaper` | Never touch the desktop wallpaper |
| `--capture N` | Save N PNG frames to the captures folder |
| `--exit-after N` | Quit after N simulated seconds |

## Tray menu

Right-click the tray icon:

- **Show Window** — the live view the wallpaper is rendered from (on by default)
- **Wallpaper Output** — stop/start writing to the desktop
- **Scene** — force a specific scene
- **Speed** — 0.5× / 1× / 2× / 4×
- **Pause**
- **New World** — regenerate terrain and start a fresh colony
- **Save Now**
- **Exit** — restores your original wallpaper on the way out

## Your wallpaper

The program records your current wallpaper at startup and puts it back when it
exits. If it's ever killed hard enough to skip that (Task Manager, power loss),
just set your wallpaper again normally.

Note that **only one program can own the desktop wallpaper.** If you run
something else that rotates wallpapers, the two will fight and you'll see
flicker. Turn the other one off, or use **Wallpaper Output** to leave your
desktop alone and just watch the live view.

## What's in there

50 implemented features, listed in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#7-the-50-features) — stickmen
gather wood and stone, haul it to a stockpile, and build a firepit, huts, a
wall, a bridge, a watchtower and eventually a totem. They eat, sleep, get cold,
huddle at the fire, talk to each other, celebrate finished buildings, and mourn
at gravestones.

They also die: falls from cliffs, lightning strikes, wildfires, mudslides,
floods. When one dies a grave is placed, the others mourn, and a new
stickman arrives with a new name and a new colour, one generation later. The
running history is kept in `chronicle.txt` next to the save file.

## Where it keeps things

`%LOCALAPPDATA%\Backgrounded\`

| File | Contents |
|---|---|
| `save.json` | The world — terrain, people, buildings, history |
| `config.json` | Your settings |
| `chronicle.txt` | Readable log of everything notable that happened |
| `backgrounded.log` | Diagnostics |
| `wallpaper_a.bmp` / `_b.bmp` | Alternating wallpaper output |

## Requirements

Python 3.11+ on Windows, plus:

```bash
pip install pygame-ce pillow numpy
```

No `pywin32` and no `pystray` — the tray icon and wallpaper handling are done
directly through `ctypes`.

## Development

`docs/ARCHITECTURE.md` is the design contract: thread model, module boundaries,
data shapes, render layer order, and the reasoning behind the light composite.
Worth reading before changing anything in `render/` or `shell/`.

Two rules keep the codebase honest:

- **Nothing in `sim/` may import pygame.** The simulation runs headless, which
  is how it gets tested.
- **Nothing in `render/` may mutate sim state.** Rendering reads only.
