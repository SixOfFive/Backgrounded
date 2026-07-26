# Backgrounded — Architecture

A Windows system-tray program that runs a persistent 2D stickman civilisation
and paints it onto the desktop wallpaper.

## 0. Hard constraints (validated by spike, do not re-litigate)

| Constraint | Source | Rule |
|---|---|---|
| `SystemParametersInfoW(SPI_SETDESKWALLPAPER)` costs **~200 ms** | measured | Wallpaper push runs on a worker thread, target 3–5 fps. Never on the sim thread. |
| Writing the *active* wallpaper file can fail `EINVAL` transiently (Defender/Explorer scan) | measured | Write to `*.tmp` then `os.replace`; alternate `wallpaper_a.bmp`/`wallpaper_b.bmp`; retry 3× with backoff; on final failure **skip the frame**, never raise. |
| `PostQuitMessage` is thread-affine | measured | The tray owns its own thread + message pump. Shut it down with `PostMessageW(hwnd, WM_CLOSE, 0, 0)`. |
| Primary display is 2560×1600 | measured | Sim renders at `RENDER_W×RENDER_H` (1280×800) and upscales. Never render the sim at native res. |

Runtime deps: `pygame-ce`, `Pillow`, `numpy`, stdlib `ctypes`. **No pywin32, no pystray.**

## 1. Process & thread model

Three threads, one owner each. All cross-thread traffic goes through queues —
no shared mutable state, no locks.

```
main thread            tray thread                 wallpaper thread
-----------            -----------                 ----------------
pygame loop @60fps     Win32 message pump          blocking I/O
  world.tick()           Shell_NotifyIconW           frame = q.get()
  renderer.draw()        TrackPopupMenu              BMP write + os.replace
  preview.present()      cmd_q.put(Command)          SystemParametersInfoW
  drain cmd_q  <---------------'                   ^
  every Nth frame: frame_q.put(surface_copy) ------'   (queue size 1, drop-oldest)
```

- `cmd_q`: `queue.Queue[Command]` — tray → main.
- `frame_q`: `queue.Queue(maxsize=1)` — main → wallpaper. Producer drops the
  pending frame rather than blocking (`get_nowait` then `put_nowait`).
- Shutdown: main sets `app.running=False`, joins wallpaper thread (with the
  original wallpaper restored), then posts `WM_CLOSE` to the tray hwnd.

## 2. Module map

```
run.pyw                     launcher, no console window
backgrounded/
  app.py            App          owns threads, loop, command dispatch
  config.py         Config       JSON @ %LOCALAPPDATA%\Backgrounded\config.json
  paths.py                       APPDATA dir, save path, wallpaper A/B paths
  shell/
    tray.py         Tray         ctypes tray + menu (thread)
    wallpaper.py    WallpaperWriter  A/B atomic writer (thread) + restore
    preview.py      Preview      pygame window, show/hide, scaling
  sim/
    world.py        World        aggregate root; tick(dt); to_dict/from_dict
    terrain.py      Terrain      heightmap + materials + deformation
    lighting.py     Lighting     LightSource registry, ambient curve
    entities.py     Stickman     stats, inventory, colour identity
    behavior.py                  utility AI scorer -> Action
    actions.py      Action       concrete behaviours (gather/build/...)
    structures.py   Structure    multi-stage buildables
    props.py        Prop         trees/rocks/bushes/water
    events.py       EventSystem  disasters
    names.py                     name + lineage generation
  render/
    renderer.py     Renderer     layer orchestration
    stickfigure.py               skeletal poser + AA draw
    sky.py                       gradient, stars, moon, aurora
    particles.py    ParticleSystem
    fx.py                        lightning bolts, shake, vignette
    atlas.py        Atlas        baked static prop sprites (startup cache)
  persist.py                     versioned save/load
```

## 3. Core data contracts

Coordinates: world space **is** render space, origin top-left,
`RENDER_W=1280`, `RENDER_H=800`. `y` increases downward. Ground is a
function of `x` — the world is a fixed screen, edges are hard walls.

```python
# terrain.py
class Terrain:
    W: int                      # == RENDER_W
    height: np.ndarray          # float32[W], y of ground surface per column
    material: np.ndarray        # uint8[W], see MAT_*
    def ground_y(self, x: float) -> float          # interpolated
    def slope(self, x: float) -> float             # dy/dx, for walk cost/climb
    def deform(self, x0: int, x1: int, dy: float)  # mudslide/dig/crater
    def is_climbable(self, x: float) -> bool       # slope within limit
    def is_cliff(self, x: float) -> bool           # slope beyond limit -> fall risk

MAT_GRASS, MAT_DIRT, MAT_STONE, MAT_SAND, MAT_SNOW, MAT_ASH, MAT_MUD, MAT_LAVA = range(8)
```

```python
# entities.py
@dataclass
class Stickman:
    id: int
    name: str
    color: tuple[int,int,int]     # identity colour, persisted
    x: float; y: float; vx: float; vy: float
    facing: int                   # -1 | +1
    hunger: float                 # 0..1, 1 = starving
    fatigue: float                # 0..1
    warmth: float                 # 0..1, 1 = freezing
    morale: float                 # 0..1
    carrying: str | None          # resource id
    carry_qty: int
    role: str                     # gatherer|builder|elder|child|lookout
    generation: int
    action: Action | None
    holds_candle: bool
    alive: bool
    anim_t: float                 # animation phase accumulator
```

Every sim object exposes `to_dict()` / `from_dict(d)`. **If it isn't in
`to_dict`, it does not survive a restart.**

```python
# lighting.py
@dataclass
class LightSource:
    x: float; y: float
    radius: float
    color: tuple[int,int,int]
    intensity: float          # 0..1
    flicker: float            # 0..1 amplitude of per-frame noise
```

`Lighting.ambient(world_time) -> float` returns 0..1. Night storm scene
clamps ambient low (~0.06) so the candle and lightning actually matter.

## 4. Render pipeline (strict layer order)

Renderer composites into one `RENDER_W×RENDER_H` surface each frame:

1. **Sky** — vertical gradient by time-of-day; stars (night); moon/sun arc; aurora.
2. **Parallax** — 2–3 ridge silhouettes, darker with depth.
3. **Weather-back** — distant rain/snow/ash.
4. **Terrain** — filled column polygon, material-coloured, with a lit rim.
5. **Props** — trees, rocks, bushes (atlas sprites), depth-sorted.
6. **Structures** — build-stage sprite + scaffolding.
7. **Agents** — stick figures, AA lines, per-agent colour.
8. **Weather-front** — near rain/snow, splashes.
9. **Particles** — embers, smoke, dust, fireflies.
10. **Light composite** — this is the important one, see below.
11. **FX** — lightning bolt geometry, screen shake offset, vignette.

### Light composite (the visual thesis)

The scene is rendered fully lit, then *darkened*, then light is added back:

```python
lightmap = Surface(size)                       # starts at ambient grey
lightmap.fill(ambient_rgb)
for src in lights:                             # candle, fire, lightning
    blit_radial_gradient(lightmap, src, special_flags=BLEND_RGB_ADD)
scene.blit(lightmap, (0,0), special_flags=BLEND_RGB_MULT)
```

So in the night-storm scene the world is near-black; the candle carves a warm
pool that moves with its bearer; a lightning strike momentarily raises the
*global* ambient to ~1.0, revealing every stickman mid-stride as a silhouette,
then drops back to black over ~400 ms. That reveal-then-dark rhythm is the
core aesthetic and everything else serves it.

## 5. Tick model

Fixed timestep, `SIM_HZ = 30`. Render is decoupled at display rate and
interpolates agent positions. `world_time` advances in *sim minutes*; a full
day/night cycle defaults to 20 real minutes (configurable).

Agent AI is **utility-based**, re-scored every `AI_HZ = 2` ticks (not every
frame). Each candidate action scores 0..1 from needs; highest wins; ties broken
by role affinity. Actions are resumable state machines so a save mid-build
reloads mid-build.

## 6. Persistence

`%LOCALAPPDATA%\Backgrounded\save.json`, `{"version": 1, ...}`. Autosave every
60 s and on exit. Load is defensive: unknown keys ignored, missing keys take
defaults, a corrupt save is renamed `save.corrupt.json` and a fresh world is
generated rather than crashing.

Persisted: terrain heightmap + materials, all agents (incl. colour, name,
generation, in-flight action state), structures + build progress, props +
harvest state, stockpiles, world_time, scene, weather state, population
statistics, graves, and the chronicle log.

## 7. The 50 features

### Stickman behaviour (1–17)
1. Wander / idle with individual gait variation
2. Gather wood from trees (multi-hit chop, tree falls)
3. Gather stone from rocks
4. Forage berries from bushes
5. Haul resources to a stockpile
6. Build a firepit
7. Build a hut (4 visible construction stages)
8. Build a wall segment
9. Build a bridge across a chasm
10. Build a watchtower + a lookout climbs it
11. Erect a totem at a population milestone
12. Repair damaged structures
13. Eat when hungry (consumes stockpile food)
14. Sleep in a hut when fatigued (night preference)
15. Huddle at the fire when cold
16. Cook food at the firepit (raw → cooked, better nutrition)
17. Plant a sapling; it grows into a harvestable tree over time

### Social & lineage (18–25)
18. Named stickmen with persistent identity colour
19. Roles: gatherer / builder / elder / child / lookout
20. Two adults near a hut produce a child; population grows
21. Children follow a parent and mature into a role
22. Conversation: two agents pause, emit symbol speech bubbles
23. Celebration dance when a structure completes
24. Mourning: agents gather at a grave, morale drops
25. Chronicle log — a persistent human-readable history of notable events

### Death & danger (26–31)
26. Climbing steep terrain with a slip chance
27. Falling from height → death → gravestone placed
28. A replacement stickman arrives (next generation, new name/colour)
29. Struck by lightning (rare, scorch mark + instant death)
30. Caught in fire → panic → burn death
31. Buried by a mudslide (terrain deformation kills)

### Light & the candle (32–36)
32. Candle-bearer with a warm radial glow that lights terrain and neighbours
33. Candle flicker + wind-driven guttering; can be blown out and relit
34. Firepit light pool, larger and warmer than the candle
35. Lightning as a global light source, silhouette reveal then darkness
36. Agents outside any light source render as near-black silhouettes

### Weather, scenes, disasters (37–45)
37. Night storm — the default scene: rain, wind, branching lightning
38. Day/night cycle with sun/moon arc and star field
39. Rain with wind shear, splashes, and puddle accumulation
40. Wildfire that spreads across vegetation and burns props away
41. Mudslide that collapses terrain and reshapes the heightmap
42. Blizzard with snow accumulating as a white terrain layer
43. Flood: a rising water line that agents flee uphill from
44. Meteor shower with impact craters that deform terrain
45. Earthquake: screen shake + terrain fissure

### World & presentation (46–50)
46. Procedurally varied terrain — hills, cliffs, plateaus, chasms (never flat)
47. Destructible/deformable terrain shared by digging, craters, and slides
48. Parallax ridges + drifting cloud layers for depth
49. Tray menu: Show Window (default **on**), scene picker, pause, speed, exit
50. Live desktop wallpaper output with the user's original wallpaper restored on exit

## 8. Conventions for implementers

- Python 3.14, type hints on public functions, dataclasses for state.
- **No pygame calls in `sim/`.** Sim is pure Python + numpy and must be
  importable and runnable headless (this is how it gets tested).
- **No world mutation in `render/`.** Render reads, never writes.
- Colours as `(r,g,b)` int tuples. Angles in radians. Time in seconds.
- Every module gets a `if __name__ == "__main__":` smoke test where sensible.
- Fail soft: a broken subsystem logs and disables itself; the wallpaper keeps
  updating. This program runs unattended for hours — a crash is the worst bug.
