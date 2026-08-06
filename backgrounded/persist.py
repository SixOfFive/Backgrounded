"""Versioned save/load.

A save file must never be able to prevent the program from starting. Anything
unreadable is moved aside and a fresh world is generated in its place. That
includes a file whose top-level JSON value is not an object: ``[1,2,3]`` used to
be the one shape that walked past the parse guard and died on ``.get`` two lines
later, taking the whole process with it and leaving the bad file in place to do
it again next launch.

The other half of that promise, and the harder half, is that starting must never
cost the user their colony. Three rules hold it up:

* **Unreadable is not the same as absent.** A save that EXISTS but could not be
  read - an antivirus scan, a backup agent, a sync client holding the handle -
  is a colony, not a blank slate. Those reads are retried, and if they still
  fail the module refuses to write for the rest of the session rather than let
  the next autosave lay a fresh world on top of a real one. See ``_block_saves``.
* **Corruption is judged on content, never on plumbing.** Only a file that was
  read and turned out to be junk is quarantined. Failing to read one is not
  evidence about what is in it.
* **The one-way door gets a key.** Loading is where a save learns how wide the
  world it is being restored into is: ``Terrain.from_dict`` grows the LAND to
  WORLD_W and ``props.migrate_world_width`` is told the width the payload
  actually carried, so the scenery counts, which are per-map, come with it.
  Neither changes SAVE_VERSION - the format is untouched, only what is done with
  the fields it always had - but the widening is not reversible from the widened
  file, so the bytes that came in are copied aside first. See
  ``_backup_premigration``.

Everything on disk here is bounded. The quarantine keeps the last few bad saves
and prunes the rest; the pre-migration copy is keyed on the width it holds, so
it is written once per historical world width and never by a launch that has
nothing new to preserve.
"""
from __future__ import annotations

import codecs
import json
import logging
import os
import shutil
import stat as _stat
import time
from pathlib import Path
from typing import Any

from . import paths
from .constants import SAVE_VERSION, WORLD_W
from .sim import props as _props
from .sim import terrain as _terrain
from .sim.world import World

log = logging.getLogger(__name__)

#: A save that could not be read is retried before it is believed. The delays
#: are short and front-loaded: the overwhelming majority of these are an on-
#: access antivirus scan or a sync client that holds the handle for a few
#: hundred milliseconds, and this is a login-time wallpaper daemon, so the
#: budget has to be small enough that a genuinely missing-permissions file does
#: not visibly stall the desktop. ~2.1 s total, then the file is declared
#: unreadable and writing is switched off for the session instead.
_READ_BACKOFF: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4, 0.6, 0.8)

#: A real save is ~90-175 KB. Anything past this is not a save that grew, it is
#: a file that went wrong, and reading it would mean a multi-second parse and a
#: memory spike during the login storm. It is moved aside unread.
SAVE_SIZE_LIMIT: int = 64 * 1024 * 1024

QUARANTINE_PREFIX = "save.corrupt."
QUARANTINE_GLOB = QUARANTINE_PREFIX + "*"
#: How much quarantined evidence to keep. Five is enough to see a *repeating*
#: failure (the interesting case - one bad save is an accident, five identical
#: ones are a bug), and 5 x ~172 KB is under a megabyte in %LOCALAPPDATA% on a
#: process that starts at every login. The byte budget is the second cap,
#: because ``SAVE_SIZE_LIMIT`` allows a single quarantined item far larger than
#: a save; the newest item is always kept even if it alone exceeds the budget,
#: since deleting the only evidence of the failure that just happened would
#: defeat the point of quarantining at all.
QUARANTINE_KEEP: int = 5
QUARANTINE_BUDGET: int = 32 * 1024 * 1024

#: The pre-migration copy is named for the width it carries, not for when it was
#: taken. That is what makes it one-time: the second launch computes the same
#: name, finds the file already there and leaves it alone, so the ORIGINAL is
#: preserved rather than the most recent pre-migration state. Keying on width
#: rather than a single fixed ``save.bak`` also means a future widening
#: (6400 -> something) cannot overwrite the 1600 px original. The set of
#: historical world widths is a property of the program, not of how long it has
#: been running, so this cannot grow with use.
PREMIGRATION_PREFIX = "save.premigration."


def _premigration_path(saved_w: float) -> Path:
    return paths.APP_DIR / f"{PREMIGRATION_PREFIX}{int(round(saved_w))}px.json"


# --------------------------------------------------------------------------
# Session state: whether the save on disk may be written over.
#
# load_world is called once, at startup, before any save_world. If it found a
# file it could not read, nothing in this process is allowed to write over that
# file - the world in memory is a FRESH one, and the autosave sixty seconds
# later would be the moment the colony was actually destroyed. Refusing costs
# this session's progress in a world the user never asked for; not refusing
# costs the world they did.
# --------------------------------------------------------------------------
_save_blocked: str = ""
_blocked_notices: int = 0


def saves_blocked() -> str:
    """Why writing is disabled this session, or "" if it is not."""
    return _save_blocked


def _block_saves(reason: str) -> None:
    global _save_blocked
    if not _save_blocked:          # the first reason is the real one
        _save_blocked = reason


def _note_blocked() -> None:
    """Say it once, loudly; then stop filling the log every autosave."""
    global _blocked_notices
    _blocked_notices += 1
    if _blocked_notices == 1:
        log.error(
            "NOT saving: %s. save.json is being left exactly as it is. Close "
            "whatever is holding it (antivirus, backup or sync software) and "
            "restart - the colony is still in that file.", _save_blocked)
    else:
        log.debug("save still refused (%d so far): %s",
                  _blocked_notices, _save_blocked)


def _log(level: str, *a, **k) -> None:
    """Log, and never let logging itself be the thing that kills the app.

    Not paranoia for its own sake: the two entry points below promise they
    cannot raise, and a log call is the one statement that appears on every path
    including the paths where everything went RIGHT. The stdlib handlers swallow
    their own errors (Handler.handleError), but a handler is something app.py
    installs, not something this module controls, and "the colony was lost
    because a log line failed" is not a sentence anyone should have to read.
    """
    try:
        getattr(log, level)(*a, **k)
    except Exception:       # noqa: BLE001 - deliberately terminal
        pass


def save_world(world: World) -> bool:
    """Atomically write the world to disk. Returns success. Never raises."""
    try:
        return _save_world(world)
    except Exception:
        _log("exception", "save failed in an unexpected way")
        return False


def _save_world(world: World) -> bool:
    if _save_blocked:
        _note_blocked()
        return False
    try:
        paths.ensure_dirs()
        data = world.to_dict()
        tmp = paths.SAVE_PATH.with_suffix(".tmp")
        payload = json.dumps(data, separators=(",", ":")).encode("utf-8")
        # Written and flushed to the platter BEFORE the rename. Without the
        # fsync the atomic replace is only atomic against this program - the
        # metadata can land ahead of the data, and a power cut then leaves a
        # save.json of the right name and the wrong length. This runs once a
        # minute on a file under 200 KB; it is not a cost worth optimising.
        with open(tmp, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(paths.SAVE_PATH)
    except Exception:
        log.exception("save failed")
        return False

    # The chronicle is also mirrored as plain text - it is the one part of the
    # save a human might actually want to read.
    try:
        paths.CHRONICLE_PATH.write_text("\n".join(world.chronicle), "utf-8")
    except Exception:
        pass
    return True


def load_world() -> World | None:
    """Load the saved world, or None if there isn't a usable one. Never raises.

    The outer guard is this module's docstring made structural. Nothing inside
    is supposed to reach it - every step below handles its own failure - but
    app.py builds the App bare inside ``return App(args).run()``, so anything
    that DID escape would kill the process with no console and no window under
    run.pyw, and would do it again at every login. Saves are switched off on the
    way out, because a swallowed failure must never be able to become a fresh
    world written over a colony this module merely failed to understand.
    """
    try:
        return _load_world()
    except Exception:
        _log("exception", "unexpected failure while loading the save")
        _block_saves("loading save.json failed in an unexpected way")
        return None


def _load_world() -> World | None:
    # stat, not exists(): one syscall that separates the three cases that used
    # to look alike. FileNotFoundError is the ordinary first run. Any other
    # OSError means the filesystem would not answer - which is not the same as
    # an answer of "nothing there", and used to end with a fresh world written
    # straight over whatever was actually sitting at that path. And a path that
    # is not a regular file is not a lock and never will be, so it can be judged
    # on the spot instead of blocking writes forever.
    # `except Exception`, not `except OSError`. This is the FIRST thing the
    # program touches on disk and the last place that can afford to be picky:
    # under run.pyw anything that gets past here dies with no console and no
    # window, and does it again at every login. OSError is only what a
    # filesystem is SUPPOSED to raise - antivirus and sync-client shims sit in
    # this call path and a MemoryError, a RecursionError or a plain ValueError
    # out of one of them would sail straight through an OSError clause into
    # App.__init__. Measured by fault injection: 6 exception types x 4 save
    # shapes at this one call, and the OSError-only version let 4 of the 6 out.
    try:
        st = paths.SAVE_PATH.stat()
    except FileNotFoundError:
        return None
    except Exception:
        log.exception("could not stat the save file")
        _block_saves("the save file exists as far as the OS is concerned but "
                     "could not be examined")
        return None

    if not _stat.S_ISREG(st.st_mode):
        log.warning("the save path is not a regular file (mode 0o%o); "
                    "moving it aside", st.st_mode)
        _quarantine()
        return None

    if st.st_size > SAVE_SIZE_LIMIT:
        log.warning("the save file is %d bytes, past the %d byte limit; "
                    "moving it aside unread", st.st_size, SAVE_SIZE_LIMIT)
        _quarantine()
        return None

    # --- read ------------------------------------------------------------
    # Retried, and NOT quarantined on failure. A read that fails tells you
    # about the handle, not about the contents; the file underneath may be a
    # perfectly healthy colony that a scanner has open for another 200 ms.
    # _quarantine's own shutil.move would fail on exactly the same lock and
    # swallow the failure, so the old path left no trace that a real save had
    # been walked past, and returned None - which app.py reads as "no save".
    raw: bytes | None = None
    last: BaseException | None = None
    for i, delay in enumerate((*_READ_BACKOFF, None)):
        try:
            raw = paths.SAVE_PATH.read_bytes()
            break
        except Exception as exc:          # OSError, and MemoryError too
            last = exc
            if delay is None:
                break
            log.debug("save read attempt %d failed (%s); retrying in %.2fs",
                      i + 1, exc, delay)
            time.sleep(delay)
    if raw is None:
        log.error("the save file exists (%d bytes) but could not be read after "
                  "%d attempts: %s", st.st_size, len(_READ_BACKOFF) + 1, last)
        _block_saves(f"save.json exists ({st.st_size} bytes) but could not be "
                     f"read at startup ({type(last).__name__})")
        return None

    # --- decode and parse ------------------------------------------------
    # Past this point a failure IS evidence about the contents, so quarantine
    # is back on the table.
    try:
        data: Any = json.loads(_decode_save(raw))
    except Exception:
        log.exception("save file unreadable")
        _quarantine()
        return None

    # JSON's top level is any VALUE, not necessarily an object, and every other
    # shape used to walk straight past the try that parsed it into
    # ``data.get("version")`` two lines down. A save file holding ``[1,2,3]``,
    # ``"hello"``, ``42``, ``null`` or ``true`` therefore raised AttributeError
    # out of load_world - which nothing up the stack catches (app.py builds the
    # App bare inside `return App(args).run()`), so under run.pyw the process
    # died with no console and no window, leaving the desktop on whatever frame
    # was written last. It then did the same on every subsequent launch, because
    # the file was never quarantined: the one file shape that could permanently
    # stop the program from starting, in the module whose docstring promises
    # exactly the opposite. Measured: 5 shapes raised, quarantined=0 for all 5.
    #
    # A non-object save is corruption like any other - move it aside, start
    # fresh. isinstance rather than a try/except around the .get() because the
    # test is the point: everything below here is entitled to assume a mapping.
    if not isinstance(data, dict):
        log.warning("save file holds a top-level %s, not an object; "
                    "treating it as corrupt", type(data).__name__)
        _quarantine()
        return None

    version = data.get("version")
    if version != SAVE_VERSION:
        log.warning("save version %r != %r; starting fresh", version, SAVE_VERSION)
        _quarantine()
        return None

    try:
        world = World.from_dict(data)
    except Exception:
        log.exception("save file could not be reconstructed")
        _quarantine()
        return None

    # The world is good from here on, so nothing below may quarantine it and
    # nothing below may raise. A save written for a NARROWER map loads its own
    # land back exactly (Terrain._restore_saved_band) and then sits in a world
    # four times wider than the density its props were authored for; see
    # props.migrate_world_width. The width is taken from the save rather than
    # guessed, and the call is wrapped because a colony that loaded is worth
    # more than a colony at the right density.
    saved_w = _saved_terrain_width(data)

    # Before that widening is written back. The migration is one-way: the wide
    # file cannot be turned back into the narrow one (the old build loads it
    # without raising and then truncates the land to 1600 while leaving props
    # out at x=6382), and the migration's own prop placements are not
    # reproducible once the narrow file is gone. The bytes that came in are what
    # gets copied - not a re-read and not a re-serialisation - so the copy is
    # the original, exactly.
    if saved_w is not None and saved_w < WORLD_W:
        _backup_premigration(raw, saved_w)

    try:
        report = _props.migrate_world_width(world, saved_w)
        if report.get("placed"):
            log.info("widened a %.0f px save to %.0f px: targets %r, planted %r",
                     _f(report.get("saved_w")), _f(report.get("world_w")),
                     report.get("targets"), report.get("placed"))
    except Exception:
        log.exception("width migration of the props failed; "
                      "the colony loads at its old density")

    # Wrapped for the same reason the shape test above exists. This line reads
    # three attributes and calls a method on a freshly rebuilt object, outside
    # any try, on the one path where the save was GOOD - so anything it raised
    # would kill the app at startup while leaving a perfectly loadable save in
    # place, and it would do it again on every launch. A log line is never
    # worth that.
    try:
        log.info("loaded world: t=%.0fs pop=%d gen=%d",
                 world.world_time, len(world.population.alive_agents()),
                 world.population.generation)
    except Exception:
        log.exception("loaded world, but could not describe it")

    # The cap is applied on the way IN as well as when quarantining, or it would
    # only ever bound the future: a machine that already collected forty of
    # these from the launches before this build shipped would keep all forty
    # until it happened to corrupt a save again.
    try:
        _prune_quarantine()
    except Exception:
        log.exception("could not prune the quarantine directory")
    return world


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------
#: UTF-32's byte-order marks START WITH UTF-16's, so the order here is load
#: bearing: b"\xff\xfe\x00\x00" has to be tested before b"\xff\xfe" or a UTF-32
#: file decodes as UTF-16 full of NULs.
#:
#: The mark is sliced off before decoding rather than left for the codec. The
#: explicit-endian codecs do NOT consume it - ``b"\xff\xfe{\x00".decode("utf-16-le")``
#: is ``"﻿{"``, and json.loads then fails on the U+FEFF with the same
#: "Expecting value" it gives for real garbage. Only the endian-agnostic aliases
#: ("utf-16", "utf-8-sig") strip it, and those cannot be used for all five.
_BOMS: tuple[tuple[bytes, str], ...] = (
    (codecs.BOM_UTF32_LE, "utf-32-le"),
    (codecs.BOM_UTF32_BE, "utf-32-be"),
    (codecs.BOM_UTF8, "utf-8"),
    (codecs.BOM_UTF16_LE, "utf-16-le"),
    (codecs.BOM_UTF16_BE, "utf-16-be"),
)


def _decode_save(raw: bytes) -> str:
    """Bytes off disk to text, tolerating what Windows editors actually write.

    ``read_text("utf-8")`` rejected a byte-order mark, so a save that had merely
    been OPENED and saved again by Notepad, VS Code or an "edit my seed" session
    came back as a UnicodeDecodeError and was quarantined as corrupt with the
    colony still perfectly intact inside it. Windows PowerShell 5.1's ``>``
    redirect is worse: it writes UTF-16LE, which is not a mangled file either,
    just one written by the shell the user's own notes tell them to use.

    A BOM is an unambiguous declaration, so it is honoured. Absent one the file
    is UTF-8, which is what this module writes; the UTF-16 fallback needs both a
    failed UTF-8 decode and NUL bytes in the first block before it is even
    tried, so it cannot hijack a file that is merely slightly mangled.
    """
    for bom, enc in _BOMS:
        if raw.startswith(bom):
            return raw[len(bom):].decode(enc)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        if b"\x00" in raw[:512]:
            for enc in ("utf-16-le", "utf-16-be"):
                try:
                    return raw.decode(enc)
                except UnicodeDecodeError:
                    pass
        raise


def _f(v: object, default: float = 0.0) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _saved_terrain_width(data: dict[str, Any]) -> float | None:
    """How many columns of land the save actually carries, or None.

    THE PAYLOAD, not the claim. ``terrain.w`` and the height array's declared
    ``shape`` are both separate assertions that a hand-edit can get wrong, and
    :meth:`Terrain.from_dict` already decides what to restore by counting the
    base64 - so this counts it the same way, through terrain's own helper, and
    the two cannot disagree about which band was restored. The ``w`` field is
    the last resort rather than the authority, exactly as it is there.
    """
    t = data.get("terrain")
    if not isinstance(t, dict):
        return None
    n: object = None
    fn = getattr(_terrain, "_array_len", None)
    if callable(fn):
        try:
            n = fn(t.get("height"))
        except Exception:
            n = None
    if n is None:
        n = t.get("w")
    try:
        w = float(n)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return w if w > 0.0 else None


# --------------------------------------------------------------------------
# The pre-migration copy
# --------------------------------------------------------------------------
def _backup_premigration(raw: bytes, saved_w: float) -> None:
    """Keep the incoming narrow save, once, before the world is widened.

    Never raises: it is called on the path where the colony has ALREADY loaded
    successfully, and no failure to make a backup is worth costing the user the
    world the backup was supposed to protect.
    """
    try:
        dst = _premigration_path(saved_w)
        if dst.exists():
            # Already have the original from an earlier upgrade. That copy is
            # older and therefore better; this launch has nothing to add.
            log.debug("pre-migration copy already exists at %s", dst)
            return
        paths.ensure_dirs()

        # Complete or absent, never half. The bytes go to a .part file that is
        # flushed to the platter, and the backup only ever comes into existence
        # through a rename, which is atomic. Killed at any point before the
        # rename there is no backup and a stray .part that the next launch
        # truncates and reuses; killed at any point after, the backup is whole.
        tmp = dst.with_name(dst.name + ".part")
        with open(tmp, "wb") as fh:
            fh.write(raw)
            fh.flush()
            os.fsync(fh.fileno())
        try:
            # os.rename, not os.replace and not shutil.move: on Windows this
            # FAILS if the destination appeared since the exists() check above,
            # which is the behaviour wanted here. An existing backup is older
            # and must win.
            os.rename(tmp, dst)
        except OSError:
            if not dst.exists():
                raise
            log.debug("a pre-migration copy appeared at %s; keeping it", dst)
            try:
                os.unlink(tmp)
            except OSError:
                pass
            return
        log.warning("this save was written for a %.0f px world and is about to "
                    "be widened to %.0f px, which cannot be undone; kept the "
                    "original at %s (%d bytes)",
                    saved_w, float(WORLD_W), dst, len(raw))
    except Exception:
        log.exception("could not keep a copy of the pre-migration save")


# --------------------------------------------------------------------------
# Quarantine
# --------------------------------------------------------------------------
def _quarantine() -> None:
    """Move a bad save aside so the next run starts clean but the evidence
    is still there to look at."""
    try:
        if paths.SAVE_PATH.exists():
            paths.ensure_dirs()
            stamped = _move_save_aside()
            if stamped is None:
                log.error("could not find a free quarantine name; "
                          "leaving the bad save in place")
            else:
                log.warning("moved unusable save to %s", stamped)
    except Exception:
        log.exception("could not quarantine save file")
    # Separate try: a prune failure must not make a successful quarantine look
    # like a failed one, and a quarantine failure must not skip the prune.
    try:
        _prune_quarantine()
    except Exception:
        log.exception("could not prune the quarantine directory")


def _move_save_aside() -> Path | None:
    """Rename the save into the first free quarantine slot, atomically.

    ``int(time.time())`` plus ``shutil.move`` was a data-destroying pair: two
    quarantines inside the same wall-clock second produced the same name, and
    shutil.move overwrites on Windows (its os.rename fails, and it falls back to
    copy2 + unlink, which does not), so the FIRST bad save vanished. Measured:
    two corrupt saves quarantined back to back left exactly one file, holding
    the second. It was the only place in the program where quarantining - the
    mechanism whose whole purpose is to preserve evidence - destroyed it.

    os.rename rather than shutil.move closes the check-then-act window as well
    as the naming one: on Windows it raises instead of overwriting, so a name
    that gets taken between the test and the move loses the race safely and the
    loop simply tries the next slot. Same directory, so it is a pure rename and
    cannot fall back to a copy.
    """
    base = f"{QUARANTINE_PREFIX}{time.strftime('%Y%m%d-%H%M%S', time.localtime())}"
    names = [f"{base}.json"]
    names += [f"{base}-{n}.json" for n in range(1, 100)]
    names.append(f"{base}-{time.time_ns()}.json")
    for name in names:
        dst = paths.APP_DIR / name
        if dst.exists():
            continue
        try:
            os.rename(paths.SAVE_PATH, dst)
            return dst
        except FileExistsError:
            continue
        except OSError:
            # Not a naming problem - a locked or vanished source. One report,
            # from the caller's except, is enough.
            raise
    return None


def _prune_quarantine() -> None:
    """Cap what quarantine leaves behind.

    This runs at login, forever, on a machine whose owner will never look in
    %LOCALAPPDATA%, and each item is a whole save file. Newest first: keep
    QUARANTINE_KEEP of them, and stop earlier if they add up to more than
    QUARANTINE_BUDGET - but always keep at least one, because the newest item is
    the evidence for the failure that just happened.

    The glob is anchored on QUARANTINE_PREFIX, so save.json itself and the
    pre-migration copies (PREMIGRATION_PREFIX, the one thing here that is
    irreplaceable) are outside it by construction; the explicit guard below says
    so out loud rather than trusting a future edit to the pattern.
    """
    try:
        found = list(paths.APP_DIR.glob(QUARANTINE_GLOB))
    except OSError:
        log.debug("could not list the quarantine directory", exc_info=True)
        return

    items: list[tuple[float, str, Path, int]] = []
    for p in found:
        if not p.name.startswith(QUARANTINE_PREFIX):
            continue
        if p.name.startswith(PREMIGRATION_PREFIX):     # cannot happen; say it
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        items.append((st.st_mtime, p.name, p, st.st_size))
    # Name breaks mtime ties, and the names are stamped %Y%m%d-%H%M%S with a
    # numeric suffix, so within one second the sort still matches the order they
    # were written in.
    items.sort(key=lambda it: (it[0], it[1]), reverse=True)

    total = 0
    for i, (_mt, name, p, size) in enumerate(items):
        total += size
        if i < QUARANTINE_KEEP and (i == 0 or total <= QUARANTINE_BUDGET):
            continue
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink()
            log.info("pruned old quarantined save %s (%d bytes)", name, size)
        except OSError:
            log.debug("could not prune %s", name, exc_info=True)
