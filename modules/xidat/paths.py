"""Where the client's files are: the game folder, Ashita's DAT overlays, and the HD copies.

    ffxi_dir()           the game folder (FFXI_DIR, else ~/FINAL FANTASY XI)
    overlay("ffxi-hd")   an XIPivot overlay beside it: <client>/Ashita/polplugins/DATs/<name>
    hd(path)             a game file's copy in the ffxi-hd overlay when there is one, else the file

LPS builds from the HD copies when they're there, so a local build matches what your client
shows. A build from the plain game files alone turns that off: LPS_HD=0 in the environment, or
set_hd(False).
"""
import os
from pathlib import Path

_hd = os.environ.get("LPS_HD", "1") != "0"


def ffxi_dir() -> Path:
    return Path(os.environ.get("FFXI_DIR", os.path.expanduser("~/FINAL FANTASY XI")))


def dats_root(ffxi: Path = None) -> Path:
    """Ashita's folder of XIPivot overlays for this client."""
    return Path(ffxi or ffxi_dir()).parent.parent / "Ashita/polplugins/DATs"


def overlay(name: str = "ffxi-hd", ffxi: Path = None) -> Path:
    return dats_root(ffxi) / name


def set_hd(on: bool) -> None:
    global _hd
    _hd = bool(on)


def hd_enabled() -> bool:
    return _hd


def hd(path, ffxi: Path = None, hd_dir: Path = None) -> Path:
    """The HD overlay's copy of a game file (a path in the game folder, or one relative to it),
    if HD is on and the copy exists; else the game's own file. A path outside the game folder is
    returned as it is."""
    ffxi = Path(ffxi or ffxi_dir())
    path = Path(path)
    if path.is_absolute():
        try:
            rel = path.relative_to(ffxi)
        except ValueError:
            return path
    else:
        rel = path
    if _hd:
        copy = Path(hd_dir or overlay(ffxi=ffxi)) / rel
        if copy.exists():
            return copy
    return ffxi / rel


# ---------------------------------------------------------------------------------------------
# the server side: where a build writes its server files
# ---------------------------------------------------------------------------------------------

def server_dir(spec: dict = None) -> Path:
    """The game server's folder (its settings/network.lua, its modules): a zone file block's
    "server", else XI_SERVER."""
    where = (spec or {}).get("server") or os.environ.get("XI_SERVER")
    if not where:
        raise SystemExit('no server folder: give the zone file block a "server", or set XI_SERVER')
    return Path(where).expanduser()


def sql_login(server) -> tuple:
    """(host, port, user, password, database) from a LandSandBoat server's settings/network.lua."""
    import re
    txt = (Path(server).expanduser() / "settings" / "network.lua").read_text()
    get = lambda k: re.search(rf"{k}\s*=\s*'([^']*)'", txt).group(1)
    port = re.search(r"SQL_PORT\s*=\s*(\d+)", txt).group(1)
    return get("SQL_HOST"), port, get("SQL_LOGIN"), get("SQL_PASSWORD"), get("SQL_DATABASE")


def npc_module(spec: dict = None) -> str:
    """Where in the server an extension writes its NPC modules (see run_hooks): a zone file
    block's "module", else XI_NPC_MODULE; relative to server_dir()."""
    where = (spec or {}).get("module") or os.environ.get("XI_NPC_MODULE")
    if not where:
        raise SystemExit('no NPC module folder: give the zone file block a "module", or set XI_NPC_MODULE')
    return where


def backups_dir() -> Path:
    """Where a build keeps the originals of files it rewrites in place (LPS_BACKUPS)."""
    return Path(os.environ.get("LPS_BACKUPS", "~/.cache/lps/backups")).expanduser()
