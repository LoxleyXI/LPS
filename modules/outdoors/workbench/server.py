#!/usr/bin/env python3
"""The architecture workbench: our pieces (outdoors/architecture.py) and any source model, in the
browser, textured from the player's own game files. Local only.

    python3 modules/outdoors/workbench/server.py [--port 8765] [--zones DIR]

then open http://localhost:8765. The page reloads itself when architecture.py changes.
http://localhost:8765/camp is the camp editor: a generated zone's camps, laid by hand (camp.html).
"""
import argparse
import importlib
import io
import json
import struct
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse, parse_qs

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))   # LPS root
sys.path.insert(0, str(HERE.parent.parent))          # modules

import numpy as np                                   # noqa: E402

import caves                                         # noqa: E402
from zonekit import writer as fz                     # noqa: E402
from caves.kits import _name                         # noqa: E402
from outdoors import architecture as A               # noqa: E402
from zonekit import textures as X                    # noqa: E402

ARCH = Path(A.__file__)
_cache = {}


def zone_sections(zid):
    if ("secs", zid) not in _cache:
        _cache[("secs", zid)] = fz.split_sections(caves.zone_dat(zid).read_bytes())
    return _cache[("secs", zid)]


def texture_png(zid, short):
    """A zone's texture by short name, as PNG bytes (RGBA)."""
    key = ("png", zid, short)
    if key in _cache:
        return _cache[key]
    from PIL import Image
    for s in zone_sections(zid):
        if s.kind != fz.T_TEXTURE:
            continue
        b = s.raw[16:]
        if b[1:17].decode("ascii", "replace").split()[-1:] != [short]:
            continue
        w, h = struct.unpack_from("<II", b, 0x15)
        if b[0] == 0xA1:
            size = struct.unpack_from("<I", b, 0x3D)[0]
            img = X.decode(b[0x45:0x45 + size], w, h, b[0x39:0x3D].decode())   # already RGB(A)
        else:
            bpp = struct.unpack_from("<H", b, 0x1F)[0]
            if bpp == 8:
                pal = np.frombuffer(b[0x3D:0x3D + 1024], np.uint8).reshape(256, 4)
                idx = np.frombuffer(b[0x3D + 1024:0x3D + 1024 + w * h], np.uint8).reshape(h, w)
                img = pal[idx][..., [2, 1, 0, 3]]
            else:
                img = np.frombuffer(b[0x3D:0x3D + w * h * 4], np.uint8).reshape(h, w, 4)[..., [2, 1, 0, 3]]
            img = img[::-1]                          # paletted ones are stored bottom up
        a = img[..., 3].astype(np.float32)
        img = img.copy()
        img[..., 3] = np.clip(a * 2, 0, 255).astype(np.uint8)   # FFXI alpha: 0x80 is opaque
        buf = io.BytesIO()
        Image.fromarray(img, "RGBA").save(buf, "PNG")
        _cache[key] = buf.getvalue()
        return _cache[key]
    return None


def piece_json(name, params):
    importlib.reload(A)
    pal = A.palette(params.pop("palette", "ronfaure"))
    m = A.build(name, pal, **{k: float(v) for k, v in params.items()})
    zone_of = {pal["texture"]: pal["from"]}
    if pal.get("banner"):
        zone_of[pal["banner"]["texture"]] = pal["banner"]["from"]
    parts = []
    for tex, (pos, nrm, uv, col, tri) in m.arrays().items():
        parts.append({"texture": tex, "zone": zone_of.get(tex, pal["from"]), "pos": pos.ravel().round(4).tolist(),
                      "nrm": nrm.ravel().round(3).tolist(), "uv": uv.ravel().round(5).tolist(),
                      "col": col.round(3).tolist(), "idx": tri.ravel().tolist()})
    return {"name": name, "parts": parts, "triangles": int(sum(len(p["idx"]) // 3 for p in parts))}


def rock_json(q):
    """The map's rock (rock.py) as the page's JSON: ?map=NAME&voxel=6&box=x0,z0,x1,z1."""
    from outdoors import rock, city
    importlib.reload(rock)
    m = city.load(q["map"])
    pal = A.palette(q.get("palette", "tahrongi"))
    box = tuple(float(v) for v in q.get("box", "-260,-160,260,160").split(","))
    mesh = rock.build(m, pal, box=box, voxel=float(q.get("voxel", 6)))
    parts = []
    for tex, (pos, nrm, uv, col, tri) in mesh.arrays().items():
        parts.append({"texture": tex, "zone": pal["from"], "pos": pos.ravel().round(3).tolist(),
                      "nrm": nrm.ravel().round(3).tolist(), "uv": uv.ravel().round(5).tolist(),
                      "col": col.round(3).tolist(), "idx": tri.ravel().tolist()})
    return {"name": "rock", "parts": parts, "triangles": int(sum(len(p["idx"]) // 3 for p in parts))}


def model_json(zid, short):
    """A source zone's model, y up, wound counter-clockwise from its front like ours."""
    tabs = fz.key_tables()
    for s in zone_sections(zid):
        if s.kind != fz.T_MESH:
            continue
        b = bytearray(s.raw[16:])
        fz.mesh_decrypt(b, tabs)
        m = fz.read_mesh(b)
        if _name(m["name"]) != short:
            continue
        parts = []
        for p in m["parts"]:
            t = fz.part_triangles(p, bool(m["config"] & 1))
            if not len(t):
                continue
            pos = p["pos"] * np.array([1, -1, 1], np.float32)
            nrm = p["nrm"] * np.array([1, -1, 1], np.float32)
            col = p["col"][:, :3].mean(1) / 128.0
            parts.append({"texture": bytes(p["texture"]).decode("ascii", "replace").split()[-1], "zone": zid,
                          "pos": pos.ravel().round(4).tolist(), "nrm": nrm.ravel().round(3).tolist(),
                          "uv": p["uv"].ravel().round(5).tolist(), "col": col.round(3).tolist(),
                          "idx": t.ravel().tolist()})
        return {"name": short, "parts": parts, "triangles": int(sum(len(p["idx"]) // 3 for p in parts))}
    return None


ZONES = HERE.parent / "zones"                         # zone files: --zones sets another folder
_camps = {}
import threading                                     # noqa: E402
_camps_lock = threading.Lock()                       # one generation at a time: a second request waits for it


def _zone_path(zone):
    """A zone file in the zone folder, by its name only (no other folder can be reached)."""
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]+", zone or ""):
        raise ValueError(f"not a zone file name: {zone!r}")
    return ZONES / f"{zone}.json"


def _zone_lands(zone):
    """The generated lands of an outdoor zone file, cached until the file changes: [{"name",
    "summary"}] (the first is the main land, then its regions)."""
    path = _zone_path(zone)
    key = (zone, path.stat().st_mtime)
    with _camps_lock:
        return _zone_lands_locked(zone, path, key)


def _zone_lands_locked(zone, path, key):
    if key not in _camps:
        import outdoors
        entry = json.loads(path.read_text())["zones"][0]
        spec = entry["outdoor"]
        kit = outdoors.kit.config(spec["kit"])
        _, summary = outdoors.generate(caves.zone_dat(kit["from"]).read_bytes(), spec, _name)
        lands = [{"name": spec.get("land", entry["name"]), "summary": summary}]
        lands += [{"name": r["name"], "summary": r} for r in summary.get("regions", [])]
        _camps.clear()
        _camps[key] = {"entry": entry, "lands": lands}
    return _camps[key]


def camp_json(q):
    """One land's camp for the editor (camp.html), in the land's own frame: ?zone=NAME&land=0."""
    from zonekit import tiles as T
    zone, k = q["zone"], int(q.get("land", 0))
    data = _zone_lands(zone)
    land = data["lands"][k]
    s = land["summary"]
    camp = (s.get("town") or {}).get("camp")
    if not camp:
        return {"error": "this land has no camp (no hamlet town)"}
    i0, j0, i1, j1, _h = camp["site"]
    pad = 1.5
    x0, z0, x1, z1 = (i0 - pad) * T.TILE, (j0 - pad) * T.TILE, (i1 + pad) * T.TILE, (j1 + pad) * T.TILE
    step = 2.0
    xs = np.arange(x0, x1 + 0.01, step)
    zs = np.arange(z0, z1 + 0.01, step)
    X, Z = np.meshgrid(xs, zs)
    H = s["surface"].at(X.ravel(), Z.ravel())
    lo = float(np.nanmin(H)) if np.isfinite(H).any() else 0.0
    H = np.where(np.isfinite(H) & (H > -900), H, lo - 6)
    spec = data["entry"]["outdoor"]
    reserve = int((spec.get("entry") or {}).get("reserve", 0))
    return {"zone": zone, "lands": [l["name"] for l in data["lands"]], "land": k, "name": land["name"],
            "shift": [float(v) for v in s.get("shift", (0, 0, 0))],
            "terrain": {"x0": float(x0), "z0": float(z0), "step": step, "nx": len(xs), "nz": len(zs),
                        "h": [round(float(v), 2) for v in H]},
            "roads": [list(c) for c in s.get("roads", []) if x0 <= c[0] * T.TILE <= x1 and z0 <= c[1] * T.TILE <= z1],
            "tile": T.TILE, "site": camp["site"], "camp": {kk: camp[kk] for kk in ("houses", "fire", "spots")},
            "fixed": bool(camp.get("fixed")), "templates": camp["templates"],
            "places": 1 + reserve, "roles": spec.get("camp_roles", [])}


def camp_save(body):
    """Write a land's camp into its zone file (town "fixed"), or drop it ("reset"): the next build
    lays it exactly."""
    zone, k = body["zone"], int(body["land"])
    path = _zone_path(zone)
    doc = json.loads(path.read_text())
    spec = doc["zones"][0]["outdoor"]
    town = (spec if k == 0 else spec["regions"][k - 1])["buildings"]["town"]
    if body.get("reset"):
        town.pop("fixed", None)
    else:
        c = body["camp"]
        town["fixed"] = {"fire": [round(float(v), 3) for v in c["fire"]],
                         "houses": [{"anchor": hh["anchor"], "variant": int(hh["variant"]), "x": round(float(hh["x"]), 3),
                                     "z": round(float(hh["z"]), 3), "yaw": round(float(hh["yaw"]), 4)} for hh in c["houses"]],
                         "spots": [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in c["spots"]]}
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return {"saved": str(path), "reset": bool(body.get("reset"))}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, kind="application/json", code=200):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        parts = [unquote(x) for x in u.path.strip("/").split("/") if x]
        try:
            if not parts:
                return self.send((HERE / "index.html").read_bytes(), "text/html")
            if parts == ["camp"]:
                return self.send((HERE / "camp.html").read_bytes(), "text/html")
            if parts == ["api", "camp"]:
                return self.send(camp_json(q))
            if parts == ["api", "version"]:
                return self.send({"version": ARCH.stat().st_mtime})
            if parts == ["api", "pieces"]:
                importlib.reload(A)
                return self.send({"pieces": sorted(A.PIECES)})
            if parts[:2] == ["api", "piece"] and len(parts) == 3:
                return self.send(piece_json(parts[2], dict(q)))
            if parts == ["api", "rock"]:
                return self.send(rock_json(q))
            if parts[:2] == ["api", "model"] and len(parts) == 4:
                j = model_json(int(parts[2]), parts[3])
                return self.send(j) if j else self.send({"error": "no such model"}, code=404)
            if parts[0] == "tex" and len(parts) == 3:
                png = texture_png(int(parts[1]), parts[2].removesuffix(".png"))
                return self.send(png, "image/png") if png else self.send(b"", code=404)
        except Exception as e:                           # shown in the page, not a dead server
            import traceback
            return self.send({"error": f"{e}", "trace": traceback.format_exc()}, code=500)
        self.send({"error": "not found"}, code=404)


def _do_post(self):
    u = urlparse(self.path)
    try:
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if u.path.strip("/") == "api/camp":
            return self.send(camp_save(body))
    except Exception as e:
        import traceback
        return self.send({"error": f"{e}", "trace": traceback.format_exc()}, code=500)
    self.send({"error": "not found"}, code=404)


Handler.do_POST = _do_post


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--zones", help="the folder of zone files for the camp editor (default: outdoors/zones)")
    a = ap.parse_args()
    if a.zones:
        ZONES = Path(a.zones).expanduser().resolve()
    print(f"workbench on http://localhost:{a.port}")
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
