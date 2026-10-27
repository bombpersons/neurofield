#!/usr/bin/env python3
"""Random levels for training data, built through the editor's MCP.

Lays out a dungeon in Python (rooms of many shapes, corridors routed between
them with stairs and ramps, balconies, pits, daises, pillars, courtyards), then
builds it in a headless editor with the MCP's level_* tools, so the editor's own
code does the carving, the stairs and the entity placement. Each level is
checked with the Issues window's checks: creatures that don't fit are shrunk or
dropped, and a level with any issue left (errors, or warnings such as views over
the triangle budget) is thrown away and rolled again.

    python3 editor/scripts/random_levels.py --count 40 --seed 1 --out levels/random

Needs a built editor (cargo build --release -p editor); --editor picks the binary.
Standard library only.
"""

import argparse
import heapq
import json
import math
import os
import random
import subprocess
import sys
import time

DIRS = {"n": (0, -1), "e": (1, 0), "s": (0, 1), "w": (-1, 0)}
OPP = {"n": "s", "s": "n", "e": "w", "w": "e"}
T = 256  # world units a tile
STEP = 64  # world units a layer
TEX_SETS = ["grounded", "mid", "deep", "strange"]
CREATURES = ["Humanoid", "Beast", "Blob", "Floater", "Mass"]
PROPS = ["Box", "Pillar", "Pyramid", "Floating"]
KEY_NAMES = ["iron key", "bone key", "rusted key", "brass key", "black key", "silver key", "moon key", "crypt key"]
LIGHT_COLOURS = [
    (1.0, 0.75, 0.5), (1.0, 0.6, 0.3), (1.0, 0.85, 0.65), (0.55, 0.7, 1.0), (0.5, 1.0, 0.6),
    (0.8, 0.5, 1.0), (1.0, 1.0, 0.92), (1.0, 0.4, 0.35), (0.6, 0.95, 0.95),
]


class Mcp:
    """A headless editor spoken to over stdio."""

    def __init__(self, exe, size):
        self.p = subprocess.Popen(
            [exe, "--mcp", "--headless", "--software", "--size", size],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
        )
        self.next = 0
        self.rpc("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "random_levels", "version": "1"}})
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def send(self, msg):
        self.p.stdin.write(json.dumps(msg) + "\n")
        self.p.stdin.flush()

    def rpc(self, method, params):
        self.next += 1
        self.send({"jsonrpc": "2.0", "id": self.next, "method": method, "params": params})
        while True:
            line = self.p.stdout.readline()
            if not line:
                raise RuntimeError("the editor closed its stdout")
            msg = json.loads(line)
            if msg.get("id") == self.next:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg["result"]

    def call(self, name, args=None, ok_fail=False):
        r = self.rpc("tools/call", {"name": name, "arguments": args or {}})
        text = "".join(c.get("text", "") for c in r.get("content", []) if c.get("type") == "text")
        if r.get("isError"):
            if ok_fail:
                return None
            raise ToolError(f"{name} {json.dumps(args)}: {text}")
        try:
            return json.loads(text)
        except ValueError:
            return text

    def close(self):
        self.p.stdin.close()
        try:
            self.p.wait(5)
        except subprocess.TimeoutExpired:
            self.p.kill()


class ToolError(Exception):
    pass


# ---------------------------------------------------------------- layout

def neighbours(t):
    x, z = t
    for d, (dx, dz) in DIRS.items():
        yield d, (x + dx, z + dz)


def connected(tiles):
    tiles = set(tiles)
    if not tiles:
        return False
    seen, todo = set(), [next(iter(tiles))]
    while todo:
        t = todo.pop()
        if t in seen:
            continue
        seen.add(t)
        todo += [n for _, n in neighbours(t) if n in tiles and n not in seen]
    return len(seen) == len(tiles)


def largest_component(tiles):
    tiles, best = set(tiles), set()
    while tiles:
        seen, todo = set(), [tiles.pop()]
        while todo:
            t = todo.pop()
            if t in seen:
                continue
            seen.add(t)
            todo += [n for _, n in neighbours(t) if n in tiles and n not in seen]
        tiles -= seen
        best = max(best, seen, key=len)
    return best


def shape_mask(rng, kind, w, d):
    """Footprint tiles (0..w, 0..d) of a room shape."""
    full = {(x, z) for x in range(w) for z in range(d)}
    if kind == "rect":
        return full
    if kind == "L" and w >= 4 and d >= 4:
        cw, cd = rng.randint(2, w - 2), rng.randint(2, d - 2)
        cx, cz = rng.choice([0, w - cw]), rng.choice([0, d - cd])
        return {t for t in full if not (cx <= t[0] < cx + cw and cz <= t[1] < cz + cd)}
    if kind == "cross" and w >= 5 and d >= 5:
        bw, bd = rng.randint(1, (w - 1) // 2), rng.randint(1, (d - 1) // 2)
        return {(x, z) for x, z in full if bw <= x < w - bw or bd <= z < d - bd}
    if kind == "T" and w >= 5 and d >= 4:
        bar = rng.randint(1, 2)
        stem = rng.randint(1, max(1, w // 3))
        mid = w // 2
        return {(x, z) for x, z in full if z < bar or abs(x - mid + 0.5) < stem + 0.5}
    if kind == "oval":
        cx, cz = (w - 1) / 2, (d - 1) / 2
        return {(x, z) for x, z in full if ((x - cx) / (w / 2)) ** 2 + ((z - cz) / (d / 2)) ** 2 <= 1.05}
    if kind == "octagon" and w >= 4 and d >= 4:
        c = rng.randint(1, min(w, d) // 3)
        return {(x, z) for x, z in full if min(x, w - 1 - x) + min(z, d - 1 - z) >= c}
    if kind == "ring" and w >= 5 and d >= 5:
        band = rng.choice([1, 2]) if min(w, d) >= 6 else 1
        return {(x, z) for x, z in full if min(x, w - 1 - x, z, d - 1 - z) < band}
    if kind == "cave":
        cells = {t for t in full if rng.random() < 0.62}
        for _ in range(3):
            nxt = set()
            for x, z in full:
                n = sum((x + a, z + b) in cells for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b)
                if n >= 5 or ((x, z) in cells and n >= 4):
                    nxt.add((x, z))
            cells = nxt
        return largest_component(cells)
    return full


class Room:
    def __init__(self, idx, x0, z0, w, d, kind, rng):
        self.idx, self.x0, self.z0, self.w, self.d, self.kind = idx, x0, z0, w, d, kind
        self.sky = kind == "courtyard"
        shape = rng.choice(["rect", "rect", "L", "cross", "T", "oval", "octagon", "ring", "cave"]) if not self.sky else rng.choice(["rect", "oval", "octagon", "cave"])
        if kind == "hall" and rng.random() < 0.6:
            shape = "rect"  # room for a balcony
        self.shape = shape
        local = shape_mask(rng, shape, w, d)
        if len(local) < 6 or not connected(local):
            local, self.shape = shape_mask(rng, "rect", w, d), "rect"
        self.mask = {(x0 + x, z0 + z) for x, z in local}
        if self.sky:
            self.h = rng.randint(10, 16)
        elif kind == "hall":
            self.h = rng.randint(18, 26)
        else:
            self.h = rng.choice([8, 9, 10, 10, 12, 12, 14, 16])
        # Each tile's open runs, relative to the room's base floor; tags for what it is.
        self.cols = {t: [(0, self.h)] for t in self.mask}
        self.tags = {t: set() for t in self.mask}
        self.stairs = []  # (tile, rel floor, dir, steps, rise, ramp, flat ceiling)
        self.base = None
        self.label = {"courtyard": "outdoors", "hall": "hall"}.get(kind) or rng.choice(["room", "room", "shrine", "crypt", "chamber", "cell", "library", "well room"])

    def interior(self, t):
        return all(n in self.mask for _, n in neighbours(t))

    def nodes(self):
        """Walkable places: (tile, run index) -> rel floor."""
        out = {}
        for t, runs in self.cols.items():
            if "stair" in self.tags[t]:
                continue
            for k, (a, b) in enumerate(runs):
                if b - a >= 5:
                    out[(t, k)] = a
        return out

    def walk_graph(self, blocked=()):
        nodes = {n: f for n, f in self.nodes().items() if n[0] not in blocked}
        adj = {n: [] for n in nodes}
        for (t, k), f in nodes.items():
            a, b = self.cols[t][k]
            for _, u in neighbours(t):
                for k2 in range(len(self.cols.get(u, []))):
                    m = (u, k2)
                    if m not in nodes:
                        continue
                    a2, b2 = self.cols[u][k2]
                    if abs(a2 - a) <= 1 and min(b, b2) - max(a, a2) >= 5:
                        adj[(t, k)].append(m)
        for (t, rf, d, n, r, _ramp, _flat) in self.stairs:
            dx, dz = DIRS[d]
            entry = (t[0] - dx, t[1] - dz)
            land = (t[0] + n * dx, t[1] + n * dz)
            e = [m for m in nodes if m[0] == entry and nodes[m] == rf]
            l = [m for m in nodes if m[0] == land and nodes[m] == rf + r * n]
            if e and l:
                adj[e[0]].append(l[0])
                adj[l[0]].append(e[0])
        return nodes, adj

    def all_reach(self, blocked=()):
        """Every walkable place outside pits joins up."""
        nodes, adj = self.walk_graph(blocked)
        want = set(nodes)
        if not want:
            return False
        seen, todo = set(), [next(iter(want))]
        while todo:
            n = todo.pop()
            if n in seen:
                continue
            seen.add(n)
            todo += adj[n]
        return want <= seen

    def features(self, rng):
        inner = [t for t in self.mask if self.interior(t)]
        # A balcony along one side of a tall rectangular hall, with stairs up to it.
        if self.kind == "hall" and self.shape == "rect" and self.w >= 6 and self.d >= 4:
            for _ in range(10):
                r = rng.randint(2, 4)
                n = rng.randint(2, min(4, self.w - 4))
                hb = r * n
                if 6 <= hb <= self.h - 9:
                    break
            else:
                hb = None
            if hb:
                z = self.z0 if rng.random() < 0.5 else self.z0 + self.d - 1
                inward = 1 if z == self.z0 else -1
                depth = rng.randint(1, 2)
                # A flat tile at the foot, inside the room, to step on from.
                self.tags[(self.x0, z)].add("foot")
                for i in range(1, n + 1):
                    self.tags[(self.x0 + i, z)].add("stair")
                self.stairs.append(((self.x0 + 1, z), 0, "e", n, r, False, True))
                for x in range(self.x0 + n + 1, self.x0 + self.w):
                    for j in range(depth):
                        t = (x, z + j * inward)
                        self.cols[t] = [(0, hb - 1), (hb, self.h)]
                        self.tags[t].add("balcony")
        # Raised dais, maybe in two tiers.
        if inner and rng.random() < 0.35 and not self.sky:
            sub = self.sub_rect(rng, inner)
            for t in sub:
                if not self.tags[t]:
                    self.cols[t] = [(1, self.h)]
                    self.tags[t].add("dais")
            if len(sub) >= 9 and rng.random() < 0.5:
                core = [t for t in sub if all(n in sub for _, n in neighbours(t)) and "dais" in self.tags[t]]
                for t in core:
                    self.cols[t] = [(2, self.h)]
        # Sunken floor, one or two layers down.
        elif inner and rng.random() < 0.3:
            sub = self.sub_rect(rng, inner)
            drop = rng.choice([1, 1, 2])
            for t in sub:
                if not self.tags[t]:
                    self.cols[t] = [(-1, self.h)]
                    self.tags[t].add("sunken")
            if drop == 2:
                for t in [t for t in sub if all(n in sub for _, n in neighbours(t)) and "sunken" in self.tags[t]]:
                    self.cols[t] = [(-2, self.h)]
        # A pit, with stairs (or a ramp) climbing out of it to the room's floor.
        inner = {t for t in self.mask if self.interior(t) and not self.tags[t]}
        if inner and rng.random() < 0.25:
            for _ in range(20):
                d = rng.choice("nesw")
                dx, dz = DIRS[d]
                r, n = rng.choice([2, 3, 4]), rng.randint(1, 3)
                length, wide = n + rng.randint(1, 2), rng.choice([1, 1, 2])
                rim = rng.choice(sorted(self.cols))
                if self.tags[rim]:
                    continue
                pit = {(rim[0] - k * dx + s * dz, rim[1] - k * dz + s * dx) for k in range(1, length + 1) for s in range(wide)}
                if not pit <= inner:
                    continue
                stair = [(rim[0] - k * dx, rim[1] - k * dz) for k in range(1, n + 1)]
                for t in pit:
                    self.cols[t] = [(-r * n, self.h)]
                    self.tags[t].add("pit")
                for t in stair:
                    self.tags[t].add("stair")
                self.stairs.append((stair[-1], -r * n, d, n, r, r <= 2 and rng.random() < 0.5, True))
                if self.all_reach():
                    break
                self.stairs.pop()
                for t in pit:
                    self.cols[t] = [(0, self.h)]
                    self.tags[t] -= {"pit", "stair"}
        # Vaulted middle, or low beams.
        if not self.sky:
            if rng.random() < 0.35:
                v = rng.randint(2, 6)
                cx, cz = self.x0 + self.w / 2, self.z0 + self.d / 2
                for t in self.mask:
                    if abs(t[0] + 0.5 - cx) < self.w / 3 and abs(t[1] + 0.5 - cz) < self.d / 3:
                        self.cols[t] = [(a, b + v if k == len(self.cols[t]) - 1 else b) for k, (a, b) in enumerate(self.cols[t])]
            elif rng.random() < 0.3 and self.h >= 11:
                axis = rng.choice([0, 1])
                every = rng.randint(2, 3)
                for t in self.mask:
                    if (t[axis] - (self.x0, self.z0)[axis]) % every == 1:
                        runs = self.cols[t]
                        a, b = runs[-1]
                        if b - 2 - a >= 8:
                            runs[-1] = (a, b - 2)
        # Pillars, where they don't cut the room in two.
        if rng.random() < 0.5:
            cands = [t for t in self.mask if self.interior(t) and not self.tags[t]]
            rng.shuffle(cands)
            for t in cands[: rng.randint(1, 6)]:
                if any("pillar" in self.tags[n] for _, n in neighbours(t)):
                    continue
                old = self.cols.pop(t)
                if self.all_reach():
                    self.tags[t].add("pillar")
                else:
                    self.cols[t] = old

    def sub_rect(self, rng, tiles, small=False):
        xs, zs = [t[0] for t in tiles], [t[1] for t in tiles]
        lo_x, hi_x, lo_z, hi_z = min(xs), max(xs), min(zs), max(zs)
        w = rng.randint(1, max(1, (hi_x - lo_x + 1) // (3 if small else 1)))
        d = rng.randint(1, max(1, (hi_z - lo_z + 1) // (3 if small else 1)))
        x = rng.randint(lo_x, max(lo_x, hi_x - w + 1))
        z = rng.randint(lo_z, max(lo_z, hi_z - d + 1))
        ts = set(tiles)
        return {(a, b) for a in range(x, x + w) for b in range(z, z + d) if (a, b) in ts}

    def door_tiles(self):
        """Tiles a corridor may come in at, with the run it meets."""
        out = []
        for t, runs in self.cols.items():
            if self.tags[t] & {"stair", "pit"}:
                continue
            for k, (a, b) in enumerate(runs):
                if b - a >= 8 and not (k == 0 and len(runs) > 1):
                    out.append((t, k, a))
        return out


class Corridor:
    def __init__(self, a, b, path, ta, tb):
        self.a, self.b, self.path = a, b, path
        self.ta, self.tb = ta, tb  # (room tile, run, rel floor) at each end
        self.floor = {}  # path tile -> absolute floor (flat tiles and stair starts)
        self.stairs = []  # (tile, abs floor, dir, steps, rise, ramp)
        self.doors = []


def route(rng, size, blocked, near, starts, goals):
    """A* over tiles from any start to any goal, turning as little as it can.
    `starts`/`goals` map a corridor end tile to the room tile it meets."""
    W, D = size

    def free(t, end_ok):
        x, z = t
        if not (1 <= x < W - 1 and 1 <= z < D - 1) or t in blocked:
            return False
        return end_ok or t not in near

    goal_set = set(goals)
    pq, best, prev = [], {}, {}
    for s in starts:
        st = (s, None)
        best[st] = 0
        heapq.heappush(pq, (0, rng.random(), 0, st))
    gx = sum(g[0] for g in goals) / len(goals)
    gz = sum(g[1] for g in goals) / len(goals)
    while pq:
        _, _, cost, st = heapq.heappop(pq)
        t, d = st
        if cost > best.get(st, 1e9):
            continue
        if t in goal_set and d is not None:
            path = [t]
            while st in prev:
                st = prev[st]
                path.append(st[0])
            return path[::-1]
        if cost > 220:
            break
        for nd, u in neighbours(t):
            end = u in goal_set
            if not free(u, end) and not end:
                continue
            c = cost + 1 + (3 if d is not None and nd != d else 0) + rng.random() * 0.5
            nst = (u, nd)
            if c < best.get(nst, 1e9):
                best[nst] = c
                prev[nst] = st
                h = abs(u[0] - gx) + abs(u[1] - gz)
                heapq.heappush(pq, (c + h * 0.5, rng.random(), c, nst))
    return None


def segments(path):
    """Straight runs of the path: (first index, last index, dir), indices inclusive,
    each run including its turn tiles at both ends."""
    out, i = [], 0
    while i < len(path) - 1:
        dx, dz = path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]
        d = next(k for k, v in DIRS.items() if v == (dx, dz))
        j = i + 1
        while j < len(path) - 1 and (path[j + 1][0] - path[j][0], path[j + 1][1] - path[j][1]) == (dx, dz):
            j += 1
        out.append((i, j, d))
        i = j
    return out


def plan_flights(rng, path, want=None):
    """Stairs and ramps along a corridor: [(first index, steps, rise, up?, ramp?)].
    With `want`, they must climb exactly that many layers (or None if they can't);
    otherwise a random climb."""
    segs = [(i, j, d) for i, j, d in segments(path) if j - i - 1 >= 1]
    if want is not None:
        if want == 0:
            return []
        for i, j, d in sorted(segs, key=lambda s: s[0] - s[1]):
            cap = j - i - 1
            for r in (4, 3, 2, 1):
                if abs(want) % r == 0 and abs(want) // r <= cap:
                    n = abs(want) // r
                    return [(i + 1 + rng.randint(0, cap - n), n, r, want > 0, r <= 2 and rng.random() < 0.5)]
        return None
    if rng.random() < 0.3 or not segs:
        return []
    up = rng.random() < 0.5
    out = []
    rng.shuffle(segs)
    for i, j, d in segs[: rng.choice([1, 1, 2])]:
        cap = j - i - 1
        n = rng.randint(1, min(cap, 6))
        ramp = rng.random() < 0.3
        r = rng.randint(1, 2) if ramp else rng.randint(2, 4)
        out.append((i + 1 + rng.randint(0, cap - n), n, r, up, ramp))
    return out


def layout(rng):
    """Rooms and corridors of one level, all in tiles and absolute layers."""
    W = D = rng.randint(40, 64)
    want = rng.randint(5, 11)
    rooms = []
    for _ in range(400):
        if len(rooms) >= want:
            break
        kind = rng.choices(["room", "hall", "courtyard"], [6, 2, 1.2])[0]
        w = rng.randint(3, 8) if kind == "room" else rng.randint(6, 10)
        d = rng.randint(3, 8) if kind == "room" else rng.randint(4, 9)
        x0, z0 = rng.randint(2, W - w - 2), rng.randint(2, D - d - 2)
        if any(x0 < r.x0 + r.w + 3 and r.x0 < x0 + w + 3 and z0 < r.z0 + r.d + 3 and r.z0 < z0 + d + 3 for r in rooms):
            continue
        rooms.append(Room(len(rooms), x0, z0, w, d, kind, rng))
    for r in rooms:
        r.features(rng)

    # Which rooms to join: a spanning tree by distance, then a few loops.
    cen = [(r.x0 + r.w / 2, r.z0 + r.d / 2) for r in rooms]
    dist = lambda a, b: abs(cen[a][0] - cen[b][0]) + abs(cen[a][1] - cen[b][1])
    order, tree = [0], []
    while len(order) < len(rooms):
        a, b = min(((a, b) for a in order for b in range(len(rooms)) if b not in order), key=lambda e: dist(*e))
        tree.append((a, b))
        order.append(b)
    extra = [(a, b) for a in range(len(rooms)) for b in range(a + 1, len(rooms)) if (a, b) not in tree and (b, a) not in tree]
    extra.sort(key=lambda e: dist(*e))
    extra = [e for e in extra[:6] if rng.random() < 0.5][:3]

    rooms[0].base = 0
    corridors = []
    used = set()  # corridor tiles
    joined = {0}

    def occupied():
        occ = set(used)
        for r in rooms:
            occ |= r.mask
        return occ

    def near_map():
        near = {}
        for t in occupied():
            for _, n in neighbours(t):
                near[n] = True
        return near

    def try_join(a, b, fixed):
        ra, rb = rooms[a], rooms[b]
        occ, near = occupied(), near_map()

        def ends(room):
            out = {}
            for t, k, f in room.door_tiles():
                for d, u in neighbours(t):
                    if u in occ:
                        continue
                    # The corridor end touches this room tile and nothing else built.
                    touching = [n for _, n in neighbours(u) if n in occ]
                    if touching == [t]:
                        out.setdefault(u, (t, k, f, OPP[d]))
            return out

        sa, sb = ends(ra), ends(rb)
        if not sa or not sb:
            return False
        path = route(rng, (W, D), occ, near, list(sa), list(sb))
        if not path or len(path) < 3:
            return False
        # No tile of it may touch another but its neighbours along it, or another room.
        idx = {t: i for i, t in enumerate(path)}
        for i, t in enumerate(path):
            for _, n in neighbours(t):
                j = idx.get(n)
                if j is not None and abs(j - i) > 1:
                    return False
                if n in occ and not ((i == 0 and n == sa[path[0]][0]) or (i == len(path) - 1 and n == sb[path[-1]][0])):
                    return False
        ea, eb = sa[path[0]], sb[path[-1]]
        fa = ra.base + ea[2]
        if fixed:
            flights = plan_flights(rng, path, want=rb.base + eb[2] - fa)
            if flights is None:
                return False
        else:
            flights = plan_flights(rng, path)
        c = Corridor(a, b, path, ea, eb)
        # Steep stairs need more than 8 layers of headroom to climb.
        c.h = rng.randint(9 if any(f[2] == 4 for f in flights) else 8, 11)
        f = fa
        stair_at = {}
        for (i, n, r, up, ramp) in flights:
            stair_at[i] = (n, r, up, ramp)
        i = 0
        while i < len(path):
            if i in stair_at:
                n, r, up, ramp = stair_at[i]
                dx, dz = path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1]
                d = next(k for k, v in DIRS.items() if v == (dx, dz))
                if up:
                    c.stairs.append((path[i], f, d, n, r, ramp))
                    c.floor[path[i]] = f
                    f += r * n
                else:
                    f -= r * n
                    low = path[i + n - 1]
                    c.stairs.append((low, f, OPP[d], n, r, ramp))
                    c.floor[low] = f
                i += n
                continue
            c.floor[path[i]] = f
            i += 1
        if not fixed:
            rb.base = f - eb[2]
            if abs(rb.base) > 48:
                return False
        elif f != rb.base + eb[2]:
            return False
        corridors.append(c)
        used.update(path)
        return True

    # Join rooms in tree order; one that can't reach its parent tries the nearest joined rooms.
    for _ in range(2):
        for a, b in tree:
            if b in joined:
                continue
            cands = ([a] if a in joined else []) + sorted(joined - {a}, key=lambda j: dist(j, b))[:4]
            for j in cands:
                if any(try_join(j, b, False) for _ in range(2)):
                    joined.add(b)
                    break
    for a, b in extra:
        if a in joined and b in joined:
            try_join(a, b, True)
    rooms = [r for r in rooms if r.idx in joined]
    return (W, D), rooms, corridors


# ---------------------------------------------------------------- building

def rects(tiles):
    """Cover a set of tiles with rectangles, greedily: [(x0, z0, x1, z1)] exclusive."""
    left, out = set(tiles), []
    while left:
        x, z = min(left, key=lambda t: (t[1], t[0]))
        x1 = x
        while (x1 + 1, z) in left:
            x1 += 1
        z1 = z
        while all((i, z1 + 1) in left for i in range(x, x1 + 1)):
            z1 += 1
        for i in range(x, x1 + 1):
            for j in range(z, z1 + 1):
                left.discard((i, j))
        out.append((x, z, x1 + 1, z1 + 1))
    return out


def jitter(rng, c, amount):
    return [max(0.0, min(1.0, v + rng.uniform(-amount, amount))) for v in c]


ALLOW_WARNINGS = False


def build(m, rng, name, out_dir):
    size, rooms, corridors = layout(rng)
    if len(rooms) < 3:
        raise ToolError("too few rooms joined")
    stats = {"rooms": len(rooms), "corridors": len(corridors), "shapes": sorted({r.shape for r in rooms}),
             "stairs": sum(len(c.stairs) for c in corridors) + sum(len(r.stairs) for r in rooms)}
    m.call("editor_open", {"new": True, "discard": True})

    # Settings.
    tex = rng.choices(TEX_SETS, [4, 3, 2, 1.5])[0]
    lo = min(r.base for r in rooms) - 8
    hi = max(r.base + r.h for r in rooms) + 8
    pts = sorted({lo, hi, rng.randint(lo, hi)})
    depth = [[float(p), round(rng.choice([0.0, 0.0, 0.05, 0.15, 0.3, 0.5, 0.7, 0.9]) * rng.random() ** 0.3, 2)] for p in pts]
    amb = rng.uniform(0.1, 0.32)
    settings = {
        "name": name,
        "textures": tex,
        "ambient": jitter(rng, [amb, amb * 0.92, amb * 0.9], 0.04),
        "fog": [round(rng.uniform(1.2, 4.0), 2), round(rng.uniform(5.5, 11.0), 2)],
        "surface": {"floor": rng.randrange(8), "wall": rng.randrange(8), "ceiling": rng.randrange(8), "tint": jitter(rng, [0.82, 0.78, 0.74], 0.12)},
        "strangeness": {"depth": depth, "noise": [round(rng.uniform(0.0, 0.25), 2), round(rng.uniform(0.4, 1.2), 2)], "seed": rng.randrange(1 << 30)},
    }
    if any(r.sky for r in rooms):
        el = rng.uniform(0.3, 1.0)
        az = rng.uniform(0, 2 * math.pi)
        settings["sun"] = {"dir": [math.cos(az) * math.cos(el), math.sin(el), math.sin(az) * math.cos(el)], "colour": jitter(rng, [0.95, 0.85, 0.7], 0.15)}
    m.call("level_settings", {"set": settings})

    # Rooms.
    for r in rooms:
        groups = {}
        for t, runs in r.cols.items():
            lo_, hi_ = runs[0][0], runs[-1][1]
            groups.setdefault((lo_, hi_), set()).add(t)
        for (a, b), tiles in groups.items():
            for x0, z0, x1, z1 in rects(tiles):
                m.call("level_box", {"min": [x0, r.base + a, z0], "max": [x1, r.base + b, z1]})
        # Slabs: the solid between two runs of a column (balconies).
        slabs = {}
        for t, runs in r.cols.items():
            for (a0, b0), (a1, b1) in zip(runs, runs[1:]):
                slabs.setdefault((b0, a1), set()).add(t)
        for (a, b), tiles in slabs.items():
            for x0, z0, x1, z1 in rects(tiles):
                m.call("level_box", {"min": [x0, r.base + a, z0], "max": [x1, r.base + b, z1], "fill": True})
        for (t, rf, d, n, rise, ramp, flat) in r.stairs:
            m.call("level_stairs", {"cell": [t[0], r.base + rf, t[1]], "dir": d, "steps": n, "rise": rise, "ramp": ramp, "flat_ceiling": flat})
        if r.sky:
            for t in r.cols:
                m.call("level_sky", {"cell": [t[0], r.base + r.cols[t][-1][0], t[1]], "sky": True}, ok_fail=True)

    # Corridors.
    for c in corridors:
        groups = {}
        for t, f in c.floor.items():
            groups.setdefault(f, set()).add(t)
        for f, tiles in groups.items():
            for x0, z0, x1, z1 in rects(tiles):
                m.call("level_box", {"min": [x0, f, z0], "max": [x1, f + c.h, z1]})
        for (t, f, d, n, rise, ramp) in c.stairs:
            m.call("level_stairs", {"cell": [t[0], f, t[1]], "dir": d, "steps": n, "rise": rise, "ramp": ramp})

    # Areas, with their own surfaces; some pinned or pushed in strangeness.
    for r in rooms:
        lo_ = r.base + min(a for runs in r.cols.values() for a, _ in runs)
        hi_ = r.base + max(b for runs in r.cols.values() for _, b in runs)
        boxes = [[[x0, lo_, z0], [x1, hi_, z1]] for x0, z0, x1, z1 in rects(r.mask)]
        args = {"name": f"{r.label} {r.idx}", "label": r.label, "boxes": boxes, "notes": f"random: {r.shape} {r.kind}"}
        if rng.random() < 0.75:
            tint = jitter(rng, [0.82, 0.78, 0.74], 0.1) if rng.random() < 0.7 else [round(rng.uniform(0.4, 1.0), 2) for _ in range(3)]
            args["surface"] = {"floor": rng.randrange(8), "wall": rng.randrange(8), "ceiling": rng.randrange(8), "tint": tint}
        if rng.random() < 0.15:
            args["strangeness"] = {"Pin": round(rng.uniform(0.0, 1.0), 2)}
        elif rng.random() < 0.15:
            args["strangeness"] = {"Add": round(rng.uniform(-0.3, 0.5), 2)}
        m.call("level_add_area", args)
    for i, c in enumerate(corridors):
        if rng.random() < 0.5:
            fs = list(c.floor.values()) + [s[1] + s[3] * s[4] for s in c.stairs]
            boxes = [[[x0, min(fs), z0], [x1, max(fs) + c.h, z1]] for x0, z0, x1, z1 in rects(c.path)]
            args = {"name": f"corridor {i}", "label": "stairwell" if c.stairs else "corridor", "boxes": boxes}
            if rng.random() < 0.7:
                args["surface"] = {"floor": rng.randrange(8), "wall": rng.randrange(8), "ceiling": rng.randrange(8), "tint": jitter(rng, [0.8, 0.76, 0.72], 0.12)}
            m.call("level_add_area", args)

    # Accent paint: a wall here and there, and patterned floors.
    for r in rooms:
        if rng.random() < 0.5:
            slot = rng.randrange(8)
            edge = [(t, d) for t in r.cols for d, n in neighbours(t) if n not in r.cols and n not in r.mask]
            for t, d in rng.sample(edge, min(len(edge), rng.randint(1, 3))):
                cell = [t[0], r.base + r.cols[t][-1][0] + 1, t[1]]
                m.call("level_paint", {"faces": [{"cell": cell, "face": d}], "slot": slot, "fill": True}, ok_fail=True)
        if rng.random() < 0.3:
            slot = rng.randrange(8)
            faces = [{"cell": [t[0], r.base + r.cols[t][0][0], t[1]], "face": "floor"} for t in r.cols
                     if (t[0] + t[1]) % 2 == 0 and not r.tags[t] & {"stair"}]
            if faces:
                m.call("level_paint", {"faces": faces, "slot": slot}, ok_fail=True)

    # Strangeness auras.
    for _ in range(rng.choice([0, 0, 1, 2, 3])):
        r = rng.choice(rooms)
        t = rng.choice(list(r.cols))
        m.call("level_strange", {"centre": [t[0] + 0.5, (r.base + r.h / 2) / 4, t[1] + 0.5], "radius": rng.uniform(1.5, 5),
                                 "amount": round(rng.uniform(-0.4, 0.8), 2), "softness": rng.random()})

    # Lights.
    def light(x, z, floor, ceil):
        y = -(floor + (ceil - floor) * rng.uniform(0.55, 0.85)) * STEP
        col = jitter(rng, rng.choice(LIGHT_COLOURS), 0.06)
        m.call("level_add_light", {"pos": [(x + rng.uniform(0.2, 0.8)) * T, y, (z + rng.uniform(0.2, 0.8)) * T], "colour": col, "radius": rng.uniform(2.0, 5.5) * T})

    nl = 0
    for r in rooms:
        for _ in range(rng.choice([0, 1, 1, 2, 2, 3]) if not r.sky else rng.choice([0, 1])):
            t = rng.choice(list(r.cols))
            light(t[0], t[1], r.base + r.cols[t][-1][0], r.base + r.cols[t][-1][1])
            nl += 1
    for c in corridors:
        flat = list(c.floor.items())
        for k in range(0, len(flat), rng.randint(5, 9)):
            if rng.random() < 0.5:
                t, f = flat[k]
                light(t[0], t[1], f, f + c.h)
                nl += 1
    stats["lights"] = nl

    # Entities. Places taken per room tile.
    taken = set()
    walk = {}
    for r in rooms:
        nodes, _ = r.walk_graph()
        walk[r.idx] = [(t, f) for (t, k), f in nodes.items() if not r.tags[t] & {"pit", "stair"}]
    room_of = {r.idx: r for r in rooms}

    solid = set()  # tiles holding creatures, chests and props, which block the way

    def spot(r, blocks=False, ok=lambda t: True):
        """A free floor cell in the room; one for something that blocks must leave
        every place in the room still joined up."""
        cands = [(t, f) for t, f in walk[r.idx] if t not in taken and ok(t)]
        rng.shuffle(cands)
        for t, f in cands[:12]:
            if blocks and not r.all_reach(blocked={u for u in solid if u in r.cols} | {t}):
                continue
            taken.add(t)
            if blocks:
                solid.add(t)
            return [t[0], r.base + f, t[1]]
        return None

    def add(kind, cell, **kw):
        res = m.call("level_add_entity", {"kind": kind, "cell": cell, **kw}, ok_fail=True)
        return res["id"] if res else None

    start = rooms[0]
    # Keep the corridor mouths clear.
    for c in corridors:
        taken.add(c.ta[0])
        taken.add(c.tb[0])
    cell = spot(start)
    add("player start", cell, facing=rng.choice("nesw"))

    # Doors at corridor mouths.
    doors = []  # (id, corridor index)
    for ci, c in enumerate(corridors):
        for end, (rt, k, rf, d) in ((c.path[0], c.ta), (c.path[-1], c.tb)):
            if rng.random() < 0.55:
                hinge = rng.choice(["Left", "Right", "Up"])
                kind = {"door": {"hinge": hinge, "height": min(8, c.h), "open": rng.random() < 0.2}}
                did = add(kind, [end[0], c.floor[end], end[1]], edge=d)
                if did:
                    doors.append((did, ci))

    # Locks: each locked door's key (or lever) is reachable with it and every
    # later locked door shut.
    def reach(shut):
        edges = {}
        for ci, c in enumerate(corridors):
            if ci in shut:
                continue
            edges.setdefault(c.a, []).append(c.b)
            edges.setdefault(c.b, []).append(c.a)
        seen, todo = set(), [rooms[0].idx]
        while todo:
            a = todo.pop()
            if a in seen:
                continue
            seen.add(a)
            todo += edges.get(a, [])
        return seen

    closed = [d for d in doors if rng.random() < 0.6][: rng.randint(0, 3)]
    nkeys = 0
    for i, (did, ci) in enumerate(closed):
        region = reach({c for _, c in closed[i:]})
        r = room_of[rng.choice(sorted(region))]
        cell = spot(r)
        if not cell:
            continue
        if rng.random() < 0.7:
            kid = add({"key": {"name": rng.choice(KEY_NAMES)}}, cell, yaw=rng.uniform(-3, 3))
            if kid:
                m.call("level_link", {"from": did, "to": kid})
                nkeys += 1
        else:
            lid = add("lever", cell, facing=rng.choice("nesw"))
            if lid:
                m.call("level_link", {"from": lid, "to": did})
                m.call("level_edit_entity", {"id": did, "set": {"kind": {"Door": {"locked": True, "open": False}}}})
    stats["doors"], stats["locked"] = len(doors), len(closed)

    # Chests, some locked, with their key on the near side.
    for _ in range(rng.choice([0, 1, 1, 2])):
        r = rng.choice(rooms)
        cell = spot(r, blocks=True)
        if not cell:
            continue
        cid = add({"chest": {"contents": [rng.choice(["potion", "gold", "dagger", "scroll", "ring"])]}}, cell, facing=rng.choice("nesw"))
        if cid and rng.random() < 0.35:
            kc = spot(rng.choice([room_of[i] for i in sorted(reach({c for _, c in closed}))]))
            if kc:
                kid = add({"key": {"name": rng.choice(KEY_NAMES)}}, kc)
                if kid:
                    m.call("level_link", {"from": cid, "to": kid})

    # Creatures.
    ncr = 0
    for r in rooms:
        if rng.random() < 0.7:
            for _ in range(rng.choice([1, 1, 2, 3])):
                cell = spot(r, blocks=True)
                if not cell:
                    break
                top = min(b for (a, b) in r.cols[(cell[0], cell[2])] if a <= cell[1] - r.base < b)
                head = (r.base + top - cell[1]) / 4
                size = round(min(rng.uniform(0.5, 1.5), max(0.4, head * 0.45)), 2)
                kind = {"creature": {"kind": rng.choice(CREATURES), "size": size, "seed": rng.randrange(1 << 31)}}
                if add(kind, cell, yaw=rng.uniform(-math.pi, math.pi)):
                    ncr += 1

    # Props, only where they don't block the way.
    for r in rooms:
        for _ in range(rng.choice([0, 0, 1, 2, 3, 4])):
            cell = spot(r, blocks=True, ok=lambda t: r.interior(t) and not r.tags[t] & {"balcony", "stair"})
            if not cell:
                break
            shape = rng.choice(PROPS)
            s = [rng.uniform(50, 180), rng.uniform(60, 220), rng.uniform(50, 180)]
            if shape == "Pillar":
                s = [rng.uniform(50, 130), 0.0, rng.uniform(50, 130)]
            kind = {"prop": {"shape": shape, "size": s, "tex": rng.randrange(8), "tint": jitter(rng, [0.75, 0.72, 0.68], 0.2)}}
            add(kind, cell, yaw=rng.choice([0.0, math.pi / 2, rng.uniform(-math.pi, math.pi)]))
    stats["creatures"] = ncr

    # Fix what the checks find: creatures that don't fit shrink or go; entities in
    # rock or out of reach go. Any other error throws the level away.
    for _ in range(4):
        issues = m.call("editor_level", {"part": "issues"})["issues"]
        fixable = [i for i in issues if i.get("entity") and i["group"] in ("creatures", "entities")]
        if not fixable:
            break
        for i in fixable:
            eid = i["entity"]
            if "pressed against the ceiling" in i["text"]:
                ents = m.call("editor_level", {"part": "entities"})["entities"]
                e = next((e for e in ents if e["id"] == eid), None)
                if e and e["kind"].get("Creature") and e["kind"]["Creature"]["size"] > 0.45:
                    m.call("level_edit_entity", {"id": eid, "set": {"kind": {"Creature": {"size": round(e["kind"]["Creature"]["size"] * 0.7, 2)}}}}, ok_fail=True)
                    continue
            m.call("level_delete_entity", {"id": eid}, ok_fail=True)
    issues = m.call("editor_level", {"part": "issues"})["issues"]
    bad = issues if not ALLOW_WARNINGS else [i for i in issues if i["severity"] == "error"]
    stats["warnings"] = len(issues) - len([i for i in issues if i["severity"] == "error"])
    if bad:
        raise ToolError("issues: " + "; ".join(i["text"] for i in bad[:4]))
    path = os.path.abspath(os.path.join(out_dir, name + ".ron"))
    m.call("editor_save", {"path": path})
    return path, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="levels/random")
    ap.add_argument("--prefix", default="random")
    ap.add_argument("--editor", default="target/release/editor")
    ap.add_argument("--size", default="320x240", help="headless window size (small is faster)")
    ap.add_argument("--tries", type=int, default=12, help="rolls per level before giving up on it")
    ap.add_argument("--allow-warnings", action="store_true", help="keep levels whose checks only warn (by default any issue re-rolls)")
    a = ap.parse_args()
    global ALLOW_WARNINGS
    ALLOW_WARNINGS = a.allow_warnings
    os.makedirs(a.out, exist_ok=True)
    m = Mcp(a.editor, a.size)
    report = []
    try:
        for i in range(a.count):
            name = f"{a.prefix}_{a.seed}_{i:03}"
            for k in range(a.tries):
                rng = random.Random(f"{a.seed}/{i}/{k}")
                t0 = time.time()
                try:
                    path, stats = build(m, rng, name, a.out)
                except ToolError as e:
                    print(f"{name} roll {k}: {e}", file=sys.stderr)
                    continue
                stats.update(name=name, roll=k, seconds=round(time.time() - t0, 1))
                report.append(stats)
                print(json.dumps(stats), flush=True)
                break
            else:
                print(f"{name}: gave up", file=sys.stderr)
    finally:
        m.close()
        with open(os.path.join(a.out, "report.jsonl"), "a") as f:
            for s in report:
                f.write(json.dumps(s) + "\n")


if __name__ == "__main__":
    main()
