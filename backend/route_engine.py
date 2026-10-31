import heapq
import json
import math

R = 6371000.0
NODE_MERGE_M = 2.0
START_CONNECT_M = 60.0
DEST_CONNECT_M = 60.0
CROSS_CONNECT_M = 4.0
NEAR_CONNECT_MANUAL_M = 3.0
# Only very small gaps are treated as a real junction. A larger
# "nearby" tolerance can incorrectly connect paths across walls/buildings.
NEAR_CONNECT_MIXED_M = 3.0
MAX_DETOUR_FACTOR = 6.0
MAX_EXTRA_DETOUR_M = 2500.0
REFERENCE_COST = 1.15
ACCESS_SLACK_M = 6.0   # only paths about as near as the nearest one may be used to enter/leave the network

import re
Y_NAME = re.compile(r"(^|\W)y[\s-]*(point|junction|fork)|\bfork\b", re.I)
STEM = re.compile(r"stem\s*[:=]?\s*(NNE|ENE|ESE|SSE|SSW|WSW|WNW|NNW|NE|SE|SW|NW|N|E|S|W)\b", re.I)
COMPASS = {"N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
           "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5, "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5}


def _bearing(a, b):
    p1, p2, dl = math.radians(a[0]), math.radians(b[0]), math.radians(b[1] - a[1])
    return (math.degrees(math.atan2(math.sin(dl) * math.cos(p2),
                                    math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl))) + 360) % 360


def _y_hints(data):
    """Y pins. The 'stem' is the single path that splits into two branches; give it as
    "stem_bearing" (degrees) in the JSON or in the pin name, e.g. "Y Point (stem NE)"."""
    out = []
    for p in data.get("points", []):
        name = str(p.get("name", ""))
        if not Y_NAME.search(name):
            continue
        stem = p.get("stem_bearing")
        if stem is None:
            m = STEM.search(name)
            stem = COMPASS[m.group(1).upper()] if m else None
        out.append((name, (float(p["lat"]), float(p["lng"])), None if stem is None else float(stem)))
    return out


def _y_steps(coords, cum, y_points):
    """'Keep left/right' only where the walker arrives along the stem and must pick a branch.
    Arriving along a branch is just a merge (nothing to choose), so those spots are returned as
    'merges' and any ordinary turn there is suppressed."""
    found, merges = [], []
    for name, yp, stem in y_points:
        if stem is None:
            continue                                   # stem unknown: never guess
        best = None
        for i in range(1, len(coords)):
            q = project(yp, coords[i - 1], coords[i])
            if best is None or q["distance"] < best[0]:
                best = (q["distance"], i, q["t"], q["point"])
        if best is None or best[0] > 12:
            continue
        _, i, t, pt = best
        pos = cum[i - 1] + t * (cum[i] - cum[i - 1])
        if pos < 8 or pos > cum[-1] - 8:
            continue

        def at(s):
            s = max(0.0, min(cum[-1], s))
            j = 1
            while j < len(cum) - 1 and cum[j] < s:
                j += 1
            f = (s - cum[j - 1]) / ((cum[j] - cum[j - 1]) or 1.0)
            a, b = coords[j - 1], coords[j]
            return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)

        before, after = at(pos - 20), at(pos + 20)
        if abs((_bearing(pt, before) - stem + 180) % 360 - 180) > 40:
            merges.append(pos)                         # came from a branch: nothing to decide
            continue
        if angle_change(before, pt, after) < 8:
            continue
        kx, ky = 111320.0 * math.cos(math.radians(pt[0])), 110540.0
        v1 = ((pt[1] - before[1]) * kx, (pt[0] - before[0]) * ky)
        v2 = ((after[1] - pt[1]) * kx, (after[0] - pt[0]) * ky)
        side = "left" if v1[0] * v2[1] - v1[1] * v2[0] > 0 else "right"
        found.append((pos, {"maneuver": {"type": "fork", "modifier": side, "location": [pt[1], pt[0]]},
                            "name": name, "distance": 0}))
    return found, merges


def hav(a, b):
    lat1, lon1 = a
    lat2, lon2 = b
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    q = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(max(0.0, min(1.0, q))))


def _local_xy(p, lat0):
    kx = 111320.0 * math.cos(math.radians(lat0))
    ky = 110540.0
    return p[1] * kx, p[0] * ky


def _local_ll(x, y, lat0):
    kx = 111320.0 * math.cos(math.radians(lat0))
    ky = 110540.0
    return y / ky, x / kx


def project(pos, a, b):
    lat0 = (pos[0] + a[0] + b[0]) / 3.0
    ax, ay = _local_xy(a, lat0)
    bx, by = _local_xy(b, lat0)
    px, py = _local_xy(pos, lat0)
    dx, dy = bx - ax, by - ay
    den = dx * dx + dy * dy
    t = 0.0 if den == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / den))
    qx, qy = ax + t * dx, ay + t * dy
    point = _local_ll(qx, qy, lat0)
    return {"distance": math.hypot(px - qx, py - qy), "point": point, "t": t}


def segment_intersection(a, b, c, d):
    lat0 = (a[0] + b[0] + c[0] + d[0]) / 4.0
    ax, ay = _local_xy(a, lat0)
    bx, by = _local_xy(b, lat0)
    cx, cy = _local_xy(c, lat0)
    dx, dy = _local_xy(d, lat0)
    rx, ry = bx - ax, by - ay
    sx, sy = dx - cx, dy - cy
    den = rx * sy - ry * sx
    if abs(den) < 1e-9:
        return None
    qx, qy = cx - ax, cy - ay
    t = (qx * sy - qy * sx) / den
    u = (qx * ry - qy * rx) / den
    if -1e-7 <= t <= 1.0000001 and -1e-7 <= u <= 1.0000001:
        point = _local_ll(ax + t * rx, ay + t * ry, lat0)
        return point, t, u
    return None


def angle_change(a, b, c):
    lat0 = math.radians((a[0] + b[0] + c[0]) / 3.0)
    kx = 111320.0 * math.cos(lat0)
    ky = 110540.0
    v1 = ((b[1] - a[1]) * kx, (b[0] - a[0]) * ky)
    v2 = ((c[1] - b[1]) * kx, (c[0] - b[0]) * ky)
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    dot = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(dot))


def build_connected_graph(data):
    """Build one geometric graph from every KML LineString.

    Important: paths are not merely kept as independent polylines. Their
    intersections and sufficiently-close crossings become graph junctions.
    This is what allows a manually drawn walkway to join a saved Directions
    trace when the two lines meet on the map.
    """
    raw_segments = []
    for path_index, path in enumerate(data.get("paths", [])):
        name = str(path.get("name", "")).strip()
        kind = str(path.get("kind", "manual")).lower()
        reference = kind == "direction_reference" or name.lower().startswith("directions from")
        coords = path.get("coordinates", [])
        for segment_index, (a, b) in enumerate(zip(coords, coords[1:])):
            a = (float(a[0]), float(a[1]))
            b = (float(b[0]), float(b[1]))
            if hav(a, b) < 0.2:
                continue
            raw_segments.append({
                "a": a,
                "b": b,
                "path_index": path_index,
                "segment_index": segment_index,
                "kind": "direction_reference" if reference else "manual",
                "name": name or "Mapped VNIT pedestrian path",
                "marks": [(0.0, a), (1.0, b)],
            })

    # Exact geometric intersections become true junctions.
    for i, first in enumerate(raw_segments):
        for j in range(i + 1, len(raw_segments)):
            second = raw_segments[j]
            if first["path_index"] == second["path_index"]:
                continue
            hit = segment_intersection(first["a"], first["b"], second["a"], second["b"])
            if hit is not None:
                point, t1, t2 = hit
                first["marks"].append((t1, point))
                second["marks"].append((t2, point))

    # Only genuinely touching/near-touching path endpoints are connected.
    # Do NOT bridge arbitrary nearby paths: visual proximity does not imply a
    # walkable connection (there may be a wall, building, gate, or road between them).
    connectors = []
    for i, first in enumerate(raw_segments):
        for j in range(i + 1, len(raw_segments)):
            second = raw_segments[j]
            if first["path_index"] == second["path_index"]:
                continue
            # We only need endpoint-to-segment and endpoint-to-endpoint checks
            # for these short pedestrian gaps; exact crossings were handled above.
            candidates = []
            for point in (first["a"], first["b"]):
                q = project(point, second["a"], second["b"])
                candidates.append((q["distance"], point, q["point"], None, q["t"]))
            for point in (second["a"], second["b"]):
                q = project(point, first["a"], first["b"])
                candidates.append((q["distance"], q["point"], point, q["t"], None))
            distance, first_point, second_point, first_t, second_t = min(candidates, key=lambda x: x[0])
            mixed = first["kind"] != second["kind"]
            limit = NEAR_CONNECT_MIXED_M if mixed else NEAR_CONNECT_MANUAL_M
            if distance <= limit:
                if first_t is None:
                    first_projection = project(first_point, first["a"], first["b"])
                    first_t, first_point = first_projection["t"], first_projection["point"]
                if second_t is None:
                    second_projection = project(second_point, second["a"], second["b"])
                    second_t, second_point = second_projection["t"], second_projection["point"]
                first["marks"].append((first_t, first_point))
                second["marks"].append((second_t, second_point))
                connectors.append((first_point, second_point, distance, "path_junction"))

    nodes = []
    adjacency = []

    def node_id(point):
        best = -1
        best_distance = NODE_MERGE_M
        for index, existing in enumerate(nodes):
            distance = hav(point, existing)
            if distance < best_distance:
                best_distance = distance
                best = index
        if best >= 0:
            return best
        nodes.append((float(point[0]), float(point[1])))
        adjacency.append({})
        return len(nodes) - 1

    def add_edge(u, v, weight=None, kind="manual", name=""):
        if u == v:
            return
        distance = hav(nodes[u], nodes[v])
        if distance <= 0.01:
            return
        multiplier = REFERENCE_COST if kind == "direction_reference" else 1.0
        weighted = float(weight if weight is not None else distance * multiplier)
        old = adjacency[u].get(v)
        if old is None or weighted < old[0]:
            info = (weighted, distance, kind, name)
            adjacency[u][v] = info
            adjacency[v][u] = info

    # Split every original segment at every discovered junction.
    for segment in raw_segments:
        marks = sorted(segment["marks"], key=lambda x: x[0])
        unique = []
        for t, point in marks:
            if not unique or abs(t - unique[-1][0]) > 1e-7:
                unique.append((t, point))
        ids = [node_id(point) for _, point in unique]
        for u, v in zip(ids, ids[1:]):
            add_edge(u, v, kind=segment["kind"], name=segment["name"])

    # Add the actual short connector between close paths.
    for first_point, second_point, distance, kind in connectors:
        add_edge(node_id(first_point), node_id(second_point), distance, kind, "Connected KML pedestrian paths")

    return nodes, adjacency, raw_segments, node_id, add_edge


def _nearest_segment_candidates(nodes, segments, pos, max_distance):
    candidates = []
    for segment in segments:
        q = project(pos, segment["a"], segment["b"])
        if q["distance"] <= max_distance:
            candidates.append((q["distance"], q["point"], segment))
    candidates.sort(key=lambda x: x[0])
    return candidates[:5]


def _splice(segment, point, node_id, add_edge):
    """Put an attach point INTO its segment so a walker can enter or leave in the middle of it.
    (Before, a pin beside the middle of a long line only reached the line's far vertices, which
    produced detours that overshoot the destination and come back.)"""
    w = node_id(point)
    q = project(point, segment["a"], segment["b"])["t"]
    marks = sorted(segment["marks"], key=lambda m: m[0])
    for (t0, p0), (t1, p1) in zip(marks, marks[1:]):
        if t1 - t0 > 1e-9 and t0 - 1e-9 <= q <= t1 + 1e-9:
            u, v = node_id(p0), node_id(p1)
            if w not in (u, v):
                add_edge(u, w, kind=segment["kind"], name=segment["name"])
                add_edge(w, v, kind=segment["kind"], name=segment["name"])
            break
    return w


def _attach_candidates(nodes, adjacency, segments, pos, max_distance, node_id, add_edge, limit=8):
    """Attach a selected start/destination to several nearby mapped segments.

    This does NOT connect those segments to one another. The only new edges are
    from the selected point to the actual projected point on an existing KML
    segment. That makes route building more reliable without inventing junctions
    between two unrelated walkways.
    """
    candidates = _nearest_segment_candidates(nodes, segments, pos, max_distance)
    if not candidates:
        return -1, []
    endpoint = node_id(pos)
    attached = []
    seen = set()
    nearest_d = candidates[0][0]
    for distance, point, segment in candidates:
        if distance > nearest_d + ACCESS_SLACK_M:
            continue   # a farther path may be on the other side of a building / wall
        path_node = _splice(segment, point, node_id, add_edge)
        if path_node in seen:
            continue
        seen.add(path_node)
        add_edge(endpoint, path_node, distance, segment["kind"], segment["name"])
        attached.append((path_node, distance, point, segment))
        if len(attached) >= limit:
            break
    return endpoint, attached


def _destination_candidates(nodes, adjacency, segments, pos, max_distance, node_id, add_edge):
    candidates = _nearest_segment_candidates(nodes, segments, pos, max_distance)
    result = []
    seen = set()
    for distance, point, segment in candidates[:8]:
        if distance > candidates[0][0] + ACCESS_SLACK_M:
            continue
        arrival = _splice(segment, point, node_id, add_edge)
        if arrival in seen:
            continue
        seen.add(arrival)
        result.append((arrival, distance, point, segment))
    return result, (candidates[0][0] if candidates else None)

def astar(nodes, adjacency, start, goal):
    g_score = {start: 0.0}
    previous = {}
    queue = [(hav(nodes[start], nodes[goal]), 0.0, start)]
    closed = set()
    while queue:
        _, current_cost, u = heapq.heappop(queue)
        if u in closed:
            continue
        closed.add(u)
        if u == goal:
            break
        for v, info in adjacency[u].items():
            weight = info[0]
            new_cost = current_cost + weight
            if new_cost < g_score.get(v, float("inf")):
                g_score[v] = new_cost
                previous[v] = u
                heuristic = hav(nodes[v], nodes[goal])
                heapq.heappush(queue, (new_cost + heuristic, new_cost, v))
    if goal not in g_score:
        return None
    path = [goal]
    while path[-1] != start:
        parent = previous.get(path[-1])
        if parent is None:
            return None
        path.append(parent)
    path.reverse()
    return path, g_score[goal]


def _make_route(nodes, result, arrival_distance, mode, y_points=()):
    ids, weighted_cost = result
    coords = [(nodes[i][0], nodes[i][1]) for i in ids]
    simplified = [coords[0]]
    for point in coords[1:-1]:
        if hav(simplified[-1], point) >= 1.5:
            simplified.append(point)
    simplified.append(coords[-1])
    coords = simplified
    actual_distance = sum(hav(coords[i - 1], coords[i]) for i in range(1, len(coords)))

    cum = [0.0]
    for i in range(1, len(coords)):
        cum.append(cum[-1] + hav(coords[i - 1], coords[i]))

    steps = []
    for i in range(1, len(coords) - 1):
        a, b, c = coords[i - 1], coords[i], coords[i + 1]
        angle = angle_change(a, b, c)
        if angle < 28:
            continue
        lat0 = math.radians(b[0])
        kx, ky = 111320.0 * math.cos(lat0), 110540.0
        v1 = ((b[1] - a[1]) * kx, (b[0] - a[0]) * ky)
        v2 = ((c[1] - b[1]) * kx, (c[0] - b[0]) * ky)
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        modifier = "uturn" if angle > 150 else ("left" if cross > 0 else "right")
        steps.append({
            "maneuver": {"type": "turn", "modifier": modifier, "location": [b[1], b[0]]},
            "name": "Mapped VNIT pedestrian path",
            "distance": 0,
        })

    y_found, y_merge = _y_steps(coords, cum, y_points)
    if y_found or y_merge:   # a Y instruction (or a plain merge) replaces any ordinary turn within 15 m
        pos_of = {(c[1], c[0]): cum[k] for k, c in enumerate(coords)}
        near = [yp for yp, _ in y_found] + y_merge
        keyed = [(pos_of.get(tuple(s["maneuver"]["location"]), 0.0), s) for s in steps]
        keyed = [(p, s) for p, s in keyed if all(abs(p - yp) > 15 for yp in near)] + y_found
        keyed.sort(key=lambda x: x[0])
        steps = [s for _, s in keyed]

    filtered = []
    for step in steps:
        point = (step["maneuver"]["location"][1], step["maneuver"]["location"][0])
        if not filtered:
            filtered.append(step)
        else:
            previous = filtered[-1]["maneuver"]["location"]
            previous_point = (previous[1], previous[0])
            if hav(point, previous_point) > 12:
                filtered.append(step)

    arrival = coords[-1]
    return {
        "geometry": {
            "type": "LineString",
            "coordinates": [[p[1], p[0]] for p in coords],
        },
        "distance": actual_distance,
        "weighted_cost": weighted_cost,
        "legs": [{"steps": filtered}],
        "arrival_point": {
            "lat": arrival[0],
            "lng": arrival[1],
            "distance_from_destination_m": round(arrival_distance or 0, 1),
            "type": "nearest_mapped_pedestrian_access",
        },
        "routing_mode": mode,
        "engine_version": "y-stem-2",
    }


def route(data, start, dest):
    nodes, adjacency, segments, node_id, add_edge = build_connected_graph(data)
    base_nodes = len(nodes)   # every node with an id >= this is a synthetic pin, not part of a path

    # A selected POI may sit inside a building or beside more than one mapped
    # path. Attach it to several nearby segments, but never connect those
    # segments to each other. A* then chooses the best genuinely connected
    # route through the KML graph.
    start_id, start_candidates = _attach_candidates(
        nodes, adjacency, segments, start, START_CONNECT_M, node_id, add_edge, limit=8
    )
    if start_id < 0:
        return None, f"The starting point is not within {START_CONNECT_M:.0f} m of any supplied pedestrian path. Add or extend a walkway in My Maps."

    destination_options, nearest_destination_distance = _destination_candidates(
        nodes, adjacency, segments, dest, DEST_CONNECT_M, node_id, add_edge
    )
    if not destination_options:
        return None, f"The destination is not within {DEST_CONNECT_M:.0f} m of any supplied pedestrian path. Add or extend a walkway in My Maps."

    best = None
    best_cost = None
    for arrival_id, access_distance, point, segment in destination_options:
        result = astar(nodes, adjacency, start_id, arrival_id)
        if result is None:
            continue
        weighted = result[1] + access_distance    # include the final hop from the path to the destination
        # Prefer a genuinely connected path. The weighted cost already includes
        # the small preference against direction-reference traces.
        if best is None or weighted < best_cost:   # (was compared with access_distance, so the nearest segment always won)
            best_cost = weighted
            best = (result, access_distance, point, segment)

    if best is None:
        return None, "The selected places are not connected by the supplied KML pedestrian network. No artificial junction was added. Extend the relevant walkway in My Maps until the paths actually meet."

    straight = hav(start, dest)
    ids = best[0][0]
    # The drawn route must contain pedestrian-path geometry ONLY. The start pin may be inside a
    # building, so the straight hop from the pin to the path is not drawn / not walked.
    if len(ids) > 2 and ids[0] >= base_nodes:
        ids = ids[1:]
        best = ((ids, best[0][1]), best[1], best[2], best[3])
    route_distance = sum(hav(nodes[ids[i - 1]], nodes[ids[i]]) for i in range(1, len(ids)))
    if straight > 1 and route_distance > max(straight * MAX_DETOUR_FACTOR, straight + MAX_EXTRA_DETOUR_M):
        return None, "The supplied KML produces an excessively indirect route. I did not invent a road; add the missing pedestrian connection to the KML."

    route_result = _make_route(nodes, best[0], best[1], "connected_kml_astar", _y_hints(data))
    route_result["graph_info"] = {
        "path_count": len(data.get("paths", [])),
        "graph_nodes": len(nodes),
        "graph_edges": sum(len(x) for x in adjacency) // 2,
        "paths_are_connected_at_intersections": True,
        "junction_rule": "exact intersections and <=3 m endpoint gaps only; no arbitrary nearby-path bridges",
        "start_access_candidates": len(start_candidates),
        "destination_access_candidates": len(destination_options),
    }
    return route_result, None

