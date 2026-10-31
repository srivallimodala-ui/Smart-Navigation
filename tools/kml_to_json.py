#!/usr/bin/env python3
"""Convert a My Maps KML export into data/vnit_pedestrian_map.json and refresh the
LOCAL_VNIT_POINTS / LOCAL_VNIT_PATHS constants (browser fallback) in website/index.html.

Usage:  python tools/kml_to_json.py data/Untitled_map.kml
Pins in 'Directions from ...' layers are ignored (they only repeat place pins); lines named
'Directions from ...' become kind 'direction_reference', everything else kind 'manual'.
A Y pin keeps its stem direction from the previous JSON (or from '(stem NE)' in its name)."""
import json, math, re, sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JSON_OUT = ROOT / "data" / "vnit_pedestrian_map.json"
HTML = ROOT / "website" / "index.html"
NS = {"k": "http://www.opengis.net/kml/2.2"}


def hav(a, b):
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp, dl = p2 - p1, math.radians(b[1] - a[1])
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371000 * 2 * math.asin(math.sqrt(h))


def convert(kml_path):
    old_stems = {}
    if JSON_OUT.exists():
        for p in json.load(open(JSON_OUT, encoding="utf-8")).get("points", []):
            if p.get("stem_bearing") is not None:
                old_stems[p["name"]] = p["stem_bearing"]
    points, paths = [], []

    def walk(el, layer):
        for c in el:
            t = c.tag.split("}")[1]
            if t == "Folder":
                walk(c, (c.findtext("k:name", "", NS) or "").strip())
            elif t == "Document":
                walk(c, layer)
            elif t == "Placemark":
                name = (c.findtext("k:name", "", NS) or "").strip()
                pt = c.find(".//k:Point/k:coordinates", NS)
                ls = c.find(".//k:LineString/k:coordinates", NS)
                if pt is not None and not layer.startswith("Directions from"):
                    lng, lat = map(float, pt.text.strip().split(",")[:2])
                    if not any(q["name"].lower() == name.lower() and hav((lat, lng), (q["lat"], q["lng"])) < 15 for q in points):
                        q = {"name": name, "lat": round(lat, 7), "lng": round(lng, 7)}
                        if name in old_stems:
                            q["stem_bearing"] = old_stems[name]
                        points.append(q)
                elif ls is not None:
                    coords = [[round(float(v), 7) for v in s.split(",")[:2][::-1]] for s in ls.text.split()]
                    paths.append({"name": name, "coordinates": coords,
                                  "kind": "direction_reference" if name.startswith("Directions from") else "manual"})

    walk(ET.parse(kml_path).getroot(), "")
    return {"points": points, "paths": paths}


if __name__ == "__main__":
    data = convert(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "Untitled_map.kml")
    json.dump(data, open(JSON_OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    s = HTML.read_text(encoding="utf-8")
    pts = json.dumps(data["points"], separators=(",", ":"), ensure_ascii=False)
    pth = json.dumps(data["paths"], separators=(",", ":"), ensure_ascii=False)
    s, n1 = re.subn(r"^const LOCAL_VNIT_POINTS = .*$", lambda m: "const LOCAL_VNIT_POINTS = " + pts + ";", s, count=1, flags=re.M)
    s, n2 = re.subn(r"^const LOCAL_VNIT_PATHS = .*$", lambda m: "const LOCAL_VNIT_PATHS = " + pth + ";", s, count=1, flags=re.M)
    assert n1 == 1 and n2 == 1, "could not find the LOCAL_VNIT_* constants in index.html"
    HTML.write_text(s, encoding="utf-8")
    print(f"{len(data['points'])} places, {len(data['paths'])} paths written; browser fallback refreshed")
