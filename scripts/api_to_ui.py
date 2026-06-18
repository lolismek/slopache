#!/usr/bin/env python3
"""Convert an API-format ComfyUI workflow into UI (graph) format so the browser can
DRAW it as a node diagram. API files store only logic (node -> node wiring); the UI
needs each node's slot layout + on-screen positions. We read the slot definitions
from a running ComfyUI's /object_info and lay nodes out in columns by dependency
depth.

Usage (on the box, ComfyUI running):
    python scripts/api_to_ui.py workflows/flux2_lora_txt2img_api.json \
        --out workflows/ui/flux2_lora_txt2img_ui.json
"""
import argparse
import json
import os
import urllib.request

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


def is_widget(t):
    """An input is a widget (value box) if its type is an enum list or a primitive;
    otherwise it's a connection slot (MODEL, CLIP, LATENT, IMAGE, AUDIO, ...)."""
    return isinstance(t, list) or t in WIDGET_TYPES


def get_info(server):
    return json.load(urllib.request.urlopen(f"http://{server}/object_info", timeout=30))


def ordered_inputs(spec):
    inp = spec.get("input", {})
    items = []
    for grp in ("required", "optional"):
        for name, d in inp.get(grp, {}).items():
            items.append((name, d))  # d == [type, opts?]
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("api")
    ap.add_argument("--out", required=True)
    ap.add_argument("--server", default="127.0.0.1:8188")
    args = ap.parse_args()

    info = get_info(args.server)
    with open(args.api) as f:
        wf = json.load(f)

    def deps(nid):
        ins = wf[nid].get("inputs", {})
        return [str(v[0]) for v in ins.values()
                if isinstance(v, list) and len(v) == 2 and isinstance(v[1], int)]

    depth = {}

    def d(nid):
        if nid in depth:
            return depth[nid]
        depth[nid] = 0
        depth[nid] = 1 + max((d(x) for x in deps(nid)), default=-1)
        return depth[nid]

    for nid in wf:
        d(nid)

    order_nids = sorted(wf, key=lambda n: (depth[n], int(n)))
    col_rows, pos = {}, {}
    for nid in order_nids:
        c = depth[nid]
        r = col_rows.get(c, 0)
        col_rows[c] = r + 1
        pos[nid] = [c * 380, r * 260]

    nodes, links, link_id = [], [], 0
    for order, nid in enumerate(order_nids):
        node = wf[nid]
        ct = node["class_type"]
        spec = info.get(ct, {})
        api_in = node.get("inputs", {})
        in_slots, widgets = [], []
        for name, dd in ordered_inputs(spec):
            t = dd[0]
            opts = dd[1] if len(dd) > 1 and isinstance(dd[1], dict) else {}
            val = api_in.get(name)
            if is_widget(t):
                widgets.append(val if val is not None else opts.get("default"))
                if name in ("seed", "noise_seed"):
                    widgets.append("fixed")  # ComfyUI's control_after_generate widget
            else:
                link = None
                if isinstance(val, list) and len(val) == 2:
                    link_id += 1
                    src, slot = str(val[0]), val[1]
                    src_out = info.get(wf[src]["class_type"], {}).get("output", [])
                    typ = src_out[slot] if slot < len(src_out) else t
                    links.append([link_id, int(src), slot, int(nid), len(in_slots), typ])
                    link = link_id
                in_slots.append({"name": name, "type": t, "link": link})

        otypes = spec.get("output", [])
        onames = spec.get("output_name", otypes)
        outs = [{"name": (onames[i] if i < len(onames) else ot), "type": ot,
                 "links": [], "slot_index": i} for i, ot in enumerate(otypes)]

        nodes.append({"id": int(nid), "type": ct, "pos": pos[nid], "size": [320, 200],
                      "flags": {}, "order": order, "mode": 0, "inputs": in_slots,
                      "outputs": outs, "properties": {"Node name for S&R": ct},
                      "widgets_values": widgets,
                      "title": node.get("_meta", {}).get("title") or ct})

    by_id = {n["id"]: n for n in nodes}
    for lid, src, slot, dst, dslot, typ in links:
        outs = by_id[src]["outputs"]
        if slot < len(outs):
            outs[slot]["links"].append(lid)

    ui = {"last_node_id": max(int(n) for n in wf), "last_link_id": link_id,
          "nodes": nodes, "links": links, "groups": [], "config": {}, "extra": {},
          "version": 0.4}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(ui, f, indent=2)
    print(f"[api_to_ui] wrote {args.out} ({len(nodes)} nodes, {link_id} links)")


if __name__ == "__main__":
    main()
