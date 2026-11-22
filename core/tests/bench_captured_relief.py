"""Evaluate held-out geometry from two CC-BY-4.0 photographic splat captures."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splatcore import preview, repair as rp                 # noqa: E402
from splatcore.formats import load_splat                    # noqa: E402
from periodic_relief_experiment import apply_height_trial  # noqa: E402
from periodic_phase_trial import apply_phase_trial          # noqa: E402


def compare(cloud, pixel, pick, out):
    fov = 42
    view = np.array((0.0, 1.8, -0.1))
    view /= np.linalg.norm(view)
    x, y = pixel
    idx = pick[y, x]
    if idx < 0:
        raise ValueError(f"no splat at reference pixel {pixel}")
    centre = cloud.xyz[idx].astype(float)
    r = 0.11
    crop = cloud.take(np.linalg.norm(cloud.xyz - centre, axis=1) < 0.52)
    core_volume = {"view": view.tolist(),
                   "capsules": [[*centre, *centre, r, 0.11]]}
    brush_volume = {"view": view.tolist(),
                    "capsules": [[*centre, *centre, 1.30 * r, 0.15]]}
    removed = rp.as_volume(core_volume).inside(crop.xyz)
    if removed.sum() < 100:
        raise ValueError(f"too few removed splats at {pixel}")
    hole = crop.take(~removed)
    reference = crop.take(removed)
    select = np.flatnonzero(rp.as_volume(brush_volume).inside(hole.xyz))
    out.mkdir(parents=True, exist_ok=True)

    # Use the same frontal view for selection and matched oblique cameras.
    cameras = {
        "front": centre - 0.80 * view,
        "left": centre - 0.72 * view + np.array([-0.35, 0.0, 0.0]),
        "right": centre - 0.72 * view + np.array([0.35, 0.0, 0.0]),
    }
    gold = {}
    for name, eye in cameras.items():
        kwargs = dict(W=144, H=144, fov=fov, up=(0, 0, -1))
        original, _ = preview.render(crop, eye, centre, **kwargs)
        deleted, _ = preview.render(hole, eye, centre, **kwargs)
        _, alpha = preview.render(reference, eye, centre, **kwargs)
        mask = alpha > 0.5
        if not mask.any():
            raise ValueError(f"no visible removed area for {name}")
        gold[name] = (original.astype(float), mask)
        preview.write_png(out / f"{name}-original.png", original)
        preview.write_png(out / f"{name}-hole.png", deleted)

    variants = (("lattice", True, None), ("no-lattice", False, None),
                ("height-trial", True, apply_height_trial),
                ("phase-trial", True, apply_phase_trial))
    for name, lattice, method in variants:
        start = perf_counter()
        result = rp.repair(hole, select, rp.RepairParams(mode="patch", seed=1,
                           lattice=lattice, volume=brush_volume, view=view.tolist()))
        changed = method(hole, result) if method else 0
        seconds = perf_counter() - start
        result_set = result.apply_to(hole)
        added = result.added.xyz if result.backing is None else result.added.xyz[~result.backing]
        distance = cKDTree(added).query(reference.xyz, workers=1)[0]
        distance_p95 = float(np.percentile(distance, 95))
        render_mae = {}
        for view_name, eye in cameras.items():
            rendered, _ = preview.render(result_set, eye, centre, W=144, H=144,
                                         fov=fov, up=(0, 0, -1))
            actual, mask = gold[view_name]
            render_mae[view_name] = float(np.abs(actual[mask] - rendered[mask]).mean() / 255)
            preview.write_png(out / f"{view_name}-{name}.png", rendered)
        stats = dict(scan=out.parent.name, pixel=[x, y], centre=centre.tolist(),
                     removed=int(removed.sum()), marked=len(select), added=result.added.n,
                     changed=changed, variant=name, seconds=seconds,
                     p95_nearest_distance=distance_p95,
                     image_mae=render_mae)
        print(json.dumps(stats, allow_nan=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", type=Path, help="Folder with textile_ball.splat and shell.splat")
    ap.add_argument("output", type=Path, help="Destination for the image comparison")
    ap.add_argument("--asset", choices=("textile_ball", "shell"))
    args = ap.parse_args()
    cases = (("textile_ball", ((130, 85), (95, 110))),
             ("shell", ((95, 60), (61, 114))))
    for name, pixels in cases:
        if args.asset and args.asset != name:
            continue
        cloud = load_splat(args.folder / f"{name}.splat")
        pick, _, _ = preview.pick_buffer(cloud, (0, -1.8, 0.1), (0, 0, 0),
                                         up=(0, -1, 0.03), W=240, H=240, fov=42)
        for x, y in pixels:
            compare(cloud, (x, y), pick, args.output / name / f"{x}-{y}")


if __name__ == "__main__":
    main()
