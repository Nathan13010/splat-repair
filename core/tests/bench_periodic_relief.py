"""Reproduce the periodic relief audit without external scans or a GPU.

Run from the repository root: python core/tests/bench_periodic_relief.py
Use --holdout for the reserved scenes, --validation for the additional
warps, --images for rendered errors or --repeats 3 for median repair times.
Each stdout line is one JSON record.
"""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from statistics import median
import sys
from time import perf_counter
from typing import Callable

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from splatcore import preview                                 # noqa: E402
from splatcore import repair as rp                            # noqa: E402
from periodic_phase_trial import apply_phase_trial            # noqa: E402
from periodic_relief_experiment import apply_height_trial     # noqa: E402
from test_fill_volume import _disc_volume                     # noqa: E402
from test_repair import disc, make_surface                    # noqa: E402


@dataclass
class Scene:
    name: str
    height: Callable
    seed: int = 0
    radius: float = 1.4
    period: float = 1.1
    centre_x: float = 0.0
    volume: bool = False
    colour: str = "uniform"
    thickness: float | None = None


def relief(period, bend):
    freq = 2 * np.pi / period
    return lambda x, y: 0.35 * np.sin(freq * (x + bend * y ** 2)) * np.cos(freq * y)


def scenes(holdout=False, validation=False, phase_holdout=False):
    if phase_holdout:
        for seed, f1, f2, b1, b2, degrees in (
                (20, 5.3, 6.3, 0.16, -0.23, 24), (21, 7.5, 5.6, 0.22, -0.19, 55),
                (22, 5.6, 5.4, -0.21, 0.22, -64), (23, 5.2, 4.7, 0.10, -0.21, -30),
                (24, 4.4, 6.4, 0.20, -0.15, 13)):
            c, s = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))

            def warped(x, y, c=c, s=s, f1=f1, f2=f2, b1=b1, b2=b2):
                u, v = c * x - s * y, s * x + c * y
                return 0.31 * np.sin(f1 * (u + b1 * v ** 2)) * np.cos(f2 * (v + b2 * u ** 2))

            yield Scene("two-axis-holdout", warped, seed, radius=1.37)

        def directional(x, y):
            return 0.31 * np.sin(6.1 * (x + 0.20 * y ** 2)) * np.cos(5 * (y - 0.18 * x ** 2))

        yield Scene("two-axis-directional", directional, seed=25, radius=1.37,
                    colour="directional")
        return
    if validation:
        for seed, radius, period, bend, degrees in (
                (12, 1.25, 0.81, 0.18, 12), (13, 1.55, 1.18, 0.29, -32),
                (14, 1.45, 0.96, -0.24, 55), (15, 1.6, 1.05, 0.24, -12),
                (16, 1.35, 0.72, 0.30, 28)):
            wave = relief(period, bend)
            angle = np.deg2rad(degrees)
            c, s = np.cos(angle), np.sin(angle)

            def rotated(x, y, wave=wave, c=c, s=s):
                return wave(c * x - s * y, s * x + c * y)

            yield Scene("rotated-curved", rotated, seed, radius, period)
        # The directions vary in both coordinates. There need not be a single
        # translation that preserves the phase across the whole rim.
        yield Scene("two-axis-warp", lambda x, y: (
            0.35 * np.sin(5.6 * (x + 0.22 * y ** 2))
            * np.cos(5.6 * (y - 0.20 * x ** 2))), seed=17)
        return
    if not holdout:
        for seed in range(3):
            for name, period, bend in (("straight", 1.1, 0.0),
                                       ("curved", 1.1, 0.24),
                                       ("short-curved", 0.72, 0.24)):
                yield Scene(name, relief(period, bend), seed, period=period)
    else:
        for seed, radius, period in ((3, 1.25, 1.1), (4, 1.4, 0.72), (5, 1.55, 1.0),
                                      (6, 1.4, 0.88), (7, 1.6, 1.1)):
            yield Scene("curved-holdout", relief(period, 0.24), seed, radius, period)
    flat = lambda x, y: np.zeros_like(x)
    yield Scene("physical-plane", flat, volume=True, colour="patterned")
    yield Scene("physical-curved", relief(1.1, 0.24), volume=True, colour="patterned")
    if holdout:
        yield Scene("dome", lambda x, y: 0.05 * (x ** 2 + y ** 2), seed=8)
        yield Scene("double-skin", flat, seed=9, thickness=0.6)
        yield Scene("free-edge", flat, seed=10, radius=1.2, centre_x=5.4,
                    volume=True, colour="patterned")
        yield Scene("material-junction", flat, seed=11, colour="junction")


def _dist(xyz, scene):
    return np.hypot(xyz[:, 0] - scene.centre_x, xyz[:, 1])


def _fixture(scene):
    truth = make_surface(scene.height, seed=scene.seed, period=scene.period,
                         thickness=scene.thickness)
    if scene.colour == "uniform":
        truth.f_dc[:] = 0
    if scene.colour == "directional":
        truth.f_dc[:] = 0
        x, y = truth.xyz[:, :2].T.astype(float)
        e = 0.001
        dx = (scene.height(x + e, y) - scene.height(x - e, y)) / (2 * e)
        dy = (scene.height(x, y + e) - scene.height(x, y - e)) / (2 * e)
        # The first SH band evaluates [-y, z, -x]. Aim its lobe along the
        # true surface normal, so a wrong turn shows up in oblique images.
        n = np.column_stack((dy, np.ones_like(dx), dx))
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        truth.f_rest[:] = 0
        truth.f_rest[:, :3, :] = (0.5 * n)[..., None]
    if scene.colour == "junction":
        truth.f_dc[truth.xyz[:, 0] > 0] = (0.2, -0.5, 0.6)
    missing = disc(truth, cx=scene.centre_x, r=scene.radius)
    if not scene.volume:
        return truth, missing, truth, np.flatnonzero(missing), None
    cloud = truth.take(~missing)
    depth = 0.8
    volume = _disc_volume(scene.centre_x, 0, scene.radius + 0.4, depth=depth)
    selected = np.flatnonzero(rp.as_volume(volume).inside(cloud.xyz))
    return truth, missing, cloud, selected, volume


def _views(scene, truth, missing, out_dir):
    target = np.array([scene.centre_x, 0.0, 0.0])
    hole = truth.take(missing)
    bare = truth.take(~missing)
    for name, dx, dz in (("front", 0, 6), ("oblique-left", -3, 5.196152),
                         ("oblique-right", 3, 5.196152)):
        eye = target + (dx, 0, dz)

        def render(cloud, eye=eye):
            return preview.render(cloud, eye, target, W=128, H=128, fov=40)

        gold, _ = render(truth)
        _, alpha = render(hole)
        mask = alpha > 0.5
        if not mask.any():
            raise ValueError(f"{scene.name}, {name}: no visible hole pixels")
        baseline, _ = render(bare)
        if out_dir:
            preview.write_png(out_dir / f"{name}-reference.png", gold)
            preview.write_png(out_dir / f"{name}-hole.png", baseline)
        bare_mae = float(np.abs(baseline[mask] - gold[mask].astype(float)).mean() / 255)
        yield name, render, gold[mask].astype(float), mask, bare_mae


def _geometry(scene, truth, missing, result):
    added = result.added.xyz
    face_mask = np.ones(len(added), bool)
    if result.backing is not None:
        face_mask &= ~result.backing
    # Gap closure reuses the original positions. Report these separately so
    # they cannot artificially lower the error of newly created geometry.
    closed = result.info.get("gaps_closed", 0)
    if closed:
        face_mask[-closed:] = False
    face = added[face_mask]
    radius = _dist(face, scene)
    inside = face[radius < scene.radius]
    want = truth.xyz[missing, :2]
    if not len(inside):
        return {"face_splats": 0, "height_p95": None, "coverage": 0.0}
    zerror = np.abs(inside[:, 2] - scene.height(inside[:, 0], inside[:, 1]))
    local = truth.xyz[(_dist(truth.xyz, scene) < scene.radius + 0.4)
                      & (truth.xyz[:, 2] > -0.5)]
    spacing = float(np.median(cKDTree(local).query(local, k=2)[0][:, 1]))
    p95 = float(np.percentile(zerror, 95))
    coverage = float((cKDTree(inside[:, :2]).query(want)[0] < 0.12).mean())
    seam = zerror[radius[radius < scene.radius] >= scene.radius - 0.12]
    slope = _slope_error(inside, scene.height)
    return {"face_splats": len(inside), "height_p95": p95, "spacing": spacing,
            "normal_error_p95_deg": slope,
            "height_p95_spacings": p95 / spacing, "coverage": coverage,
            "seam_height_mae": float(seam.mean()) if len(seam) else None}


def _slope_error(points, height):
    """Compare local least-squares normals against the known height gradient."""
    if len(points) < 10:
        return None
    tree = cKDTree(points[:, :2])
    dist, near = tree.query(points[:, :2], k=10)
    good = dist[:, -1] < 0.16
    if not good.any():
        return None
    patch = points[near[good]].astype(float)
    patch -= patch.mean(axis=1, keepdims=True)
    tangent = patch[..., :2]
    singular = np.linalg.svd(tangent, compute_uv=False)
    use = singular[:, 1] > 0.016
    if not use.any():
        return None
    gradients = (np.linalg.pinv(tangent[use]) @ patch[use, :, 2, None])[..., 0]
    x, y = points[good][use, :2].astype(float).T
    step = 0.001
    true = np.column_stack(((height(x + step, y) - height(x - step, y)) / (2 * step),
                            (height(x, y + step) - height(x, y - step)) / (2 * step)))
    cosine = (1 + np.sum(gradients * true, axis=1)) / (
        np.sqrt(1 + np.sum(gradients ** 2, axis=1))
        * np.sqrt(1 + np.sum(true ** 2, axis=1)))
    return float(np.percentile(np.rad2deg(np.arccos(np.clip(cosine, -1, 1))), 95))


def compare(scene, images, repeats, out_dir, experiment, phase_experiment):
    truth, missing, cloud, selected, volume = _fixture(scene)
    if out_dir:
        out_dir = out_dir / f"{scene.name}-{scene.seed}"
        out_dir.mkdir(parents=True, exist_ok=True)
    views = list(_views(scene, truth, missing, out_dir)) if images else []
    variants = [("lattice", True, None), ("no-lattice", False, None)]
    if experiment:
        variants.append(("height-trial", True, apply_height_trial))
    if phase_experiment:
        variants.append(("phase-trial", True, apply_phase_trial))
    for variant, lattice, trial in variants:
        params = rp.RepairParams(mode="patch", seed=1, view=[0, 0, -1],
                                 lattice=lattice, volume=volume)
        times = []
        for _ in range(repeats):
            begin = perf_counter()
            result = rp.repair(cloud, selected, params)
            changed = trial(cloud, result) if trial else 0
            times.append(perf_counter() - begin)
        stats = dict(scene=scene.name, seed=scene.seed, radius=scene.radius,
                     period=scene.period, variant=variant, repair_seconds=median(times),
                     trial_adjusted_splats=changed, selected_splats=len(selected),
                     removed_splats=len(result.removed), gaps_closed=result.info.get("gaps_closed", 0),
                     unfolded=bool(result.info.get("unfold")))
        stats.update(_geometry(scene, truth, missing, result))
        if views:
            filled = result.apply_to(cloud)
            view_scores = {}
            for name, render, gold, mask, bare_mae in views:
                img, _ = render(filled)
                if out_dir:
                    preview.write_png(out_dir / f"{name}-{variant}.png", img)
                view_scores[name] = {"hole": bare_mae,
                                     "fill": float(np.abs(img[mask] - gold).mean() / 255)}
            stats["views"] = view_scores
            stats["worst_oblique_mae"] = max(v["fill"] for k, v in view_scores.items()
                                             if k.startswith("oblique"))
        print(json.dumps(stats, allow_nan=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    split = ap.add_mutually_exclusive_group()
    split.add_argument("--holdout", action="store_true", help="Reserved seeds and controls")
    split.add_argument("--validation", action="store_true",
                       help="Additional unseen rotations and a two-axis warp")
    split.add_argument("--phase-holdout", action="store_true",
                       help="Five additional warps whose two directions bend")
    ap.add_argument("--phase-experiment", action="store_true",
                    help="Also evaluate the offline phase-matched donor trial")
    ap.add_argument("--images", action="store_true", help="Measure front and oblique images")
    ap.add_argument("--save-images", type=Path, help="Also save images under this directory")
    ap.add_argument("--experiment", action="store_true",
                    help="Also evaluate the offline height trial")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--scene", action="append", help="Restrict the run to these scene names")
    args = ap.parse_args()
    if args.repeats < 1:
        ap.error("--repeats must be positive")
    seen = set()
    for scene in scenes(args.holdout, args.validation, args.phase_holdout):
        if not args.scene or scene.name in args.scene:
            compare(scene, args.images or args.save_images, args.repeats,
                    args.save_images, args.experiment, args.phase_experiment)
            seen.add(scene.name)
    if args.scene and set(args.scene) - seen:
        ap.error(f"unknown scenes: {sorted(set(args.scene) - seen)}")


if __name__ == "__main__":
    main()
