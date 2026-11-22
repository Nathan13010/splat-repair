"""Offline experiment: continue height using repeated, trusted rim samples."""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from splatcore import repair as rp


def predict_height(uvh, grid, cell, target, query):
    """Predict missing heights from a continuous shift of the observed surface.

    Each proposed displacement must agree on heights in multiple independent
    sectors. Unknown and empty cells carry no evidence. Return NaN for queries
    with fewer than two matching sources or where their predictions disagree.
    """
    lo_u, lo_v, nu, nv = grid
    xy, z = uvh[:, :2], uvh[:, 2]
    gu, gv, valid = rp._cells(xy[:, 0], xy[:, 1], cell, lo_u, lo_v, nu, nv)
    known = valid.copy()
    known[valid] &= ~target[gv[valid], gu[valid]]
    xy, z, gu, gv = xy[known], z[known], gu[known], gv[known]
    if len(z) < 400 or nv * nu > 25000:
        return None
    span = float(np.std(z))
    if span < cell * 0.45:
        return None

    sums, counts = rp._accum(z, gv, gu, nv, nu)
    known_cells = counts > 0
    corr = (sums / np.maximum(counts, 1))[:, :, None]
    candidate_pixels = rp._dominant_offsets(corr, known_cells, k=18, smooth=0.7)
    if not candidate_pixels:
        return None
    near = ndimage.binary_dilation(known_cells) & ~ndimage.binary_dilation(target)
    fine = _height_grid(xy, z, near, cell / 2, lo_u, lo_v)
    centre = np.asarray(np.where(target)[::-1]).mean(1) * cell + (lo_u, lo_v)
    angles = np.arctan2(*(xy - centre).T[::-1])
    sector = np.floor((angles + np.pi) / (np.pi / 4)).astype(int) % 8
    perm = np.random.default_rng(0).permutation(len(xy))
    train = perm[::2][:384]
    validation = perm[1::2][:1280]

    def sample(here):
        coords = ((here - (lo_u, lo_v)) / (cell / 2) - 0.5).T[::-1]
        return ndimage.map_coordinates(fine, coords, order=1,
                                       mode="constant", cval=np.nan)

    def match(delta, idx):
        err = sample(xy[idx] + delta) - z[idx]
        finite = np.isfinite(err)
        return err, finite

    def cost(delta):
        err, good = match(delta, train)
        if good.sum() < 28:
            return 2 * span
        residual = err[good] - np.median(err[good])
        return float(np.sort(np.abs(residual))[:int(good.sum() * 0.9)].mean())

    estimates = []
    for dv, du in candidate_pixels:
        # A periodic displacement can be used in either direction.
        if dv < 0 or (dv == 0 and du < 0):
            continue
        delta0 = np.array([du, dv]) * cell
        if cost(delta0) > 0.55 * span:
            continue
        delta = delta0.copy()
        for step in (0.35 * cell, 0.12 * cell):
            deltas = delta + step * np.array([[ix, iy] for ix in (-1, 0, 1)
                                                        for iy in (-1, 0, 1)])
            delta = min(deltas, key=cost)
        for sign in (-1, 1):
            err, good = match(sign * delta, validation)
            if good.sum() < 80 or len(np.unique(sector[validation][good])) < 4:
                continue
            bias = np.median(err[good])
            mae = np.median(np.abs(err[good] - bias))
            sector_ok = []
            for i in np.unique(sector[validation][good]):
                e = err[good & (sector[validation] == i)]
                if len(e) >= 16:
                    sector_ok.append(np.median(np.abs(e - bias)) < 0.48 * span)
            if mae > 0.3 * span or len(sector_ok) < 3 or not all(sector_ok):
                continue
            estimate = sample(query + sign * delta) - bias
            estimates.append(estimate)
        if len(estimates) >= 6:
            break

    if len(estimates) < 2:
        return None
    all_h = np.asarray(estimates)
    valid = np.isfinite(all_h)
    hits = valid.sum(axis=0)
    mid = np.nansum(all_h, axis=0) / np.maximum(hits, 1)
    spread = np.where(valid, np.abs(all_h - mid), 0).max(axis=0)
    return np.where((hits >= 2) & (spread < 0.5 * span), mid, np.nan)


def _height_grid(xy, z, known, step, lo_u, lo_v):
    """Fit local planes at half-cell spacing, where measurements support them."""
    fine_known = np.repeat(np.repeat(known, 2, axis=0), 2, axis=1)
    gy, gx = np.where(fine_known)
    here = np.column_stack((lo_u + (gx + 0.5) * step,
                            lo_v + (gy + 0.5) * step))
    dist, j = cKDTree(xy).query(here, k=6)
    keep = dist[:, 0] < step * 1.4
    gy, gx, here, dist, j = gy[keep], gx[keep], here[keep], dist[keep], j[keep]
    vals = z[j]
    delta = xy[j] - here[:, None, :]
    weights = 1 / (dist ** 2 + (0.45 * step) ** 2)
    weights /= weights.sum(axis=1, keepdims=True)
    x0, y0 = np.einsum("ij,ijk->ki", weights, delta)
    z0 = np.sum(weights * vals, axis=1)
    x = delta[..., 0] - x0[:, None]
    y = delta[..., 1] - y0[:, None]
    err = vals - z0[:, None]
    # Small ridge controls nearly collinear sample neighbourhoods.
    ridge = (0.15 * step) ** 2
    xx = np.sum(weights * x * x, axis=1) + ridge
    yy = np.sum(weights * y * y, axis=1) + ridge
    cross = np.sum(weights * x * y, axis=1)
    xz = np.sum(weights * x * err, axis=1)
    yz = np.sum(weights * y * err, axis=1)
    det = xx * yy - cross * cross
    h = z0 - (x0 * (yy * xz - cross * yz) + y0 * (xx * yz - cross * xz)) / det
    fine = np.full(fine_known.shape, np.nan)
    fine[gy, gx] = h
    return fine


def apply_height_trial(cloud, result):
    """Measure a geometry-only continuation on an existing single-chart patch.

    This trial keeps existing donor attributes and splat coverage. It is an
    offline geometry experiment and does not recompute rotations or SH.
    """
    ctx = result.context
    known_idx = ctx.get("surface_idx")
    cells = ctx.get("cell_key")
    if (known_idx is None or result.info.get("charts") != 1
            or result.info.get("mode") != "patch" or result.info.get("unfold")
            or cells is None
            or len(cells) != result.added.n or result.added.n == 0):
        return 0
    origin, basis = ctx["origin"], ctx["basis"]
    # Volume repairs can include artificial stand-ins outside the input.
    survived = np.ones(cloud.n, bool)
    survived[result.removed] = False
    known_idx = known_idx[known_idx < cloud.n]
    known = (cloud.xyz[known_idx[survived[known_idx]]] - origin) @ basis.T
    uvh = (result.added.xyz - origin) @ basis.T
    face = cells >= 0
    if result.backing is not None:
        face &= ~result.backing
    positions = uvh[face]
    h = predict_height(known, ctx["grid"], ctx["cell"], ctx["target"] | ctx["holefoot"],
                       positions[:, :2])
    if h is None:
        return 0
    valid = np.isfinite(h)
    changed = int(valid.sum())
    if not changed or changed < 0.5 * len(positions):
        return 0
    positions[valid, 2] = h[valid]
    result.added.xyz[np.flatnonzero(face)[valid]] = positions[valid] @ basis + origin
    return changed
