"""Offline donor experiment for periodic surfaces that bend in two axes."""
import numpy as np
from scipy.optimize import differential_evolution, least_squares
from scipy.spatial import cKDTree
from splatcore import repair as rp


def _phase_basis(xy):
    x, y = xy.T
    return np.column_stack((x, y, x**2, x*y, y**2))


def _fit_phases(xy, h, radius, cell):
    """Fit two bending wave directions and verify them on an unseen inner ring."""
    R = np.linalg.norm(xy, axis=1) / radius
    above = (R > 1.30) & (R < 2.02)
    below = (R > 1.02) & (R < 1.29)
    if above.sum() < 450 or below.sum() < 130:
        return None
    t = np.flatnonzero(above)
    t = np.random.default_rng(36).choice(t, min(len(t), 512), replace=False)
    transform = radius * 1.8
    v = xy / transform
    a = _phase_basis(v[t])
    b = np.column_stack((np.ones(len(a)), a[:, :2]))
    init_trend = np.linalg.lstsq(b, h[t], rcond=None)[0]
    train = h[t]
    residue = train - b @ init_trend
    span = np.sqrt(np.mean(residue ** 2))
    if span < 0.25 * cell:
        return None
    bounds = [(0, 28), (-28, 28), (-11, 11), (-13, 13), (-11, 11)]
    models = []

    def frequency_loss(shifts):
        # Shapes (5, candidates) from the optimizer and (point, 5) from the rim.
        shifts = np.asarray(shifts)
        raw = a @ shifts
        cos = np.cos(raw)
        sin = np.sin(raw)
        score = (residue @ cos) ** 2 + (residue @ sin) ** 2
        k = np.linalg.norm(shifts[:2], axis=0)
        return -np.where((k > 5) & (k < 34), score, -0.1)

    for seed in (0, 1):
        fit = differential_evolution(frequency_loss, bounds, popsize=10, maxiter=58,
                                     updating="deferred", vectorized=True, seed=seed)
        raw = a @ fit.x
        C = np.column_stack((b, np.cos(raw), np.sin(raw)))
        coeff = np.linalg.lstsq(C, residue, rcond=None)[0]
        new = residue - C @ coeff
        if np.linalg.norm(new) > 0.95 * np.linalg.norm(residue):
            return None
        models.append(np.r_[coeff[-2:], fit.x])
        residue = new

    def calc(p, a, jac=False):
        result = p[0] + a[:, :2] @ p[1:3]
        columns = [np.ones((len(a), 1)), a[:, :2]] if jac else None
        for m in p[3:].reshape(2, 7):
            phase = a @ m[2:]
            cs, sn = np.cos(phase), np.sin(phase)
            result += m[0] * cs + m[1] * sn
            if jac:
                columns.extend([cs[:, None], sn[:, None],
                                (m[1] * cs - m[0] * sn)[:, None] * a])
        return np.concatenate(columns, 1) if jac else result

    best = least_squares(lambda p: calc(p, a) - train,
                         np.concatenate([init_trend, *models]),
                         jac=lambda p: calc(p, a, True), max_nfev=90)
    held = calc(best.x, _phase_basis(v[below]))
    error = np.abs(held - h[below])
    sectors = (np.floor((np.arctan2(*v[below].T[::-1]) + np.pi) / (np.pi/4)).astype(int)) % 8
    counts = np.bincount(sectors, minlength=8)
    if (counts < 10).any():
        return None
    weakest_sector = max(np.median(error[sectors == i]) for i in range(8))
    rmse = np.sqrt(np.mean((calc(best.x, a) - train) ** 2))
    if rmse > 0.23 * span or np.median(error) > 0.16 * span or weakest_sector > 0.28 * span:
        return None

    def height(u, v):
        uv = np.column_stack((u, v))
        return calc(best.x, _phase_basis(uv / transform))

    def features(uv):
        x = uv / transform
        phases = _phase_basis(x) @ best.x[3:].reshape(2, 7)[:, 2:].T
        return np.column_stack((np.cos(phases), np.sin(phases), 0.25 * x))

    return height, features, span


def apply_phase_trial(cloud, r):
    """Pick phase-matched donors and transport their attributes on a chart.

    The existing repair supplies the target positions, DC tone and opacity
    fade. The donor supplies size, rotation, directional colour and extras.
    The chart model moves the donor's local height residual and turns its
    axes and spherical harmonics together.
    """
    ctx = r.context
    known = ctx.get("surface_idx")
    cells = ctx.get("cell_key")
    if (known is None or cells is None or len(cells) != r.added.n
            or r.info.get("mode") != "patch" or r.info.get("unfold")
            or r.info.get("charts") != 1):
        return 0
    survives = np.ones(cloud.n, bool)
    survives[r.removed] = False
    known = known[known < cloud.n]
    known = known[survives[known]]
    uvh = (cloud.xyz[known] - ctx["origin"]) @ ctx["basis"].T
    u0, v0 = ctx["uv_centre"]
    xyz = (r.added.xyz - ctx["origin"]) @ ctx["basis"].T
    origin_uv = np.array([u0, v0])
    source_uv = uvh[:, :2] - origin_uv
    pred = _fit_phases(source_uv, uvh[:, 2], ctx["radius"], ctx["cell"])
    if pred is None:
        return 0
    height, features, span = pred
    source_residual = uvh[:, 2] - height(*source_uv.T)
    radius = np.linalg.norm(source_uv, axis=1) / ctx["radius"]
    donor = np.flatnonzero((np.abs(source_residual) < 0.22 * span) & (radius < 2.1))
    face_mask = cells >= 0
    if r.backing is not None:
        face_mask &= ~r.backing
    face = np.flatnonzero(face_mask)
    if len(donor) < 100 or not len(face):
        return 0
    query = xyz[face, :2] - origin_uv
    dist, idx = cKDTree(features(source_uv[donor])).query(features(query))
    use = dist < 0.45
    if use.mean() < 0.98:
        return 0
    face, query, picked = face[use], query[use], donor[idx[use]]
    src = known[picked]
    xyz[face, 2] = height(*query.T) + source_residual[picked]
    quats, _, turn_sh = rp._turn_copies(
        cloud.rot[src], cloud.xyz[src], xyz[face, :2], height, ctx["origin"],
        ctx["basis"], u0, v0, ctx["cell"])
    r.added.xyz[face] = xyz[face] @ ctx["basis"] + ctx["origin"]
    r.added.scale[face] = cloud.scale[src]
    r.added.rot[face] = quats
    r.added.f_rest[face] = turn_sh(cloud.f_rest[src])
    for name, values in r.added.extra.items():
        values[face] = cloud.extra[name][src]
    return int(len(face))
