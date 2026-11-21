"""Deep repair: let a 2-D inpainter decide what the patch should look like.

The geometric repair borrows appearance from somewhere else on the same
surface. That is the right answer when such a place exists, and a guess when it
does not -- it cannot invent the continuation of a logo, a seam, or a shadow it
has never seen. A 2-D inpainting model can, because it has seen millions of
surfaces.

The pipeline is deliberately the short one:

1. Repair geometrically first. That gives the tangent frame, the fitted surface
   and Gaussians already in the right places with coherent covariances.
2. Render one reference view straight down the patch normal.
3. Mask the patch and let **LaMa** fill it.
4. Push that image back onto the Gaussians by least squares.
5. Check from other angles that nothing got worse, and fall back if it did.

Two choices worth stating.

**LaMa, not a diffusion model.** The only independent head-to-head on 3DGS
inpainting found reconstruction-based inpainters beat generative ones on
multi-view consistency by a wide margin (FID 142.9 vs 291.8 under finetuning):
diffusion output looks sharper in 2-D and does not survive the lift to 3-D,
because each view hallucinates something different.

**One reference view, not many.** The DC term of a Gaussian is view-independent,
so a single view determines it completely. Inpainting several views separately
and averaging is how the published pipelines get mush -- the views disagree, and
the average of disagreement is grey. The remaining view-dependence (SH bands
1..3) stays as the geometric pass set it, inherited whole from a real donor.

Further pictures can be given (``references``, from a multi-view sheet filled
in one call), each a train view of its own. Measured on the stand hole and the
autotest stroke (November 2026) they did not look better with LaMa and cost
time, so the wide view asks for one unless told otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .io import SplatSet, SH_C0
from .repair import RepairParams, RepairResult, RepairError, repair
from . import preview


@dataclass
class DeepParams:
    views: int = 6                  # kept for callers; the check uses two held-out views
    resolution: int = 0             # 0 = two pixels per splat spacing
    tilt_deg: float = 16.0
    frame_fraction: float = 0.55    # how much of the frame the patch fills
    backend: str = "auto"           # "auto" | "gpu" | "cpu"
    steps: int = 250                # gradient steps; 400 over-fits the one view
    iterations: int = 4             # least-squares passes on the CPU fallback
    damping: float = 0.9
    mask_grow: int = 6              # pixels the inpaint mask grows past the hole
    inpainter: str = "auto"         # "auto" | "lama" | "opencv"
    guard: bool = True
    multiview: float = 0.3          # weight of the picture carried into tilted views
    detail: float = -1.0            # share of grain from the geometric fill; < 0 = measure
    tone_scale: float = 0.0         # spacings; tone coarser than this from the geometric fill
    # Hiding them measured ahead on all six views of bench_deep (108.7
    # against 110.3 summed RMS) and lifted the stand's patch from 0.265 to
    # 0.285 against 0.30 in the ring.
    hide_dark: float = 0.4          # hide rim blobs darker than this share of the ring from LaMa
    # Above this share of grain lost, LaMa returned mush and the copied fill
    # stays. On the hold-out discs LaMa loses 0 to 0.2 of the grain; over the
    # stand it lost 0.95, and its picture there was a darker smear on top of
    # a lattice the copied fill had continued in phase.
    mush: float = 0.6
    # Both audits advised freezing these two, judging by agreement with the
    # inpainted picture carried into other views. Against the real surface
    # (bench_deep.py) training them came out slightly ahead -- 106.4 against
    # 107.1 summed RMS over six views -- so they train.
    train_backing: bool = True      # let the slab behind the face change colour
    train_sh: bool = True           # let the view-dependent bands change
    # Pictures from further views (``deep_repair(references=[...])``, a
    # multi-view sheet): each is a train view of its own, at this weight
    # against the main picture's 1.
    extra_weight: float = 1.0
    # On a picture whose grain had to come from the copy (``reference
    # ["grain"]``, the share `viewfill.grain_from_copy` put back, at least
    # ``copy_grain``): the copy's splats keep shape, relief and colour detail
    # finer than ``copy_structure`` spacings and take only what is coarser
    # from the picture (`_tone_onto_copy`) instead of being fitted to it.
    # Stand hole 0.56: on. Hole without splats 0.05, which LaMa filled well
    # and where the copy had a pale blemish the fit removes: off. 0 = off.
    copy_structure: float = 1.0
    copy_grain: float = 0.3
    seed: int = 0


class DeepUnavailable(RuntimeError):
    """Raised when no inpainter can be loaded, with the reason."""


# ------------------------------------------------------------- inpainting

_LAMA = {}

# The LaMa weights simple-lama-inpainting downloads, pinned: torch loads this
# file as code, so it is checked before loading instead of trusted by URL.
LAMA_URL = ("https://github.com/enesmsahin/simple-lama-inpainting/"
            "releases/download/v0.1.0/big-lama.pt")
LAMA_SHA256 = "7ba7aa7ac37a4d41fdbbeba3a2af7ead18058552997e3a3cd1a3b2210c9e6b4c"


def _lama_weights() -> str:
    """Path of the verified LaMa weights, downloading them on first use."""
    import hashlib
    import os
    import torch
    path = os.path.join(torch.hub.get_dir(), "checkpoints", "big-lama.pt")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.hub.download_url_to_file(LAMA_URL, path, hash_prefix=LAMA_SHA256)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != LAMA_SHA256:
        raise DeepUnavailable("the LaMa weights at %s do not match their checksum; "
                              "delete the file and Deep Fill downloads it again" % path)
    return path


def load_inpainter(kind: str = "auto"):
    """Return ``(fn(rgb_uint8, mask_uint8) -> rgb_uint8, name)``."""
    if kind in ("auto", "lama"):
        try:
            if "lama" not in _LAMA:
                import os
                from simple_lama_inpainting import SimpleLama
                os.environ["LAMA_MODEL"] = _lama_weights()
                _LAMA["lama"] = SimpleLama()
            lama = _LAMA["lama"]

            def run_lama(rgb, mask):
                from PIL import Image
                out = lama(Image.fromarray(rgb), Image.fromarray(mask))
                return np.asarray(out)[:rgb.shape[0], :rgb.shape[1]]

            return run_lama, "LaMa"
        except Exception as e:                   # noqa: BLE001
            if kind == "lama":
                raise DeepUnavailable("LaMa could not be loaded: %s" % e) from e

    try:
        import cv2
    except Exception as e:                       # noqa: BLE001
        raise DeepUnavailable(
            "No inpainter available (neither LaMa nor OpenCV): %s" % e) from e

    has_fsr = hasattr(cv2, "xphoto") and hasattr(cv2.xphoto, "inpaint")

    def run_cv(rgb, mask):
        # Frequency-selective reconstruction: not as good as LaMa on structure,
        # but it continues stochastic texture instead of smearing it, which
        # Telea and Navier-Stokes both do.
        try:
            return cv2.xphoto.inpaint(rgb, 255 - mask, cv2.xphoto.INPAINT_FSR_BEST)
        except Exception:                        # noqa: BLE001
            return cv2.inpaint(rgb, mask, 5, cv2.INPAINT_TELEA)

    # Say which one actually runs. Without the contrib build there is no
    # xphoto, and the fallback is Telea -- which the comment above calls a
    # smear -- while the interface went on reporting FSR.
    return run_cv, ("OpenCV FSR" if has_fsr else "OpenCV Telea (weak)")


# ------------------------------------------------------ rasterise + solve

def _visible_subset(s: SplatSet, centre, radius):
    """Splats near the patch -- everything that can plausibly cover it."""
    d = np.linalg.norm(s.xyz.astype(np.float64) - centre, axis=1)
    return np.where(d < radius)[0]


def _raster(s: SplatSet, idx, cam_eye, cam_target, up, W, H, fov,
            residual=None, accum_for=None):
    """Composite front-to-back; optionally accumulate a least-squares update.

    With ``residual`` and ``accum_for`` (a boolean mask over ``idx``), also
    returns per-splat ``sum(w)`` and ``sum(w * residual)``, where ``w`` is the
    splat's actual contribution to a pixel -- alpha times the transmittance in
    front of it.

    The residual passed in must already be divided by the total coverage at
    each pixel. Solving each splat's colour as if it alone had to close the gap
    overshoots by however many splats share the pixel, which on a surface with
    three or four overlapping splats makes the patch worse, not better.
    """
    Rv, eye = preview.look_at(cam_eye, cam_target, up)
    sub = s.take(idx)
    cam, z, f, sx, sy = preview._project(sub, Rv, eye, W, H, fov)
    keep0 = np.where(z > 1e-3)[0]
    if len(keep0) == 0:
        raise RepairError("The spot is not visible from this direction")
    cam, z, sx, sy = cam[keep0], z[keep0], sx[keep0], sy[keep0]
    inv, rad, ok = preview._conics(sub, Rv, cam, z, f, keep0)
    cutoff = 3.0
    rad = np.minimum(cutoff * rad, 64)
    a = sub.alpha[keep0]
    keep = ok & (a > 0.012) & (rad > 0.3) & (sx > -rad) & (sx < W + rad) \
        & (sy > -rad) & (sy < H + rad)
    sel = keep0[keep]
    sx, sy, z, inv, rad, a = sx[keep], sy[keep], z[keep], inv[keep], rad[keep], a[keep]
    order = np.argsort(z)
    sel, sx, sy, inv, rad, a = (sel[order], sx[order], sy[order], inv[order],
                                rad[order], a[order])

    dirs = sub.xyz[sel].astype(np.float64) - eye
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
    col = preview._colour(sub, sel, dirs, 3)

    img = np.zeros((H, W, 3))
    T = np.ones((H, W))
    sum_w = np.zeros(len(idx)) if residual is not None else None
    sum_wr = np.zeros((len(idx), 3)) if residual is not None else None
    want = None if accum_for is None else accum_for

    ri = np.ceil(rad).astype(np.int32)
    lim = cutoff * cutoff * 0.5
    for k in range(len(sel)):
        r = ri[k]
        x0, x1 = max(0, int(sx[k]) - r), min(W, int(sx[k]) + r + 1)
        y0, y1 = max(0, int(sy[k]) - r), min(H, int(sy[k]) + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        Tl = T[y0:y1, x0:x1]
        if Tl.max() < 0.004:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx, dy = xx - sx[k], yy - sy[k]
        I = inv[k]
        p = 0.5 * (I[0, 0] * dx * dx + 2 * I[0, 1] * dx * dy + I[1, 1] * dy * dy)
        al = np.where(p > lim, 0.0, np.minimum(a[k] * np.exp(-p), 0.995))
        w = al * Tl
        img[y0:y1, x0:x1] += w[..., None] * col[k]
        if residual is not None and (want is None or want[sel[k]]):
            rr = residual[y0:y1, x0:x1]
            sum_w[sel[k]] += float(w.sum())
            sum_wr[sel[k]] += (w[..., None] * rr).sum(axis=(0, 1))
        T[y0:y1, x0:x1] = Tl * (1.0 - al)
    return img, 1.0 - T, sum_w, sum_wr


# ----------------------------------------------------------- refinement

def _refine_cpu(filled, near, local_new, eye, centre, up, W, H, fov,
                target, mask, img0, before, deep, say):
    """Least-squares colour fit, no GPU required.

    Each new Gaussian takes the coverage-weighted mean of the residual over the
    pixels it contributes to. Dividing the residual by the total coverage first
    is what keeps it stable: without that, every splat sharing a pixel tries to
    close the whole gap on its own and the patch overshoots.
    """
    work = filled.copy()
    history = [before]
    resid_mask = mask[..., None].astype(np.float64)
    img, alpha = img0, None
    for it in range(max(deep.iterations, 1)):
        if alpha is None:
            _, alpha, _, _ = _raster(work, near, eye, centre, up, W, H, fov)
        r = (target - img) / np.maximum(alpha, 0.15)[..., None]
        r *= resid_mask
        _, _, sum_w, sum_wr = _raster(work, near, eye, centre, up, W, H, fov,
                                      residual=r, accum_for=local_new)
        good = (sum_w > 1e-9) & local_new
        if not good.any():
            break
        delta = sum_wr[good] / sum_w[good][:, None]
        gi = near[good]
        work.f_dc[gi] = (work.f_dc[gi].astype(np.float64)
                         + deep.damping * delta / SH_C0).astype(np.float32)
        img, alpha, _, _ = _raster(work, near, eye, centre, up, W, H, fov)
        err = float(np.abs((target - img) * resid_mask).sum() / max(mask.sum(), 1))
        history.append(err)
        say("fitting colors (%d/%d)" % (it + 1, deep.iterations),
            0.55 + 0.25 * (it + 1) / max(deep.iterations, 1))
        if err > history[-2] * 0.995:
            break
    return work, history, "CPU fit"


def _project(P, cam):
    """World points -> (x, y, z) in a gsplat camera's pixel frame."""
    X = np.asarray(P, np.float64) @ cam.view[:3, :3].T + cam.view[:3, 3]
    z = X[:, 2]
    zs = np.where(z > 1e-9, z, np.nan)
    x = cam.K[0, 0] * X[:, 0] / zs + cam.K[0, 2]
    y = cam.K[1, 1] * X[:, 1] / zs + cam.K[1, 2]
    return x, y, z


def _colour_from_picture(s: SplatSet, ids, cam, target, mask, px_sp, shift=0.0, k=3.0,
                         texture=6.0):
    """Start the new splats near the picture's colour, not at the copy's.

    The fit only nudges colours (250 steps, a DC step of 2.5e-3), so it
    starts from what the copy brought. Over the EHBO letters the copy had
    taken thin, dark splats from the shutter's rail beside the hole -- a
    needle runs up to sixteen cells across the fill -- and the fit got them
    a third of the way to the wall's cream: the letters' outlines stayed as
    dark and pale lines, in the quick fill, the refined one and the deep one
    alike, although LaMa's picture behind them was clean.

    So each new splat whose centre the reference view sees inside the mask
    may differ from the picture there by at most ``k`` times the texture of
    the surface round the mask: the robust spread, per channel, of the
    picture's detail finer than ``texture`` spacings in a ring outside the
    mask. Further off it is pulled in to that bound and loses the copy's
    view-dependent bands. A copy within it keeps its colour, and with it
    the grain the picture may lack: reset to the picture wholesale, the
    tread lattice copied into the shoe's edge stroke went flat brown; at
    two spreads it was still a little soft, at three it is as before. The
    rail's needles sit about five spreads off the cream wall. A copy away
    from the mask keeps its own colour.

    Returns the set, how many splats were pulled in, and the bound.
    """
    from scipy import ndimage
    ids = np.asarray(ids, np.int64)
    if len(ids) == 0:
        return s, 0, None
    x, y, z = _project(s.xyz[ids], cam)
    H, W = mask.shape[:2]
    ok = np.isfinite(x) & np.isfinite(y) & (z > 0)
    xi = np.clip(np.floor(np.nan_to_num(x) - shift).astype(np.int64), 0, W - 1)
    yi = np.clip(np.floor(np.nan_to_num(y) - shift).astype(np.int64), 0, H - 1)
    ok &= (x >= 0) & (x < W) & (y >= 0) & (y < H)
    m = np.asarray(mask, bool)
    ok &= m[yi, xi]
    ring = ndimage.binary_dilation(m, np.ones((3, 3)), iterations=12) & ~m
    if not ok.any() or ring.sum() < 64:
        return s, 0, None
    tgt = np.asarray(target, np.float64)
    sigma = max(texture * px_sp, 2.0)
    lo = np.stack([ndimage.gaussian_filter(tgt[..., c], sigma) for c in range(3)], -1)
    hi = (tgt - lo)[ring]
    spread = 1.4826 * np.median(np.abs(hi - np.median(hi, 0)), 0)
    bound = k * np.maximum(spread, 1.0 / 255.0)
    pic = np.stack([ndimage.gaussian_filter(tgt[..., c], 1.0) for c in range(3)], -1)
    want = np.clip(pic[yi[ok], xi[ok]], 0, 1)
    have = s.rgb[ids[ok]].astype(np.float64)
    pulled = np.clip(have, want - bound, want + bound)
    moved = np.abs(pulled - have).max(1) > 0
    if not moved.any():
        return s, 0, bound
    out = s.copy()
    g = ids[ok][moved]
    out.f_dc[g] = ((np.clip(pulled[moved], 0, 1) - 0.5) / SH_C0).astype(np.float32)
    if out.f_rest.shape[1]:
        out.f_rest[g] = 0.0
    return out, int(moved.sum()), bound


def _tone_onto_copy(s: SplatSet, ids, near, cam, target, mask, sigma, deg):
    """The copy's splats as they are, with only the picture's coarse colour.

    The wide view LaMa fills is about 1.4 pixels per splat spacing, and its
    lattice comes back as a hint (`viewfill.grain_from_copy` puts some of the
    copy's grain back, but only at the picture's resolution). Fitted to that
    picture, every new splat took the blurred colour of its pixel: the
    lattice's ridges, the knurl between them and the shading on their flanks
    -- all carried splat by splat by the copy -- went soft from every angle,
    and flattening the relief to the picture took the ridges' shape too.

    So nothing is fitted. The picture minus the copy's own rendering in the
    same view, both blurred by ``sigma`` pixels inside the grown mask, is a
    colour offset field; each new splat takes the offset at its centre, faded
    out past the mask. Arc, plaque and tone come from the picture, everything
    finer than ``sigma`` stays the copy's, in every view alike. Splats the
    front view does not see (ridge flanks) get the same offset as their
    neighbours, so they keep their shading relative to them.
    """
    from scipy import ndimage
    ids = np.asarray(ids, np.int64)
    rgb, _, _ = _gpu_draw(s, near, [cam], deg)
    m = np.asarray(mask, bool)
    grown = ndimage.binary_dilation(m, np.ones((3, 3)), iterations=max(int(2 * sigma), 1))
    w = ndimage.gaussian_filter(grown.astype(np.float64), sigma)

    def lo(a):
        b = np.stack([ndimage.gaussian_filter(a[..., c] * grown, sigma) for c in range(3)], -1)
        return b / np.maximum(w, 1e-6)[..., None]

    off = (lo(np.asarray(target, np.float64)) - lo(rgb[0])) * w[..., None]
    x, y, z = _project(s.xyz[ids], cam)
    ok = np.isfinite(x) & np.isfinite(y) & (z > 0)
    xs = np.clip(np.nan_to_num(x) - 0.5, 0, cam.width - 1)
    ys = np.clip(np.nan_to_num(y) - 0.5, 0, cam.height - 1)
    o = np.stack([ndimage.map_coordinates(off[..., c], [ys, xs], order=1, mode="nearest")
                  for c in range(3)], -1)
    o[~ok] = 0.0
    out = s.copy()
    out.f_dc[ids] = (out.f_dc[ids].astype(np.float64) + o / SH_C0).astype(np.float32)
    return out


def _hole_points(ctx):
    """The hole as points on the fitted surface, one per filled grid cell.

    This is what the inpainter is allowed to repaint. It used to be wherever
    any new splat landed, grown by six pixels -- and the geometric fill
    deliberately lays new splats past the hole, under the surviving surface,
    to fade into it. The mask came out 39 % larger than the hole; 27,000
    pixels of intact sole were handed to LaMa to invent again, the row of
    dots along the sole's edge was erased and re-imagined, and LaMa, with its
    context pushed 10 to 40 pixels further away, filled the real hole 12 %
    softer. The hole is the hole: the cells the geometric pass had to fill.
    """
    lo_u, lo_v, nu, nv = ctx["grid"]
    cell = float(ctx["cell"])
    cu0, cv0 = ctx["uv_centre"]
    ty, tx = np.where(ctx["target"])
    # four samples per cell, so the mask has no gaps at any reasonable scale
    off = np.array([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]])
    u = (lo_u + (tx[:, None] + off[None, :, 0]) * cell).ravel()
    v = (lo_v + (ty[:, None] + off[None, :, 1]) * cell).ravel()
    h = ctx["surface"](u - cu0, v - cv0)
    return ctx["origin"] + np.stack([u, v, h], 1) @ ctx["basis"]


def _mask_from_points(P, cam, W, H, grow, shift=0.0):
    from scipy import ndimage
    x, y, z = _project(P, cam)
    ok = np.isfinite(x) & (z > 0) & (x >= 0) & (x < W) & (y >= 0) & (y < H)
    m = np.zeros((H, W), bool)
    m[(y[ok] - shift).astype(np.int64).clip(0, H - 1),
      (x[ok] - shift).astype(np.int64).clip(0, W - 1)] = True
    m = ndimage.binary_closing(m, np.ones((3, 3)), iterations=2)
    m = ndimage.binary_fill_holes(m)
    if grow > 0:
        m = ndimage.binary_dilation(m, np.ones((3, 3)), iterations=grow)
    return m


def volume_mask(vol, cam, W, H, depth=None, grow=0):
    """What was painted, as seen from ``cam``: the volume on the visible surface.

    ``vol.mask_in_view`` is every pixel whose ray passes through the painted
    volume. Seen from another angle than the one it was painted from, that
    shadow runs on over intact surface in front of or behind the volume, so
    with the view's ``depth`` (camera z, inf where nothing is drawn) a pixel
    counts when the point it shows lies inside the volume. Pixels that show
    nothing -- past the object's edge -- are not part of it; a hole in the
    scan is added by the caller from the fitted surface.

    With ``depth`` the rays are not traced at all: only the shown points
    inside the volume's projected box are tested. Tracing every ray through
    a stroke of 1290 capsules took 1.2 s a view and was half the deep fill.
    """
    from scipy import ndimage
    if depth is None:
        m = np.asarray(vol.mask_in_view(cam, W, H), bool)
    else:
        box = np.isfinite(depth) & (depth > 0)
        b = vol.bounds() if hasattr(vol, "bounds") else None
        if b is not None:
            lo, hi = b
            c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                          for z in (lo[2], hi[2])]) @ cam.view[:3, :3].T + cam.view[:3, 3]
            if (c[:, 2] > 1e-6).all():
                x = cam.K[0, 0] * c[:, 0] / c[:, 2] + cam.K[0, 2]
                y = cam.K[1, 1] * c[:, 1] / c[:, 2] + cam.K[1, 2]
                x0, x1 = int(np.clip(np.floor(x.min()) - 1, 0, W)), int(np.clip(np.ceil(x.max()) + 1, 0, W))
                y0, y1 = int(np.clip(np.floor(y.min()) - 1, 0, H)), int(np.clip(np.ceil(y.max()) + 1, 0, H))
                inb = np.zeros((H, W), bool)
                inb[y0:y1, x0:x1] = True
                box &= inb
        yy, xx = np.where(box)
        out = np.zeros((H, W), bool)
        if len(yy):
            z = depth[yy, xx]
            K = cam.K
            Xc = np.stack([(xx + 0.5 - K[0, 2]) / K[0, 0] * z,
                           (yy + 0.5 - K[1, 2]) / K[1, 1] * z, z], 1)
            Pw = (Xc - cam.view[:3, 3]) @ cam.view[:3, :3]
            ok = np.asarray(vol.inside(Pw), bool)
            out[yy[ok], xx[ok]] = True
        # the per-pixel test is noisy at the volume's boundary
        m = ndimage.binary_closing(out, np.ones((3, 3)), iterations=2)
    if grow > 0 and m.any():
        m = ndimage.binary_dilation(m, np.ones((3, 3)), iterations=int(grow))
    return m


def _warp(target, depth, mask, src, dst, dst_depth, tol):
    """Carry the inpainted picture into another camera through the surface.

    Every pixel of the hole in the reference view is a ray that hits the
    geometric fill at the rendered depth; that point is projected into the
    other camera and takes its colour along. Pixels the other camera sees
    but the reference did not -- or sees at a different depth, i.e. behind
    something -- stay unsupervised. This is the depth-guided lift of InFusion
    and SPIn-NeRF, with the geometric pass's surface standing in for a depth
    network: the inpainter is still asked once, so there is no disagreement
    between views to average into grey.
    """
    from scipy import ndimage
    H, W = mask.shape
    yy, xx = np.where(mask & np.isfinite(depth) & (depth > 0))
    if len(yy) < 64:
        return None, None
    z = depth[yy, xx]
    K = src.K
    Xc = np.stack([(xx + 0.5 - K[0, 2]) / K[0, 0] * z,
                   (yy + 0.5 - K[1, 2]) / K[1, 1] * z, z], 1)
    R, t = src.view[:3, :3], src.view[:3, 3]
    Pw = (Xc - t) @ R                        # camera -> world, R orthonormal
    x, y, zt = _project(Pw, dst)
    ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    ok = np.isfinite(x) & (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H) & (zt > 0)
    ok[ok] &= np.abs(zt[ok] - dst_depth[iy[ok], ix[ok]]) < tol
    acc = np.zeros((H, W, 3))
    cnt = np.zeros((H, W))
    np.add.at(acc, (iy[ok], ix[ok]), target[yy[ok], xx[ok]])
    np.add.at(cnt, (iy[ok], ix[ok]), 1.0)
    hit = cnt > 0
    out = acc / np.maximum(cnt, 1)[..., None]
    # close the one-pixel gaps a forward splat leaves, no further
    fill = ndimage.binary_closing(hit, np.ones((3, 3))) & ~hit
    if fill.any():
        num = ndimage.uniform_filter(acc, (3, 3, 1))
        den = ndimage.uniform_filter(cnt, 3)
        out[fill] = num[fill] / np.maximum(den[fill], 1e-9)[:, None]
        hit = hit | (fill & (den > 0))
    return out, ndimage.binary_erosion(hit, np.ones((3, 3)))


def _grain(img, mask, sigma):
    """Mean fine-detail energy of an image inside a mask."""
    from scipy import ndimage
    g = img.mean(-1)
    hi = g - ndimage.gaussian_filter(g, sigma)
    return float(np.sqrt((hi[mask] ** 2).mean())) if mask.any() else 0.0


def _dark_rim(rgb, mask, share, reach=12, touch=3):
    """Dark blobs touching the mask, for LaMa not to take its tone from.

    What is left of a removed object usually sits right at the hole's edge
    -- the black foot of the stand, here -- and LaMa, which takes its tone
    from whatever touches the mask, spread it over the whole patch. A blob
    counts when it is darker than ``share`` of the ring's median and reaches
    within ``touch`` pixels of the mask. The fit still ends at the mask; only
    the inpainter is told not to look there.
    """
    from scipy import ndimage
    L = rgb.astype(np.float64).mean(-1)
    ring = ndimage.binary_dilation(mask, np.ones((3, 3)), iterations=reach) & ~mask
    if not ring.any():
        return np.zeros_like(mask)
    dark = ring & (L < share * float(np.median(L[ring])))
    lbl, n = ndimage.label(dark)
    if n == 0:
        return np.zeros_like(mask)
    near = ndimage.binary_dilation(mask, np.ones((3, 3)), iterations=touch)
    keep = np.unique(lbl[near & dark])
    out = np.isin(lbl, keep[keep > 0])
    return ndimage.binary_dilation(out, np.ones((3, 3)), iterations=2) & ~mask


def _tone_from_geo(target, geo, mask, sigma):
    """The geometric fill's tone under LaMa's structure.

    LaMa takes its tone from whatever touches the mask. Over the stand that
    was a black remnant of the stand itself at the lower rim, and the whole
    patch came out a shade darker than the sole round it (0.265 against
    0.301 in the ring). The geometric fill's tone is a field fitted to the
    ring, which is the thing to trust at that scale, so everything coarser
    than ``sigma`` is taken from it; outside the mask both images agree and
    the correction fades out over a few pixels.
    """
    from scipy import ndimage
    lo = lambda a: np.stack([ndimage.gaussian_filter(a[..., c], sigma)  # noqa: E731
                             for c in range(3)], -1)
    w = ndimage.gaussian_filter(mask.astype(np.float64), 2.0)[..., None]
    return np.clip(target + w * (lo(geo) - lo(target)), 0.0, 1.0)


def _detail_split(target, geo, mask, sigma, amount):
    """LaMa's tone and structure, the geometric fill's grain.

    LaMa is a regression model; on stochastic texture it returns the local
    mean. How much depends on the hole: on a small disc with rich context
    it keeps most of the grain, and keeping its output measured best against
    the truth there; over the stand's hole it returned brown mush where the
    copied fill had a clear lattice. So ``amount`` may be negative, meaning
    "measure it": compare the fine-detail energy LaMa left inside the hole
    with the energy of the surface around it, and put back from the
    geometric fill exactly the share that went missing. Everything coarser
    than ``sigma`` stays LaMa's.
    """
    from scipy import ndimage
    if amount < 0:
        ring = ndimage.binary_dilation(mask, np.ones((3, 3)), iterations=12) & ~mask
        have = _grain(target, mask, sigma)
        want = _grain(target, ring, sigma)
        # Two unrelated grains mixed a : 1 - a lose energy to the mixing
        # itself, so putting back exactly the missing share still looks soft
        # (it did, at 0.5 over the stand); twice the missing share, capped,
        # gives the stand nearly all the copied grain and a well-filled small
        # hole almost none.
        amount = float(np.clip(2.0 * (1.0 - have / max(want, 1e-6)), 0.0, 1.0))
    if amount <= 0:
        return target, 0.0
    lo = lambda a: np.stack([ndimage.gaussian_filter(a[..., c], sigma)  # noqa: E731
                             for c in range(3)], -1)
    hi_t, hi_g = target - lo(target), geo - lo(geo)
    mixed = target + amount * (hi_g - hi_t)
    m = mask[..., None]
    return np.clip(np.where(m, mixed, target), 0.0, 1.0), amount


def _refine_gpu(filled, near, train, frozen_new, cams, target, mask, warped,
                hole_masks, deep, deg, say, extra=()):
    """Fit the new Gaussians to the inpainted view with real gradients.

    Only the face of the new patch is a parameter. Everything around it is
    rendered with it -- occlusion and blending have to be right -- but
    frozen, so the solve cannot improve the patch by quietly editing the
    scan. Three things are frozen that used to train:

    * the backing behind the face. Adam rescales every parameter's step, so
      splats that barely show took full-size steps and changed as much as
      the visible ones (DC 0.200 against 0.220);
    * the SH bands. One view cannot tell a view-dependent term from the DC
      term, and the optimiser used them as spare colour freedom;
    * runaway opacity, which now pays for leaving where the geometric pass
      put it.

    Geometry stays where the geometric pass put it. Letting positions move
    would let the optimiser chase the inpainted image off the surface.
    """
    torch, _ = gpu_require()
    dev = torch.device("cuda")
    sub = filled.take(near)

    def T(a, grad=False):
        t = torch.tensor(np.ascontiguousarray(a), device=dev, dtype=torch.float32)
        return t.requires_grad_(True) if grad else t

    means = T(sub.xyz)
    quats = T(sub.quat)
    sh0 = np.concatenate([sub.f_dc[:, None, :], sub.f_rest], axis=1)
    ctx = ~train

    p_dc = T(sh0[train, :1], True)
    p_logit = T(sub.opacity[train], True)
    p_logscale = T(sub.scale[train], True)
    rest_t = (T(sh0[train, 1:], deep.train_sh) if sh0.shape[1] > 1 else None)
    f_sh = T(sh0[ctx])
    f_op = T(sub.alpha[ctx])
    f_sc = T(sub.sigma[ctx])
    idx_new = torch.tensor(np.where(train)[0], device=dev, dtype=torch.long)
    idx_ctx = torch.tensor(np.where(ctx)[0], device=dev, dtype=torch.long)
    n_all = len(near)
    logit0, logscale0 = p_logit.detach().clone(), p_logscale.detach().clone()

    def assemble():
        sh = torch.zeros((n_all, sh0.shape[1], 3), device=dev)
        sh = sh.index_copy(0, idx_ctx, f_sh)
        own = p_dc if rest_t is None else torch.cat([p_dc, rest_t], dim=1)
        sh = sh.index_copy(0, idx_new, own)
        op = torch.zeros(n_all, device=dev)
        op = op.index_copy(0, idx_ctx, f_op)
        op = op.index_copy(0, idx_new, torch.sigmoid(p_logit))
        sc = torch.zeros((n_all, 3), device=dev)
        sc = sc.index_copy(0, idx_ctx, f_sc)
        sc = sc.index_copy(0, idx_new, torch.exp(p_logscale))
        return sh, op, sc

    def draw(sh, op, sc):
        out, _, _ = gpu_render({"means": means, "quats": quats, "scales": sc,
                                "opacities": op, "colors": sh}, cams,
                               sh_degree=deg, antialiased=False, with_depth=False)
        return out

    with torch.no_grad():
        anchors = draw(*assemble()).clone()

    tgt = T(target)
    m_ref = T(mask.astype(np.float32))[..., None]
    outside = T(np.stack([~h for h in hole_masks]).astype(np.float32))[..., None]
    sup = []
    for k, (w_img, w_ok) in enumerate(warped):
        if w_img is not None and deep.multiview > 0:
            sup.append((k + 1, T(w_img), T(w_ok.astype(np.float32))[..., None]))
    # pictures made for further views: supervised like the main one, each
    # in its own view (index into ``cams``)
    ex = [(k, T(img), T(ok.astype(np.float32))[..., None]) for k, img, ok in extra]
    w_ex = float(getattr(deep, "extra_weight", 1.0))

    groups = [{"params": [rest_t], "lr": 2.5e-4}] if (rest_t is not None
                                                       and deep.train_sh) else []
    opt = torch.optim.Adam(groups + [
        {"params": [p_dc], "lr": 2.5e-3},
        {"params": [p_logit], "lr": 5e-3},
        {"params": [p_logscale], "lr": 1e-3},
    ])
    steps = max(deep.steps, 1)
    history = []
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        out = draw(*assemble())
        loss_in = ((out[0] - tgt).abs() * m_ref).sum() / m_ref.sum().clamp_min(1)
        loss_mv = torch.zeros((), device=dev)
        for k, w_img, w_ok in sup:
            loss_mv = loss_mv + ((out[k] - w_img).abs() * w_ok).sum() \
                / w_ok.sum().clamp_min(1)
        if sup:
            loss_mv = loss_mv / len(sup)
        loss_ex = torch.zeros((), device=dev)
        for k, e_img, e_ok in ex:
            loss_ex = loss_ex + ((out[k] - e_img).abs() * e_ok).sum() / e_ok.sum().clamp_min(1)
        if ex:
            loss_ex = loss_ex / len(ex)
        loss_out = ((out - anchors).abs() * outside).sum() / outside.sum().clamp_min(1)
        reg = 0.5 * (p_logscale - logscale0).pow(2).mean() + \
            0.05 * (p_logit - logit0).pow(2).mean()
        loss = loss_in + deep.multiview * loss_mv + w_ex * loss_ex + 3.0 * loss_out + reg
        loss.backward()
        opt.step()
        if i % 25 == 0 or i == steps - 1:
            history.append((float(loss_in.detach()), float(loss_mv.detach()),
                            float(loss_out.detach())))
            say("optimizing (%d/%d)" % (i + 1, steps),
                0.55 + 0.25 * (i + 1) / steps)

    with torch.no_grad():
        work = filled.copy()
        gi = near[train]
        work.f_dc[gi] = p_dc.detach()[:, 0].cpu().numpy().astype(np.float32)
        if rest_t is not None and deep.train_sh:
            work.f_rest[gi] = rest_t.detach().cpu().numpy().astype(np.float32)
        work.opacity[gi] = p_logit.detach().cpu().numpy().astype(np.float32)
        work.scale[gi] = p_logscale.detach().cpu().numpy().astype(np.float32)
    return work, history, "gsplat (%d steps)" % steps


def gpu_tensors(*a, **kw):
    from . import gpu
    return gpu.to_tensors(*a, **kw)


def gpu_require():
    from . import gpu
    return gpu.require()


def gpu_render(*a, **kw):
    from . import gpu
    return gpu.render(*a, **kw)


def _gpu_draw(s, near, cams, deg):
    """Render a set from cams on the GPU: rgb, alpha, expected depth (numpy)."""
    import torch
    p = gpu_tensors(s.take(near), torch.device("cuda"))
    with torch.no_grad():
        rgb, a, d = gpu_render(p, cams, sh_degree=deg, antialiased=False)
    return (rgb.cpu().numpy().astype(np.float64), a[..., 0].cpu().numpy(),
            d.cpu().numpy().astype(np.float64))


# -------------------------------------------------------------------- main

def _reproject(image, src, dst, shift=0.0):
    """``image`` seen by camera ``src``, resampled into camera ``dst``.

    Both cameras stand at the same eye, so the map between them is a pure
    rotation and exact: a homography, no depth needed. Pixels ``src`` does not
    see come back with ``ok`` False.
    """
    from scipy import ndimage
    ys, xs = np.mgrid[0:dst.height, 0:dst.width].astype(np.float64)
    # ``shift``: where the destination renderer puts its pixel centres -- 0.5
    # less on the CPU rasteriser than on gsplat, as everywhere else in here.
    rays = np.stack([xs + 0.5 - shift, ys + 0.5 - shift, np.ones_like(xs)], -1) \
        @ np.linalg.inv(dst.K).T
    world = rays @ dst.view[:3, :3]                    # camera -> world directions
    cam = world @ src.view[:3, :3].T
    front = cam[..., 2] > 1e-9
    pix = cam @ src.K.T
    u = pix[..., 0] / np.where(front, pix[..., 2], 1.0) - 0.5
    v = pix[..., 1] / np.where(front, pix[..., 2], 1.0) - 0.5
    ok = front & (u >= 0) & (u <= src.width - 1) & (v >= 0) & (v <= src.height - 1)
    img = np.asarray(image, np.float64)
    out = np.stack([ndimage.map_coordinates(img[..., c], [v, u], order=1, mode="nearest")
                    for c in range(3)], -1)
    return np.clip(out, 0, 255).astype(np.uint8), ok


def _relief_from_picture(filled, new_idx, near, ctx, target, geo, mask, cam, sigma,
                         cams, deg):
    """Scale the new splats' relief by how much fine detail the picture keeps."""
    from scipy import ndimage

    def energy(img):
        g = img.mean(-1)
        hi = g - ndimage.gaussian_filter(g, sigma)
        return ndimage.gaussian_filter(hi * hi, 2.0 * sigma)

    m = mask.astype(np.float64)
    e_pic, e_geo = energy(target), energy(geo)
    ratio = np.sqrt(e_pic / np.maximum(e_geo, 1e-12))
    ratio = np.where(m > 0, np.clip(ratio, 0.0, 1.0), 1.0)
    origin, basis = ctx["origin"], ctx["basis"]
    cu0, cv0 = ctx["uv_centre"]
    hf = ctx["surface"]
    xyz = filled.xyz.astype(np.float64).copy()
    uvh = (xyz[new_idx] - origin) @ basis.T
    h_s = np.asarray(hf(uvh[:, 0] - cu0, uvh[:, 1] - cv0), float).reshape(-1)
    pc = xyz[new_idx] @ cam.view[:3, :3].T + cam.view[:3, 3]
    uvp = pc @ cam.K.T
    u = np.clip((uvp[:, 0] / np.maximum(uvp[:, 2], 1e-9)).astype(int), 0, cam.width - 1)
    v = np.clip((uvp[:, 1] / np.maximum(uvp[:, 2], 1e-9)).astype(int), 0, cam.height - 1)
    k = ratio[v, u]
    uvh[:, 2] = h_s + (uvh[:, 2] - h_s) * k
    xyz[new_idx] = origin + uvh @ basis
    out = filled.copy()
    out.xyz = xyz.astype(filled.xyz.dtype)
    g_rgb, g_a, g_d = _gpu_draw(out, near, cams, deg)
    return out, g_rgb[0], g_rgb, g_a, g_d


def deep_repair(s: SplatSet, selection, params: Optional[RepairParams] = None,
                deep: Optional[DeepParams] = None,
                base: Optional[RepairResult] = None,
                on_progress=None, reference: Optional[dict] = None,
                references: Optional[list] = None) -> RepairResult:
    """Geometric repair, then let an inpainter decide the patch's appearance.

    ``reference`` takes the inpainter's place: a picture of the object with
    the hole already filled -- in Photoshop, say -- and the camera it was
    rendered from (``image``, ``eye``, ``target``, ``up``, ``fov``, the
    vertical field of view in degrees). The patch's reference view is then
    put at that camera's eye, so the picture maps onto it exactly.

    ``references``: several such pictures of the same fill from different
    eyes (a multi-view sheet). The first is ``reference``; each further one
    becomes a train view of its own, standing at its picture's eye so that
    it too maps on exactly, and supervised like the first inside the patch
    (``DeepParams.extra_weight``). A picture that does not show the whole
    patch is left out. The held-out check views stay where they were.

    With a painted volume in the quick fill's context the masks are what
    was painted (`volume_mask`) and the filled cells, not the cells alone.
    """
    import time
    deep = deep or DeepParams()
    say = on_progress or (lambda *_: None)
    t0 = time.time()
    stamps = {}
    refs = [r for r in (references or []) if r is not None]
    if reference is None and refs:
        reference = refs[0]
    extras = refs[1:] if (refs and refs[0] is reference) else refs

    say("geometry", 0.05)
    base = base or repair(s, selection, params)
    ctx = base.context
    if not ctx:
        raise RepairError("The geometric pass returned no frame")
    stamps["geometrie"] = time.time() - t0

    if reference is None:
        inpaint, inpainter_name = load_inpainter(deep.inpainter)
    else:
        inpaint, inpainter_name = None, "image"
    say("inpainter: " + inpainter_name, 0.15)

    filled = base.apply_to(s)
    n_kept = s.n - len(base.removed)
    new_idx_global = np.arange(n_kept, filled.n)
    deg = int(s.sh_degree)

    # Camera straight down the patch normal, oriented away from the material.
    origin, basis = ctx["origin"], ctx["basis"]
    cu0, cv0 = ctx["uv_centre"]
    hf = ctx["surface"]
    radius = float(ctx["radius"])
    patch_centre = origin + np.array([cu0, cv0, float(hf(cu0 - cu0, cv0 - cv0))]) @ basis
    normal = basis[2] * float(ctx["outward"])
    spacing = float(ctx["spacing"])

    # Resolution follows the patch, not a constant. At a fixed 512 pixels a
    # splat spacing is 1.5 pixels on the shoe, 4.7 on a hole of thirty
    # spacings and half a pixel on one of three hundred -- below the splat
    # field's own resolution, so its texture never reaches the reference and
    # cannot be fitted. Two pixels per spacing, within what LaMa handles.
    fov = 35.0
    if deep.resolution > 0:
        W = H = int(deep.resolution)
    else:
        W = int(np.clip(4.0 * radius / (deep.frame_fraction * max(spacing, 1e-9)),
                        256, 1024)) // 8 * 8
        H = W
    dist = radius / max(np.tan(np.radians(fov) * 0.5) * deep.frame_fraction, 1e-6)
    up = basis[1]
    eye = patch_centre + normal * dist
    if reference is not None:
        # From the picture's own eye, framed on the patch as usual.
        eye = np.asarray(reference["eye"], float)
        dist = float(np.linalg.norm(eye - patch_centre))
        fov = float(np.degrees(2.0 * np.arctan(radius / max(deep.frame_fraction * dist, 1e-9))))
        up = np.asarray(reference.get("up", up), float)

    near = _visible_subset(filled, patch_centre, radius * 4.0)
    is_new = np.zeros(filled.n, bool)
    is_new[new_idx_global] = True
    local_new = is_new[near]
    if local_new.sum() == 0:
        raise RepairError("No new Gaussians within reach of the view")
    is_back = np.zeros(filled.n, bool)
    if base.backing is not None and len(base.backing) == len(new_idx_global):
        is_back[new_idx_global[base.backing]] = True
    train = local_new & ~is_back[near] if not deep.train_backing else local_new
    # Copies made along an edge (`unfold`) are the scan moved along it: the
    # fit leaves them as they are. LaMa continued the hut's corner board as a
    # dark stripe where the copy had carried it on.
    fz_mask = (ctx or {}).get("unfold_frozen")
    if fz_mask is not None and len(fz_mask) == len(new_idx_global):
        is_fz = np.zeros(filled.n, bool)
        is_fz[new_idx_global[np.asarray(fz_mask, bool)]] = True
        train = train & ~is_fz[near]
        if not train.any():
            raise RepairError("The spot was copied entirely along the edge")

    from . import gpu
    gpu_ok, gpu_why = gpu.available()
    use_gpu = deep.backend == "gpu" or (deep.backend == "auto" and gpu_ok)
    if deep.backend == "gpu" and not gpu_ok:
        raise DeepUnavailable("GPU optimization is not available: " + gpu_why)

    # One renderer from end to end. The reference, the mask and the "before /
    # after" used to come from the CPU rasteriser while the fit ran in gsplat's
    # antialiased mode with pixel centres half a pixel away -- 0.02 per channel
    # of disagreement before anything was fitted, as large as the residual the
    # fit ends on, and a renderer the viewport never uses. The two CPU renders
    # were also eleven of the sixteen seconds.
    nrm = normal / max(np.linalg.norm(normal), 1e-12)
    # The tilt axes lie in the patch plane. A reference camera's own "up"
    # (world up over a floor, say) can run along the normal; crossed with it
    # the side axis vanished, every tilted view collapsed onto the straight
    # one and the side check compared the fit with itself.
    up_t = up - nrm * float(np.dot(up, nrm))
    if np.linalg.norm(up_t) < 1e-3:
        up_t = np.asarray(basis[1], float)
    up_t = up_t / max(np.linalg.norm(up_t), 1e-12)
    side = np.cross(nrm, up_t)
    side /= max(np.linalg.norm(side), 1e-9)
    t1 = np.tan(np.radians(deep.tilt_deg))
    t2 = np.tan(np.radians(deep.tilt_deg * 1.8))

    def tilted(off):
        d = nrm + off
        d /= np.linalg.norm(d)
        return gpu.look_at_camera(patch_centre + d * dist, patch_centre, up, W, H, fov)

    ref_cam = gpu.look_at_camera(eye, patch_centre, up, W, H, fov)
    train_cams = [ref_cam, tilted(side * t1), tilted(-side * t1)]
    # Views the fit never sees, for judging it: along the other axis and
    # further over. A check made from the views it was fitted to can only
    # agree with itself, which is what the old guard did.
    check_cams = [tilted(up_t * t2), tilted(-up_t * t2)]
    n_tr = len(train_cams)
    n_ck = len(check_cams)
    hole_pts = _hole_points(ctx)
    vol = ctx.get("volume")
    # Further pictures: a camera at each picture's eye, framed on the patch
    # as the main one is (only on the GPU; the CPU fit has one view).
    ex_cams, ex_src = [], []
    if use_gpu:
        for r_k in extras:
            e_k = np.asarray(r_k["eye"], float)
            d_k = float(np.linalg.norm(e_k - patch_centre))
            if d_k < 1e-9:
                continue
            f_k = float(np.degrees(2.0 * np.arctan(radius / max(deep.frame_fraction * d_k, 1e-9))))
            u_k = np.asarray(r_k.get("up", up), float)
            ex_cams.append(gpu.look_at_camera(e_k, patch_centre, u_k, W, H, f_k))
            ex_src.append(r_k)
    all_cams = train_cams + check_cams + ex_cams

    say("rendering reference view", 0.25)
    if use_gpu:
        g_rgb, g_a, g_d = _gpu_draw(filled, near, all_cams, deg)
        img0, alpha0 = g_rgb[0], g_a[0]
        shift = 0.0
    else:
        img0, alpha0, _, _ = _raster(filled, near, eye, patch_centre, up, W, H, fov)
        shift = 0.5                          # the CPU rasteriser's pixel centres
    stamps["referenz"] = time.time() - t0
    rgb0 = (np.clip(img0, 0, 1) * 255 + 0.5).astype(np.uint8)

    def patch_mask(cam, k=None, grow=deep.mask_grow, sh=0.0):
        """The filled cells in ``cam``, and with a volume what was painted."""
        m = _mask_from_points(hole_pts, cam, W, H, grow, sh)
        if vol is not None:
            dk = None
            if use_gpu and k is not None:
                dk = np.where(g_a[k] > 0.5, g_d[k], np.inf)
            m = m | volume_mask(vol, cam, W, H, dk, grow=max(2, int(round(W / 250))))
        return m

    mask = patch_mask(ref_cam, 0, sh=shift)
    if mask.sum() < 64:
        raise RepairError("The spot is too small in the reference view")
    if mask.mean() > 0.9:
        raise RepairError("The spot fills almost the whole image")

    say("filling in 2D", 0.4)
    if reference is not None:
        img_ref = np.asarray(reference["image"])[..., :3]
        src = gpu.look_at_camera(np.asarray(reference["eye"], float),
                                 np.asarray(reference["target"], float),
                                 np.asarray(reference.get("up", up), float),
                                 img_ref.shape[1], img_ref.shape[0], float(reference["fov"]))
        target_u8, seen = _reproject(img_ref, src, ref_cam, shift)
        if float(seen[mask.astype(bool)].mean()) < 0.95:
            raise RepairError("The image does not show the whole spot")
    else:
        lama_mask = mask
        if deep.hide_dark > 0:
            lama_mask = mask | _dark_rim(rgb0, mask.astype(bool), deep.hide_dark)
        target_u8 = inpaint(rgb0, (lama_mask * 255).astype(np.uint8))
    target = target_u8.astype(np.float64) / 255.0
    px_sp = spacing * W / (2.0 * dist * np.tan(np.radians(fov) * 0.5))   # pixels per spacing
    # a picture whose grain LaMa lost and the copy put back: keep the copy
    keep_copy = (use_gpu and reference is not None and deep.copy_structure > 0
                 and float(reference.get("grain") or 0.0) >= max(deep.copy_grain, 1e-6))
    if reference is not None:
        # The picture is the answer: no grain from the copy mixed in -- it put
        # the lattice's grain back into a plaque the picture had made smooth.
        detail_used = 0.0
        # The shape follows the picture too -- unless the picture's grain
        # had to come from the copy: then the copy's shape is the better
        # guess (`_tone_onto_copy`).
        # The copy laid lattice relief
        # where the picture now shows a smooth plaque, and from the side the
        # ridges showed through the new colour. Where the picture has less
        # fine detail than the copy, the new splats' relief over the fitted
        # surface is scaled down by that ratio; where it has as much, it
        # stays. Then the views are rendered again from the new shape.
        if use_gpu and len(new_idx_global) and not keep_copy:
            face_new = new_idx_global[~is_back[new_idx_global]]     # not the slab behind
            filled, img0, g_rgb, g_a, g_d = _relief_from_picture(
                filled, face_new, near, ctx, target, np.clip(img0, 0, 1), mask,
                ref_cam, max(2.0 * px_sp, 1.0), all_cams, deg)
            rgb0 = (np.clip(img0, 0, 1) * 255 + 0.5).astype(np.uint8)
    else:
        target, detail_used = _detail_split(target, np.clip(img0, 0, 1), mask,
                               sigma=max(2.0 * px_sp, 1.0), amount=deep.detail)
    if deep.tone_scale > 0:
        px = spacing * W / (2.0 * dist * np.tan(np.radians(fov) * 0.5))
        target = _tone_from_geo(target, np.clip(img0, 0, 1), mask,
                                max(deep.tone_scale * px, 1.0))
    stamps["inpaint"] = time.time() - t0

    resid_mask = mask[..., None].astype(np.float64)
    before = float(np.abs((target - img0) * resid_mask).sum() / max(mask.sum(), 1))

    off_geo = off_gen = None
    n_from_picture, pic_bound = 0, None
    stamps_extra = None
    # a picture the user made is the answer, not a guess to be second-guessed
    mushy = reference is None and deep.mush > 0 and detail_used >= deep.mush
    if mushy:
        work, history, how, img_after = filled, [], None, np.clip(img0, 0, 1)
    elif use_gpu:
        tol = 4.0 * spacing
        warped = [_warp(target, g_d[0], mask, ref_cam, c, g_d[k + 1], tol)
                  for k, c in enumerate(train_cams[1:])]
        checks = [_warp(target, g_d[0], mask, ref_cam, c, g_d[k + n_tr], tol)
                  for k, c in enumerate(check_cams)]
        hole_masks = [patch_mask(c, k) if vol is not None else
                      _mask_from_points(hole_pts, c, W, H, deep.mask_grow)
                      for k, c in enumerate(train_cams)]
        # each further picture, mapped into its own view
        extra, fit_cams = [], []
        for j, (c, r_k) in enumerate(zip(ex_cams, ex_src)):
            k = n_tr + n_ck + j
            img_k = np.asarray(r_k["image"])[..., :3]
            src_k = gpu.look_at_camera(np.asarray(r_k["eye"], float),
                                       np.asarray(r_k["target"], float),
                                       np.asarray(r_k.get("up", up), float),
                                       img_k.shape[1], img_k.shape[0], float(r_k["fov"]))
            t_k, seen_k = _reproject(img_k, src_k, c, shift)
            m_k = patch_mask(c, k)
            if m_k.sum() < 64 or float(seen_k[m_k].mean()) < 0.95:
                continue
            extra.append((len(train_cams) + len(fit_cams), t_k.astype(np.float64) / 255.0,
                          m_k & seen_k))
            hole_masks.append(m_k)
            fit_cams.append((c, k))
        say("optimizing on the GPU", 0.55)
        filled, n_from_picture, pic_bound = _colour_from_picture(
            filled, near[train], ref_cam, target, mask, px_sp, shift)
        if keep_copy:
            sig = max(deep.copy_structure * px_sp, 1.0)
            work = _tone_onto_copy(filled, near[train], near, ref_cam, target, mask, sig, deg)
            history, how = [], "copy, tone from the image (%.1f px)" % sig
        else:
            work, history, how = _refine_gpu(
                filled, near, train, local_new & ~train, train_cams + [c for c, _ in fit_cams],
                target, mask, warped, hole_masks, deep, deg, say, extra=extra)
        a_rgb, _, _ = _gpu_draw(work, near, [ref_cam] + check_cams
                                + [c for c, _ in fit_cams], deg)
        img_after = a_rgb[0]
        if extra:
            # how close each further view came to its own picture, before and after
            e_geo, e_deep = [], []
            for j, (k_t, img_t, ok_t) in enumerate(extra):
                e_geo.append(float(np.abs(g_rgb[fit_cams[j][1]] - img_t)[ok_t].mean()))
                e_deep.append(float(np.abs(a_rgb[1 + n_ck + j] - img_t)[ok_t].mean()))
            stamps_extra = (len(extra), float(np.mean(e_geo)), float(np.mean(e_deep)))
        else:
            stamps_extra = None

        def off_axis(rgbs):
            errs = []
            for k, (w_img, w_ok) in enumerate(checks):
                if w_img is None or w_ok.sum() < 200:
                    continue
                errs.append(float(np.abs(rgbs[k] - w_img)[w_ok].mean()))
            return float(np.mean(errs)) if errs else None
        off_geo = off_axis(g_rgb[n_tr:])
        off_gen = off_axis(a_rgb[1:])
    else:
        say("fitting colors", 0.55)
        work, history, how = _refine_cpu(
            filled, near, local_new, eye, patch_centre, up, W, H, fov,
            target, mask, img0, before, deep, say)
        img_after, _, _, _ = _raster(work, near, eye, patch_centre, up, W, H, fov)
    stamps["fit"] = time.time() - t0
    after = float(np.abs((target - img_after) * resid_mask).sum() / max(mask.sum(), 1))

    # --- guard -------------------------------------------------------------
    # Two questions the fit was not optimised to answer. From views it never
    # saw, does the patch agree with the inpainted picture carried there
    # through the surface at least as well as the geometric fill did? And did
    # it get closer to that picture at all from the front? The old guard
    # measured drift outside the patch -- where every splat is frozen -- and
    # could not fire; in eighteen runs it never did.
    improved = after < before and not mushy
    worse_aside = (deep.guard and off_geo is not None and off_gen is not None
                   and off_gen > off_geo * 1.05)
    use_deep = improved and not worse_aside

    out = base
    if use_deep:
        added = work.take(np.arange(n_kept, work.n))
        back = (base.backing if base.backing is not None
                and len(base.backing) == added.n else None)
        out = RepairResult(removed=base.removed, added=added, backing=back,
                           info=dict(base.info), context=dict(base.context))
    else:
        out = RepairResult(removed=base.removed, added=base.added,
                           backing=base.backing, info=dict(base.info),
                           context=dict(base.context))
    out.info.update({
        "deep": bool(use_deep),
        "inpainter": inpainter_name,
        "backend": how,
        "resolution": int(W), "detail": round(float(detail_used), 2),
        "fromPicture": int(n_from_picture),
        "keptCopy": bool(keep_copy),
        "pictureBound": None if pic_bound is None else [round(float(b), 3) for b in pic_bound],
        "maskPixels": int(mask.sum()),
        # how far the result is from the inpainted picture, as a share of
        # full brightness; per channel, so it reads as a percentage
        "fit": round(after / 3.0, 4),
        "errBefore": round(before, 5),
        "errAfter": round(after, 5),
        "offAxisGeo": None if off_geo is None else round(off_geo, 5),
        "offAxisDeep": None if off_gen is None else round(off_gen, 5),
        # further pictures used, and how far each view ended from its own
        # picture (per channel, 0..1), before and after the fit
        "views": 1 + (stamps_extra[0] if stamps_extra else 0),
        "fitExtraGeo": None if not stamps_extra else round(stamps_extra[1], 4),
        "fitExtra": None if not stamps_extra else round(stamps_extra[2], 4),
        "seconds": {k: round(v, 2) for k, v in stamps.items()},
        "history": [list(np.round(np.atleast_1d(h), 5)) for h in history],
        "rejected": (None if use_deep else
                     ("%s returned only mush, the copied fill stays" % inpainter_name
                      if mushy else
                      "no improvement" if not improved else
                      "worse from the side than the geometric fill")),
    })
    # "picture", not "target": that key is the quick fill's hole grid
    # (`repair`), which `_hole_points` reads when this result is the base of
    # another fill
    out.context.update({"reference": rgb0, "mask": mask, "picture": target_u8})
    say("done", 1.0)
    return out
