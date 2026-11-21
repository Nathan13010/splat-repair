"""Fill a picture in the user's own Photoshop.

The app renders a view and a mask; Photoshop fills the masked area -- the
same Content-Aware Fill (or Generative Fill) a person would run by hand --
and the result comes back as a picture for ``deep_repair(reference=...)``.

Driven over COM (Windows) with an ExtendScript file, so nothing has to be
installed in Photoshop. The script opens the picture, turns the mask into a
selection (from a layer's transparency), fills, saves a PNG and closes
without touching anything else the user has open.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Optional

import numpy as np

MODES = ("content_aware", "generative")

_JSX = r"""
#target photoshop
(function () {
    var cfg = %(cfg)s;
    var oldDialogs = app.displayDialogs;
    app.displayDialogs = DialogModes.NO;
    var oldUnits = app.preferences.rulerUnits;
    app.preferences.rulerUnits = Units.PIXELS;
    var log = new File(cfg.log);
    function done(msg) {
        log.open("w"); log.write(msg); log.close();
    }
    var doc = null, maskDoc = null;
    try {
        doc = app.open(new File(cfg.image));
        maskDoc = app.open(new File(cfg.mask));
        var lay = maskDoc.activeLayer.duplicate(doc, ElementPlacement.PLACEATBEGINNING);
        maskDoc.close(SaveOptions.DONOTSAVECHANGES); maskDoc = null;
        app.activeDocument = doc;
        doc.activeLayer = lay;
        // selection = transparency of the mask layer
        var d = new ActionDescriptor();
        var r = new ActionReference();
        r.putProperty(charIDToTypeID("Chnl"), charIDToTypeID("fsel"));
        d.putReference(charIDToTypeID("null"), r);
        var r2 = new ActionReference();
        r2.putEnumerated(charIDToTypeID("Chnl"), charIDToTypeID("Chnl"), charIDToTypeID("Trsp"));
        d.putReference(charIDToTypeID("T   "), r2);
        executeAction(charIDToTypeID("setd"), d, DialogModes.NO);
        lay.remove();
        doc.activeLayer = doc.layers[doc.layers.length - 1];
        if (cfg.mode == "generative") {
            var g = new ActionDescriptor();
            var gr = new ActionReference();
            gr.putEnumerated(stringIDToTypeID("document"), stringIDToTypeID("ordinal"), stringIDToTypeID("targetEnum"));
            g.putReference(stringIDToTypeID("null"), gr);
            g.putString(stringIDToTypeID("prompt"), "");
            g.putString(stringIDToTypeID("serviceID"), "clio");
            g.putString(stringIDToTypeID("workflow"), "in_painting");
            executeAction(stringIDToTypeID("syntheticFill"), g, DialogModes.NO);
            doc.flatten();
        } else {
            var f = new ActionDescriptor();
            f.putEnumerated(charIDToTypeID("Usng"), charIDToTypeID("FlCn"), stringIDToTypeID("contentAware"));
            f.putBoolean(stringIDToTypeID("contentAwareColorAdaptationFill"), true);
            f.putUnitDouble(charIDToTypeID("Opct"), charIDToTypeID("#Prc"), 100);
            f.putEnumerated(charIDToTypeID("Md  "), charIDToTypeID("BlnM"), charIDToTypeID("Nrml"));
            executeAction(charIDToTypeID("Fl  "), f, DialogModes.NO);
        }
        doc.selection.deselect();
        var opt = new PNGSaveOptions();
        doc.saveAs(new File(cfg.out), opt, true, Extension.LOWERCASE);
        doc.close(SaveOptions.DONOTSAVECHANGES); doc = null;
        done("ok");
    } catch (e) {
        try { if (maskDoc) maskDoc.close(SaveOptions.DONOTSAVECHANGES); } catch (e2) {}
        try { if (doc) doc.close(SaveOptions.DONOTSAVECHANGES); } catch (e3) {}
        done("error: " + e);
    }
    app.preferences.rulerUnits = oldUnits;
    app.displayDialogs = oldDialogs;
})();
"""


class PhotoshopUnavailable(RuntimeError):
    """Photoshop is missing, unreachable, or did not answer in time."""


class PhotoshopFillFailed(PhotoshopUnavailable):
    """Photoshop ran the script and reported an error (no credits, offline)."""

    def __init__(self, reason: str):
        super().__init__("Photoshop fill failed: %s" % reason)
        self.reason = reason


def available() -> tuple[bool, str]:
    """Is a scriptable Photoshop registered on this machine?"""
    if sys.platform != "win32":
        return False, "Windows only"
    try:
        import winreg
        winreg.CloseKey(winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, r"Photoshop.Application\CLSID"))
        return True, ""
    except OSError:
        return False, "Photoshop is not installed"


def fill(rgb: np.ndarray, mask: np.ndarray, mode: str = "content_aware",
         timeout: float = 240.0, workdir: Optional[str] = None) -> np.ndarray:
    """Return ``rgb`` with the area where ``mask`` is set filled by Photoshop."""
    from PIL import Image
    if mode not in MODES:
        raise ValueError("the mode must be one of %s" % (MODES,))
    ok, why = available()
    if not ok:
        raise PhotoshopUnavailable(why)
    rgb = np.ascontiguousarray(np.asarray(rgb)[..., :3], np.uint8)
    m = np.asarray(mask) > 0
    if m.shape != rgb.shape[:2]:
        raise ValueError("mask and image differ in size")
    own_tmp = workdir is None
    tmp = workdir or tempfile.mkdtemp(prefix="splatrepair_ps_")
    try:
        out = _run(rgb, m, mode, timeout, tmp)
    except PhotoshopFillFailed:
        if own_tmp:
            shutil.rmtree(tmp, ignore_errors=True)
        raise
    # (after a timeout Photoshop may still write there: the folder stays)
    if own_tmp:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def _run(rgb, m, mode, timeout, tmp):
    from PIL import Image
    img_p = os.path.join(tmp, "image.png")
    mask_p = os.path.join(tmp, "mask.png")
    out_p = os.path.join(tmp, "filled.png")
    log_p = os.path.join(tmp, "log.txt")
    jsx_p = os.path.join(tmp, "fill.jsx")
    for p in (out_p, log_p):
        if os.path.exists(p):
            os.remove(p)
    Image.fromarray(rgb).save(img_p, compress_level=1)
    rgba = np.zeros(m.shape + (4,), np.uint8)
    rgba[m] = (255, 255, 255, 255)
    Image.fromarray(rgba, "RGBA").save(mask_p)
    cfg = {k: v.replace("\\", "/") for k, v in
           {"image": img_p, "mask": mask_p, "out": out_p, "log": log_p}.items()}
    cfg["mode"] = mode
    with open(jsx_p, "w", encoding="utf-8") as f:
        f.write(_JSX % {"cfg": json.dumps(cfg)})
    # The path goes in through the environment, so no character in it (a quote
    # in a user name) can end the string and become PowerShell code.
    ps = ("$ErrorActionPreference='Stop';"
          "$a = New-Object -ComObject Photoshop.Application;"
          "$null = $a.DoJavaScriptFile($env:SPLAT_REPAIR_JSX)")
    env = dict(os.environ, SPLAT_REPAIR_JSX=jsx_p)
    t = time.time()
    try:
        # PowerShell writes errors in the console's OEM code page (cp850 on
        # German Windows); decoded as cp1252 an umlaut raised in the reader
        # thread and the message was lost.
        r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                           capture_output=True, text=True, encoding="oem", errors="replace",
                           timeout=timeout, env=env,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except subprocess.TimeoutExpired as e:
        raise PhotoshopUnavailable("Photoshop did not answer within %d s" % timeout) from e
    msg = open(log_p, encoding="utf-8", errors="replace").read() if os.path.exists(log_p) else ""
    if msg != "ok" or not os.path.exists(out_p):
        if not msg:
            # The script never ran: COM could not reach or start Photoshop.
            detail = (r.stderr or r.stdout or "").strip()[:300]
            raise PhotoshopUnavailable("Photoshop could not be reached: %s" % (detail or "no message"))
        raise PhotoshopFillFailed(msg.replace("error: ", "", 1)[:300] or "no message")
    out = np.asarray(Image.open(out_p).convert("RGB"))
    if out.shape != rgb.shape:
        raise PhotoshopUnavailable("Photoshop returned an image of a different size")
    res = rgb.copy()
    res[m] = out[m]
    fill.last_seconds = time.time() - t
    return res
