"""Install everything the deep repair needs, in an order that survives.

    core/.venv/Scripts/python.exe core/setup_deep.py

Three things here are not obvious, and each one cost an hour the first time:

1. **`simple-lama-inpainting` depends on torch and will happily replace a CUDA
   build with the CPU one.** Installed with ``--no-deps`` for that reason; its
   real dependencies go in separately.
2. **gsplat from PyPI is pure Python and compiles on first use.** That needs
   MSVC and a CUDA toolkit, and it passes the GCC-only flag ``-Wno-attributes``
   to MSVC, which rejects it (``error D8021``). Patched below. The compile takes
   a couple of minutes, once.
3. **The community Windows wheels are not needed.** They work, but they are a
   third-party binary; the official package plus that one-line patch gets there
   without one.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

TORCH = "2.9.1"
TORCHVISION = "0.24.1"
# the versions this was tested with; newer ones only after someone checks them
GSPLAT = "1.5.3"
SIMPLE_LAMA = "0.1.2"
CUDA = "cu128"
INDEX = "https://download.pytorch.org/whl/" + CUDA


def run(args, **kw):
    print("   $", " ".join(args[:6]), "…" if len(args) > 6 else "")
    return subprocess.run(args, check=False, **kw)


def pip(*args):
    return run([sys.executable, "-m", "pip", *args])


def step(n, title):
    print("\n[%d] %s" % (n, title))


def main():
    print("Python:", sys.version.split()[0], "|", sys.executable)
    if sys.version_info[:2] not in ((3, 10), (3, 11), (3, 12), (3, 13)):
        print("Warning: gsplat expects Python 3.10-3.13")

    step(1, "PyTorch with CUDA")
    pip("install", "--upgrade", "torch==" + TORCH, "torchvision==" + TORCHVISION,
        "--index-url", INDEX)

    step(2, "Build tools for gsplat")
    pip("install", "--upgrade", "ninja", "packaging", "jaxtyping", "rich",
        "typing_extensions")

    step(3, "gsplat (official package, compiles on first run)")
    pip("install", "gsplat==" + GSPLAT)

    step(4, "LaMa, without touching PyTorch")
    # --no-deps: it pins torch loosely and would pull the CPU build over the
    # CUDA one we just installed, silently disabling every GPU path.
    print("   (pip may now report dependency conflicts for simple-lama-inpainting.")
    print("    That is expected: it is installed without its pinned versions on purpose.)")
    pip("install", "--no-deps", "simple-lama-inpainting==" + SIMPLE_LAMA)
    pip("install", "--upgrade", "pillow", "opencv-python-headless", "fire")

    step(5, "Remove the flag MSVC rejects from gsplat")
    patch_gsplat()

    step(6, "Verify")
    verify()


def patch_gsplat():
    try:
        import gsplat
    except Exception as e:                       # noqa: BLE001
        print("   cannot import gsplat:", e)
        return
    path = os.path.join(os.path.dirname(gsplat.__file__), "cuda", "_backend.py")
    try:
        src = open(path, encoding="utf-8").read()
    except OSError as e:
        print("   cannot read _backend.py:", e)
        return
    if "os.name != \"nt\"" in src and "-Wno-attributes" in src:
        print("   already patched")
        return
    new = re.sub(
        r'extra_cflags\s*=\s*\[opt_level,\s*"-Wno-attributes"\]',
        'extra_cflags = [opt_level]\n'
        '        # MSVC reads -Wno-attributes as /Wno-attributes and refuses it\n'
        '        # with "error D8021"; the flag is GCC/Clang only.\n'
        '        if os.name != "nt":\n'
        '            extra_cflags.append("-Wno-attributes")',
        src)
    if new == src:
        print("   nothing to patch (different gsplat version?)")
        return
    open(path, "w", encoding="utf-8").write(new)
    print("   patched:", path)


def verify():
    code = r'''
import sys
ok = True
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda,
          "available", torch.cuda.is_available())
    ok &= torch.cuda.is_available()
except Exception as e:
    print("torch MISSING:", e); ok = False
try:
    sys.path.insert(0, ".")
    from splatcore import gpu
    good, why = gpu.available()
    print("gsplat:", good, "-", why)
    ok &= good
except Exception as e:
    print("gsplat ERROR:", e); ok = False
try:
    from splatcore.deep import load_inpainter
    _, name = load_inpainter("auto")
    print("Inpainter:", name)
except Exception as e:
    print("Inpainter MISSING:", e); ok = False
print("\nDeep Fill is ready" if ok else "\nSetup is not complete yet")
'''
    here = os.path.dirname(os.path.abspath(__file__))
    run([sys.executable, "-c", code], cwd=here)


if __name__ == "__main__":
    main()
