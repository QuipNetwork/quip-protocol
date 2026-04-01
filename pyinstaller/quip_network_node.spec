# -*- mode: python ; coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors

"""PyInstaller spec for quip-network-node frozen binary."""

import glob
import importlib
import os
import platform

from PyInstaller.utils.hooks import collect_submodules, copy_metadata

block_cipher = None
proj_root = os.path.abspath(os.path.join(SPECPATH, ".."))

# ---------------------------------------------------------------------------
# Data files
# ---------------------------------------------------------------------------
datas = [
    # Genesis block configurations
    (os.path.join(proj_root, "genesis_block.json"), "."),
    (os.path.join(proj_root, "genesis_block_public.json"), "."),
    # GPU kernel source files
    (os.path.join(proj_root, "GPU", "*.cu"), "GPU"),
    (os.path.join(proj_root, "GPU", "*.metal"), "GPU"),
    # D-Wave hardware topology graphs
    (
        os.path.join(proj_root, "dwave_topologies", "topologies", "*.json.gz"),
        os.path.join("dwave_topologies", "topologies"),
    ),
]

# Pre-computed minor embeddings (directory may not exist in fresh checkouts)
_embeddings_dir = os.path.join(proj_root, "dwave_topologies", "embeddings")
if os.path.isdir(_embeddings_dir):
    datas.append((_embeddings_dir, os.path.join("dwave_topologies", "embeddings")))

# ---------------------------------------------------------------------------
# Hidden imports: collect all submodules for packages with dynamic/Cython imports
# ---------------------------------------------------------------------------
hiddenimports = []
# Project packages (conditional imports in GPU/__init__.py, dynamic in miner_worker.py)
for pkg in ("shared", "CPU", "GPU", "QPU", "dwave_topologies"):
    hiddenimports += collect_submodules(pkg)
# D-Wave packages with Cython extensions that static analysis misses
for pkg in ("dimod", "minorminer", "dwave"):
    hiddenimports += collect_submodules(pkg)
# Explicitly dynamic imports (inside function bodies)
hiddenimports += [
    "CPU.sa_filtered_miner",
    "QPU.qpu_time_manager",
    "tomli",
]
# D-Wave ecosystem runtime deps missed by static analysis
hiddenimports += [
    "fasteners",
    "homebase",
    "plucky",
    "diskcache",
    "orjson",
]


def _collect_extension_binaries(pkg_name):
    """Find all .so/.pyd in a package, including namespace packages without __init__.py."""
    try:
        pkg = importlib.import_module(pkg_name)
    except ImportError:
        return []
    if not hasattr(pkg, "__file__") or pkg.__file__ is None:
        if hasattr(pkg, "__path__"):
            search_dirs = list(pkg.__path__)
        else:
            return []
    else:
        search_dirs = [os.path.dirname(pkg.__file__)]
    # Find the site-packages root to compute correct relative paths
    # e.g. for dwave.samplers, __path__ = ['.../site-packages/dwave/samplers']
    # We need dest_dir = 'dwave/samplers/greedy', not 'samplers/greedy'
    site_root = None
    top_pkg = pkg_name.split(".")[0]
    for base in search_dirs:
        idx = base.replace(os.sep, "/").rfind("/" + top_pkg + "/")
        if idx >= 0:
            site_root = base[:idx]
            break
        idx = base.replace(os.sep, "/").rfind("/" + top_pkg)
        if idx >= 0:
            site_root = base[:idx]
            break
    exts = (".pyd", ".so")
    result = []
    for base in search_dirs:
        parent = site_root if site_root else os.path.dirname(base)
        for root, _dirs, files in os.walk(base):
            for f in files:
                if any(f.endswith(e) or (e in f) for e in exts):
                    full = os.path.join(root, f)
                    dest_dir = os.path.relpath(root, parent)
                    mod_stem = f.split(".")[0]
                    mod_path = dest_dir.replace(os.sep, ".") + "." + mod_stem
                    result.append((full, dest_dir))
                    if mod_path not in hiddenimports:
                        hiddenimports.append(mod_path)
    return result


# CUDA library filter — keep only cudart + nvrtc, strip everything else.
# The GPU code only uses cp.RawModule for custom CUDA kernels.
_cuda_keep = {"cudart", "nvrtc"}
_cuda_strip = {"cublas", "cufft", "cusparse", "cusolver", "curand", "nccl",
               "cutensor", "cutensormg", "cublaslt", "nvjitlink", "nvfatbin"}

def _is_unwanted_cuda(name):
    n = name.lower()
    return any(lib in n for lib in _cuda_strip) and not any(lib in n for lib in _cuda_keep)

# Collect compiled extensions from packages with namespace sub-packages
extra_binaries = []
for pkg in ("dimod", "minorminer", "dwave.samplers", "dwave.preprocessing"):
    found = _collect_extension_binaries(pkg)
    print(f"  {pkg}: {len(found)} extensions")
    for src, dest in found:
        print(f"    {os.path.basename(src)} -> {dest}")
    extra_binaries += found

# Collect vendored .libs directories (Windows only)
# Python wheels on Windows vendor C++ deps in <package>.libs/ next to the package dir.
# e.g. site-packages/dwave_samplers.libs/ contains DLLs that dwave/samplers/*.pyd needs.
# On Windows, bundle only the vendored DLLs that dwave/minorminer .pyd files need.
# Skip numpy.libs/scipy.libs (huge OpenBLAS) — numpy/scipy work without them in
# the frozen binary because PyInstaller bundles the .pyd files directly.
# Also skip VC++ runtime (msvcp140.dll etc.) — present on every Windows machine
# with the VC++ Redistributable installed (required prerequisite).
if platform.system() == "Windows":
    _needed_libs = {"dwave_samplers.libs", "dwave_preprocessing.libs",
                    "dimod.libs", "minorminer.libs"}
    import sys as _sys
    for p in _sys.path:
        if not os.path.isdir(p):
            continue
        for entry in os.listdir(p):
            if entry in _needed_libs:
                libs_dir = os.path.join(p, entry)
                if not os.path.isdir(libs_dir):
                    continue
                for dll in os.listdir(libs_dir):
                    full = os.path.join(libs_dir, dll)
                    if os.path.isfile(full) and not _is_unwanted_cuda(dll):
                        extra_binaries.append((full, "."))
                        print(f"  vendored: {dll} (from {entry})")

# Package metadata so importlib.metadata.version("quip-protocol") works
datas += copy_metadata("quip-protocol")

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
# Use version-stamped boot script if available (built by build.sh)
_boot = os.path.join(SPECPATH, "boot_network_node_stamped.py")
if not os.path.exists(_boot):
    _boot = os.path.join(SPECPATH, "boot_network_node.py")

a = Analysis(
    [_boot],
    pathex=[proj_root],
    binaries=extra_binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib",
        "pandas",
        "seaborn",
        "tkinter",
        "torch",
        "torchvision",
        "torchaudio",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Strip unused CUDA libraries that PyInstaller collected during Analysis
_before = len(a.binaries)
a.binaries = [b for b in a.binaries if not _is_unwanted_cuda(b[0])]
_removed = _before - len(a.binaries)
if _removed:
    print(f"  Stripped {_removed} unused CUDA libraries from bundle")

# ---------------------------------------------------------------------------
# Platform-aware binary name: quip-network-node-{os}-{arch}
# ---------------------------------------------------------------------------
_os_map = {"darwin": "macos", "linux": "linux", "windows": "windows"}
_arch_map = {
    "arm64": "arm64",
    "aarch64": "arm64",
    "x86_64": "x86_64",
    "amd64": "x86_64",
}
_os = _os_map.get(platform.system().lower(), platform.system().lower())
_arch = _arch_map.get(platform.machine().lower(), platform.machine().lower())
binary_name = f"quip-network-node-{_os}-{_arch}"

# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=binary_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
