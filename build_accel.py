"""用 MinGW gcc 编译可选的 Cython 加速模块；发布包自带预编译 pyd，只有涉及重编时才需要。"""

import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

HERE = Path(__file__).resolve().parent

MODULES = {
    # 名字: {"sources": [...], "openmp": bool}
    # openmp 默认关：.pyx 里没有 prange，挂 -fopenmp 只会白拖进
    # libgomp-1.dll / libwinpthread-1.dll；要开需保证这两个 DLL 在搜索路径里
    "sfm_accel": {"sources": ["sfm_accel.pyx"], "openmp": False},
}

# ABI 锁定在当前解释器：换版本或换平台都要重新编译
PY_VERSION = f"{sys.version_info[0]}{sys.version_info[1]}"
PY_INCLUDE = Path(sysconfig.get_paths()["include"])
PY_LIBS = Path(sys.base_prefix) / "libs"
PY_LIB = f"python{PY_VERSION}"

# 不加 -ffast-math：增量式流程对舍入敏感，C 结果要与 numpy 逐位一致
CFLAGS = [
    "-O2",
    "-fno-strict-aliasing",
    "-DMS_WIN64",
    "-D_CRT_SECURE_NO_WARNINGS",
    "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
    "-Wall",
    "-Wno-unused-function",
    "-Wno-unused-variable",
    "-Wno-maybe-uninitialized",
]


def _run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def _numpy_include():
    import numpy
    return Path(numpy.get_include())


def _mingw_bin():
    """未找到 gcc 时返回 None。"""
    exe = shutil.which("gcc")
    return Path(exe).resolve().parent if exe else None


def build(name):
    spec = MODULES[name]
    pyx = HERE / f"{name}.pyx"
    c_file = HERE / f"{name}.c"
    pyd = HERE / f"{name}.pyd"

    print(f"\n=== {name} ===")
    _run([sys.executable, "-m", "cython", "-3", "--fast-fail", pyx, "-o", c_file])

    cmd = [
        "gcc", "-shared",
        *CFLAGS,
        f"-I{PY_INCLUDE}",
        f"-I{_numpy_include()}",
        f"-I{HERE}",
    ]
    if spec.get("openmp"):
        cmd.append("-fopenmp")
    cmd += [c_file, f"-L{PY_LIBS}", f"-l{PY_LIB}", "-o", pyd]
    _run(cmd)

    if spec.get("openmp"):
        bindir = _mingw_bin()
        if bindir is not None:
            for dll in ("libgomp-1.dll", "libwinpthread-1.dll"):
                src = bindir / dll
                if src.exists():
                    shutil.copy2(src, HERE / dll)

    print(f"built {pyd} ({pyd.stat().st_size / 1024:.0f} KiB)")
    return pyd


def verify(names):
    """编译后立即 import 并跑一条投影断言，确认 ABI 与链接正常。"""
    code = "import importlib, sys, numpy as np;"
    code += "sys.path.insert(0, r'%s');" % HERE
    code += "import os;"
    code += " hasattr(os,'add_dll_directory') and os.add_dll_directory(r'%s');" % HERE
    for n in names:
        code += f" m = importlib.import_module('{n}');"
    code += (
        " R = np.eye(3); t = np.zeros(3);"
        " X = np.array([[0.,0.,2.],[1.,0.,3.]]);"
        " uv = np.zeros((2,2)); d = np.zeros(2);"
        " m.project_points(R, t, X, 500., 500., 320., 240., uv, d);"
        " assert abs(d[0] - 2.0) < 1e-12, d;"
        " assert abs(uv[0,0] - 320.0) < 1e-12, uv;"
        " print('import + self-test OK')"
    )
    _run([sys.executable, "-c", code])


def main(argv):
    names = argv[1:] or list(MODULES)
    for n in names:
        build(n)
    verify(names)


if __name__ == "__main__":
    main(sys.argv)
