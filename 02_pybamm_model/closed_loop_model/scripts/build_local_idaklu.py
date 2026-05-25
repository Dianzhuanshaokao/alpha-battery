from __future__ import annotations

import os
import sysconfig
from pathlib import Path

import casadi
import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSOLVERS_ROOT = PROJECT_ROOT / "pybamm" / "solvers" / "c_solvers"
IDAKLU_ROOT = CSOLVERS_ROOT / "idaklu"
CONDA_PREFIX = Path(os.environ["CONDA_PREFIX"])
CASADI_ROOT = Path(casadi.__file__).resolve().parent
CONDA_LIB = CONDA_PREFIX / "lib"
LIB_SEARCH_DIRS = [CONDA_LIB, CASADI_ROOT]


def _available_libraries() -> set[str]:
    libraries = set()
    for directory in LIB_SEARCH_DIRS:
        libraries.update(
            path.name[3:].split(".so", 1)[0]
            for path in directory.glob("lib*.so*")
            if path.name.startswith("lib")
        )
    return libraries


def _select_libraries() -> list[str]:
    available = _available_libraries()
    required = [
        "casadi",
        "sundials_ida",
        "sundials_idas",
        "sundials_nvecserial",
        "sundials_sunlinsolklu",
        "sundials_sunlinsoldense",
        "sundials_sunlinsolband",
        "sundials_sunlinsolspbcgs",
        "sundials_sunlinsolspfgmr",
        "sundials_sunlinsolspgmr",
        "sundials_sunlinsolsptfqmr",
        "sundials_sunmatrixband",
        "sundials_sunmatrixsparse",
        "sundials_sunmatrixdense",
        "klu",
        "amd",
        "colamd",
        "btf",
        "suitesparseconfig",
    ]
    optional = [
        "sundials_core",
        "sundials_nvecopenmp",
    ]

    missing = [name for name in required if name not in available]
    if missing:
        missing_str = ", ".join(missing)
        search_dirs = ", ".join(str(path) for path in LIB_SEARCH_DIRS)
        raise RuntimeError(
            f"Missing required libraries in [{search_dirs}]: {missing_str}"
        )

    return required + [name for name in optional if name in available]


def build_extension() -> None:
    include_dirs = [
        str(CSOLVERS_ROOT),
        str(IDAKLU_ROOT),
        sysconfig.get_paths()["include"],
        pybind11.get_include(),
        str(CONDA_PREFIX / "include"),
        str(CONDA_PREFIX / "include" / "suitesparse"),
        str(CASADI_ROOT / "include"),
    ]
    library_dirs = [
        *(str(path) for path in LIB_SEARCH_DIRS),
    ]
    libraries = _select_libraries()
    sources = [
        str(CSOLVERS_ROOT / "idaklu.cpp"),
        str(IDAKLU_ROOT / "casadi_functions.cpp"),
        str(IDAKLU_ROOT / "casadi_solver.cpp"),
        str(IDAKLU_ROOT / "casadi_sundials_functions.cpp"),
        str(IDAKLU_ROOT / "options.cpp"),
        str(IDAKLU_ROOT / "python.cpp"),
        str(IDAKLU_ROOT / "solution.cpp"),
    ]

    ext_modules = [
        Extension(
            "pybamm.solvers.idaklu",
            sources=sources,
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            runtime_library_dirs=library_dirs,
            libraries=libraries,
            define_macros=[("_GLIBCXX_USE_CXX11_ABI", "0")],
            extra_compile_args=["-O3", "-std=c++17"],
            language="c++",
        )
    ]

    setup(
        name="pybamm-local-idaklu",
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext},
        script_args=["build_ext", "--inplace", "--force"],
    )


if __name__ == "__main__":
    build_extension()
