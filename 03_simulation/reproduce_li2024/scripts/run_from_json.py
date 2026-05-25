#!/usr/bin/env python3
"""Run a pure PyBaMM degradation simulation from a JSON file."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


SIM_ROOT = Path(__file__).resolve().parents[1]
ALPHA_ROOT = Path(__file__).resolve().parents[3]
MODEL_ROOT = ALPHA_ROOT / '02_pybamm_model' / 'reproduce_li2024_model'
if str(MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_ROOT))

from sim.runner import run_simulation_case  # noqa: E402


def _normalise_pybamm_root(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if (resolved / 'pybamm' / '__init__.py').is_file():
        return resolved
    if resolved.name == 'pybamm' and (resolved / '__init__.py').is_file():
        return resolved.parent
    raise ValueError(
        f"Invalid pybamm root '{resolved}'. Expected either a repository root "
        "containing `pybamm/` or the `pybamm/` package directory itself."
    )


def _default_pybamm_root() -> Path | None:
    vendored_root = ALPHA_ROOT / 'external' / 'pybamm'
    if vendored_root.is_dir():
        return vendored_root
    return None


def _resolve_pybamm_root(cli_value: Path | None) -> tuple[Path | None, str]:
    if cli_value is not None:
        return _normalise_pybamm_root(cli_value), 'cli'

    for env_name in ('PYBAMM_ROOT', 'PYBAMM_REPO_ROOT'):
        env_value = os.environ.get(env_name)
        if env_value:
            return _normalise_pybamm_root(Path(env_value)), f'env:{env_name}'

    default_root = _default_pybamm_root()
    if default_root is not None:
        return default_root, 'default'

    return None, 'installed'


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _import_pybamm(pybamm_root: Path | None):
    if pybamm_root is not None and str(pybamm_root) not in sys.path:
        sys.path.insert(0, str(pybamm_root))

    pybamm = importlib.import_module('pybamm')

    if pybamm_root is not None:
        pybamm_source = Path(pybamm.__file__).resolve()
        if not _path_is_within(pybamm_source, pybamm_root):
            raise RuntimeError(
                f"Requested pybamm root '{pybamm_root}', but imported pybamm from "
                f"'{pybamm_source}'. Please check your environment and PYTHONPATH."
            )

    return pybamm


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run pure PyBaMM degradation simulation from JSON config.')
    parser.add_argument(
        '--config',
        type=Path,
        default=MODEL_ROOT / 'configs' / 'cases' / 'okane2023_full_cycle.json',
        help='Path to simulation config JSON.',
    )
    parser.add_argument(
        '--output-root',
        type=Path,
        default=SIM_ROOT / 'outputs',
        help='Directory to store outputs.',
    )
    parser.add_argument(
        '--pybamm-root',
        type=Path,
        default=None,
        help=(
            'Path to the PyBaMM repository root (or directly to the `pybamm/` '
            'package directory). Overrides PYBAMM_ROOT / PYBAMM_REPO_ROOT.'
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    pybamm_root, pybamm_root_source = _resolve_pybamm_root(args.pybamm_root)
    pybamm = _import_pybamm(pybamm_root)

    if pybamm_root is not None:
        print(f'[INFO] pybamm root ({pybamm_root_source}): {pybamm_root}')
    else:
        print(f'[INFO] pybamm root ({pybamm_root_source}): using installed package search path')
    print(f'[INFO] pybamm source: {pybamm.__file__}')
    print(f'[INFO] config: {config_path}')
    print(f'[INFO] output root: {output_root}')

    result = run_simulation_case(config_path, output_root, pybamm)
    summary = result['summary']

    print(f"[DONE] case dir: {result['case_dir']}")
    print(f"[DONE] final time [h]: {summary['final_time_h']:.3f}")
    print(f"[DONE] RPT metrics: {result['rpt_metrics_path']}")
    if 'final_soh_percent' in summary:
        print(f"[DONE] final SOH [%]: {summary['final_soh_percent']:.3f}")
    print(f"[DONE] saved columns: {', '.join(summary['output_variables_saved'])}")
    if summary['output_variables_skipped']:
        print(f"[DONE] skipped columns: {', '.join(summary['output_variables_skipped'])}")


if __name__ == '__main__':
    main()
