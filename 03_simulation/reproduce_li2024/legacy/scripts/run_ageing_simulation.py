#!/usr/bin/env python3
"""Run Li2024 ageing simulation from an OKane2023 JSON configuration."""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent

# Ensure this project uses the in-repo PyBaMM source tree first, then local scripts.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _load_config(
    config_path: Path,
    keep_thermal_model: bool = False,
    model_preset: str = 'full',
) -> Dict[str, Any]:
    with config_path.open('r', encoding='utf-8') as f:
        cfg = json.load(f)

    defaults = {
        'Scan No': 1,
        'Exp No.': 3,
        'Ageing temperature': 25,
        'Para_Set': 'OKane2023',
        'Cycles within RPT': 1,
        'RPT temperature': 25,
        'Mesh list': '[10, 5, 5, 100, 20]',
        # Keep compatibility across different installed PyBaMM versions.
        'Outer SEI partial molar volume [m3.mol-1]': 9.585e-05,
        'Inner SEI partial molar volume [m3.mol-1]': 9.585e-05,
        'Ratio of lithium moles to SEI moles': 2.0,
        'Electrode width [m]': 1.58,
        'Electrode height [m]': 0.065,
        'Negative electrode thickness [m]': 8.52e-05,
        'Negative electrode active material volume fraction': 0.75,
        'Negative particle radius [m]': 5.86e-06,
        'Negative electrode initial crack length [m]': 2e-08,
        'Negative electrode initial crack width [m]': 1.5e-08,
        'Negative electrode number of cracks per unit area [m-2]': 3.18e15,
        'Initial electrolyte excessive amount ratio': 0.0,
        'Nominal cell capacity [A.h]': 5.0,
        'Negative current collector surface heat transfer coefficient [W.m-2.K-1]': 10.0,
        'Positive current collector surface heat transfer coefficient [W.m-2.K-1]': 10.0,
        'Negative tab heat transfer coefficient [W.m-2.K-1]': 10.0,
        'Positive tab heat transfer coefficient [W.m-2.K-1]': 10.0,
        'Edge heat transfer coefficient [W.m-2.K-1]': 10.0,
        'Negative tab width [m]': 0.007,
        'Positive tab width [m]': 0.0069,
    }
    for key, value in defaults.items():
        cfg.setdefault(key, value)

    ageing_temp_c = cfg.get('Ageing temperature', 25)
    ageing_temp_k = float(ageing_temp_c) + 273.15
    cfg.setdefault('Ambient temperature [K]', ageing_temp_k)
    cfg.setdefault('Initial temperature [K]', ageing_temp_k)

    model_option_raw = cfg.get('Model option')
    if isinstance(model_option_raw, dict):
        model_option = dict(model_option_raw)
    elif isinstance(model_option_raw, str):
        model_option = ast.literal_eval(model_option_raw)
    else:
        model_option = None

    if model_option is None:
        raise ValueError("Missing required key 'Model option' in config JSON")

    preset_options = {
        'full': {
            'SEI': 'interstitial-diffusion limited',
            'SEI on cracks': 'true',
            'lithium plating': 'partially reversible',
            'lithium plating porosity change': 'true',
            'particle mechanics': ('swelling and cracking', 'swelling only'),
            'loss of active material': 'stress-driven',
            'contact resistance': 'true',
            'open-circuit potential': 'current sigmoid',
            'SEI film resistance': 'distributed',
            'SEI porosity change': 'true',
            'thermal': 'lumped',
        },
        'sei': {
            'SEI': 'interstitial-diffusion limited',
            'SEI on cracks': 'true',
            'lithium plating': 'none',
            'lithium plating porosity change': 'false',
            'particle mechanics': 'constant cracks',
            'loss of active material': 'none',
            'contact resistance': 'true',
            'open-circuit potential': 'current sigmoid',
            'SEI film resistance': 'distributed',
            'SEI porosity change': 'true',
            'thermal': 'lumped',
        },
        'minimal': {
            'SEI': 'interstitial-diffusion limited',
            'SEI on cracks': 'false',
            'lithium plating': 'none',
            'lithium plating porosity change': 'false',
            'particle mechanics': 'none',
            'loss of active material': 'none',
            'contact resistance': 'true',
            'open-circuit potential': 'current sigmoid',
            'SEI film resistance': 'distributed',
            'SEI porosity change': 'true',
            'thermal': 'lumped',
        },
    }
    if model_preset in preset_options:
        model_option.update(preset_options[model_preset])
        print(f'[INFO] Apply model preset: {model_preset}')

    # In many local environments, thermal can still be simplified to stabilize the first run.
    if (not keep_thermal_model) and model_option.get('thermal') == 'lumped':
        model_option['thermal'] = 'isothermal'
        print("[INFO] Override model option: thermal -> isothermal (use --keep-thermal-model to disable)")

    # Fun_NC.Para_init expects string values for these two fields.
    cfg['Model option'] = repr(model_option)
    if isinstance(cfg.get('Mesh list'), list):
        cfg['Mesh list'] = json.dumps(cfg['Mesh list'])

    return cfg


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run ageing simulation for Li2024 reproduction project.'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=REPO_ROOT / 'configs' / 'okane2023' / 'okane2023_exp3_25C.json',
        help='Path to JSON config file.',
    )
    parser.add_argument(
        '--exp-data-path',
        type=Path,
        default=REPO_ROOT / 'InputData',
        help='Path to experimental dataset root expected by Fun_NC.Read_Exp().',
    )
    parser.add_argument(
        '--output-root',
        type=Path,
        default=REPO_ROOT / 'results',
        help='Directory where simulation outputs are written.',
    )
    parser.add_argument(
        '--purpose',
        type=str,
        default=None,
        help='Output case name. Default uses config filename without suffix.',
    )
    parser.add_argument('--re-no', type=int, default=0, help='Repeat run index.')
    parser.add_argument('--runshort', type=str, default='GEM-2', help='Run mode passed to Fun_NC.')
    parser.add_argument('--hours', type=float, default=48.0, help='Timeout hours for the full simulation.')

    parser.add_argument('--add-rest', action='store_true', help='Insert rest step into ageing protocol.')
    parser.add_argument('--no-plot-exp', action='store_true', help='Disable experiment comparison plotting.')
    parser.add_argument('--no-return-sol', action='store_true', help='Disable returning full solution objects.')
    parser.add_argument('--no-check-small-time', action='store_true', help='Disable small timer diagnostics.')
    parser.add_argument('--no-r-from-gitt', action='store_true', help='Use C/2 instead of GITT for resistance.')

    parser.add_argument('--dpi', type=int, default=100, help='Plot dpi.')
    parser.add_argument('--fs', type=int, default=13, help='Plot font size.')
    parser.add_argument(
        '--keep-thermal-model',
        action='store_true',
        help='Keep thermal option from config instead of forcing isothermal fallback.',
    )
    parser.add_argument(
        '--model-preset',
        choices=['full', 'sei', 'minimal'],
        default='full',
        help='Preset model complexity for compatibility/stability.',
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')

    cfg = _load_config(
        config_path,
        keep_thermal_model=args.keep_thermal_model,
        model_preset=args.model_preset,
    )

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    exp_data_path = args.exp_data_path.expanduser().resolve()
    purpose = args.purpose or config_path.stem
    target = f'/{purpose}/'

    options = [
        False,  # On_HPC
        args.runshort,
        args.add_rest,
        not args.no_plot_exp,
        True,  # Timeout
        not args.no_return_sol,
        not args.no_check_small_time,
        not args.no_r_from_gitt,
        args.dpi,
        args.fs,
    ]
    timelimit = int(args.hours * 3600)
    # Fun_NC.Read_Exp concatenates paths using `+`, so keep a trailing separator.
    exp_data_path_str = str(exp_data_path)
    if not exp_data_path_str.endswith(os.sep):
        exp_data_path_str += os.sep

    path_list = [str(output_root), exp_data_path_str, target, purpose]

    print(f'[INFO] Running case: {purpose}')
    print(f'[INFO] Config: {config_path}')
    print(f'[INFO] Output root: {output_root}')
    print(f'[INFO] Experimental data path: {exp_data_path}')

    from Fun_NC import Run_P2_Excel  # pylint: disable=import-outside-toplevel

    midc_merge, sol_rpt, sol_age, debug_lists = Run_P2_Excel(
        cfg,
        path_list,
        args.re_no,
        timelimit,
        options,
    )

    print('[DONE] Simulation finished.')
    print(f'[DONE] midc_merge keys: {len(midc_merge) if hasattr(midc_merge, "keys") else "N/A"}')
    print(f'[DONE] sol_rpt length: {len(sol_rpt) if hasattr(sol_rpt, "__len__") else "N/A"}')
    print(f'[DONE] sol_age length: {len(sol_age) if hasattr(sol_age, "__len__") else "N/A"}')
    print(f'[DONE] debug entries: {len(debug_lists) if hasattr(debug_lists, "__len__") else "N/A"}')


if __name__ == '__main__':
    main()
