from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Dict


def _flatten_degradation_updates(value: Dict[str, Any]) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, dict):
            flattened.update(_flatten_degradation_updates(item))
        else:
            flattened[key] = item
    return flattened


def _merge_parameter_sections(cfg: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(cfg.get('parameter_updates', {}))
    merged.update(
        _flatten_degradation_updates(cfg.get('degradation_parameter_updates', {}))
    )
    return merged


def _convert_option_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_convert_option_value(v) for v in value)
    if isinstance(value, dict):
        return {k: _convert_option_value(v) for k, v in value.items()}
    return value


def _ensure_named_protocol(protocol_name: str, protocol_cfg: Dict[str, Any]) -> None:
    if not protocol_cfg.get('steps'):
        raise ValueError(f"Protocol '{protocol_name}' must define non-empty 'steps'")
    for step in protocol_cfg['steps']:
        if 'instruction' in step:
            if 'name' not in step:
                raise ValueError(
                    f"Protocol '{protocol_name}' contains a step without 'name'"
                )
        elif 'repeat' in step:
            if not step.get('steps'):
                raise ValueError(
                    f"Protocol '{protocol_name}' contains a repeat block without nested 'steps'"
                )
        else:
            raise ValueError(
                f"Protocol '{protocol_name}' step must define either 'instruction' or 'repeat'"
            )


def load_json_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open('r', encoding='utf-8') as f:
        cfg = json.load(f)

    cfg.setdefault('case_name', config_path.stem)
    cfg.setdefault('model', {})
    cfg.setdefault('parameter_updates', {})
    cfg.setdefault('degradation_parameter_updates', {})
    cfg.setdefault(
        'mesh_points',
        {'x_n': 10, 'x_s': 5, 'x_p': 5, 'r_n': 50, 'r_p': 50},
    )
    cfg.setdefault(
        'solver',
        {
            'name': 'CasadiSolver',
            'mode': 'safe',
            'dt_max': 10.0,
            'rtol': 1e-6,
            'atol': 1e-6,
        },
    )
    cfg.setdefault('simulation', {})
    cfg.setdefault('output_variables', [])
    cfg.setdefault('study', {})
    cfg.setdefault('analysis', {})

    model = cfg['model']
    model.setdefault('type', 'DFN')
    model.setdefault('parameter_set', 'OKane2023')
    model.setdefault('options', {})
    model['options'] = _convert_option_value(model['options'])

    simulation = cfg['simulation']
    simulation.setdefault('period', '30 seconds')

    study = cfg['study']
    study.setdefault('equivalent_cycles', 78)
    study.setdefault('include_initial_rpt', True)
    study.setdefault('initial_rpt_type', 'long')
    study.setdefault('rpt_schedule', {'odd': 'long', 'even': 'short'})
    study.setdefault('rpt_interval_equivalent_cycles', 1.0)
    study.setdefault('count_rpt_discharge_in_equivalent_cycles', False)

    if 'ageing_protocol' not in study:
        raise ValueError("Config study must define 'ageing_protocol'")
    if 'rpt_protocols' not in study:
        raise ValueError("Config study must define 'rpt_protocols'")

    _ensure_named_protocol('ageing_protocol', study['ageing_protocol'])
    for rpt_name, rpt_protocol in study['rpt_protocols'].items():
        _ensure_named_protocol(f'rpt_protocols.{rpt_name}', rpt_protocol)

    analysis = cfg['analysis']
    analysis.setdefault('capacity_step_name', 'rpt_0p1c_discharge')
    analysis.setdefault('resistance_step_name', 'rpt_0p5c_discharge')
    merged_parameter_updates = _merge_parameter_sections(cfg)
    cfg['resolved_parameter_updates'] = merged_parameter_updates
    analysis.setdefault(
        'nominal_capacity_ah',
        merged_parameter_updates.get('Nominal cell capacity [A.h]'),
    )

    return cfg


def load_parameter_values(pybamm, parameter_set: str):
    try:
        return pybamm.ParameterValues(parameter_set)
    except (FileNotFoundError, ValueError):
        import importlib.util
        from pathlib import Path
        
        project_root = Path(__file__).resolve().parents[3]
        param_file = project_root / "external" / "pybamm" / "input" / "parameters" / "lithium_ion" / f"{parameter_set}.py"
        if not param_file.is_file():
            param_file = project_root / "external" / "pybamm" / "pybamm" / "input" / "parameters" / "lithium_ion" / f"{parameter_set}.py"
            
        if param_file.is_file():
            spec = importlib.util.spec_from_file_location(parameter_set, str(param_file))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return pybamm.ParameterValues(module.get_parameter_values())
        else:
            module_map = {
                'OKane2023': 'pybamm.input.parameters.lithium_ion.OKane2023',
                'OKane2022': 'pybamm.input.parameters.lithium_ion.OKane2022',
            }
            if parameter_set not in module_map:
                raise
            module = importlib.import_module(module_map[parameter_set])
            return pybamm.ParameterValues(module.get_parameter_values())
