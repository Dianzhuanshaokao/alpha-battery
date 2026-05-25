from __future__ import annotations

from typing import Any, Dict, List


def _expand_protocol_steps(steps: List[Dict[str, Any]], prefix: str = '') -> List[Dict[str, str]]:
    expanded: List[Dict[str, str]] = []
    for step in steps:
        if 'instruction' in step:
            step_name = f"{prefix}{step['name']}" if prefix else step['name']
            expanded.append({'name': step_name, 'instruction': step['instruction']})
            continue

        repeat = int(step['repeat'])
        block_name = step.get('name', 'repeat')
        for idx in range(1, repeat + 1):
            nested_prefix = f"{prefix}{block_name}_{idx:02d}__"
            expanded.extend(_expand_protocol_steps(step['steps'], prefix=nested_prefix))
    return expanded


def compile_protocol(protocol_name: str, protocol_cfg: Dict[str, Any]) -> Dict[str, Any]:
    expanded_steps = _expand_protocol_steps(protocol_cfg['steps'])
    instructions = [step['instruction'] for step in expanded_steps]
    step_name_to_index = {step['name']: idx for idx, step in enumerate(expanded_steps)}
    return {
        'name': protocol_name,
        'step_names': [step['name'] for step in expanded_steps],
        'instructions': instructions,
        'step_name_to_index': step_name_to_index,
    }


def _select_rpt_type(checkpoint_index: int, rpt_schedule: Dict[str, str]) -> str:
    parity = 'odd' if checkpoint_index % 2 == 1 else 'even'
    return rpt_schedule[parity]


def build_study_plan(cfg: Dict[str, Any]) -> Dict[str, Any]:
    study = cfg['study']
    ageing_protocol = compile_protocol('ageing', study['ageing_protocol'])
    rpt_protocols = {
        name: compile_protocol(name, protocol_cfg)
        for name, protocol_cfg in study['rpt_protocols'].items()
    }

    return {
        'target_equivalent_cycles': float(study['equivalent_cycles']),
        'rpt_interval_equivalent_cycles': float(
            study.get('rpt_interval_equivalent_cycles', 1.0)
        ),
        'include_initial_rpt': bool(study['include_initial_rpt']),
        'initial_rpt_type': study['initial_rpt_type'],
        'count_rpt_discharge_in_equivalent_cycles': bool(
            study.get('count_rpt_discharge_in_equivalent_cycles', False)
        ),
        'rpt_schedule': dict(study['rpt_schedule']),
        'compiled_protocols': {'ageing': ageing_protocol, **rpt_protocols},
    }
