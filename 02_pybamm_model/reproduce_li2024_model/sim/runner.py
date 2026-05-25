from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .analysis import (
    build_rpt_metrics,
    cycle_discharged_capacity_ah,
    extract_rpt_cycle_record,
)
from .config import load_json_config, load_parameter_values
from .protocol import _select_rpt_type, build_study_plan


DEFAULT_OUTPUTS = [
    'Time [h]',
    'Current [A]',
    'Terminal voltage [V]',
    'Discharge capacity [A.h]',
    'Throughput capacity [A.h]',
    'Loss of lithium to SEI [mol]',
    'Loss of capacity to SEI [A.h]',
    'Loss of capacity to SEI on cracks [A.h]',
    'Loss of capacity to lithium plating [A.h]',
    'Total lithium lost [mol]',
    'Negative electrode capacity [A.h]',
    'Positive electrode capacity [A.h]',
    'Total lithium capacity in particles [A.h]',
    'X-averaged negative electrode porosity',
]


def _build_model(pybamm, model_cfg: Dict[str, Any]):
    model_type = model_cfg['type']
    builders = {
        'DFN': pybamm.lithium_ion.DFN,
        'SPMe': pybamm.lithium_ion.SPMe,
        'SPM': pybamm.lithium_ion.SPM,
    }
    if model_type not in builders:
        raise ValueError(f'Unsupported model type: {model_type}')
    return builders[model_type](options=model_cfg.get('options', {}))


def _build_solver(pybamm, solver_cfg: Dict[str, Any]):
    name = solver_cfg.get('name', 'CasadiSolver')
    if name == 'CasadiSolver':
        return pybamm.CasadiSolver(
            mode=solver_cfg.get('mode', 'safe'),
            dt_max=solver_cfg.get('dt_max', 30.0),
            rtol=solver_cfg.get('rtol', 1e-6),
            atol=solver_cfg.get('atol', 1e-6),
        )
    if name == 'IDAKLUSolver':
        return pybamm.IDAKLUSolver(
            rtol=solver_cfg.get('rtol', 1e-6),
            atol=solver_cfg.get('atol', 1e-6),
            root_method=solver_cfg.get('root_method', 'casadi'),
            root_tol=solver_cfg.get('root_tol', 1e-6),
            extrap_tol=solver_cfg.get('extrap_tol'),
            options=solver_cfg.get('options'),
        )
    raise ValueError(f'Unsupported solver: {name}')


def _build_var_pts(pybamm, mesh_cfg: Dict[str, int]):
    var = pybamm.standard_spatial_vars
    return {
        var.x_n: int(mesh_cfg.get('x_n', 10)),
        var.x_s: int(mesh_cfg.get('x_s', 5)),
        var.x_p: int(mesh_cfg.get('x_p', 5)),
        var.r_n: int(mesh_cfg.get('r_n', 50)),
        var.r_p: int(mesh_cfg.get('r_p', 50)),
    }


def _extract_timeseries(solution, output_variables: List[str]) -> tuple[pd.DataFrame, List[str]]:
    data = {}
    skipped = []

    # Ensure time is always present
    try:
        time_h = solution['Time [h]'].entries
    except KeyError:
        time_h = solution.t / 3600
    data['Time [h]'] = time_h
    n_time = len(time_h)

    for name in output_variables:
        if name == 'Time [h]':
            continue
        try:
            entries = solution[name].entries
        except Exception:
            skipped.append(name)
            continue
        if getattr(entries, 'ndim', 1) != 1 or len(entries) != n_time:
            skipped.append(name)
            continue
        data[name] = entries

    return pd.DataFrame(data), skipped


def _solve_experiment_cycle(
    pybamm,
    model,
    parameter_values,
    solver,
    var_pts,
    period: str,
    protocol: Dict[str, Any],
    *,
    starting_solution=None,
    initial_soc: float | None = None,
):
    simulation = pybamm.Simulation(
        model,
        experiment=pybamm.Experiment([tuple(protocol['instructions'])], period=period),
        parameter_values=parameter_values,
        solver=solver,
        var_pts=var_pts,
    )
    solve_kwargs: Dict[str, Any] = {'calc_esoh': False}
    if starting_solution is not None:
        solve_kwargs['starting_solution'] = starting_solution
    elif initial_soc is not None:
        solve_kwargs['initial_soc'] = initial_soc
    solution = simulation.solve(**solve_kwargs)
    return solution, solution.cycles[-1]


def run_simulation_case(config_path: Path, output_root: Path, pybamm) -> Dict[str, Any]:
    cfg = load_json_config(config_path)
    case_name = cfg['case_name']
    case_dir = output_root / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    model = _build_model(pybamm, cfg['model'])
    parameter_values = load_parameter_values(pybamm, cfg['model']['parameter_set'])
    parameter_values.update(cfg.get('resolved_parameter_updates', {}), check_already_exists=False)

    solver = _build_solver(pybamm, cfg.get('solver', {}))
    var_pts = _build_var_pts(pybamm, cfg.get('mesh_points', {}))
    study_plan = build_study_plan(cfg)
    period = cfg['simulation'].get('period', '30 seconds')
    analysis_cfg = cfg['analysis']
    nominal_capacity_ah = analysis_cfg.get('nominal_capacity_ah')
    if nominal_capacity_ah is None:
        nominal_capacity_ah = parameter_values['Nominal cell capacity [A.h]']
    nominal_capacity_ah = float(nominal_capacity_ah)
    initial_soc = cfg.get('initial_soc')

    solution = None
    cycle_plan: List[Dict[str, Any]] = []
    rpt_records = []
    cumulative_discharged_capacity_ah = 0.0
    next_rpt_efc = float(study_plan['rpt_interval_equivalent_cycles'])
    checkpoint_index = 0
    ageing_block_index = 0
    experiment_cycle = 0

    def append_cycle_meta(
        role: str,
        protocol: Dict[str, Any],
        *,
        equivalent_cycle: float,
        equivalent_cycle_actual: float,
        checkpoint_index_value: int | None = None,
        rpt_type: str | None = None,
        ageing_block_value: int | None = None,
        block_discharged_capacity_ah: float = 0.0,
        cumulative_discharged_capacity_ah_value: float = 0.0,
    ) -> Dict[str, Any]:
        return {
            'experiment_cycle': experiment_cycle,
            'role': role,
            'protocol_name': protocol['name'],
            'equivalent_cycle': equivalent_cycle,
            'equivalent_cycle_actual': equivalent_cycle_actual,
            'checkpoint_index': checkpoint_index_value,
            'rpt_type': rpt_type,
            'ageing_block_index': ageing_block_value,
            'block_discharged_capacity_ah': block_discharged_capacity_ah,
            'cumulative_discharged_capacity_ah': cumulative_discharged_capacity_ah_value,
            'step_names': list(protocol['step_names']),
            'step_name_to_index': dict(protocol['step_name_to_index']),
        }

    def run_protocol(protocol: Dict[str, Any]):
        nonlocal solution, experiment_cycle
        experiment_cycle += 1
        solution, cycle_solution = _solve_experiment_cycle(
            pybamm,
            model,
            parameter_values,
            solver,
            var_pts,
            period,
            protocol,
            starting_solution=solution,
            initial_soc=initial_soc,
        )
        return cycle_solution

    compiled_protocols = study_plan['compiled_protocols']
    if study_plan['include_initial_rpt']:
        initial_rpt_type = study_plan['initial_rpt_type']
        cycle_solution = run_protocol(compiled_protocols[initial_rpt_type])
        cycle_meta = append_cycle_meta(
            'rpt',
            compiled_protocols[initial_rpt_type],
            equivalent_cycle=0.0,
            equivalent_cycle_actual=0.0,
            checkpoint_index_value=0,
            rpt_type=initial_rpt_type,
            cumulative_discharged_capacity_ah_value=0.0,
        )
        cycle_plan.append(cycle_meta)
        rpt_records.append(
            extract_rpt_cycle_record(
                cycle_solution,
                cycle_meta,
                analysis_cfg,
                nominal_capacity_ah,
            )
        )

    while checkpoint_index < int(study_plan['target_equivalent_cycles']):
        ageing_block_index += 1
        cycle_solution = run_protocol(compiled_protocols['ageing'])
        block_discharged_capacity_ah = cycle_discharged_capacity_ah(cycle_solution)
        cumulative_discharged_capacity_ah += block_discharged_capacity_ah
        equivalent_cycle_actual = cumulative_discharged_capacity_ah / nominal_capacity_ah
        cycle_plan.append(
            append_cycle_meta(
                'ageing',
                compiled_protocols['ageing'],
                equivalent_cycle=equivalent_cycle_actual,
                equivalent_cycle_actual=equivalent_cycle_actual,
                ageing_block_value=ageing_block_index,
                block_discharged_capacity_ah=block_discharged_capacity_ah,
                cumulative_discharged_capacity_ah_value=cumulative_discharged_capacity_ah,
            )
        )

        if block_discharged_capacity_ah > nominal_capacity_ah + 1e-9:
            cycle_plan[-1]['warning'] = (
                'Single ageing block discharged more than 1 nominal capacity; '
                'RPT checkpoints may lag the exact EFC threshold.'
            )

        while (
            equivalent_cycle_actual + 1e-12 >= next_rpt_efc
            and checkpoint_index < int(study_plan['target_equivalent_cycles'])
        ):
            checkpoint_index += 1
            rpt_type = _select_rpt_type(checkpoint_index, study_plan['rpt_schedule'])
            rpt_solution = run_protocol(compiled_protocols[rpt_type])
            rpt_discharged_capacity_ah = (
                cycle_discharged_capacity_ah(rpt_solution)
                if study_plan['count_rpt_discharge_in_equivalent_cycles']
                else 0.0
            )
            if rpt_discharged_capacity_ah > 0:
                cumulative_discharged_capacity_ah += rpt_discharged_capacity_ah
                equivalent_cycle_actual = (
                    cumulative_discharged_capacity_ah / nominal_capacity_ah
                )
            cycle_meta = append_cycle_meta(
                'rpt',
                compiled_protocols[rpt_type],
                equivalent_cycle=next_rpt_efc,
                equivalent_cycle_actual=equivalent_cycle_actual,
                checkpoint_index_value=checkpoint_index,
                rpt_type=rpt_type,
                block_discharged_capacity_ah=rpt_discharged_capacity_ah,
                cumulative_discharged_capacity_ah_value=cumulative_discharged_capacity_ah,
            )
            cycle_plan.append(cycle_meta)
            rpt_records.append(
                extract_rpt_cycle_record(
                    rpt_solution,
                    cycle_meta,
                    analysis_cfg,
                    nominal_capacity_ah,
                )
            )
            next_rpt_efc += float(study_plan['rpt_interval_equivalent_cycles'])

    output_variables = cfg.get('output_variables') or DEFAULT_OUTPUTS
    timeseries_df, skipped = _extract_timeseries(solution, output_variables)
    timeseries_path = case_dir / 'timeseries.csv'
    timeseries_df.to_csv(timeseries_path, index=False)

    cycle_plan_df = pd.DataFrame(cycle_plan)
    cycle_plan_df.to_csv(case_dir / 'cycle_plan.csv', index=False)

    rpt_metrics_df = build_rpt_metrics(rpt_records)
    rpt_metrics_path = case_dir / 'rpt_metrics.csv'
    rpt_metrics_df.to_csv(rpt_metrics_path, index=False)

    summary = {
        'case_name': case_name,
        'config_path': str(config_path),
        'model_type': cfg['model']['type'],
        'parameter_set': cfg['model']['parameter_set'],
        'target_equivalent_cycles': float(study_plan['target_equivalent_cycles']),
        'rpt_interval_equivalent_cycles': float(
            study_plan['rpt_interval_equivalent_cycles']
        ),
        'equivalent_cycle_definition': (
            'Equivalent cycle = cumulative discharged capacity during ageing '
            '/ nominal cell capacity'
        ),
        'count_rpt_discharge_in_equivalent_cycles': bool(
            study_plan['count_rpt_discharge_in_equivalent_cycles']
        ),
        'n_experiment_cycles': int(len(cycle_plan)),
        'n_ageing_blocks_completed': int(ageing_block_index),
        'n_rpt_checkpoints': int(len(rpt_metrics_df)),
        'output_variables_saved': list(timeseries_df.columns),
        'output_variables_skipped': skipped,
        'n_time_points': int(len(timeseries_df)),
        'final_time_h': float(timeseries_df['Time [h]'].iloc[-1]),
        'total_ageing_discharged_capacity_ah': float(cumulative_discharged_capacity_ah),
        'final_equivalent_cycle_actual': float(
            cumulative_discharged_capacity_ah / nominal_capacity_ah
        ),
        'timeseries_path': str(timeseries_path),
        'rpt_metrics_path': str(rpt_metrics_path),
    }
    if not rpt_metrics_df.empty:
        final_rpt = rpt_metrics_df.iloc[-1]
        summary['initial_rpt_capacity_ah'] = float(rpt_metrics_df.iloc[0]['rpt_discharge_capacity_ah'])
        summary['final_rpt_capacity_ah'] = float(final_rpt['rpt_discharge_capacity_ah'])
        summary['final_soh_percent'] = float(final_rpt['soh_percent'])
        summary['final_lli_percent'] = float(final_rpt['lli_percent'])
        summary['final_lam_ne_percent'] = float(final_rpt['lam_ne_percent'])
        summary['final_lam_pe_percent'] = float(final_rpt['lam_pe_percent'])
        if pd.notna(final_rpt['internal_resistance_mohm']):
            summary['final_internal_resistance_mohm'] = float(final_rpt['internal_resistance_mohm'])
    for col in timeseries_df.columns:
        if col != 'Time [h]':
            try:
                summary[f'final::{col}'] = float(timeseries_df[col].iloc[-1])
            except Exception:
                pass

    with (case_dir / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (case_dir / 'resolved_config.json').open('w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    # Keep a raw solution for deeper offline inspection.
    with (case_dir / 'solution.pkl').open('wb') as f:
        pickle.dump(solution, f)

    return {
        'case_dir': case_dir,
        'summary': summary,
        'timeseries_path': timeseries_path,
        'rpt_metrics_path': rpt_metrics_path,
    }
