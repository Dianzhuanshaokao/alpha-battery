from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _safe_last_state_value(cycle_solution, variable_name: str) -> float | None:
    try:
        return float(cycle_solution.last_state[variable_name].data[0])
    except Exception:
        return None


def _step_capacity_delta_ah(step_solution) -> float:
    entries = step_solution['Discharge capacity [A.h]'].entries
    return float(abs(entries[-1] - entries[0]))


def _step_discharged_capacity_ah(step_solution) -> float:
    entries = step_solution['Discharge capacity [A.h]'].entries
    return float(max(entries[-1] - entries[0], 0.0))


def cycle_discharged_capacity_ah(cycle_solution) -> float:
    total = 0.0
    for step_solution in getattr(cycle_solution, 'steps', []):
        try:
            total += _step_discharged_capacity_ah(step_solution)
        except Exception:
            continue
    return float(total)


def _extract_capacity_from_named_step(cycle_solution, step_name_to_index: Dict[str, int], step_name: str) -> float | None:
    step_index = step_name_to_index.get(step_name)
    if step_index is None:
        return None
    try:
        return _step_capacity_delta_ah(cycle_solution.steps[step_index])
    except Exception:
        return None


def _extract_resistance_from_0p5c_step(
    cycle_solution,
    step_name_to_index: Dict[str, int],
    step_name: str,
    reference_capacity_ah: float | None,
) -> float | None:
    step_index = step_name_to_index.get(step_name)
    if step_index is None:
        return None

    try:
        step = cycle_solution.steps[step_index]
        discharge_capacity = abs(
            step['Discharge capacity [A.h]'].entries[0]
            - step['Discharge capacity [A.h]'].entries
        )
        capacity_for_soc = (
            reference_capacity_ah
            if reference_capacity_ah and reference_capacity_ah > 0
            else float(discharge_capacity[-1])
        )
        soc = (1 - discharge_capacity / capacity_for_soc) * 100
        v_ohmic = (
            step['Battery open-circuit voltage [V]'].entries
            - step['Terminal voltage [V]'].entries
            + step['Battery particle concentration overpotential [V]'].entries
            + step['X-averaged battery concentration overpotential [V]'].entries
        )
        current = step['Current [A]'].entries[0]
        resistance_mohm = v_ohmic / current * 1e3
        return float(np.interp(50, np.flip(soc), np.flip(resistance_mohm)))
    except Exception:
        return None


def extract_rpt_cycle_record(
    cycle_solution,
    cycle_meta: Dict[str, Any],
    analysis_cfg: Dict[str, Any],
    reference_capacity_ah: float | None,
) -> Dict[str, Any]:
    step_name_to_index = cycle_meta['step_name_to_index']
    capacity_step_name = analysis_cfg['capacity_step_name']
    resistance_step_name = analysis_cfg['resistance_step_name']

    return {
        'experiment_cycle': cycle_meta['experiment_cycle'],
        'equivalent_cycle': cycle_meta['equivalent_cycle'],
        'equivalent_cycle_actual': cycle_meta.get('equivalent_cycle_actual'),
        'checkpoint_index': cycle_meta['checkpoint_index'],
        'rpt_type': cycle_meta['rpt_type'],
        'throughput_capacity_ah': _safe_last_state_value(
            cycle_solution, 'Throughput capacity [A.h]'
        ),
        'rpt_discharge_capacity_ah': _extract_capacity_from_named_step(
            cycle_solution, step_name_to_index, capacity_step_name
        ),
        'negative_electrode_capacity_ah': _safe_last_state_value(
            cycle_solution, 'Negative electrode capacity [A.h]'
        ),
        'positive_electrode_capacity_ah': _safe_last_state_value(
            cycle_solution, 'Positive electrode capacity [A.h]'
        ),
        'total_lithium_capacity_in_particles_ah': _safe_last_state_value(
            cycle_solution, 'Total lithium capacity in particles [A.h]'
        ),
        'loss_of_capacity_to_sei_ah': _safe_last_state_value(
            cycle_solution, 'Loss of capacity to SEI [A.h]'
        ),
        'loss_of_capacity_to_sei_on_cracks_ah': _safe_last_state_value(
            cycle_solution, 'Loss of capacity to SEI on cracks [A.h]'
        ),
        'loss_of_capacity_to_lithium_plating_ah': _safe_last_state_value(
            cycle_solution, 'Loss of capacity to lithium plating [A.h]'
        ),
        'resistance_50soc_mohm': _extract_resistance_from_0p5c_step(
            cycle_solution,
            step_name_to_index,
            resistance_step_name,
            reference_capacity_ah,
        ),
    }


def build_rpt_metrics(records: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df

    df = df.sort_values('equivalent_cycle').reset_index(drop=True)
    baseline = df.iloc[0]

    reference_capacity = baseline['rpt_discharge_capacity_ah']
    if not reference_capacity or pd.isna(reference_capacity):
        raise ValueError('Baseline RPT capacity is missing; cannot compute SOH')

    baseline_neg = baseline['negative_electrode_capacity_ah']
    baseline_pos = baseline['positive_electrode_capacity_ah']
    baseline_li = baseline['total_lithium_capacity_in_particles_ah']
    baseline_sei = baseline['loss_of_capacity_to_sei_ah']
    baseline_sei_cracks = baseline['loss_of_capacity_to_sei_on_cracks_ah']
    baseline_plating = baseline['loss_of_capacity_to_lithium_plating_ah']

    df['soh_percent'] = df['rpt_discharge_capacity_ah'] / reference_capacity * 100.0
    df['lam_ne_percent'] = (1 - df['negative_electrode_capacity_ah'] / baseline_neg) * 100.0
    df['lam_pe_percent'] = (1 - df['positive_electrode_capacity_ah'] / baseline_pos) * 100.0
    df['lli_percent'] = (
        1 - df['total_lithium_capacity_in_particles_ah'] / baseline_li
    ) * 100.0

    df['lli_sei_percent'] = (
        (df['loss_of_capacity_to_sei_ah'] - baseline_sei) / baseline_li * 100.0
    )
    df['lli_sei_on_cracks_percent'] = (
        (df['loss_of_capacity_to_sei_on_cracks_ah'] - baseline_sei_cracks)
        / baseline_li
        * 100.0
    )
    df['lli_plating_percent'] = (
        (df['loss_of_capacity_to_lithium_plating_ah'] - baseline_plating)
        / baseline_li
        * 100.0
    )
    df['lli_due_to_lam_percent'] = (
        df['lli_percent']
        - df['lli_sei_percent']
        - df['lli_sei_on_cracks_percent']
        - df['lli_plating_percent']
    )

    df['internal_resistance_mohm'] = df['resistance_50soc_mohm']
    df['capacity_fade_percent'] = 100.0 - df['soh_percent']
    return df
