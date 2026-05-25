"""Baseline DFN model with lumped thermal coupling for BOL simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, Tuple

EXTERNAL_ROOT = Path(__file__).resolve().parents[4] / "external"
if str(EXTERNAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EXTERNAL_ROOT))

import pybamm


@dataclass
class BOLParameters:
    """Beginning-of-life (BOL) descriptors used for baseline initialization."""

    initial_capacity_ah: float = 5.0
    initial_internal_resistance_ohm: float = 0.01
    initial_sei_thickness_m: float = 5e-9
    initial_lithium_inventory_ratio: float = 1.0


@dataclass
class ThermalParameters:
    """Thermal parameters for lumped thermal behavior (0-60 C operating range)."""

    specific_heat_capacity_j_kgk: float = 950.0 # Specific heat capacity J.kg-1.K-1
    thermal_conductivity_w_mk: float = 1.0      # Thermal conductivity W.m-1.K-1
    heat_transfer_coefficient_w_m2k: float = 12.0 # Heat transfer coefficient W.m-2.K-1


class BaselineDFNThermalModel:
    """Factory and utilities for the DFN model with lumped thermal coupling."""

    def __init__(self, bol: BOLParameters | None = None, thermal: ThermalParameters | None = None):
        self.bol = bol or BOLParameters()
        self.thermal = thermal or ThermalParameters()
        self.model = pybamm.lithium_ion.DFN(options={"thermal": "lumped"})
        self.parameter_values = pybamm.ParameterValues("Chen2020")
        self._configure_bol_and_thermal()

    def _safe_update(self, updates: Dict[str, float]) -> None:
        """Update known parameter names and skip unknown keys safely."""
        keys = set(self.parameter_values.keys())
        filtered = {k: v for k, v in updates.items() if k in keys}
        if filtered:
            self.parameter_values.update(filtered, check_already_exists=False)

    def _configure_bol_and_thermal(self) -> None:
        """Apply BOL and thermal parameters to the selected parameter set."""
        # Thermal block for lumped model behavior across broad ambient range.
        thermal_updates = {
            "Specific heat capacity [J.kg-1.K-1]": self.thermal.specific_heat_capacity_j_kgk,
            "Thermal conductivity [W.m-1.K-1]": self.thermal.thermal_conductivity_w_mk,
            "Total heat transfer coefficient [W.m-2.K-1]": self.thermal.heat_transfer_coefficient_w_m2k,
        }
        self._safe_update(thermal_updates)

        # BOL descriptors for capacity, resistance, SEI thickness and lithium inventory.
        bol_updates = {
            "Nominal cell capacity [A.h]": self.bol.initial_capacity_ah,
            "Contact resistance [Ohm]": self.bol.initial_internal_resistance_ohm,
            "Initial SEI thickness [m]": self.bol.initial_sei_thickness_m,
            "Lithium inventory ratio": self.bol.initial_lithium_inventory_ratio,
        }
        self._safe_update(bol_updates)

    def apply_degradation_feedback(self, updated_capacity_ah: float, updated_resistance_ohm: float) -> None:
        """Update DFN parameters after degradation step."""
        updates = {
            "Nominal cell capacity [A.h]": max(0.05, updated_capacity_ah),
            "Contact resistance [Ohm]": max(1e-5, updated_resistance_ohm),
        }
        self._safe_update(updates)

    def create_simulation(self, experiment: pybamm.Experiment) -> pybamm.Simulation:
        return pybamm.Simulation(
            self.model,
            parameter_values=self.parameter_values,
            experiment=experiment,
        )


def pick_first_variable(solution: pybamm.Solution, candidates: Iterable[str]) -> Tuple[str, pybamm.ProcessedVariable]:
    """Pick the first available variable from a list of possible solution keys."""
    for name in candidates:
        try:
            variable = solution[name]
        except KeyError:
            continue
        else:
            return name, variable
    raise KeyError(f"None of candidate variables found: {list(candidates)}")
