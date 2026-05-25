import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybamm as pb
import os
import yaml

try:
    from . import Fun_Sim_Only as fun
except ImportError:
    import Fun_Sim_Only as fun

class BatteryEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This environment simulates a lithium-ion battery aging process using PyBaMM.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, task_id=None, log_dir=None, solver_type=None, env_rank=None):
        super(BatteryEnv, self).__init__()
        self.env_rank = int(env_rank) if env_rank is not None else -1
        
        # Set log directory
        if log_dir is None:
            self.log_dir = os.path.dirname(__file__)
        else:
            self.log_dir = log_dir
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir, exist_ok=True)
        
        # Load configuration from YAML file (per-task when task_id is provided)
        base_dir = os.path.dirname(__file__)
        if task_id is not None:
            try:
                task_id_str = str(int(task_id))
            except (TypeError, ValueError):
                task_id_str = str(task_id)
            config_name = f"config_T{task_id_str}.yaml"
        else:
            config_name = "config_T25.yaml"

        config_path = os.path.join(base_dir, config_name)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Get parameters from config
        para_dict = self.config['para_dict'].copy()
        
        if solver_type is not None:
            para_dict['Solver_Type'] = solver_type

        self.solver_type = para_dict.get("Solver_Type", "Casadi")
        
        # Set simulation parameters from config
        self.V_min = 2.5
        self.V_max = 4.2
        self.cap_0 = 4.86491 # Initial capacity (Ah)

        # Timeout settings
        self.timeout_text = "Timed out. The simulation did not complete within the specified time limit."
        self.timelimit_seconds = int(60*10) # 10 minutes for charging + discharge
        self.enable_timeout = True
        
        self.ageing_temp = int(para_dict.get('Ageing temperature', 25))
        # Print aging temperature
        print(f"Aging T is {self.ageing_temp}degC")
        
        tot_cyc = para_dict.get('Total ageing cycles', 256)
        cyc_age = para_dict.get('Ageing cycles between RPT', 1)
        update = para_dict.get('Update cycles for ageing', 1)
        para_dict["Total ageing cycles"]       = int(tot_cyc)
        para_dict["Ageing cycles between RPT"] = int(cyc_age)
        para_dict["Update cycles for ageing"]  = int(update)

        CyclePack, self.para_0_base = fun.Para_init(para_dict, self.cap_0)
        
        [self.Total_Cycles, _, _, _, self.Temper_i, _, self.mesh_list,
         self.submesh_strech, self.model_options, self.cap_increase] = CyclePack
         
        # Define Action Space
        # [Charge Rate 1 (0-20% SOC), Charge Rate 2 (20-80% SOC), CV Terminal Current (C-rate)]
        # Discrete values:
        # - Charge Rates: 0.1, 0.2, ..., 1.9, 2.0 (C-rate)
        # - Terminal Current: 0.01 (C/100), 0.02 (C/50), 0.05 (C/20), 0.1 (C/10)
        # Previous action space (for reference):
        # - Charge Rates: 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0
        #self.charge_rate_values = np.arange(0.5, 4.01, 0.25)
        #self.charge_rate_values = np.arange(0.1, 2.05, 0.1)
        self.charge_rate_values = np.arange(0.1, 4.05, 0.1)
        self.terminal_current_values = np.array([0.01, 0.02, 0.05, 0.1])
        self.num_charge_rates = len(self.charge_rate_values)
        self.num_terminal_currents = len(self.terminal_current_values)
        # Use MultiDiscrete for 3 independent discrete actions
        self.action_space = spaces.MultiDiscrete([self.num_charge_rates, self.num_charge_rates, self.num_terminal_currents])
        
        # Define Observation Space
        # [SOH, Charge Time Total, CV Time, Average Temperature, Cycle Count, Throughput Capacity,
        # Terminal Voltage (from discharge rest 3h),
        # LLI (Loss of Lithium Inventory, %), LAM_neg (%), LAM_pos (%),
        # Last Charge Rate 1, Last Charge Rate 2, Last CV Terminal Current]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, np.inf, np.inf, 50.0, np.inf, np.inf,
                           5.0, 100.0, 100.0, 100.0, 5.0, 5.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        
        # Internal State
        self.model = None
        self.solution = None
        self.para_0 = None
        self.mdic_dry = None
        
        # Initial Capacity States for Aging Calculation
        self.Q_neg_0 = None
        self.Q_pos_0 = None
        self.Q_Li_0 = None

        # Reward configuration: priority weights
        self.reward_config = self.config.get('reward', {})
        # weights applied to the outputs r_soh, r_time, and r_soh_val respectively
        self.w_soh_weight = float(self.reward_config.get('w_soh_weight', 0.4))
        self.w_time_weight = float(self.reward_config.get('w_time_weight', 0.4))
        self.w_soh_val_weight = float(self.reward_config.get('w_soh_val_weight', 0.2))

        self._reset_tracking_state()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize DryOut Model State
        self.para_0 = self.para_0_base.copy()
        Int_ElelyExces_Ratio = self.para_0["Initial electrolyte excessive amount ratio"]
        self.mdic_dry, self.para_0 = fun.Initialize_mdic_dry(self.para_0, Int_ElelyExces_Ratio)
        
        # Run Break-in Cycle
        # Construct Break-in Experiment
        # Use Initialize_exp_text from Fun_Sim_Only to ensure consistency
        exp_texts = fun.Initialize_exp_text(self.V_max, self.V_min, False)
        exp_breakin_text = exp_texts[4] + exp_texts[8]\
            + [(f"Discharge at 0.1C until {self.V_min}V",
             "Rest for 3 hours")]
        
        experiment_breakin = pb.Experiment(exp_breakin_text)
        
        # Run Break-in
        Result_list_breakin = fun.Run_Breakin(
            self.model_options, experiment_breakin, 
            self.para_0, self.mesh_list, self.submesh_strech, self.cap_increase,
            solver_type=self.solver_type
        )
        
        [Model_0, Sol_0, Call_Breakin] = Result_list_breakin
        self.model = Model_0
        self.solution = Sol_0
        
        # --- Extract Initial Capacities for LLI/LAM Calculation ---
        try:
            if hasattr(Sol_0, 'cycles') and len(Sol_0.cycles) > 0:
                # The break-in experiment ends with a cycle containing (Discharge, Rest).
                # This corresponds to the last cycle.
                last_cycle = Sol_0.cycles[-1]
                steps = last_cycle.steps
                
                # Assuming the last cycle has [Discharge, Rest]
                # We need the end of Discharge step.
                if len(steps) >= 1:
                    # Usually Discharge is the step before Rest. 
                    # If steps = [Discharge, Rest], index 0 is Discharge.
                    # Or check for "Discharge" in step description if available, 
                    # but index approach is usually robust for fixed experiments.
                    # We take the first step of the last cycle (Discharge part).
                    step_discharge = steps[0]
                    
                    self.Q_neg_0 = step_discharge["Negative electrode capacity [A.h]"].entries[-1]
                    self.Q_pos_0 = step_discharge["Positive electrode capacity [A.h]"].entries[-1]
                    self.Q_Li_0  = step_discharge["Total lithium capacity in particles [A.h]"].entries[-1]
                    
                    print(f"Initial Capacities extracted: Q_neg={self.Q_neg_0:.4f}, Q_pos={self.Q_pos_0:.4f}, Q_Li={self.Q_Li_0:.4f}")
                else:
                    print("Warning: Last cycle has no steps for initial capacity extraction.")
            else:
                 print("Warning: Sol_0 has no cycles.")
        except Exception as e:
            print(f"Error extracting initial capacities: {e}")
            # Fallback or critical failure? 
            # If critical, re-raise. For now, set to None/1.0 to avoid crash
            self.Q_neg_0 = 1.0 # Placeholder
            self.Q_pos_0 = 1.0
            self.Q_Li_0 = 1.0
        
        self._reset_tracking_state()
        self.terminal_voltage = self._extract_terminal_voltage(self.solution)

        obs = self._get_observation()

        info = {}
        return obs, info

    def step(self, action):
        # 1. Parse Action
        # Three stages:
        # 1. CC (0% SOC - 20% SOC) - charge rate 1
        # 2. CC (20% SOC - 80% SOC) - charge rate 2
        # 3. CV (80% SOC - 100% SOC) - terminal current
        # Convert action indices to actual values
        chargerate1 = self.charge_rate_values[action[0]]  # 0% to 20% SOC
        chargerate2 = self.charge_rate_values[action[1]]  # 20% to 80% SOC
        terminal_current = self.terminal_current_values[action[2]]  # CV terminal current

        self.last_action = np.array([chargerate1, chargerate2, terminal_current], dtype=np.float32)

        print()
        print(f"Action: {action} → Rates: [{chargerate1:.2f}, {chargerate2:.2f}]C, Terminal Current: {terminal_current:.3f}C")

        # 2. Construct Experiment for this step (Multi-stage CC-CV Charging)
        # Calculate duration for SOC-based steps using current SOH to maintain relative SOC windows
        # Duration (hours) = (Target_SOC_Delta * SOH) / C_rate   
        # Stage 1: CC at rate1 until 20% SOC (Delta = 0.2)
        duration1_sec = 0.2  / chargerate1 * 3600.0
        
        # Stage 2: CC at rate2 until 80% SOC (Delta = 0.6)
        duration2_sec = 0.6  / chargerate2 * 3600.0
        
        # Stage 3: CV at V_max until terminal current
        # No duration calculation needed for CV stage
        
        # Split experiment to ensure DryOut update happens at Top of Charge (100% SOC)
        
        # Part 1: Charge to 100%
        exp_charge_text = [(
            f"Charge at {chargerate1}C for {duration1_sec:.2f} seconds or until {self.V_max*0.9} V",
            f"Charge at {chargerate2}C for {duration2_sec:.2f} seconds or until {self.V_max} V",
            f"Hold at {self.V_max}V until {terminal_current}C",
            f"Rest for 3 hours"
        )]
        exp_charge = pb.Experiment(exp_charge_text)
        Charge_Indexes = [0, 1, 2]
        
        # Part 2: Discharge to 0% (Performance Test)
        exp_discharge_text = [(
            f"Discharge at 0.1C until {self.V_min}V",
            "Rest for 3 hours"
        )]
        exp_discharge = pb.Experiment(exp_discharge_text)
        
        # 4. Run Simulation for this step
        try:
            # --- Part 1: Run Charge (Action) ---
            # This step starts from previous state (0% SOC) and ends at 100% SOC
            # Run_Model_Base_On_Last_Solution handles carrying over state and throughput capacity
            print("Charging...")
            
            # Use the single-process signal-based timeout mechanism if enabled
            if self.enable_timeout:
                charge_caller = fun.TimeoutFunc(
                    fun.Run_Model_Base_On_Last_Solution,
                    timeout=self.timelimit_seconds,
                    timeout_val=self.timeout_text,
                )
            else:
                charge_caller = fun.Run_Model_Base_On_Last_Solution

            Result_List_Charge = charge_caller(
                self.model, self.solution, self.para_0, exp_charge,
                1, self.Temper_i, self.mesh_list, self.submesh_strech,
                solver_type=self.solver_type
            )
            [Model_Chg, Sol_Chg, Call_Chg, _] = Result_List_Charge
            
            if isinstance(Sol_Chg, str):
                print(f"Charge Failed: {Sol_Chg}")
                return self._get_observation(), -1000.0, True, False, {"error": Sol_Chg}
            if hasattr(Call_Chg, 'success') and not Call_Chg.success:
                print("Charge Failed: Infeasible")
                return self._get_observation(), -1000.0, True, False, {"error": "Infeasible Charge"}
            
            # --- Update Parameters (At Top of Charge / 100% SOC) ---
            # Now we are at 100% SOC, safe to call Cal_new_con_Update for correct DryOut logic
            Data_Pack, Paraupdate = fun.Cal_new_con_Update(
                Sol_Chg, self.para_0
            )
            self.mdic_dry = fun.Update_mdic_dry(Data_Pack, self.mdic_dry)
            self.para_0 = Paraupdate
            
            # --- Part 2: Run Discharge (Performance Test) ---
            # This step starts from 100% SOC and ends at 0% SOC
            print("Discharging...")
            
            if self.enable_timeout:
                discharge_caller = fun.TimeoutFunc(
                    fun.Run_Model_Base_On_Last_Solution_RPT,
                    timeout=self.timelimit_seconds,
                    timeout_val=self.timeout_text,
                )
            else:
                discharge_caller = fun.Run_Model_Base_On_Last_Solution_RPT

            Result_List_Discharge = discharge_caller(
                Model_Chg, Sol_Chg, self.para_0, exp_discharge,
                1, self.Temper_i, self.mesh_list, self.submesh_strech,
                solver_type=self.solver_type
            )
            [Model_Dis, Sol_Dis, Call_Dis, _] = Result_List_Discharge
            
            if isinstance(Sol_Dis, str):
                print(f"Discharge Failed: {Sol_Dis}")
                return self._get_observation(), -1000.0, True, False, {"error": Sol_Dis}
            if hasattr(Call_Dis, 'success') and not Call_Dis.success:
                print("Discharge Failed: Infeasible")
                return self._get_observation(), -1000.0, True, False, {"error": "Infeasible Discharge"}

            self.model = Model_Dis
            self.solution = Sol_Dis # Final state is 0% SOC, ready for next step
            
            # Always update these (every step)
            self.cycle_count += 1
            
            # 6. Calculate Metrics (Reward & Observation)
            d_cap = self._extract_discharge_capacity(self.solution)
            if d_cap == 0.0 or np.isnan(d_cap):
                print("Failed to calculate discharge capacity")
                return self._get_observation(), -1000.0, True, False, {"error": "Zero discharge capacity"}

            self.soh_measured = float(d_cap / self.cap_0)

            charge_metrics = self._extract_charge_metrics(Sol_Chg, Charge_Indexes)
            if charge_metrics is None:
                print("No charge steps found in the last cycle")
                return self._get_observation(), -1000.0, True, False, {"error": "No charge steps"}
            self.charge_time = charge_metrics["charge_time_total"]
            self.charge_time_stage1 = charge_metrics["charge_time_stage1"]
            self.charge_time_stage2 = charge_metrics["charge_time_stage2"]
            self.charge_time_stage3 = charge_metrics["charge_time_stage3"]
            self.avg_temperature = charge_metrics["avg_temperature"]

            # Get Throughput capacity from solution (Accumulated)
            self.throughput_capacity = self.solution["Throughput capacity [A.h]"].entries[-1]

            self.terminal_voltage = self._extract_terminal_voltage(self.solution)
            
            self._update_aging_metrics(self.solution)
            self.soh_param = self.lli + self.lam_pos + (self.lli * self.lam_pos)

        except Exception as e:
            print(f"Step failed with exception: {e}")
            import traceback
            traceback.print_exc()
            return self._get_observation(), -1000.0, True, False, {"error": str(e)}

        # 7. Calculate reward using only SOH and total charge time.
        reward_components = self._compute_reward(
            soh=self.soh_measured,
            charge_time=self.charge_time,
            soh_param=self.soh_param,
        )
        reward = reward_components["total"]
        
        # 8. Check Termination
        terminated = False
        truncated = False
        
        if self.cycle_count >= self.Total_Cycles:
            terminated = True

        obs = self._get_observation()
        info = {
            "env_rank": self.env_rank,
            "cycle_count": self.cycle_count,
            "soh_measured": self.soh_measured,
            "soh_param": self.soh_param,
            "delta_soh": reward_components["delta_soh"],
            "charge_time": self.charge_time,
            "charge_time_stage1": self.charge_time_stage1,
            "charge_time_stage2": self.charge_time_stage2,
            "charge_time_stage3": self.charge_time_stage3,
            "avg_temperature": self.avg_temperature,
            "throughput": self.throughput_capacity,
            "terminal_voltage": self.terminal_voltage,
            "lli": self.lli,
            "lam_neg": self.lam_neg,
            "lam_pos": self.lam_pos,
            "reward_components": reward_components,
            "tensorboard_metrics": {
                "cycle": float(self.cycle_count),
                "chargerate1_c": float(chargerate1),
                "chargerate2_c": float(chargerate2),
                "terminal_current_c": float(terminal_current),
                "soh_measured": float(self.soh_measured),
                "soh_param": float(self.soh_param),
                "charge_time_s1": float(self.charge_time_stage1),
                "charge_time_s2": float(self.charge_time_stage2),
                "charge_time_s3": float(self.charge_time_stage3),
                "charge_time_s_total": float(self.charge_time),
                "avg_temp_c": float(self.avg_temperature),
                "throughput_ah": float(self.throughput_capacity),
                "discharge_capacity_ah": float(d_cap),
                "terminal_voltage_v": float(self.terminal_voltage),
                "reward_total": float(reward_components["total"]),
                "reward_soh": float(reward_components["r_soh"]),
                "reward_time": float(reward_components["r_time"]),
                "reward_soh_val": float(reward_components["r_soh_val"]),
                "delta_soh": float(reward_components["delta_soh"]),
                "lli": float(self.lli),
                "lam_neg": float(self.lam_neg),
                "lam_pos": float(self.lam_pos),
            },
        }
        
        self.render(info=info)
        
        return obs, reward, terminated, truncated, info

    def _compute_reward(self, soh, charge_time, soh_param):
        delta_soh = float(soh_param) - self.last_soh_param
        x_time = float(charge_time)
        x_soh_val = float(soh)

        if self.ageing_temp == 10:
            r_soh = 1.0 / (1.0 + np.exp(21.0 * (delta_soh - 0.5)))
            r_time = 1.0 / (1.0 + np.exp(0.01 * (x_time - 3700.0)))
            r_soh_val = 1.0 - 1.0 / (1.0 + np.exp(47.3 * (x_soh_val - 0.8)))

        elif self.ageing_temp == 25:
            r_soh = 1.0 / (1.0 + np.exp(31.0 * (delta_soh - 0.18)))
            r_time = 1.0 / (1.0 + np.exp(0.01 * (x_time - 2800.0)))
            r_soh_val = 1.0 - 1.0 / (1.0 + np.exp(47.3 * (x_soh_val - 0.8)))

        else:
            r_soh = 1.0 / (1.0 + np.exp(37.0 * (delta_soh - 0.1)))
            r_time = 1.0 / (1.0 + np.exp(0.01 * (x_time - 2400.0)))
            r_soh_val = 1.0 - 1.0 / (1.0 + np.exp(47.3 * (x_soh_val - 0.8)))

        w_soh = getattr(self, 'w_soh_weight', 0.4)
        w_time = getattr(self, 'w_time_weight', 0.4)
        w_soh_val = getattr(self, 'w_soh_val_weight', 0.2)

        r_total = w_soh * r_soh + w_time * r_time + w_soh_val * r_soh_val

        reward_components = {
            "total": float(r_total),
            "r_soh": float(r_soh),
            "r_time": float(r_time),
            "r_soh_val": float(r_soh_val),
            "delta_soh": float(delta_soh),
        }
        self.last_soh = float(soh)
        self.last_soh_param = float(soh_param)
        return reward_components

    def _reset_tracking_state(self):
        self.cycle_count = 0
        self.soh_measured = 1.0
        self.last_soh = 1.0
        self.soh_param = 0.0
        self.last_soh_param = 0.0
        self.lli = 0.0
        self.lam_neg = 0.0
        self.lam_pos = 0.0
        self.charge_time = 0.0
        self.charge_time_stage1 = 0.0
        self.charge_time_stage2 = 0.0
        self.charge_time_stage3 = 0.0
        self.avg_temperature = self.Temper_i - 273.15
        self.throughput_capacity = 0.0
        self.terminal_voltage = float(self.V_min)
        self.last_action = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def _extract_discharge_capacity(self, solution):
        if hasattr(solution, 'cycles') and len(solution.cycles) > 0:
            last_cycle = solution.cycles[-1]
            if len(last_cycle.steps) > 0:
                step_dis = last_cycle.steps[0]
                return step_dis["Discharge capacity [A.h]"].entries[-1] - step_dis["Discharge capacity [A.h]"].entries[0]
        return 0.0

    def _extract_charge_metrics(self, charge_solution, charge_indexes):
        if not hasattr(charge_solution, 'cycles') or len(charge_solution.cycles) == 0:
            return None

        last_cycle = charge_solution.cycles[-1]
        charge_time_total = 0.0
        charge_time_stage1 = 0.0
        charge_time_stage2 = 0.0
        charge_time_stage3 = 0.0
        temperatures = []
        for stage_i, index in enumerate(charge_indexes[:3]):
            if index < len(last_cycle.steps):
                step = last_cycle.steps[index]
                try:
                    time = step["Time [s]"].entries[-1] - step["Time [s]"].entries[0]
                    charge_time_total += time
                    if stage_i == 0:
                        charge_time_stage1 = time
                    elif stage_i == 1:
                        charge_time_stage2 = time
                    elif stage_i == 2:
                        charge_time_stage3 = time
                    try:
                        temperatures.extend(step["Volume-averaged cell temperature [C]"].entries)
                    except (KeyError, TypeError):
                        pass
                except (TypeError, KeyError):
                    continue

        avg_temperature = float(np.mean(np.array(temperatures))) if temperatures else 25.0
        return {
            "charge_time_total": charge_time_total,
            "charge_time_stage1": charge_time_stage1,
            "charge_time_stage2": charge_time_stage2,
            "charge_time_stage3": charge_time_stage3,
            "avg_temperature": avg_temperature,
        }

    def _extract_terminal_voltage(self, solution):
        if hasattr(solution, 'cycles') and len(solution.cycles) > 0:
            last_cycle = solution.cycles[-1]
            if len(last_cycle.steps) > 0:
                step_rest = last_cycle.steps[1] if len(last_cycle.steps) > 1 else last_cycle.steps[-1]
                for key in ("Terminal voltage [V]", "Voltage [V]"):
                    try:
                        return float(step_rest[key].entries[-1])
                    except KeyError:
                        continue
        return float(self.V_min)

    def _update_aging_metrics(self, solution):
        try:
            if hasattr(solution, 'cycles') and len(solution.cycles) > 0:
                last_cycle = solution.cycles[-1]
                if len(last_cycle.steps) > 0:
                    step_discharge = last_cycle.steps[0]
                    q_neg_curr = step_discharge["Negative electrode capacity [A.h]"].entries[-1]
                    q_pos_curr = step_discharge["Positive electrode capacity [A.h]"].entries[-1]
                    q_li_curr = step_discharge["Total lithium capacity in particles [A.h]"].entries[-1]
                    self.lam_neg = (1.0 - (q_neg_curr / self.Q_neg_0)) * 100.0
                    self.lam_pos = (1.0 - (q_pos_curr / self.Q_pos_0)) * 100.0
                    self.lli = (1.0 - (q_li_curr / self.Q_Li_0)) * 100.0
        except Exception as e:
            print(f"Warning: Failed to extract aging params: {e}")

    def _get_observation(self):
        # Observation keeps measured SOH from the latest discharge capacity calculation.
        obs_base = np.array(
            [
                self.soh_measured,
                self.charge_time,
                self.charge_time_stage3,
                self.avg_temperature,
                self.cycle_count,
                self.throughput_capacity,
                self.terminal_voltage,
                self.lli,
                self.lam_neg,
                self.lam_pos,
            ],
            dtype=np.float32,
        )
        return np.concatenate((obs_base, self.last_action)).astype(np.float32)

    def render(self, mode='human', info=None):
        if info is None:
            return
        
        # 直接硬编码解析 info 和 tensorboard_metrics
        tb = info.get("tensorboard_metrics", {})
        
        cr1 = tb.get("chargerate1_c", 0.0)
        cr2 = tb.get("chargerate2_c", 0.0)
        tc = tb.get("terminal_current_c", 0.0)
        
        ct1 = info.get("charge_time_stage1", 0.0)
        ct2 = info.get("charge_time_stage2", 0.0)
        ct3 = info.get("charge_time_stage3", 0.0)
        
        rc = info.get("reward_components", {})
        r_tot = rc.get("total", 0.0)
        r_soh = rc.get("r_soh", 0.0)
        r_time = rc.get("r_time", 0.0)
        r_soh_val = rc.get("r_soh_val", 0.0)
        d_soh = rc.get("delta_soh", 0.0)
        
        avg_t = info.get("avg_temperature", 0.0)
        thru = info.get("throughput", 0.0)
        d_cap = tb.get("discharge_capacity_ah", 0.0)
        t_v = info.get("terminal_voltage", 0.0)
        
        lli = info.get("lli", 0.0)
        lam_neg = info.get("lam_neg", 0.0)
        lam_pos = info.get("lam_pos", 0.0)

        print(f"Cycle: {self.cycle_count}, SOH(measured): {self.soh_measured:.6f}")
        print(f"Charge Rates: Stage1={cr1:.2f}C, Stage2={cr2:.2f}C | Terminal Current={tc:.3f}C")
        print(f"Stage Times: S1={ct1:.2f}s, S2={ct2:.2f}s, S3={ct3:.2f}s")
        print(f"Rewards: Total={r_tot:.6f}, r_soh={r_soh:.6f}, r_time={r_time:.6f}, r_soh_val={r_soh_val:.6f}, delta_soh={d_soh:.6f}")
        print(f"Metrics: Avg Temp={avg_t:.2f}°C, Throughput={thru:.4f}Ah, Discharge Cap={d_cap:.4f}Ah, Terminal V={t_v:.4f}V")
        print(f"Aging: LLI={lli:.6f}%, LAM_neg={lam_neg:.6f}%, LAM_pos={lam_pos:.6f}%")

    def close(self):
        pass
