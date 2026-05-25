import multiprocessing as mp
import yaml, os, sys
import numpy as np
import gc
import json
from bayes_opt import BayesianOptimization
from bayes_opt.acquisition import UpperConfidenceBound, ConstantLiar
import warnings
from pathlib import Path

# Suppress PyBaMM output if needed, but logging is fine
warnings.filterwarnings("ignore")

# Load base config
BasicPath = str(Path(__file__).resolve().parent)
os.chdir(BasicPath)

# Ensure logs directory exists to prevent FileNotFoundError
os.makedirs(os.path.join(BasicPath, 'logs'), exist_ok=True)

config_path = os.path.join(BasicPath, 'config.yaml')
if not os.path.exists(config_path):
    raise FileNotFoundError(
        f"Bayesian optimization config not found: {config_path}. "
        "This entrypoint is preserved during repository reorganization, "
        "but still requires a dedicated config.yaml."
    )

with open(config_path, 'r') as file:
    config_data = yaml.safe_load(file)
Para_dict_base = config_data['para_dict']

# Import the actual run function
try:
    from .Fun_NC import Run_P2_Excel
except ImportError:
    from Fun_NC import Run_P2_Excel

# Define the evaluation function wrapper
def evaluate_pybamm_simulation(kwargs_dict):
    params = kwargs_dict['params']
    temp = kwargs_dict['temp']
    scan_no = kwargs_dict['scan_no']

    # Redirect stdout/stderr to a dedicated log file for this worker
    import sys
    log_path = os.path.join(BasicPath, f"logs/worker_T{int(temp)}_Scan{scan_no}.log")
    f_log = open(log_path, 'w')
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = f_log, f_log

    try:
        # Update PyBaMM parameters
        Para_dict_i = Para_dict_base.copy()
        Para_dict_i.update(params)
        Para_dict_i["Ageing temperature"] = temp
        Para_dict_i["Scan No"] = scan_no

        Timelimit = int(3600*48)
        Options = [True, "GEM-2", False, False, True, True, True, True, 100, 13]
        # Fixed path string to prevent os.path.join from jumping to root directory
        Path_List = [BasicPath, "InputData/", f"BO_Temp{int(temp)}_Scan{scan_no}/", f"BO_Run_T{int(temp)}_Scan{scan_no}"]
        
        midc_merge, Sol_RPT, Sol_AGE, DeBug_Lists = Run_P2_Excel(
            Para_dict_i, Path_List, 0, Timelimit, Options
        )
        mpe_tot = float(midc_merge[0].get("Error tot %", np.nan))
        
        # Free memory immediately after getting the target metric
        del midc_merge
        del Sol_RPT
        del Sol_AGE
        del DeBug_Lists
        gc.collect()
        
        # Save current hyperparameters alignment with PyBaMM output structure
        save_dir = os.path.join(BasicPath, f"BayesianOptimizationrunRes/T{int(temp)}")
        os.makedirs(save_dir, exist_ok=True)
        param_save_path = os.path.join(save_dir, f"params_scan_{scan_no}.json")
        try:
            with open(param_save_path, 'w', encoding='utf-8') as f_out:
                json.dump({
                    "temp": int(temp),
                    "scan_no": scan_no,
                    "target_mpe": mpe_tot,
                    "params": {k: float(v) for k, v in params.items()}
                }, f_out, indent=4)
        except Exception as e_json:
            print(f"Failed to save JSON for temp {temp} scan {scan_no}: {e_json}")
        
        # Protect against failures
        if np.isnan(mpe_tot) or mpe_tot > 1e6:
            return {"target": -1e6, "params": params, "temp": temp}
            
        print(f"[Done] Temp {temp}°C | Scan {scan_no} | MPE: {mpe_tot:.4f}")
        return {"target": -mpe_tot, "params": params, "temp": temp}
    except Exception as e:
        print(f"Simulation failed for temp {temp} scan {scan_no}: {e}")
        return {"target": -1e6, "params": params, "temp": temp}
    finally:
        sys.stdout, sys.stderr = orig_stdout, orig_stderr
        f_log.close()

def run_bo():
    bounds = {
        "Inner SEI lithium interstitial diffusivity [m2.s-1]": (1e-20, 1e-17),
        "Dead lithium decay constant [s-1]": (1e-08, 1e-06),
        "Lithium plating kinetic rate constant [m.s-1]": (1e-12, 1e-08),
        "Negative electrode LAM constant proportional term [s-1]": (1e-10, 1e-08),
        "Positive electrode LAM constant proportional term [s-1]": (1e-19, 1e-17),
        "Negative electrode cracking rate": (1e-26, 1e-24),
        "Outer SEI partial molar volume [m3.mol-1]": (1e-05, 1e-04),
        "SEI growth activation energy [J.mol-1]": (1000.0, 10000.0),
        "Negative cracking growth activation energy [J.mol-1]": (0.0, 5000.0),
        "Negative electrode diffusivity activation energy [J.mol-1]": (10000.0, 30000.0),
        "Positive electrode diffusivity activation energy [J.mol-1]": (5000.0, 20000.0),
    }

    opts = {}
    for temp in [10, 25, 40]:
        acq = UpperConfidenceBound(kappa=2.5)
        cl = ConstantLiar(base_acquisition=acq, strategy="mean")
        opt_instance = BayesianOptimization(
            f=None, 
            pbounds=bounds, 
            acquisition_function=cl,
            random_state=temp, 
            allow_duplicate_points=True
        )
        
        # Define checkpoint file path
        log_path = f"logs/bo_checkpoint_temp{temp}.json"
        
        # Load previous logs if they exist (resume capability)
        if os.path.exists(log_path):
            print(f"Loading previous optimization state for {temp}°C from {log_path}...")
            try:
                with open(log_path, 'r') as f_in:
                    old_logs = json.load(f_in)
                for entry in old_logs:
                    try:
                        opt_instance.register(params=entry['params'], target=entry['target'])
                    except KeyError:
                        pass
            except Exception as e:
                print(f"Failed to load logs: {e}")
        
        opts[temp] = opt_instance

    EPOCHS = 5
    WORKERS_PER_TEMP = 8  # Total cores = 24 (8 * 3)

    for epoch in range(EPOCHS):
        print(f"\n{'='*40}")
        print(f"=== EPOCH {epoch+1}/{EPOCHS} ===")
        print(f"{'='*40}")
        tasks = []
        
        # 1. Ask GP for parameters
        for temp in [10, 25, 40]:
            for w in range(WORKERS_PER_TEMP):
                next_point = opts[temp].suggest()
                # Use current log size to determine correct scan number automatically
                scan_no = len(opts[temp].res) + w + 1
                tasks.append({
                    "params": next_point,
                    "temp": temp,
                    "scan_no": scan_no
                })

        # 2. Run parallel evaluations
        total_tasks = len(tasks)
        print(f"Submitting {total_tasks} tasks to localized joblib pool (using 24 processes)...")
        results = []
        
        # Import Loky backend for robust exception handling (e.g., OOM kill -> TerminatedWorkerError)
        from joblib import Parallel, delayed
        
        try:
            results = Parallel(n_jobs=24, backend='loky')(
                delayed(evaluate_pybamm_simulation)(task) for task in tasks
            )
        except Exception as e:
            print(f"Joblib multiprocessing encountered an error (likely OOM): {e}")
            # Filter out tasks that failed entirely to ensure continuity
            results = [{"target": -1e6, "params": t["params"], "temp": t["temp"]} for t in tasks]
            
        for i, res in enumerate(results):
            print(f"  [Progress] Task {i+1}/{total_tasks} processed. (Temp {res['temp']} MPE target: {res['target']})")
        
        # 3. Tell GP the results
        for res in results:
            opt_inst = opts[res["temp"]]
            try:
                opt_inst.register(params=res["params"], target=res["target"])
            except KeyError:
                pass # 避免重复注册触发KeyError
            
            # 手动实现断点记录保存，转换为可序列化的原生类型
            checkpoint_path = f"logs/bo_checkpoint_temp{res['temp']}.json"
            serializable_res = []
            for r in opt_inst.res:
                s_params = {k: float(v) for k, v in r['params'].items()}
                s_target = float(r['target']) if r['target'] is not None else None
                serializable_res.append({"target": s_target, "params": s_params})
            
            with open(checkpoint_path, "w", encoding="utf-8") as f_out:
                json.dump(serializable_res, f_out, indent=4)

        # 4. Dump current state
        print("\n--- Current Best Results ---")
        for temp in [10, 25, 40]:
            if opts[temp].max is not None:
                best_mpe = -opts[temp].max['target']
                print(f"  Temp {temp}°C Best MPE: {best_mpe:.4f}")
            else:
                print(f"  Temp {temp}°C Best MPE: N/A")

    # ==========================================
    # --- 保存最终优化后的超参数及全过程记录至本地文件 ---
    # ==========================================
    print("\n[Output] 正在提取并保存各温度下的最优超参数及全过程采样记录...")
    
    # 获取时间戳后缀（可选，用于区分不同跑批任务防止文件完全覆盖）
    import datetime
    time_suffix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for temp in [10, 25, 40]:
        temp_key = f"Temp_{temp}"
        
        # 1. 记录最优参数 (独立分片存储)
        if opts[temp].max is not None:
            final_best_params = {
                "Best_MPE": -opts[temp].max['target'],
                "Optimized_Parameters": {k: float(v) for k, v in opts[temp].max['params'].items()}
            }
            output_best_path = os.path.join(BasicPath, "logs", f"final_best_parameters_T{temp}_{time_suffix}.json")
            with open(output_best_path, "w", encoding="utf-8") as f_out:
                json.dump(final_best_params, f_out, indent=4, ensure_ascii=False)
            
            # 同时将最优超参数存入与 PyBaMM 输出对应的 T(温度) 文件夹内，从而保证结构清晰
            align_best_path = os.path.join(BasicPath, f"BayesianOptimizationrunRes/T{temp}", "optimal_parameters.json")
            os.makedirs(os.path.dirname(align_best_path), exist_ok=True)
            with open(align_best_path, "w", encoding="utf-8") as f_out:
                json.dump(final_best_params, f_out, indent=4, ensure_ascii=False)
                
            print(f"[Output] Temp {temp}°C 最优超参数已保存至: {output_best_path} 及 {align_best_path}")
            
        # 2. 记录该温度下所有迭代过程的参数与结果 (独立分片存储)
        history_list = []
        for idx, res in enumerate(opts[temp].res):
            mpe_val = float(-res['target']) if res['target'] is not None else None
            history_list.append({
                "Scan_No": idx + 1,
                "MPE": None if (mpe_val is not None and mpe_val >= 1e6) else mpe_val,
                "Parameters": {k: float(v) for k, v in res['params'].items()}
            })
            
        output_history_path = os.path.join(BasicPath, "logs", f"all_evaluated_parameters_T{temp}_{time_suffix}.json")
        with open(output_history_path, "w", encoding="utf-8") as f_out:
            json.dump(history_list, f_out, indent=4, ensure_ascii=False)
        print(f"[Output] Temp {temp}°C 全过程超参数记录已保存至: {output_history_path}")

if __name__ == "__main__":
    mp.set_start_method('spawn')
    run_bo()
