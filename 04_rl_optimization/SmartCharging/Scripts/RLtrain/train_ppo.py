import os
# 强制限制底层数学计算库的内部多线程，彻底消除与 SubprocVecEnv 及其底层求解器的进程级并发冲突
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
# Delay import of BatteryEnv to inside the worker function to avoid global state pollution
# from BatteryEnv import BatteryEnv 
import numpy as np
import os
import sys
import warnings
import json
import yaml
from pathlib import Path

# Filter purely annoying warnings from third-party libs
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=UserWarning, module="pybamm")


# Global Variables for paths
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
BOARD_ROOT = PACKAGE_ROOT.parent
ROOT_SAVE_DIR = BOARD_ROOT / "outputs" / "training" / "overhauled_train_iter2"

TEMPERATURE_MAP = {0: 10, 1: 25, 2: 40}
DEFAULT_TB_LOG_NAME = "PPO"
DEFAULT_LR_FINAL_RATIO = 0.1


def env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def resolve_temperature():
    temp_str = os.environ.get("SLURM_ARRAY_TASK_ID")
    try:
        task_id = int(temp_str) if temp_str else 1
        return TEMPERATURE_MAP.get(task_id, 25)
    except ValueError:
        return 25


def resolve_num_cpu():
    slurm_cpus_per_task = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus_per_task:
        try:
            return max(1, int(slurm_cpus_per_task))
        except ValueError:
            pass

    slurm_job_cpus = os.environ.get("SLURM_JOB_CPUS_PER_NODE")
    if slurm_job_cpus:
        first_chunk = slurm_job_cpus.split(",", 1)[0]
        first_value = first_chunk.split("(", 1)[0]
        try:
            return max(1, int(first_value))
        except ValueError:
            pass

    detected_cpu_count = os.cpu_count() or 1
    return max(1, int(detected_cpu_count))


def default_model_path(temperature):
    return ROOT_SAVE_DIR / f"T{temperature}" / f"ppo_battery_aging_model_T{temperature}.zip"


def load_runtime_config(temperature):
    config_file_name = f"config_T{temperature}.yaml"
    config_path = os.path.join(os.path.dirname(__file__), config_file_name)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_learning_rate_schedule(learning_rate_config):
    if not isinstance(learning_rate_config, (int, float)):
        raise ValueError(
            "ppo.learning_rate must be a scalar base learning rate; train_ppo.py now applies automatic decay."
        )

    initial_value = float(learning_rate_config)
    if initial_value <= 0.0:
        raise ValueError("ppo.learning_rate must be positive.")

    final_value = initial_value * DEFAULT_LR_FINAL_RATIO

    def cosine_decay_schedule(progress_remaining):
        clipped_progress = float(np.clip(progress_remaining, 0.0, 1.0))
        cosine_scale = 0.5 * (1.0 + np.cos(np.pi * (1.0 - clipped_progress)))
        return final_value + (initial_value - final_value) * cosine_scale

    return cosine_decay_schedule, {
        "mode": "cosine_decay",
        "initial_value": initial_value,
        "final_value": final_value,
        "final_ratio": DEFAULT_LR_FINAL_RATIO,
    }


def resolve_ppo_config(runtime_config):
    ppo_config = runtime_config.get("ppo", {}).copy()
    learning_rate, learning_rate_metadata = build_learning_rate_schedule(
        ppo_config.get("learning_rate", 3e-4)
    )
    ppo_config["learning_rate"] = learning_rate
    return ppo_config, learning_rate_metadata


def format_learning_rate_metadata(learning_rate_metadata):
    return (
        f"{learning_rate_metadata['mode']} "
        f"initial={learning_rate_metadata['initial_value']:.6g}, "
        f"final={learning_rate_metadata['final_value']:.6g}, "
        f"final_ratio={learning_rate_metadata['final_ratio']:.3f}"
    )


class TensorboardMetricsCallback(BaseCallback):
    def _on_step(self):
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        per_env_metric_keys = [
            "cycle",
            "chargerate1_c",
            "chargerate2_c",
            "terminal_current_c",
            "soh_measured",
            "delta_soh",
            "charge_time_s1",
            "charge_time_s2",
            "charge_time_s3",
            "charge_time_s_total",
            "avg_temp_c",
            "throughput_ah",
            "discharge_capacity_ah",
            "reward_total",
            "reward_soh",
            "reward_time",
            "lli",
            "lam_neg",
            "lam_pos",
        ]
        for env_index, info in enumerate(infos):
            if not isinstance(info, dict):
                continue
            metrics = info.get("tensorboard_metrics")
            if not metrics:
                continue
            env_rank = int(info.get("env_rank", env_index))
            for key in per_env_metric_keys:
                value = metrics.get(key)
                if value is not None:
                    self.logger.record(
                        f"envs/env_{env_rank}/{key}",
                        float(value),
                        exclude=("stdout", "log"),
                    )

        return True


def save_training_metadata(
    save_dir,
    temperature,
    num_cpu,
    total_timesteps,
    tensorboard_log,
    tb_log_name,
    runtime_config,
    learning_rate_metadata,
    resumed_from=None,
):
    os.makedirs(save_dir, exist_ok=True)

    metadata = {
        "algorithm": "PPO",
        "temperature": temperature,
        "num_cpu": num_cpu,
        "total_timesteps": total_timesteps,
        "tb_log_name": tb_log_name,
        "tensorboard_log": tensorboard_log,
        "solver_type": "IDAKLU",
        "ppo_hyperparameters": runtime_config.get("ppo", {}),
        "resolved_learning_rate": learning_rate_metadata,
        "resumed_from": resumed_from,
        "reward_config": runtime_config.get("reward", {}),
        "para_dict": runtime_config.get("para_dict", {}),
    }

    metadata_path = os.path.join(save_dir, "training_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Training metadata saved to {metadata_path}")

def make_env_function(rank, temperature):
    """
    Utility function for multiprocessed env.
    
    :param rank: (int) index of the subprocess
    :param temperature: (int) temperature for the environment
    :return: (Callable)
    """
    def _init():
        # Define output directory for this environment
        log_dir = os.path.join(ROOT_SAVE_DIR, f"T{temperature}")
        os.makedirs(log_dir, exist_ok=True)
        
        # Redirect output to file
        log_file_path = os.path.join(log_dir, f"env_{rank}.log")
        
        # Robust redirection:
        # 2. Open the log file
        log_file = open(log_file_path, "w") # Default block buffering
        
        # 3. Redirect stdout (fd 1) and stderr (fd 2) to the log file's fd
        os.dup2(log_file.fileno(), sys.stdout.fileno())
        os.dup2(log_file.fileno(), sys.stderr.fileno())
        
        # 4. Replace sys.stdout/stderr objects to ensure Python plays nice with the new FD settings
        sys.stdout = log_file 
        sys.stderr = log_file
        
        print(f"Environment {rank} initialized. Logging to {log_file_path}")
        
        # Local import to prevent main process warnings/output
        try:
            try:
                from .BatteryEnv import BatteryEnv
            except ImportError:
                from BatteryEnv import BatteryEnv
            env = BatteryEnv(task_id=temperature, log_dir=log_dir, solver_type="IDAKLU", env_rank=rank)
            print("BatteryEnv instantiated successfully.")
            return env
        except Exception as e:
            print(f"Failed to initialize BatteryEnv: {e}", file=sys.stderr)
            raise e

    return _init

def main():
    # 1. Configuration
    temperature = resolve_temperature()
    runtime_config = load_runtime_config(temperature)
    ppo_config, learning_rate_metadata = resolve_ppo_config(runtime_config)

    resume_training = env_flag("RESUME_TRAINING", default=True)
    total_timesteps = int(runtime_config.get("total_timesteps", 30720))
    tb_log_name = DEFAULT_TB_LOG_NAME
        
    num_cpu = resolve_num_cpu()
    
    # Print to main process stdout (which goes to SLURM general log)
    print(f"Starting Training for Temperature {temperature}C with {num_cpu} cores.")
    print(f"Configured total_timesteps: {total_timesteps}")
    print(f"Configured PPO learning rate: {format_learning_rate_metadata(learning_rate_metadata)}")
    
    # 2. Create Vectorized Environment
    # Create list of callables
    env_methods = [make_env_function(i, temperature) for i in range(num_cpu)]
    
    # Create the vectorized environment
    # start_method='fork' is default on Linux, usually fine but 'spawn' might be safer for PyBaMM if issues arise
    vec_env = SubprocVecEnv(env_methods)
    
    # 3. Define PPO Agent
    # Tensorboard log directory
    save_dir = os.path.join(ROOT_SAVE_DIR, f"T{temperature}")
    tensorboard_log = os.path.join(save_dir, "tensorboard")
    os.makedirs(tensorboard_log, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    resume_model_path = Path(os.environ.get("RESUME_MODEL_PATH", str(default_model_path(temperature)))).expanduser()
    resumed_from = None

    if resume_training and resume_model_path.exists():
        print(f"Loading existing model for T={temperature}C from {resume_model_path} and continuing training.")
        model = PPO.load(
            str(resume_model_path),
            env=vec_env,
            tensorboard_log=tensorboard_log,
            verbose=1,
            **ppo_config,
        )
        resumed_from = str(resume_model_path)
    else:
        if resume_training:
            print(f"No existing model found at {resume_model_path}; starting a new model for T={temperature}C.")
        else:
            print(f"RESUME_TRAINING is disabled; starting a new model for T={temperature}C.")

        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=tensorboard_log,
            **ppo_config,
        )
    
    # 4. Train Agent
    print(f"Starting PPO Training for T={temperature}C...")
    save_training_metadata(save_dir=save_dir,
                           temperature=temperature,
                           num_cpu=num_cpu,
                           total_timesteps=total_timesteps,
                           tensorboard_log=tensorboard_log,
                           tb_log_name=tb_log_name,
                           runtime_config=runtime_config,
                           learning_rate_metadata=learning_rate_metadata,
                           resumed_from=resumed_from)
    model.learn(
        total_timesteps=total_timesteps,
        callback=TensorboardMetricsCallback(),
        reset_num_timesteps=not bool(resumed_from),
        tb_log_name=tb_log_name,
    )
    
    # 5. Save Model
    model_name = f"ppo_battery_aging_model_T{temperature}"
    save_path = os.path.join(save_dir, model_name)
    
    model.save(save_path)
    print(f"Model saved to {save_path}.zip")
    
    # Close environments
    vec_env.close()

if __name__ == "__main__":
    main()
