import sys
from pathlib import Path

def main():
    sbatch_path = Path(__file__).resolve().parent / "run_reproduce_li2024_gpu.sbatch"
    if not sbatch_path.is_file():
        print(f"Error: {sbatch_path} does not exist.")
        sys.exit(1)
    
    content = sbatch_path.read_text(encoding="utf-8")
    
    # 验证关键配置
    checks = {
        "#SBATCH --partition=GPUcompute": False,
        "#SBATCH --gres=gpu:1": False,
        "PROJECT_ROOT=\"/home/zxsun/AlphaBattery\"": False,
        "ENV_NAME=\"pp\"": False,
    }
    
    for key in checks:
        if key in content:
            checks[key] = True
    
    # 确保不含有旧的 yanli 路径
    has_old_path = "/home/yanli" in content
    
    failed = False
    for key, passed in checks.items():
        if not passed:
            print(f"Failed check: [{key}] not found in sbatch script")
            failed = True
    
    if has_old_path:
        print("Failed check: Found old path /home/yanli in sbatch script")
        failed = True
        
    if failed:
        sys.exit(1)
        
    print("All sbatch validation checks passed successfully.")
    sys.exit(0)

if __name__ == "__main__":
    main()
