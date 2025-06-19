import os
import subprocess
import torch

def setup_gpu(config):
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if "gpu_ids" not in config:
            config["gpu_ids"] = [0]
        elif type(config["gpu_ids"]) == int:
            config["gpu_ids"] = [config["gpu_ids"]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(x) for x in config["gpu_ids"]]
        )
        print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Using cpu")

    first_gpu_id = config["gpu_ids"][0]
    device = torch.device(
        f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu"
    )
    return device

def log_semaphore_count(label=""):
    try:
        result = subprocess.check_output("ls /dev/shm | grep -c ^sem\\.", shell=True)
        count = int(result.decode().strip())
        print(f"[DEBUG] 🔍 /dev/shm semaphore count {label}: {count}")
        return count
    except Exception as e:
        print(f"[WARN] Could not check /dev/shm semaphores: {e}")
        return -1

def clean_stale_semaphores(threshold=40):
    count = log_semaphore_count("before cleanup")
    if count > threshold:
        print(f"⚠️ Too many semaphores ({count}), cleaning...")
        subprocess.run("ls /dev/shm/sem.* 2>/dev/null | xargs -r rm -v", shell=True)