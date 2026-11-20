#!/usr/bin/env python3
"""Record the environment of a run as JSON, so every result can be traced back to
the exact hardware and software that produced it.

Usage:
  python3 collect_env.py --out results/env.json [--note "batch size 8, fan off"]
"""
import argparse
import datetime
import importlib.metadata as md
import json
import os
import platform
import shutil
import subprocess
import sys

# Packages worth recording when installed; missing ones are skipped.
PACKAGES = [
    "torch", "transformers", "vllm", "sglang", "tensorrt_llm", "llama_cpp_python",
    "jax", "jaxlib", "tpu-inference", "accelerate", "numpy", "pandas", "httpx", "aiohttp",
]


def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def run_json(cmd):
    """Run a command that prints JSON and return it parsed, so env.json holds an object
    rather than escaped text. Falls back to the raw output (e.g. an error message)."""
    text = run(cmd)
    try:
        return json.loads(text) if text else text
    except ValueError:
        return text


def read(path):
    try:
        with open(path) as f:
            return f.read().strip().replace("\x00", "")
    except OSError:
        return None


def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith(("model name", "hardware", "cpu model")):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def mem_total_gb():
    try:
        with open("/proc/meminfo") as f:
            return round(int(f.readline().split()[1]) / 2**20, 1)
    except (OSError, ValueError, IndexError):
        return None


def gpus():
    out = {}
    if shutil.which("nvidia-smi"):
        out["nvidia"] = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit",
                             "--format=csv,noheader"])
    if shutil.which("amd-smi"):
        out["amd_smi_static"] = run_json(["amd-smi", "static", "--asic", "--json"])
    if shutil.which("rocminfo"):
        names = run(["sh", "-c", "rocminfo | grep -m3 'Marketing Name'"])
        out["rocm_agents"] = names
    if shutil.which("tpu-info"):
        out["tpu_info"] = run(["tpu-info"])
    pi_model = read("/proc/device-tree/model")
    if pi_model:
        out["board_model"] = pi_model
    return out or None


def packages():
    found = {}
    for name in PACKAGES:
        try:
            found[name] = md.version(name)
        except md.PackageNotFoundError:
            pass
    return found


def lab_git():
    here = os.path.dirname(os.path.abspath(__file__))
    commit = run(["git", "-C", here, "rev-parse", "HEAD"])
    dirty = run(["git", "-C", here, "status", "--porcelain"])
    return {"commit": commit, "dirty": bool(dirty)} if commit else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="", help="free-text note about this run")
    args = ap.parse_args()

    env = {
        "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "os": platform.platform(),
        "kernel": platform.release(),
        "cpu": cpu_model(),
        "cpu_count": os.cpu_count(),
        "mem_total_gb": mem_total_gb(),
        "python": sys.version.split()[0],
        "packages": packages(),
        "accelerators": gpus(),
        "cpu_governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "lab_git": lab_git(),
        "note": args.note,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(env, f, indent=2)
    print(f"environment written to {args.out}")


if __name__ == "__main__":
    main()
