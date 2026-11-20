#!/usr/bin/env python3
"""Sample hardware state to a CSV while an experiment runs.

Works on any Linux box in the lab with no dependencies:
  - CPU utilisation, average clock, temperature, memory  (all machines)
  - AMD GPU busy %, VRAM/GTT use, power, temperature, clock  (EVO-X2, via sysfs)
  - NVIDIA GPU utilisation, memory, power, temperature, clock  (cloud GPUs, via nvidia-smi)
  - Raspberry Pi throttle flags and ARM clock  (via vcgencmd)
Columns that don't apply to a machine are left empty.

Usage:
  python3 sample_hw.py --out results/hw.csv --interval 1.0
Stop it with Ctrl-C or SIGTERM; the file is flushed after every row.
"""
import argparse
import csv
import glob
import os
import shutil
import signal
import subprocess
import sys
import time


def read(path, default=None):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return default


def cpu_times():
    with open("/proc/stat") as f:
        parts = f.readline().split()[1:]
    vals = [int(x) for x in parts]
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
    return idle, sum(vals)


def cpu_freq_mhz():
    freqs = [read(p) for p in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq")]
    freqs = [int(f) for f in freqs if f and f.isdigit()]
    return round(sum(freqs) / len(freqs) / 1000) if freqs else ""


def cpu_temp_c():
    # Prefer a zone that looks like the CPU package; fall back to zone 0.
    best = None
    for zone in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        kind = (read(f"{zone}/type") or "").lower()
        temp = read(f"{zone}/temp")
        if not temp or not temp.lstrip("-").isdigit():
            continue
        if any(k in kind for k in ("x86_pkg", "cpu", "k10temp", "soc")):
            return int(temp) / 1000
        best = best if best is not None else int(temp) / 1000
    return best if best is not None else ""


def mem_used_mb():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            info[k] = int(v.split()[0])
    return round((info["MemTotal"] - info.get("MemAvailable", info["MemFree"])) / 1024)


def amdgpu():
    """First amdgpu card found in sysfs (covers the integrated GPU on the EVO-X2)."""
    for dev in sorted(glob.glob("/sys/class/drm/card[0-9]*/device")):
        if read(f"{dev}/vendor") != "0x1002":
            continue
        out = {
            "amd_busy_pct": read(f"{dev}/gpu_busy_percent", ""),
            "amd_vram_used_mb": "",
            "amd_gtt_used_mb": "",
            "amd_power_w": "",
            "amd_temp_c": "",
            "amd_sclk_mhz": "",
        }
        for key, name in (("amd_vram_used_mb", "mem_info_vram_used"), ("amd_gtt_used_mb", "mem_info_gtt_used")):
            v = read(f"{dev}/{name}")
            if v and v.isdigit():
                out[key] = round(int(v) / 2**20)
        for hw in glob.glob(f"{dev}/hwmon/hwmon*"):
            p = read(f"{hw}/power1_average") or read(f"{hw}/power1_input")
            if p and p.isdigit():
                out["amd_power_w"] = round(int(p) / 1e6, 1)
            t = read(f"{hw}/temp1_input")
            if t and t.isdigit():
                out["amd_temp_c"] = int(t) / 1000
            fq = read(f"{hw}/freq1_input")
            if fq and fq.isdigit():
                out["amd_sclk_mhz"] = round(int(fq) / 1e6)
        return out
    return {}


NV_FIELDS = "utilization.gpu,memory.used,power.draw,temperature.gpu,clocks.sm"


def nvidia():
    if not shutil.which("nvidia-smi"):
        return {}
    try:
        line = subprocess.run(
            ["nvidia-smi", f"--query-gpu={NV_FIELDS}", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        util, mem, power, temp, sm = [x.strip() for x in line.split(",")]
        return {"nv_util_pct": util, "nv_mem_used_mb": mem, "nv_power_w": power, "nv_temp_c": temp, "nv_sm_mhz": sm}
    except (subprocess.SubprocessError, ValueError):
        return {}


def raspberry_pi():
    if not shutil.which("vcgencmd"):
        return {}
    try:
        thr = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=5).stdout
        clk = subprocess.run(["vcgencmd", "measure_clock", "arm"], capture_output=True, text=True, timeout=5).stdout
        flags = int(thr.strip().split("=")[1], 16)
        return {
            "pi_throttled_hex": hex(flags),
            "pi_undervolt_now": flags & 0x1,
            "pi_freq_capped_now": (flags >> 1) & 0x1,
            "pi_throttled_now": (flags >> 2) & 0x1,
            "pi_soft_temp_limit_now": (flags >> 3) & 0x1,
            "pi_arm_mhz": round(int(clk.strip().split("=")[1]) / 1e6),
        }
    except (subprocess.SubprocessError, ValueError, IndexError):
        return {}


COLUMNS = [
    "ts", "cpu_util_pct", "cpu_mhz", "cpu_temp_c", "mem_used_mb",
    "amd_busy_pct", "amd_vram_used_mb", "amd_gtt_used_mb", "amd_power_w", "amd_temp_c", "amd_sclk_mhz",
    "nv_util_pct", "nv_mem_used_mb", "nv_power_w", "nv_temp_c", "nv_sm_mhz",
    "pi_throttled_hex", "pi_undervolt_now", "pi_freq_capped_now", "pi_throttled_now",
    "pi_soft_temp_limit_now", "pi_arm_mhz",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="CSV file to write")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between samples")
    ap.add_argument("--count", type=int, default=0, help="stop after N samples (0 = until stopped)")
    args = ap.parse_args()

    stop = False

    def handle(_sig, _frm):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        prev_idle, prev_total = cpu_times()
        n = 0
        while not stop:
            time.sleep(args.interval)
            idle, total = cpu_times()
            dt = total - prev_total
            util = round(100 * (1 - (idle - prev_idle) / dt), 1) if dt else ""
            prev_idle, prev_total = idle, total
            row = {"ts": round(time.time(), 3), "cpu_util_pct": util, "cpu_mhz": cpu_freq_mhz(),
                   "cpu_temp_c": cpu_temp_c(), "mem_used_mb": mem_used_mb()}
            row.update(amdgpu())
            row.update(nvidia())
            row.update(raspberry_pi())
            w.writerow(row)
            f.flush()
            n += 1
            if args.count and n >= args.count:
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
