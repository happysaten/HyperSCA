"""Launch, record, and clean up tune.py training processes in batch."""

from pathlib import Path
from datetime import datetime, timezone
import torch
import sys, os, subprocess, time, platform, errno

BASE_DIR = Path.cwd()
VENV_PY = BASE_DIR / ".venv" / "bin" / "python"
PYTHON_BIN = (
    str(VENV_PY)
    if VENV_PY.is_file() and os.access(str(VENV_PY), os.X_OK)
    else sys.executable
)
SCRIPT_PATH = BASE_DIR / "tune.py"
OUTPUT_DIR = BASE_DIR / "outputs"
HOST = platform.node()
PID_LIST_FILE = OUTPUT_DIR / f"run_tune_pids_{HOST}.txt"


def is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError as e:
        return e.errno == errno.EPERM


def cleanup_and_list() -> None:
    """Clean up stale PID files and rebuild the current run list."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    active_lines = []
    for pfile in OUTPUT_DIR.glob(f"run_*_{HOST}*.pid"):
        # Derive the corresponding output...txt file from run_...pid.
        ofile = OUTPUT_DIR / pfile.name.replace("run_", "output").replace(
            ".pid", ".txt"
        )
        try:
            meta = dict(
                l.split("=", 1) for l in pfile.read_text().splitlines() if "=" in l
            )
            pid = int(meta["pid"])
            if is_running(pid):
                parts = pfile.stem.split("_")
                idx, ts = (parts[1], parts[-1]) if len(parts) >= 4 else ("?", "?")
                active_lines.append(
                    f"run_{idx}_{ts}: {pid} gpu:{meta.get('gpu', '?')} out:{ofile.name} pidf:{pfile.name}\n"
                )
                continue
        except Exception:
            pass  # Treat parse failures or missing fields as invalid files.

        # Delete files belonging to invalid or stopped processes.
        pfile.unlink(missing_ok=True)
        ofile.unlink(missing_ok=True)
        print(f"Cleaned stale: {pfile.name}")

    PID_LIST_FILE.write_text("".join(active_lines), encoding="utf-8")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in ("stop", "--stop"):
        print(f"Stopping {PYTHON_BIN} {SCRIPT_PATH}...")
        subprocess.run(["pkill", "-f", f"{PYTHON_BIN} {SCRIPT_PATH}"], check=False)
        return

    if len(sys.argv) < 2 or not sys.argv[1].isdigit():
        print(f"Usage: {sys.argv[0]} NUM_PROCESS [[DEV_LIST]]")
        sys.exit(1)

    num_proc = int(sys.argv[1])
    n_gpus = torch.cuda.device_count()
    print(f"Detected {n_gpus} GPU(s).")

    devs = list(range(max(1, n_gpus)))
    if len(sys.argv) > 2:
        try:
            devs = [int(x) for x in sys.argv[2].strip("[]").split(",") if x.strip()]
        except ValueError:
            print("Invalid DEV_LIST. Example: [0,1,3]")
            sys.exit(1)
    print(f"Using devices: {devs}")

    cleanup_and_list()
    env = os.environ.copy()

    for i in range(1, num_proc + 1):
        gpu = devs[(i - 1) % len(devs)]
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        pfile = OUTPUT_DIR / f"run_{i}_{HOST}_{ts}.pid"
        ofile = OUTPUT_DIR / f"output{i}_{HOST}_{ts}.txt"

        print(f"[{i}/{num_proc}] Starting on GPU {gpu} -> {ofile.name}")
        with ofile.open("wb") as f:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            p = subprocess.Popen(
                [PYTHON_BIN, str(SCRIPT_PATH)],
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=env,
            )

        pfile.write_text(
            f"pid={p.pid}\ngpu={gpu}\nindex={i}\nhost={HOST}\ntime={ts}\nscript={SCRIPT_PATH.name}\n",
            encoding="utf-8",
        )

        with PID_LIST_FILE.open("a", encoding="utf-8") as f:
            f.write(
                f"run_{i}_{ts}: {p.pid} gpu:{gpu} out:{ofile.name} pidf:{pfile.name}\n"
            )

        if i < num_proc:
            print("Waiting 30s...")
            time.sleep(30)

    print(
        f"---\nAll submitted. Stop cmd:\n  kill $(awk '{{print $2}}' {PID_LIST_FILE})"
    )


if __name__ == "__main__":
    main()
