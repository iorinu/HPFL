"""YAML 実験の検証と隔離実行。Nix は依存環境だけを担当する。"""
import argparse
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from configs.experiment import load_experiment, hpfl_config
from utils.final_artifact import exclusive_run

RUN_ROOT = ROOT / ".local-experiments"
launch_worker = subprocess.Popen


def execute_run(cfg, device, source, command, output=None, timeout=None, overrides=()):
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if output is None:
        run = Path(tempfile.mkdtemp(prefix="run-", dir=RUN_ROOT))
    else:
        run = Path(output).resolve()
        if not run.is_relative_to(RUN_ROOT.resolve()):
            raise ValueError("出力先は .local-experiments 内が必要です")
        run.mkdir(parents=True, exist_ok=False)
    status = 1
    process = None
    try:
        (run / "original.yaml").write_text(Path(source).read_text())
        (run / "effective.yaml").write_text(cfg.dump())
        (run / "command.json").write_text(json.dumps(command, ensure_ascii=False) + "\n")
        bound = timeout if timeout is not None else (180 if cfg.runner.mode == "synthetic" else None)
        (run / "execution.json").write_text(json.dumps(dict(device=device, mode=cfg.runner.mode,
                                                              timeout_seconds=bound,
                                                              execution_backend=cfg.runner.execution_backend,
                                                              max_concurrent_clients=cfg.runner.max_concurrent_clients,
                                                              client_timeout_seconds=cfg.runner.client_timeout_seconds,
                                                              model_residency=cfg.model_residency,
                                                              capacity_policy=cfg.capacity_policy,
                                                              debug_tensor_dumps=cfg.debug_tensor_dumps,
                                                              rng_policy=hpfl_config(cfg).client_rng_policy), indent=2) + "\n")
        artifacts = run / "artifacts"
        artifacts.mkdir()
        worker = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
                  "--config", str(run / "original.yaml"), "--device", device,
                  "--output", str(artifacts), *overrides]
        if bound is not None:
            worker = ["timeout", "--signal=TERM", "--kill-after=10s", str(bound), *worker]
        with (run / "console.log").open("w") as log:
            process = launch_worker(worker, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, start_new_session=True)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            status = process.wait()
            if status < 0:
                status = 128 - status
    except BaseException:
        status = 1
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
        raise
    finally:
        (run / "exit-status").write_text(str(status) + "\n")
        print(f"出力: {run}\n終了値: {status}", flush=True)
    return run, status


@exclusive_run
def dispatch(cfg, device, output):
    from tools.local_smoke import run_smoke, select_device

    if cfg.runner.mode == "synthetic":
        return run_smoke(cfg, device, output)
    import torch
    import wandb

    select_device(device)
    flat = hpfl_config(cfg)
    flat.gpu_index = 0
    resolved = output / "hpfl.yaml"
    resolved.write_text(flat.dump())
    entry = str(ROOT / "experiments/standalone/main.py")
    previous = Path.cwd()
    try:
        os.chdir(output)
        (output / "result").mkdir()
        with patch.dict(os.environ, dict(WANDB_MODE="disabled", WANDB_DISABLED="true",
                                        HOME=str(output), XDG_CACHE_HOME=str(output / "cache"),
                                        MPLCONFIGDIR=str(output / "matplotlib"))), \
             patch.object(sys, "argv", [entry, "--config_file", str(resolved)]), \
             patch.object(torch.cuda, "is_available", return_value=device == "cuda"):
            # 上流の無条件 summary 書き込みに必要。外部記録・login は行わない。
            run = wandb.init(mode="disabled")
            status = "failed"
            try:
                namespace = runpy.run_path(entry, run_name="__main__")
                artifact = namespace["artifact_info"]
                if not artifact:
                    raise RuntimeError("最終推論成果物が公開されませんでした")
                artifact["path"] = str((namespace["run_output"] / artifact["path"]).relative_to(output))
                status = "passed"
            finally:
                try:
                    (output / "summary.json").write_text(json.dumps(
                        dict(status=status, synthetic=False, device=device, metrics=dict(run.summary),
                             artifact=artifact if status == "passed" else None,
                             model_residency=cfg.model_residency, capacity_policy=cfg.capacity_policy),
                        indent=2, default=lambda value: value.tolist(), ensure_ascii=False) + "\n")
                finally:
                    run.finish()
    finally:
        os.chdir(previous)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/smoke.yaml")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--check", "--dry-run", action="store_true")
    parser.add_argument("--timeout", type=float, help="実行上限秒。合成は既定180秒、実データは既定上限なし")
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="YACS KEY VALUE の明示上書き")
    options = parser.parse_args(argv)
    try:
        cfg = load_experiment(options.config, options.opts)
        if options.timeout is not None:
            import math
            if not math.isfinite(options.timeout) or options.timeout <= 0:
                raise ValueError("timeout は有限の正数が必要です")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(dict(mode=cfg.runner.mode, device=options.device,
                          rounds=cfg.comm_round, local_epochs=cfg.global_epochs_per_round,
                          head_epochs=cfg.HPFL_local_iteration, clients=cfg.client_num_in_total,
                          participating=cfg.client_num_per_round, data_dir=cfg.data_dir,
                          batch_size=cfg.batch_size), ensure_ascii=False))
    if options.check:
        print(cfg.dump())
        return 0
    if options.worker:
        dispatch(cfg, options.device, options.output.resolve())
        return 0
    _, status = execute_run(cfg, options.device, options.config,
                            list(sys.argv[1:] if argv is None else argv), output=options.output,
                            timeout=options.timeout, overrides=options.opts)
    return status


if __name__ == "__main__":
    sys.exit(main())
