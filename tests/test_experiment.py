"""実験設定と起動境界の回帰テスト。"""
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_loader_precedence(tmp_path):
    from configs.experiment import load_experiment

    source = tmp_path / "experiment.yaml"
    source.write_text("runner:\n  mode: real\nlr: 0.2\n")
    cfg = load_experiment(source, ["lr", "0.3"])
    assert cfg.lr == 0.3
    assert cfg.batch_size == 128
    assert cfg.runner.mode == "real"
    assert Path(cfg.data_dir).is_absolute()
    assert cfg.wandb_record is False


@pytest.mark.parametrize("text, overrides", [
    ("unknown: 1", []), ("batch_size: wrong", []),
    ("runner:\n  mode: other", []), ("runner: {mode: 2}", []),
    ("client_num_per_round: 11", []), ("num_classes: 2", []),
    ("lr: .nan", []), ("lr: -0.1", []), ("batch_size: 0", []),
    ("batch_size: true", []), ("seed: -1", []),
    ("wandb_record: true", []), ("wandb_offline: false", []),
    ("wandb_key: forbidden", []), ("runner: {token: forbidden}", []),
    ("{}", ["missing", "1"]), ("{}", ["lr", "wrong"]),
    ("{}", ["lr"]), ("{}", ["wandb_key", "forbidden"]),
])
def test_invalid_configuration(tmp_path, text, overrides):
    from configs.experiment import load_experiment

    source = tmp_path / "bad.yaml"
    source.write_text(text)
    with pytest.raises(ValueError):
        load_experiment(source, overrides)


def test_presets_and_synthetic_bounds():
    from configs.experiment import load_experiment, hpfl_config

    smoke = ROOT / "configs/experiments/smoke.yaml"
    cfg = load_experiment(smoke)
    assert cfg.runner.mode == "synthetic"
    assert (cfg.client_num_in_total, cfg.runner.train_per_client, cfg.runner.test_per_client) == (2, 16, 8)
    assert cfg.batch_size == 8 and cfg.seed == 7 and cfg.lr == 0.01
    assert "runner" not in hpfl_config(cfg)
    for pairs in (["runner.train_per_client", "1"], ["runner.test_per_client", "1"],
                  ["runner.test_per_client", "9"], ["batch_size", "1"],
                  ["runner.train_per_client", "0"], ["client_num_in_total", "100"],
                  ["model", "other"], ["HPFL", "False"], ["comm_round", "2", "max_epochs", "1"]):
        with pytest.raises(ValueError):
            load_experiment(smoke, pairs)
    changed = load_experiment(smoke, ["client_num_in_total", "3", "client_num_per_round", "3",
                                     "comm_round", "2", "batch_size", "4"])
    assert changed.comm_round == 2 and changed.client_num_in_total == 3
    for clients in (10, 100):
        real = load_experiment(ROOT / f"configs/experiments/cifar10-{clients}clients.yaml")
        assert real.runner.mode == "real"
        assert real.client_num_in_total == clients and real.client_num_per_round == 5
        assert real.comm_round == real.max_epochs == 1000
        assert real.num_classes == real.model_output_dim == 10
        assert real.noise_on_feat and not real.noise_on_test_feat


def test_check_has_no_side_effects(tmp_path, monkeypatch, capsys):
    from tools import run_experiment

    monkeypatch.setattr(run_experiment, "RUN_ROOT", tmp_path / "outputs")
    monkeypatch.setattr(run_experiment, "dispatch", lambda *a: pytest.fail("学習は禁止"))
    for device in ("cpu", "cuda"):
        assert run_experiment.main(["--config", str(ROOT / "configs/experiments/cifar10-10clients.yaml"),
                                    "--device", device, "--dry-run"]) == 0
    text = capsys.readouterr().out
    assert "real" in text and "1000" in text and ".local-data/cifar10" in text
    assert not (tmp_path / "outputs").exists()
    with pytest.raises(SystemExit):
        run_experiment.main(["--device", "auto", "--check"])


def test_yaml_controls_synthetic_data(tmp_path):
    from configs.experiment import load_experiment
    from tools.local_smoke import synthetic_data

    text = (ROOT / "configs/experiments/smoke.yaml").read_text()
    source = tmp_path / "three.yaml"
    source.write_text(text.replace("client_num_in_total: 2", "client_num_in_total: 3")
                      .replace("client_num_per_round: 2", "client_num_per_round: 3")
                      .replace("train_per_client: 16", "train_per_client: 24"))
    cfg = load_experiment(source, ["batch_size", "4", "num_classes", "3", "model_output_dim", "3"])
    data = synthetic_data(cfg)
    assert data[:2] == [72, 24]
    assert len(data[5]) == 3 and data[7] == 3
    assert len(data[5][0]) == 6 and len(data[6][2]) == 2
    assert set(data[5][1].dataset.targets) == {0, 1, 2}


@pytest.mark.parametrize("backend", ["sequential", "process"])
def test_real_dispatch_uses_official_main(tmp_path, monkeypatch, backend):
    import json
    import os
    import sys
    import yaml
    import torch
    import wandb
    from configs.experiment import load_experiment
    from tools.run_experiment import dispatch

    cfg = load_experiment(ROOT / "configs/experiments/cifar10-10clients.yaml",
                          ["HPFL_local_iteration", "10", "runner.execution_backend", backend])
    output = tmp_path / "artifacts"
    output.mkdir()
    seen = []

    def official(path, run_name):
        assert path == str(ROOT / "experiments/standalone/main.py") and run_name == "__main__"
        assert Path.cwd() == output
        assert not torch.cuda.is_available()
        assert wandb.run.disabled
        assert os.environ["WANDB_DISABLED"] == "true"
        assert sys.argv[1] == "--config_file"
        resolved = yaml.safe_load(Path(sys.argv[2]).read_text())
        assert "runner" not in resolved and resolved["HPFL_local_iteration"] == 10
        assert resolved["gpu_index"] == 0 and resolved["wandb_record"] is False
        assert resolved["data_dir"] == str(ROOT / ".local-data/cifar10")
        assert resolved["client_execution_backend"] == backend
        assert resolved["client_rng_policy"] == ("task-sha256-v1" if backend == "process" else "historical-global")
        wandb.run.summary["local_metric"] = 0.25
        seen.append(True)

    monkeypatch.setattr("runpy.run_path", official)
    cwd = Path.cwd()
    dispatch(cfg, "cpu", output)
    assert seen == [True] and Path.cwd() == cwd
    assert json.loads((output / "summary.json").read_text())["metrics"]["local_metric"] == 0.25


def test_unique_outputs_effective_settings_and_failure_log(tmp_path, monkeypatch):
    import json
    import subprocess
    import yaml
    from configs.experiment import load_experiment
    from tools import run_experiment

    monkeypatch.setattr(run_experiment, "RUN_ROOT", tmp_path)
    source = ROOT / "configs/experiments/smoke.yaml"
    cfg = load_experiment(source, ["batch_size", "4"])
    calls = []

    original_popen = subprocess.Popen
    def fake_launch(command, **kwargs):
        calls.append(command)
        return original_popen(["bash", "-c", "echo expected-failure; exit 9"], **kwargs)

    monkeypatch.setattr(run_experiment, "launch_worker", fake_launch)
    first, status = run_experiment.execute_run(cfg, "cpu", source, ["--config", str(source)])
    second, _ = run_experiment.execute_run(cfg, "cuda", source, [])
    assert first != second and status == 9
    assert (first / "exit-status").read_text().strip() == "9"
    assert "expected-failure" in (first / "console.log").read_text()
    assert (first / "original.yaml").read_text() == source.read_text()
    effective = yaml.safe_load((first / "effective.yaml").read_text())
    assert effective["batch_size"] == 4 and effective["runner"]["mode"] == "synthetic"
    assert json.loads((second / "execution.json").read_text())["device"] == "cuda"
    assert "180" in calls[0] and "--worker" in calls[0]
    with pytest.raises(FileExistsError):
        run_experiment.execute_run(cfg, "cpu", source, [], output=first)


def test_shell_propagates_nix_and_tee_failures(tmp_path):
    import shutil
    import subprocess

    repo = tmp_path / "repo"
    (repo / "tools").mkdir(parents=True)
    shutil.copy(ROOT / "tools/run-experiment.sh", repo / "tools/run-experiment.sh")
    binaries = tmp_path / "bin"
    binaries.mkdir()
    nix = binaries / "nix"
    nix.write_text("#!/bin/bash\necho simulated-nix \"$@\"\nexit 23\n")
    nix.chmod(0o755)
    import os
    env = dict(os.environ, PATH=f"{binaries}:" + os.defpath)
    result = subprocess.run(["bash", "tools/run-experiment.sh"], cwd=repo, env=env, capture_output=True)
    assert result.returncode == 23
    first = next((repo / ".local-experiments").glob("launch-*"))
    assert (first / "exit-status").read_text().strip() == "23"
    assert "simulated-nix" in (first / "launcher.log").read_text()
    assert "develop . --command" in (first / "launcher.log").read_text()
    nix.write_text("#!/bin/bash\necho \"$@\"\nexit 0\n")
    tee = binaries / "tee"
    tee.write_text("#!/bin/bash\ncat >/dev/null\nexit 31\n")
    tee.chmod(0o755)
    result = subprocess.run(["bash", "tools/run-experiment.sh"], cwd=repo, env=env, capture_output=True)
    assert result.returncode == 31
    launches = list((repo / ".local-experiments").glob("launch-*"))
    assert len(launches) == 2
    second = next(path for path in launches if path != first)
    assert (second / "exit-status").read_text().strip() == "31"
    result = subprocess.run(["bash", "tools/run-experiment.sh", "--check"], cwd=repo, env=env)
    assert result.returncode == 0 and len(list((repo / ".local-experiments").iterdir())) == 2
    result = subprocess.run(["bash", "tools/run-experiment.sh", "--device", "cuda", "--check"],
                            cwd=repo, env=env, capture_output=True, text=True)
    assert result.returncode == 0 and "develop .#gpu --command" in result.stdout
    assert "path:" not in result.stdout


@pytest.mark.parametrize("pairs", [
    ["checkpoint_root_path", "../escape"], ["image_save_path", "/tmp/escape"],
    ["data_dir", ".local-experiments/data"], ["pretrained", "True"],
    ["load_backbone_from", "FedAvg"], ["gpu_index", "2"],
    ["runner.train_per_client", "24"], ["algorithm", "FedPer"],
])
def test_reject_unsafe_or_ignored_real_options(pairs):
    from configs.experiment import load_experiment

    with pytest.raises(ValueError):
        load_experiment(ROOT / "configs/experiments/cifar10-10clients.yaml", pairs)


def test_python_log_close_failure_is_not_success(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from configs.experiment import load_experiment
    from tools import run_experiment
    import subprocess

    monkeypatch.setattr(run_experiment, "RUN_ROOT", tmp_path)
    original_open = Path.open
    @contextmanager
    def broken_log(path, *args, **kwargs):
        with original_open(path, *args, **kwargs) as stream:
            yield stream
        raise OSError("ログ保存失敗")

    def open_path(path, *args, **kwargs):
        if path.name == "console.log":
            return broken_log(path, *args, **kwargs)
        return original_open(path, *args, **kwargs)

    original_popen = subprocess.Popen
    monkeypatch.setattr(run_experiment, "launch_worker", lambda command, **kw:
                        original_popen(["bash", "-c", "exit 0"], **kw))
    monkeypatch.setattr(Path, "open", open_path)
    source = ROOT / "configs/experiments/smoke.yaml"
    cfg = load_experiment(source)
    with pytest.raises(OSError, match="ログ保存失敗"):
        run_experiment.execute_run(cfg, "cpu", source, [])
    run = next(tmp_path.glob("run-*"))
    assert (run / "exit-status").read_text().strip() != "0"


@pytest.mark.parametrize("text", ["lr: '-0.1'", "data_dir: null", "runner: null", "noise_type: null"])
def test_yacs_coercion_cannot_bypass_validation(tmp_path, text):
    from configs.experiment import load_experiment

    source = tmp_path / "bad.yaml"
    source.write_text(text)
    with pytest.raises(ValueError):
        load_experiment(source)


def test_max_epochs_is_derived_only_when_not_explicit(tmp_path):
    from configs.experiment import load_experiment

    source = tmp_path / "smoke.yaml"
    text = (ROOT / "configs/experiments/smoke.yaml").read_text().replace("max_epochs: 1\n", "")
    source.write_text(text)
    cfg = load_experiment(source, ["comm_round", "2"])
    assert cfg.comm_round == cfg.max_epochs == 2
    cfg = load_experiment(source, ["comm_round", "2", "global_epochs_per_round", "3"])
    assert cfg.max_epochs == 6
    source.write_text(text.replace("comm_round: 1", "comm_round: 3")
                      .replace("global_epochs_per_round: 1", "global_epochs_per_round: 2"))
    assert load_experiment(source).max_epochs == 6
    source.write_text(text + "max_epochs: 1\n")
    with pytest.raises(ValueError, match="max_epochs"):
        load_experiment(source, ["comm_round", "2"])
    assert load_experiment(source, ["comm_round", "2", "max_epochs", "2"]).max_epochs == 2
    with pytest.raises(ValueError, match="max_epochs"):
        load_experiment(source, ["max_epochs", "2"])


@pytest.mark.parametrize("key", ["lr", "seed", "wd", "data_dir", "runner.mode",
                                "runner.train_per_client", "runner.test_per_client"])
@pytest.mark.parametrize("via_cli", [False, "None", "null"])
def test_null_values_are_rejected(tmp_path, key, via_cli):
    from configs.experiment import load_experiment

    text = (ROOT / "configs/experiments/smoke.yaml").read_text()
    overrides = []
    if via_cli:
        overrides = [key, via_cli]
    else:
        field = key.split(".")[-1]
        text = "\n".join(line.split(":")[0] + ": null" if line.lstrip().startswith(field + ":")
                         else line for line in text.splitlines())
        if key in ("wd",):
            text += "\nwd: null\n"
    source = tmp_path / "null.yaml"
    source.write_text(text)
    with pytest.raises(ValueError):
        load_experiment(source, overrides)
