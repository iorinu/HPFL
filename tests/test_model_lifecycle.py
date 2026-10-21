"""常駐状態と成功時だけの推論成果物を検証する。"""
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_head_failure_has_no_model_writes(tmp_path, monkeypatch):
    from configs.experiment import load_experiment
    from tools.local_smoke import run_smoke
    from algorithms_standalone.fedavg.FedAVGManager import FedAVGManager

    cfg = load_experiment(ROOT / "configs/experiments/smoke.yaml")
    original = FedAVGManager.algorithm_train

    def fail_head(self, *args, **kwargs):
        if self.server_timer.global_comm_round_idx == self.max_comm_round:
            raise RuntimeError("head 失敗")
        return original(self, *args, **kwargs)

    writes = []
    save = torch.save

    def observe(*args, **kwargs):
        writes.append(str(args[1] if len(args) > 1 else kwargs["f"]))
        return save(*args, **kwargs)

    monkeypatch.setattr(FedAVGManager, "algorithm_train", fail_head)
    monkeypatch.setattr(torch, "save", observe)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="head 失敗"):
        run_smoke(cfg, "cpu", tmp_path)
    assert writes == []
    assert not list(tmp_path.rglob("*.pt"))
    assert not list(tmp_path.rglob("*.pth"))


def test_parameter_observation_does_not_offload(monkeypatch):
    from trainers.normal_trainer import NormalTrainer
    from types import SimpleNamespace

    model = torch.nn.Linear(2, 2)
    monkeypatch.setattr(model, "cpu", lambda: pytest.fail("観測による offload は禁止"))
    state = NormalTrainer.get_model_params(SimpleNamespace(model=model))
    assert set(state) == {"weight", "bias"}


@pytest.mark.parametrize("pairs", [["model_residency", "disk"], ["capacity_policy", "spill"],
                                   ["checkpoint_save", "True"]])
def test_reject_unsupported_retention_and_recovery(pairs):
    from configs.experiment import load_experiment

    with pytest.raises(ValueError):
        load_experiment(ROOT / "configs/experiments/smoke.yaml", pairs)


def test_stable_worker_routing_with_subsets():
    from algorithms_standalone.basePS.client_executor import BoundedExecutor
    from tests.test_parallel_clients import scheduled_task

    executor = BoundedExecutor(2, 20, scheduled_task)
    try:
        first = executor.run([dict(client_id=i, worker_id=i % 2, delay=0) for i in (2, 1, 0)])
        second = executor.run([dict(client_id=i, worker_id=i % 2, delay=0) for i in (0, 2, 1)])
        assert {r["client_id"]: r["pid"] for r in first} == {r["client_id"]: r["pid"] for r in second}
        assert first[0]["pid"] == first[2]["pid"] != first[1]["pid"]
    finally:
        executor.close()


def test_persistent_client_matches_reconstruction_rng_and_state(tmp_path, monkeypatch):
    from configs.experiment import load_experiment, hpfl_config
    from tools.local_smoke import synthetic_data
    from algorithms_standalone.fedavg.FedAVGManager import FedAVGManager
    from algorithms_standalone.fedavg import process_worker as worker

    cfg = load_experiment(ROOT / "configs/experiments/smoke.yaml",
                          ["runner.execution_backend", "process", "momentum", "0.9", "sched", "StepLR"])
    # この単体比較だけでshuffleとmarker noiseのRNG消費も検証する。
    cfg.noise_on_feat = True
    cfg.noise_type = "random"
    args = hpfl_config(cfg)
    args.server_index = args.client_index = -1
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("algorithms_standalone.basePS.basePSmanager.load_data", lambda **kw: synthetic_data(cfg))
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda n: None)
    torch.set_num_threads(1)
    manager = FedAVGManager(torch.device("cpu"), args)
    manager.train_data_local_dict[0] = torch.utils.data.DataLoader(
        manager.train_data_local_dict[0].dataset, batch_size=cfg.batch_size, shuffle=True)
    initial = worker.export_client(manager.client_list[0])
    weights = worker.cpu_tree(manager.aggregator.trainer.model.state_dict())
    worker.initialize_worker(worker.export_loaders(manager.train_data_local_dict),
                             worker.export_loaders(manager.test_data_local_dict), manager.train_data_local_num_dict)
    state = initial
    identity = None
    for round_index, phase in enumerate(("shared", "shared", "head")):
        if phase == "head":
            manager.freeze()
            args.freeze_bn = True
        task = dict(client_id=0, round=round_index, phase=phase, config=dict(args), device="cpu",
                    weights=weights, weights_sha256=worker.weights_digest(weights), state=initial if round_index == 0 else None,
                    update={}, time=manager.server_timer.get_time_info_to_send(), debug=True,
                    flags=worker.export_client(manager.client_list[0])["requires_grad"])
        actual = worker.train_task(task)
        client = worker._CLIENTS[0]
        if identity is not None:
            assert id(client) == identity
        identity = id(client)
        retained = worker._CLIENTS
        worker._CLIENTS = {}
        reference = worker.train_task(dict(task, state=state))
        worker._CLIENTS = retained
        for key in actual["state"]["model"]:
            torch.testing.assert_close(actual["state"]["model"][key], reference["state"]["model"][key], rtol=0, atol=0)
        for key, value in actual["state"]["optimizer"]["state"].items():
            torch.testing.assert_close(value["momentum_buffer"], reference["state"]["optimizer"]["state"][key]["momentum_buffer"], rtol=0, atol=0)
        assert actual["state"]["scheduler"] == reference["state"]["scheduler"]
        assert actual["state"]["timer"] == reference["state"]["timer"]
        if phase == "head":
            import numpy as np
            for a, b in zip(actual["marker"], reference["marker"], strict=True):
                np.testing.assert_array_equal(a, b)
        state = actual["state"]
    assert len(worker._CLIENTS) == 1


@pytest.mark.parametrize("phase", ["selection", "evaluation", "serialization", "oom", "interrupt"])
def test_late_failures_do_not_publish_success(tmp_path, monkeypatch, phase):
    from configs.experiment import load_experiment
    from tools.local_smoke import run_smoke
    from algorithms_standalone.fedavg.FedAVGManager import FedAVGManager
    from algorithms_standalone.fedavg.aggregator import FedAVGAggregator
    import json

    cfg = load_experiment(ROOT / "configs/experiments/smoke.yaml")
    def fail(*args, **kwargs):
        if phase == "interrupt":
            raise KeyboardInterrupt()
        if phase == "oom":
            raise torch.OutOfMemoryError("容量不足")
        raise RuntimeError("意図した失敗")
    if phase == "selection":
        monkeypatch.setattr(FedAVGAggregator, "select_pluggable", fail)
    elif phase == "serialization":
        monkeypatch.setattr(torch, "save", fail)
    else:
        monkeypatch.setattr(FedAVGManager, "test_with_personalized_batch", fail)
    monkeypatch.chdir(tmp_path)
    with pytest.raises((RuntimeError, KeyboardInterrupt)):
        run_smoke(cfg, "cpu", tmp_path)
    assert not list(tmp_path.rglob("*.pt")) and not list(tmp_path.rglob("*.pth"))
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "failed"


def test_real_lifecycle_with_synthetic_loader_publishes_after_all_evaluations(tmp_path, monkeypatch):
    from configs.experiment import load_experiment
    from tools.local_smoke import synthetic_data
    from tools.run_experiment import dispatch
    import json

    cfg = load_experiment(ROOT / "configs/experiments/smoke.yaml")
    monkeypatch.setattr("algorithms_standalone.basePS.basePSmanager.load_data", lambda **kw: synthetic_data(cfg))
    cfg.runner.mode = "real"
    monkeypatch.chdir(tmp_path)
    dispatch(cfg, "cpu", tmp_path)
    artifacts = list(tmp_path.rglob("final-model.pt"))
    assert len(artifacts) == 1
    assert not list(tmp_path.rglob("*.pth"))
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "passed" and summary["artifact"]["model_write_count"] == 1


def test_post_link_failure_rolls_back_own_artifact_only(tmp_path, monkeypatch):
    from utils.final_artifact import atomic_save
    import os

    link = os.link
    def interrupted_link(source, target):
        link(source, target)
        raise KeyboardInterrupt()
    monkeypatch.setattr(os, "link", interrupted_link)
    with pytest.raises(KeyboardInterrupt):
        atomic_save({"x": torch.ones(1)}, tmp_path / "final-model.pt")
    assert list(tmp_path.iterdir()) == []


def test_summary_failure_revokes_final(tmp_path, monkeypatch):
    from configs.experiment import load_experiment
    from tools.local_smoke import run_smoke

    cfg = load_experiment(ROOT / "configs/experiments/smoke.yaml")
    write = Path.write_text
    def broken(path, text, *args, **kwargs):
        if path.name == "summary.json" and '"passed"' in text:
            raise OSError("summary 保存失敗")
        return write(path, text, *args, **kwargs)
    monkeypatch.setattr(Path, "write_text", broken)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(OSError, match="summary"):
        run_smoke(cfg, "cpu", tmp_path)
    assert not list(tmp_path.rglob("*.pt"))


def test_existing_output_is_not_modified(tmp_path, monkeypatch):
    from configs.experiment import load_experiment
    from tools.local_smoke import run_smoke

    target = tmp_path / "final-model.pt"
    target.write_bytes(b"preserve")
    cfg = load_experiment(ROOT / "configs/experiments/smoke.yaml")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileExistsError):
        run_smoke(cfg, "cpu", tmp_path)
    assert target.read_bytes() == b"preserve"
    assert list(tmp_path.iterdir()) == [target]


@pytest.mark.parametrize("residency", ["device", "cpu"])
def test_default_process_has_only_final_and_bounded_snapshots(tmp_path, monkeypatch, residency):
    from configs.experiment import load_experiment
    from tools.local_smoke import run_smoke
    from algorithms_standalone.fedavg.FedAVGManager import FedAVGManager
    import json
    import copy

    cfg = load_experiment(ROOT / "configs/experiments/smoke.yaml",
                          ["runner.execution_backend", "process", "comm_round", "2",
                           "momentum", "0.9", "sched", "StepLR", "model_residency", residency])
    train = FedAVGManager.train
    process_round = FedAVGManager._process_round
    deepcopy = copy.deepcopy
    ipc_weights = []
    def observe_round(manager, client_ids, weights, *args, **kwargs):
        ipc_weights.append(weights)
        return process_round(manager, client_ids, weights, *args, **kwargs)
    def bounded_copy(value, *args, **kwargs):
        assert not any(value is weights for weights in ipc_weights), "process 結果取込で未使用の全モデル複製は禁止"
        return deepcopy(value, *args, **kwargs)
    monkeypatch.setattr(FedAVGManager, "_process_round", observe_round)
    monkeypatch.setattr(copy, "deepcopy", bounded_copy)
    def observe(manager):
        result = train(manager)
        assert all("weights" not in h for h in manager.process_history)
        assert manager.aggregator.model_dict == {}
        assert all(not c.trainer.optimizer.state for c in manager.client_list)
        return result
    monkeypatch.setattr(FedAVGManager, "train", observe)
    monkeypatch.chdir(tmp_path)
    run_smoke(cfg, "cpu", tmp_path)
    events = [json.loads(line) for line in (tmp_path / "client-execution.jsonl").read_text().splitlines()]
    clients = [e for e in events if e["event"] == "client"]
    assert [e["initialized"] for e in clients] == [False] * 6
    assert all(e["resident_clients"] <= 2 for e in clients)
    assert all(e["retained_model_devices"] == ["cpu"] and e["retained_optimizer_devices"] == ["cpu"] for e in clients)
    assert all(e["boundary_model_devices"] == ["cpu"] and e["boundary_optimizer_devices"] in ([], ["cpu"]) for e in clients)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["artifact"]["model_write_count"] == 1
    assert 0 < summary["artifact"]["bytes"] < 60_000_000
    assert summary["artifact"]["allocated_bytes"] < 61_000_000
    assert [p.name for p in tmp_path.rglob("*.pt")] == ["final-model.pt"]


def test_process_initializes_inactive_clients_and_routes_subsets(tmp_path, monkeypatch):
    from configs.experiment import load_experiment, hpfl_config
    from tools.local_smoke import synthetic_data
    from algorithms_standalone.fedavg.FedAVGManager import FedAVGManager

    cfg = load_experiment(ROOT / "configs/experiments/smoke.yaml", ["runner.execution_backend", "process",
                          "runner.max_concurrent_clients", "2", "client_num_in_total", "3", "client_num_per_round", "3"])
    args = hpfl_config(cfg)
    args.server_index = args.client_index = -1
    args.client_num_per_round = 1
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("algorithms_standalone.basePS.basePSmanager.load_data", lambda **kw: synthetic_data(cfg))
    manager = FedAVGManager(torch.device("cpu"), args)
    weights = manager.aggregator.trainer.get_model_params()
    owners = {}
    try:
        for round_index, ids in enumerate(([2], [0, 2], [2, 1, 0])):
            manager.server_timer.global_comm_round_idx = round_index
            manager.max_comm_round = 2
            if round_index == 2:
                manager.freeze()
                args.freeze_bn = True
            results = manager._process_round(ids, weights, "model", {}, {}, manager.server_timer.get_time_info_to_send(), {})
            assert manager.worker_initialized_clients == {0, 1, 2}
            assert all(next(c.trainer.model.parameters()).device.type == "meta" for c in manager.client_list)
            for result in results:
                event = result["observation"]
                assert event["resident_clients"] == (2 if result["client_id"] % 2 == 0 else 1)
                if result["client_id"] in owners:
                    assert owners[result["client_id"]] == event["pid"]
                owners[result["client_id"]] = event["pid"]
    finally:
        manager.client_executor.close()


def test_sequential_device_policy_never_calls_offload_and_observes_boundaries(tmp_path, monkeypatch):
    from configs.experiment import load_experiment
    from tools.local_smoke import run_smoke
    from algorithms_standalone.fedavg.client import FedAVGClient
    import json

    cfg = load_experiment(ROOT / "configs/experiments/smoke.yaml")
    monkeypatch.setattr(FedAVGClient, "move_to_cpu", lambda *a: pytest.fail("既定での offload は禁止"))
    monkeypatch.chdir(tmp_path)
    run_smoke(cfg, "cpu", tmp_path)
    events = [json.loads(line) for line in (tmp_path / "client-execution.jsonl").read_text().splitlines()]
    assert len(events) == 4
    assert all(e["retained_model_devices"] == ["cpu"] and e["model_residency"] == "device" for e in events)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="実 CUDA 専用の常駐回帰")
@pytest.mark.parametrize("residency", ["device", "cpu"])
def test_sequential_cuda_aggregation_snapshot_residency(tmp_path, monkeypatch, residency):
    from configs.experiment import load_experiment, hpfl_config
    from tools.local_smoke import synthetic_data
    from algorithms_standalone.fedavg.FedAVGManager import FedAVGManager

    cfg = load_experiment(ROOT / "configs/experiments/smoke.yaml",
                          ["model_residency", residency, "momentum", "0.9", "sched", "StepLR"])
    args = hpfl_config(cfg)
    args.server_index = args.client_index = -1
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("algorithms_standalone.basePS.basePSmanager.load_data", lambda **kw: synthetic_data(cfg))
    manager = FedAVGManager(torch.device("cuda:0"), args)
    client = manager.client_list[0]
    # 実際の学習返却値を集約器へ渡し、生モデルとは別に保持先を確認する。
    result = client.train({}, manager.server_timer.get_time_info_to_send(), 0,
                          manager.aggregator.trainer.get_model_params(), traininig_start=True)
    weights, indexes, count, other = result[:4]
    manager.aggregator.add_local_trained_result(0, weights, indexes, count, other)
    expected = "cuda" if residency == "device" else "cpu"
    assert {p.device.type for p in client.trainer.model.state_dict().values()} == {expected}
    optimizer_tensors = [v for state in client.trainer.optimizer.state.values()
                         for v in state.values() if isinstance(v, torch.Tensor)]
    assert optimizer_tensors
    assert {v.device.type for v in optimizer_tensors} == {expected}
    assert {v.device.type for v in weights.values()} == {expected}
    assert {v.device.type for v in manager.aggregator.model_dict[0].values()} == {expected}


def test_teardown_failure_revokes_success_summary(tmp_path, monkeypatch):
    from configs.experiment import load_experiment
    from tools.local_smoke import run_smoke
    import wandb
    import json

    initialize = wandb.init
    def init(*args, **kwargs):
        run = initialize(*args, **kwargs)
        monkeypatch.setattr(run, "finish", lambda: (_ for _ in ()).throw(RuntimeError("終了処理失敗")))
        return run
    monkeypatch.setattr(wandb, "init", init)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="終了処理失敗"):
        run_smoke(load_experiment(ROOT / "configs/experiments/smoke.yaml"), "cpu", tmp_path)
    assert not list(tmp_path.rglob("*.pt"))
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("field,value", [("fedavg_cfl", True), ("if_OOD", True),
                                        ("reduce_plug", True), ("finetune_FedAvg", True),
                                        ("load_backbone_from", "FedAvg")])
def test_unsupported_legacy_inference_paths_fail_before_loading(field, value, monkeypatch):
    from configs.experiment import load_experiment, hpfl_config
    from algorithms_standalone.fedavg.FedAVGManager import FedAVGManager

    args = hpfl_config(load_experiment(ROOT / "configs/experiments/smoke.yaml"))
    args[field] = value
    monkeypatch.setattr("algorithms_standalone.basePS.basePSmanager.load_data",
                        lambda **kw: pytest.fail("非対応経路のデータ取得は禁止"))
    with pytest.raises(ValueError, match="推論"):
        FedAVGManager(torch.device("cpu"), args)


def test_real_debug_dump_is_not_silently_ignored():
    from configs.experiment import load_experiment

    with pytest.raises(ValueError):
        load_experiment(ROOT / "configs/experiments/cifar10-10clients.yaml", ["debug_tensor_dumps", "True"])


def test_runner_does_not_import_torch_before_device_environment():
    import subprocess
    import sys

    result = subprocess.run([sys.executable, "-c", "import sys; import tools.run_experiment; assert 'torch' not in sys.modules"],
                            cwd=ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


def test_temporary_cleanup_failure_is_not_a_published_model(tmp_path, monkeypatch):
    from utils.final_artifact import atomic_save

    unlink = Path.unlink
    failed = []
    def interrupt(path, *args, **kwargs):
        result = unlink(path, *args, **kwargs)
        if path.suffix == ".tmp" and not failed:
            failed.append(True)
            raise OSError("temp cleanup 失敗")
        return result
    monkeypatch.setattr(Path, "unlink", interrupt)
    with pytest.raises(OSError):
        atomic_save({"value": torch.ones(1)}, tmp_path / "final-model.pt")
    assert list(tmp_path.iterdir()) == []


def test_process_startup_timeout_leaves_failed_summary_and_no_artifact(tmp_path, monkeypatch):
    from configs.experiment import load_experiment
    from tools.local_smoke import run_smoke
    from algorithms_standalone.fedavg.FedAVGManager import FedAVGManager
    import json

    cfg = load_experiment(ROOT / "configs/experiments/smoke.yaml", ["runner.execution_backend", "process",
                          "runner.client_timeout_seconds", "0.001"])
    train = FedAVGManager.train
    managers = []
    def observe(manager):
        managers.append(manager)
        return train(manager)
    monkeypatch.setattr(FedAVGManager, "train", observe)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(TimeoutError):
        run_smoke(cfg, "cpu", tmp_path)
    assert all(not p.is_alive() for p in managers[0].client_executor.processes)
    assert not list(tmp_path.rglob("*.pt"))
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "failed"


def test_sigterm_during_selection_is_a_failed_run(tmp_path, monkeypatch):
    from configs.experiment import load_experiment
    from tools.local_smoke import run_smoke
    from algorithms_standalone.fedavg.aggregator import FedAVGAggregator
    import os
    import signal
    import json

    cfg = load_experiment(ROOT / "configs/experiments/smoke.yaml")
    monkeypatch.setattr(FedAVGAggregator, "select_pluggable", lambda *a: os.kill(os.getpid(), signal.SIGTERM))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="終了要求"):
        run_smoke(cfg, "cpu", tmp_path)
    assert not list(tmp_path.rglob("*.pt"))
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "failed"


def test_legacy_relative_data_path_keeps_original_base(tmp_path, monkeypatch):
    from configs.experiment import load_experiment, hpfl_config
    from tools.local_smoke import synthetic_data
    import runpy
    import sys
    import wandb

    cfg = load_experiment(ROOT / "configs/experiments/smoke.yaml")
    flat = hpfl_config(cfg)
    flat.data_dir = "expected-data"
    source = tmp_path / "legacy.yaml"
    source.write_text(flat.dump())
    def loader(**kwargs):
        assert kwargs["args"].data_dir == str(tmp_path / "expected-data")
        return synthetic_data(cfg)
    monkeypatch.setattr("algorithms_standalone.basePS.basePSmanager.load_data", loader)
    monkeypatch.setattr(sys, "argv", ["main.py", "--config_file", str(source)])
    monkeypatch.chdir(tmp_path)
    run = wandb.init(mode="disabled")
    try:
        result = runpy.run_path(str(ROOT / "experiments/standalone/main.py"), run_name="__main__")
        assert (result["run_output"] / "final-model.pt").exists()
    finally:
        run.finish()


@pytest.mark.parametrize("failure", ["serialize", "publish", "interrupt", "existing"])
def test_atomic_publication_failure_preserves_outputs(tmp_path, monkeypatch, failure):
    from utils.final_artifact import atomic_save
    import os

    target = tmp_path / "final-model.pt"
    if failure == "existing":
        target.write_bytes("既存".encode())
    elif failure == "publish":
        monkeypatch.setattr(os, "link", lambda *a: (_ for _ in ()).throw(OSError("公開失敗")))
    else:
        def broken(value, stream):
            stream.write(b"partial")
            raise KeyboardInterrupt() if failure == "interrupt" else OSError("保存失敗")
        monkeypatch.setattr(torch, "save", broken)
    with pytest.raises((OSError, KeyboardInterrupt)):
        atomic_save({"version": 1}, target)
    assert list(tmp_path.iterdir()) == ([target] if failure == "existing" else [])
    if failure == "existing":
        assert target.read_bytes() == "既存".encode()


def test_success_writes_only_reloadable_final_after_evaluation(tmp_path, monkeypatch):
    from configs.experiment import load_experiment
    from tools.local_smoke import run_smoke, synthetic_data
    from utils import final_artifact
    import numpy as np

    cfg = load_experiment(ROOT / "configs/experiments/smoke.yaml")
    publish = final_artifact.publish_final
    checked = []
    writes = []
    save = torch.save
    def record_save(*args, **kwargs):
        writes.append(1)
        return save(*args, **kwargs)
    monkeypatch.setattr(torch, "save", record_save)

    def verify(manager, output):
        assert not list(tmp_path.rglob("*.pt")) and not list(tmp_path.rglob("*.pth"))
        info = publish(manager, output)
        restored = final_artifact.load_final(tmp_path / "final-model.pt")
        loader = synthetic_data(cfg)[6][0]
        x, _ = next(iter(loader))
        manager.aggregator.trainer.model.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored.logits(x), manager.aggregator.trainer.model(x))
            for index, head in enumerate(manager.aggregator.model_list):
                import copy
                expected_model = copy.deepcopy(manager.aggregator.trainer.model)
                list(expected_model.children())[-1].load_state_dict(head[0].state_dict())
                expected = expected_model.eval()(x)
                torch.testing.assert_close(restored.logits(x, index), expected)
        a, i = manager.aggregator.select_pluggable(loader, "MMD")
        b, j = restored.select(loader)
        assert i == j
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
        assert restored.client_ids == [0, 1]
        checked.append(info)
        return info

    monkeypatch.setattr(final_artifact, "publish_final", verify)
    monkeypatch.chdir(tmp_path)
    run_smoke(cfg, "cpu", tmp_path)
    assert checked
    assert len(writes) == 1
    assert [p.name for p in tmp_path.rglob("*.pt")] == ["final-model.pt"]
    assert not list(tmp_path.rglob("*.pth"))
