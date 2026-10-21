"""spawn client 実行の設定・状態・同期境界を検証する。"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_execution_defaults_and_real_transfer():
    from configs.experiment import load_experiment, hpfl_config

    source = ROOT / "configs/experiments/cifar10-10clients.yaml"
    cfg = load_experiment(source)
    assert cfg.runner.execution_backend == "sequential"
    assert cfg.runner.max_concurrent_clients == 1
    cfg = load_experiment(source, ["runner.execution_backend", "process",
                                   "runner.max_concurrent_clients", "2"])
    assert cfg.client_rng_policy == "task-sha256-v1"
    flat = hpfl_config(cfg)
    assert "runner" not in flat
    assert flat.client_execution_backend == "process"
    assert flat.max_concurrent_clients == 2
    assert flat.client_rng_policy == "task-sha256-v1"


@pytest.mark.parametrize("pairs", [
    ["runner.max_concurrent_clients", value] for value in ("0", "True", "None", "1.5", "3")
] + [["runner.execution_backend", "threads"],
     ["runner.max_concurrent_clients", "2"],
     ["runner.client_timeout_seconds", "0"],
     ["runner.client_timeout_seconds", "None"]])
def test_invalid_execution_settings(pairs):
    from configs.experiment import load_experiment

    with pytest.raises(ValueError):
        load_experiment(ROOT / "configs/experiments/cifar10-10clients.yaml", pairs)


@pytest.mark.parametrize("key,value", [
    ("scaffold", "True"), ("VHL", "True"), ("if_OOD", "True"),
    ("fedprox", "True"), ("instantiate_all", "False"), ("fedavg_cfl", "True"),
    ("fedavg_local_step_type", "fixed"), ("compression", "topk"),
    ("record_dataframe", "True"), ("trainer_param_groups", "True"),
    ("client_optimizer", "SAM"), ("central_trainning_epoch", "1"),
    ("data_sampler", "imbalance"), ("model", "resnet20_cifar"),
    ("Contrastive", "SimCLR"), ("data_transform", "FLTransform"),
    ("data_efficient_load", "False"), ("reduce_plug", "True"),
    ("model_input_channels", "1"), ("client_select", "ood_score"),
    ("frequency_of_the_test", "2"),
])
def test_unsupported_process_options(key, value):
    from configs.experiment import load_experiment

    with pytest.raises(ValueError):
        load_experiment(ROOT / "configs/experiments/cifar10-10clients.yaml",
                        ["runner.execution_backend", "process", key, value])


def scheduled_task(task):
    import os
    import time

    start = time.monotonic()
    time.sleep(task["delay"])
    if task.get("crash"):
        os._exit(17)
    if task.get("error"):
        raise RuntimeError("意図した worker エラー / CUDA out of memory")
    return dict(client_id=task["client_id"], start=start, end=time.monotonic(), pid=os.getpid())


def test_spawn_barrier_order_overlap_and_cleanup():
    from algorithms_standalone.basePS.client_executor import BoundedExecutor

    executor = BoundedExecutor(2, timeout=20, handler=scheduled_task)
    try:
        results = executor.run([dict(client_id=7, delay=0.5), dict(client_id=3, delay=0.1)])
        assert [r["client_id"] for r in results] == [7, 3]
        assert len({r["pid"] for r in results}) == 2
        assert max(r["start"] for r in results) < min(r["end"] for r in results)
        assert results[1]["end"] < results[0]["end"]
        assert set(r["pid"] for r in executor.run([dict(client_id=1, delay=0)])) <= {
            r["pid"] for r in results}
    finally:
        executor.close()
    assert all(not p.is_alive() for p in executor.processes)


@pytest.mark.parametrize("failure", ["error", "timeout", "crash"])
def test_worker_failure_discards_barrier_and_cleans_up(failure):
    from algorithms_standalone.basePS.client_executor import BoundedExecutor

    executor = BoundedExecutor(2, timeout=0.3 if failure == "timeout" else 20,
                               handler=scheduled_task)
    with pytest.raises((RuntimeError, TimeoutError)):
        executor.run([dict(client_id=0, delay=0),
                      dict(client_id=1, delay=2 if failure == "timeout" else 0,
                           error=failure == "error", crash=failure == "crash")])
    assert all(not p.is_alive() for p in executor.processes)


def test_cpu_process_one_two_rounds_heads_and_momentum(tmp_path):
    import json
    import subprocess
    import sys
    import torch

    runs = []
    for workers, residency in ((1, "device"), (2, "device"), (2, "cpu")):
        output = tmp_path / f"process{workers}-{residency}"
        output.mkdir()
        command = [sys.executable, "-m", "tools.run_experiment", "--worker", "--config",
                   str(ROOT / "configs/experiments/smoke.yaml"), "--output", str(output),
                   "runner.execution_backend", "process", "runner.max_concurrent_clients", str(workers),
                   "comm_round", "2", "momentum", "0.9", "sched", "StepLR", "debug_tensor_dumps", "True",
                   "model_residency", residency]
        with (tmp_path / f"process{workers}-{residency}.log").open("w") as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        assert completed.returncode == 0, (tmp_path / f"process{workers}-{residency}.log").read_text()
        summary = json.loads((output / "summary.json").read_text())
        assert summary["status"] == "passed" and summary["markers"] == 2
        assert summary["shared_rounds"] == 2 and len(summary["mmd"]) == 2
        events = [json.loads(line) for line in (output / "client-execution.jsonl").read_text().splitlines()]
        tasks = [event for event in events if event["event"] == "client"]
        assert len(tasks) == 6
        assert [e["client_id"] for e in tasks] == [0, 1] * 3
        assert all(e["rng_policy"] == "task-sha256-v1" and e["device"] == "cpu" for e in tasks)
        assert all(e["forward_calls"] > 0 and e["host_peak_rss_kib"] > 0 for e in tasks)
        assert all(e["loss_calls"] == e["forward_calls"] and e["target_device"] == "cpu" for e in tasks)
        assert all(tasks[i]["start_weights_sha256"] == tasks[i+1]["start_weights_sha256"] for i in (0, 2, 4))
        assert all(e["start_weights_sha256"] == e["loaded_weights_sha256"] for e in tasks)
        assert len({e["pid"] for e in tasks}) == workers
        rounds = [e for e in events if e["event"] == "round"]
        assert len(rounds) == 3
        assert all(e["status"] == "passed" for e in rounds)
        assert any(e["event"] == "resources" and e["host_current_total_rss_bytes"] > 0 for e in events)
        assert len([e for e in events if e["event"] == "parent_phases"]) == 3
        if workers == 2:
            assert any(max(t["start"] for t in tasks[i:i+2]) < min(t["end"] for t in tasks[i:i+2])
                       for i in (0, 2, 4))
        state = torch.load(output / "process-contract.pt", weights_only=False)
        assert all(s["optimizer"]["state"] for s in state["clients"])
        assert all(s["timer"]["local_comm_round_idx"] == 3 for s in state["clients"])
        assert all(s["scheduler"] is not None for s in state["clients"])
        runs.append((state, summary, tasks))

    maxima = []

    def compare(a, b):
        if isinstance(a, torch.Tensor):
            maxima.append(float((a.double() - b.double()).abs().max()))
            torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                compare(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                compare(x, y)
        else:
            assert a == b

    compare(runs[0][0], runs[1][0])
    compare(runs[0][0], runs[2][0])
    import numpy as np
    for run in runs[1:]:
        for a, b in zip(runs[0][1]["mmd"], run[1]["mmd"]):
            assert a["choice"] == b["choice"]
            np.testing.assert_allclose(a["score"], b["score"], rtol=1e-6, atol=1e-7)
        assert [t["seed"] for t in runs[0][2]] == [t["seed"] for t in run[2]]
    output = tmp_path / "legacy"
    output.mkdir()
    with (tmp_path / "legacy.log").open("w") as log:
        legacy = subprocess.run([sys.executable, "-m", "tools.run_experiment", "--worker", "--config",
                                 str(ROOT / "configs/experiments/smoke.yaml"), "--output", str(output),
                                 "comm_round", "2", "momentum", "0.9", "sched", "StepLR", "debug_tensor_dumps", "True"],
                                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=120)
    assert legacy.returncode == 0
    historical = torch.load(output / "legacy-contract.pt", weights_only=False)
    differences = [float((v.double() - historical["global_model"][k].double()).abs().max())
                   for k, v in runs[0][0]["global_model"].items()]
    (tmp_path / "comparison.json").write_text(json.dumps(dict(
        max_tensor_difference=max(maxima), legacy_global_max_difference=max(differences),
        legacy_rng_policy="historical-global", process_rng_policy="task-sha256-v1")))


def failing_train_task(task):
    from algorithms_standalone.fedavg.process_worker import train_task

    if task["client_id"] == 1:
        raise RuntimeError("意図した CUDA out of memory")
    return train_task(task)


def test_failed_round_never_ingests_or_checkpoints(tmp_path, monkeypatch):
    import json
    import torch
    import wandb
    from configs.experiment import load_experiment, hpfl_config
    from tools.local_smoke import synthetic_data
    from algorithms_standalone.fedavg.FedAVGManager import FedAVGManager

    cfg = load_experiment(ROOT / "configs/experiments/smoke.yaml",
                          ["runner.execution_backend", "process", "runner.max_concurrent_clients", "2"])
    args = hpfl_config(cfg)
    args.server_index = args.client_index = -1
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("algorithms_standalone.basePS.basePSmanager.load_data", lambda **kw: synthetic_data(cfg))
    monkeypatch.setattr("algorithms_standalone.fedavg.process_worker.train_task", failing_train_task)
    monkeypatch.setattr("algorithms_standalone.fedavg.FedAVGManager.save_checkpoint",
                        lambda *a, **kw: pytest.fail("失敗 round の checkpoint は禁止"))
    torch.set_num_threads(1)
    run = wandb.init(mode="disabled")
    try:
        manager = FedAVGManager(torch.device("cpu"), args)
        monkeypatch.setattr(manager.aggregator, "add_local_trained_result",
                            lambda *a, **kw: pytest.fail("partial ingest は禁止"))
        monkeypatch.setattr(manager.aggregator, "aggregate",
                            lambda *a, **kw: pytest.fail("partial aggregation は禁止"))
        with pytest.raises(RuntimeError, match="out of memory"):
            manager.train()
        assert manager.server_timer.global_comm_round_idx == 0
        assert all(not c.trainer.optimizer.state for c in manager.client_list)
        assert manager.aggregator.model_list == manager.aggregator.dist_list == []
        assert not list(tmp_path.rglob("*.pth"))
        assert all(not p.is_alive() for p in manager.client_executor.processes)
        event = json.loads((tmp_path / "client-execution.jsonl").read_text().splitlines()[-1])
        assert event["status"] == "failed"
    finally:
        run.finish()


def test_partition_export_keeps_indices_transform_and_shared_storage():
    import numpy as np
    import torch
    from types import SimpleNamespace
    from torch.utils.data import DataLoader
    from data_preprocessing.cifar10.datasets import CIFAR10_truncated_WO_reload
    from torchvision import transforms
    from algorithms_standalone.fedavg.process_worker import export_loaders, make_loader

    full = SimpleNamespace(data=np.arange(8*32*32*3, dtype=np.uint8).reshape(8, 32, 32, 3),
                           targets=np.arange(8) % 2)
    transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    loaders = {i: DataLoader(CIFAR10_truncated_WO_reload("未使用", dataidxs=indices,
                           full_dataset=full, transform=transform), batch_size=2)
               for i, indices in enumerate(([7, 1, 5, 3], [6, 0, 4, 2]))}
    specs = export_loaders(loaders)
    assert specs[0]["data"].is_shared()
    assert specs[0]["data"].data_ptr() == specs[1]["data"].data_ptr()
    for i in loaders:
        torch.manual_seed(17)
        expected = list(loaders[i])
        torch.manual_seed(17)
        actual = list(make_loader(specs[i]))
        for (x, y), (a, b) in zip(expected, actual):
            torch.testing.assert_close(x, a, rtol=0, atol=0)
            torch.testing.assert_close(y, b, rtol=0, atol=0)


def test_state_roundtrip_preserves_momentum_scheduler_bn_and_flags():
    from types import SimpleNamespace
    import torch
    from algorithms_standalone.fedavg.process_worker import export_client, restore_client
    from lr_scheduler.steplr_scheduler import StepLR

    def client():
        model = torch.nn.Sequential(torch.nn.BatchNorm1d(2), torch.nn.Linear(2, 1))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        return SimpleNamespace(trainer=SimpleNamespace(model=model, optimizer=optimizer,
                               lr_scheduler=StepLR(optimizer, base_lr=0.1, lr_decay_rate=0.5, step_size=1)),
                               client_timer=SimpleNamespace(args=None, local_comm_round_idx=2),
                               global_epochs_per_round=1)

    torch.manual_seed(41)
    original, restored = client(), client()
    x = torch.randn(4, 2)
    original.trainer.model(x).sum().backward()
    original.trainer.optimizer.step()
    original.trainer.optimizer.zero_grad()
    original.trainer.lr_scheduler.step(2)
    original.trainer.model[0].eval()
    original.trainer.model[0].weight.requires_grad = False
    snapshot = export_client(original)
    restore_client(restored, snapshot)
    for instance in (original, restored):
        instance.trainer.model(x).sum().backward()
        instance.trainer.optimizer.step()
    for a, b in zip(original.trainer.model.state_dict().values(), restored.trainer.model.state_dict().values()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert restored.trainer.optimizer.param_groups[0]["lr"] == 0.025
    assert restored.client_timer.local_comm_round_idx == 2


@pytest.mark.parametrize("key,value", [("max_concurrent_clients", "2"),
                                      ("client_execution_backend", "process"),
                                      ("client_rng_policy", "task-sha256-v1")])
def test_runner_execution_keys_cannot_be_bypassed(key, value):
    from configs.experiment import load_experiment

    with pytest.raises(ValueError):
        load_experiment(ROOT / "configs/experiments/smoke.yaml", [key, value])


def test_process_device_rejected_before_dataset_load(monkeypatch):
    import torch
    from configs.experiment import load_experiment, hpfl_config
    from algorithms_standalone.fedavg.FedAVGManager import FedAVGManager

    cfg = load_experiment(ROOT / "configs/experiments/smoke.yaml", ["runner.execution_backend", "process"])
    monkeypatch.setattr("algorithms_standalone.basePS.basePSmanager.load_data",
                        lambda **kw: pytest.fail("device 検証前のデータ取得は禁止"))
    with pytest.raises(ValueError):
        FedAVGManager(torch.device("cuda:1"), hpfl_config(cfg))


@pytest.mark.parametrize("fails", [False, True])
def test_parent_process_evaluation_releases_client_model(monkeypatch, fails):
    from types import SimpleNamespace
    import torch
    from configs.experiment import hpfl_config, load_experiment
    from trainers.normal_trainer import NormalTrainer

    args = hpfl_config(load_experiment(ROOT / "configs/experiments/smoke.yaml",
                                      ["runner.execution_backend", "process"]))
    model = torch.nn.Linear(2, 2)
    trainer = SimpleNamespace(model=model, role="client", args=args, criterion=torch.nn.CrossEntropyLoss())
    calls = []
    monkeypatch.setattr(model, "cpu", lambda: calls.append("cpu"))
    data = [(torch.ones(2, 2), torch.tensor([0, 1]))]
    if fails:
        monkeypatch.setattr(model, "forward", lambda *a: (_ for _ in ()).throw(RuntimeError("評価失敗")))
        with pytest.raises(RuntimeError, match="評価失敗"):
            NormalTrainer.test(trainer, data, device=torch.device("cpu"))
    else:
        NormalTrainer.test(trainer, data, device=torch.device("cpu"))
    assert calls == ["cpu"]


def test_process_rng_with_canonical_random_augmentation_and_marker_noise(tmp_path, monkeypatch):
    import numpy as np
    import torch
    from types import SimpleNamespace
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from configs.experiment import load_experiment, hpfl_config
    from algorithms_standalone.fedavg.FedAVGManager import FedAVGManager
    from data_preprocessing.cifar10.datasets import CIFAR10_truncated_WO_reload

    source = ROOT / "configs/experiments/cifar10-10clients.yaml"
    cfg = load_experiment(source, ["runner.execution_backend", "process", "comm_round", "2",
                                  "client_num_in_total", "2", "client_num_per_round", "2",
                                  "num_classes", "2", "model_output_dim", "2", "batch_size", "2",
                                  "data_load_num_workers", "0", "noise_type", "random"])
    generator = np.random.default_rng(23)
    full = SimpleNamespace(data=generator.integers(0, 256, size=(8, 32, 32, 3), dtype=np.uint8),
                           targets=np.arange(8) % 2)
    transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    loaders = {i: DataLoader(CIFAR10_truncated_WO_reload("未使用", dataidxs=indices,
                           full_dataset=full, transform=transform), batch_size=2, shuffle=True)
               for i, indices in enumerate(([7, 1, 5, 3], [6, 0, 4, 2]))}
    monkeypatch.setattr("algorithms_standalone.basePS.basePSmanager.load_data", lambda **kw:
                        [8, 8, loaders[0], loaders[1], {0: 4, 1: 4}, loaders, loaders, 2, {}])
    torch.set_num_threads(1)
    runs = []
    for workers in (1, 2):
        output = tmp_path / str(workers)
        output.mkdir()
        monkeypatch.chdir(output)
        args = hpfl_config(cfg)
        args.max_concurrent_clients = workers
        args.server_index = args.client_index = -1
        manager = FedAVGManager(torch.device("cpu"), args)
        manager.capture_process_history = True
        manager.server_timer.global_comm_round_idx = manager.max_comm_round
        manager.freeze()
        try:
            runs.append(manager._process_round(
                [1, 0], manager.aggregator.get_global_model_params(), "model", {}, {},
                manager.server_timer.get_time_info_to_send(), {}))
        finally:
            manager.client_executor.close()
    assert [r["client_id"] for r in runs[0]] == [1, 0]
    for a, b in zip(*runs):
        for x, y in zip(a["marker"], b["marker"]):
            np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-7)
        for key in a["state"]["model"]:
            torch.testing.assert_close(a["state"]["model"][key], b["state"]["model"][key], rtol=1e-6, atol=1e-7)
        assert a["observation"]["seed"] == b["observation"]["seed"]


@pytest.mark.parametrize("custom", ["collate", "replacement", "sampler_rng"])
def test_custom_loader_semantics_are_not_silently_dropped(custom):
    import torch
    from torch.utils.data import DataLoader, TensorDataset, RandomSampler
    from algorithms_standalone.fedavg.process_worker import export_loaders

    dataset = TensorDataset(torch.ones(4, 2), torch.zeros(4, dtype=torch.long))
    kwargs = {}
    if custom == "collate":
        kwargs["collate_fn"] = tuple
    else:
        kwargs["sampler"] = RandomSampler(dataset, replacement=custom == "replacement",
                                           generator=torch.Generator() if custom == "sampler_rng" else None)
    with pytest.raises(ValueError):
        export_loaders({0: DataLoader(dataset, batch_size=2, **kwargs)})


def slow_initializer():
    import time

    time.sleep(2)


def test_startup_timeout_does_not_block_on_large_task_send():
    import time
    from algorithms_standalone.basePS.client_executor import BoundedExecutor

    executor = BoundedExecutor(1, timeout=0.2, handler=scheduled_task, initializer=slow_initializer)
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        executor.run([dict(client_id=0, delay=0, payload=b"x" * 262144)])
    assert time.monotonic() - start < 1
    assert all(not p.is_alive() for p in executor.processes)
