"""合成画像だけで公式 HPFL 経路を確認する。実データ評価ではない。"""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import random
import resource
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"
os.environ["MPLBACKEND"] = "Agg"
# torch の import と CUDA context 作成より前に設定する。
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def select_device(name):
    import torch

    if name not in ("cpu", "cuda"):
        raise ValueError("device は cpu または cuda で指定してください")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA が利用できません。CPU への自動切替は行いません")
    return torch.device("cuda:0" if name == "cuda" else "cpu")


def attach_device_probe(trainer, device, evidence, handles):
    import torch

    counts = dict(forward_calls=0, training_forwards=0, shared_forwards=0,
                  head_forwards=0, loss_calls=0, input_devices=[],
                  parameter_devices=[], target_devices=[])
    evidence.append(counts)

    def record(values, value):
        if value not in values:
            values.append(value)

    def model_hook(model, inputs, output):
        x = inputs[0]
        assert x.device == device and output.device == device
        parameter_devices = {str(p.device) for p in model.parameters()}
        assert parameter_devices == {str(device)}
        counts["forward_calls"] += 1
        record(counts["input_devices"], str(x.device))
        for actual in parameter_devices:
            record(counts["parameter_devices"], actual)
        if model.training and torch.is_grad_enabled():
            counts["training_forwards"] += 1
            phase = "head" if not model.conv1.weight.requires_grad else "shared"
            counts[f"{phase}_forwards"] += 1

    def loss_hook(criterion, inputs):
        logits, target = inputs[:2]
        assert logits.device == device and target.device == device
        counts["loss_calls"] += 1
        record(counts["target_devices"], str(target.device))

    handles.append(trainer.model.register_forward_hook(model_hook))
    handles.append(trainer.criterion.register_forward_pre_hook(loss_hook))
    return counts


def synthetic_data(cfg, **kwargs):
    import torch

    # CIFAR は形状指定だけ。元データの取得・部分抽出は行わない。
    generator = torch.Generator().manual_seed(cfg.seed)
    trains, tests, counts = {}, {}, {}
    for client in range(cfg.client_num_in_total):
        for loaders, count in ((trains, cfg.runner.train_per_client), (tests, cfg.runner.test_per_client)):
            labels = torch.arange(count) % cfg.num_classes
            images = torch.randn(count, 3, 32, 32, generator=generator)
            images += client * 0.25
            dataset = torch.utils.data.TensorDataset(images, labels)
            dataset.targets = labels.numpy()
            loaders[client] = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size)
        counts[client] = cfg.runner.train_per_client

    def global_loader(loaders):
        return torch.utils.data.DataLoader(
            torch.utils.data.ConcatDataset([loader.dataset for loader in loaders.values()]),
            batch_size=cfg.batch_size)

    return [cfg.runner.train_per_client * cfg.client_num_in_total,
            cfg.runner.test_per_client * cfg.client_num_in_total,
            global_loader(trains), global_loader(tests), counts, trains, tests, cfg.num_classes, {}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    options = parser.parse_args()
    from configs.experiment import load_experiment

    cfg = load_experiment(ROOT / "configs/experiments/smoke.yaml")
    output = options.output.resolve()
    if not output.is_relative_to(ROOT / ".local-smoke"):
        parser.error("出力先はリポジトリ内 .local-smoke 以下にしてください")
    output.mkdir(parents=True, exist_ok=False)
    (output / "effective.yaml").write_text(cfg.dump())
    run_smoke(cfg, options.device, output)


def run_smoke(cfg, device_name, output):
    from configs.experiment import hpfl_config

    output = Path(output).resolve()
    # 上流の相対 checkpoint/result 出力も固有ディレクトリに閉じ込める。
    os.chdir(output)
    (output / "result").mkdir()
    os.environ["HOME"] = str(output)
    os.environ["XDG_CACHE_HOME"] = str(output / "cache")
    os.environ["MPLCONFIGDIR"] = str(output / "matplotlib")
    import numpy as np
    import torch
    import wandb
    from algorithms_standalone.fedavg.FedAVGManager import FedAVGManager
    from utils.mmd_loss import MMDLoss

    device = select_device(device_name)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    args = hpfl_config(cfg)
    args.server_index = -1
    args.client_index = -1

    start = time.monotonic()
    run = wandb.init(mode="disabled")
    handles, evidence = [], []
    mmd_evidence = dict(calls=0, input_devices=[])

    def mmd_hook(module, inputs):
        if isinstance(module, MMDLoss):
            mmd_evidence["calls"] += 1
            for tensor in inputs:
                actual = str(tensor.device)
                if actual not in mmd_evidence["input_devices"]:
                    mmd_evidence["input_devices"].append(actual)

    try:
        # モデル初期化と再配分も、記録した seed から開始する。
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        with patch("algorithms_standalone.basePS.basePSmanager.load_data", lambda **kw: synthetic_data(cfg)):
            manager = FedAVGManager(device, args)
        for client in manager.client_list:
            attach_device_probe(client.trainer, device, evidence, handles)
        attach_device_probe(manager.aggregator.trainer, device, evidence, handles)
        handles.append(torch.nn.modules.module.register_module_forward_pre_hook(mmd_hook))
        manager.train()
        assert manager.server_timer.global_comm_round_idx == cfg.comm_round + 1
        assert len(manager.aggregator.model_list) == cfg.client_num_in_total
        assert len(manager.aggregator.dist_list) == cfg.client_num_in_total
        manager.generate_personalized_data()
        manager.test_locally()
        manager.test_with_personalized_batch("index")
        scores = []
        for loader in manager.test_data_local_dict.values():
            score, choice = manager.aggregator.select_pluggable(loader, "MMD")
            assert np.isfinite(score).all() and 0 <= choice < cfg.client_num_in_total
            scores.append({"score": score.tolist(), "choice": int(choice)})
        manager.test_with_personalized_batch("MMD")
        batches = cfg.runner.train_per_client // cfg.batch_size
        for counts in evidence[:cfg.client_num_in_total]:
            assert counts["shared_forwards"] == batches * cfg.global_epochs_per_round * cfg.comm_round
            assert counts["head_forwards"] == batches * cfg.HPFL_local_iteration
            assert counts["loss_calls"] > 0
        assert mmd_evidence["calls"] > 0
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.monotonic() - start
        cuda_info = None
        if device.type == "cuda":
            cuda_info = dict(name=torch.cuda.get_device_name(device),
                             capability=list(torch.cuda.get_device_capability(device)),
                             torch_cuda=torch.version.cuda,
                             nccl_version=list(torch.cuda.nccl.version()),
                             peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                             peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
            assert cuda_info["peak_allocated_bytes"] > 0
        checkpoints = sorted(str(path.relative_to(output)) for path in output.glob("*.pth"))
        assert checkpoints
        result = dict(status="passed", synthetic=True, device=device_name, seed=args.seed,
                      shared_rounds=cfg.comm_round, local_epochs=cfg.global_epochs_per_round,
                      head_epochs=cfg.HPFL_local_iteration, clients=cfg.client_num_in_total,
                      train_per_client=cfg.runner.train_per_client, test_per_client=cfg.runner.test_per_client,
                      batch_size=cfg.batch_size, num_classes=cfg.num_classes,
                      model=args.model, markers=len(manager.aggregator.dist_list), mmd=scores, checkpoints=checkpoints,
                      seconds=elapsed, device_evidence=evidence, mmd_evidence=mmd_evidence,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      vram=cuda_info if cuda_info else "未使用（CPU）", python=sys.version,
                      versions={name: importlib.metadata.version(name) for name in
                                ["torch", "torchvision", "numpy", "scipy", "scikit-learn", "wandb"]})
        (output / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    finally:
        for handle in handles:
            handle.remove()
        run.finish()


if __name__ == "__main__":
    main()
