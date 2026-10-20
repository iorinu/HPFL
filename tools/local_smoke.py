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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    options = parser.parse_args()
    output = options.output.resolve()
    if not output.is_relative_to(ROOT / ".local-smoke"):
        parser.error("出力先はリポジトリ内 .local-smoke 以下にしてください")
    output.mkdir(parents=True, exist_ok=False)
    # 上流の相対 checkpoint/result 出力も固有ディレクトリに閉じ込める。
    os.chdir(output)
    (output / "result").mkdir()
    os.environ["HOME"] = str(output)
    os.environ["XDG_CACHE_HOME"] = str(output / "cache")
    os.environ["MPLCONFIGDIR"] = str(output / "matplotlib")
    import numpy as np
    import torch
    import wandb
    from configs import get_cfg
    from algorithms_standalone.fedavg.FedAVGManager import FedAVGManager
    from utils.mmd_loss import MMDLoss

    device = select_device(options.device)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    args = get_cfg()
    for key, value in dict(
        client_num_in_total=2, client_num_per_round=2, batch_size=8,
        comm_round=1, max_epochs=1, global_epochs_per_round=1,
        HPFL=True, HPFL_local_iteration=1, algorithm="FedAvg",
        model="resnet18_v2", dataset="cifar10", num_classes=2,
        model_output_dim=2, client_optimizer="sgd", server_optimizer="no",
        server_index=-1, client_index=-1,
        psgd_exchange="model", data_load_num_workers=0, seed=7, lr=0.01,
        wandb_record=False, wandb_save_record_dataframe=False,
        noise_on_feat=False, noise_on_test_feat=False, freeze_layer=1,
        checkpoint_save=False,
    ).items():
        setattr(args, key, value)

    def synthetic_data(**kwargs):
        # CIFAR は形状指定だけ。元データの取得・部分抽出は行わない。
        generator = torch.Generator().manual_seed(args.seed)
        trains, tests, counts = {}, {}, {}
        for client in range(2):
            for loaders, count in ((trains, 16), (tests, 8)):
                labels = torch.arange(count) % 2
                images = torch.randn(count, 3, 32, 32, generator=generator)
                images += client * 0.25
                dataset = torch.utils.data.TensorDataset(images, labels)
                dataset.targets = labels.numpy()
                loaders[client] = torch.utils.data.DataLoader(dataset, batch_size=8)
            counts[client] = 16
        def global_loader(loaders):
            return torch.utils.data.DataLoader(
                torch.utils.data.ConcatDataset([loader.dataset for loader in loaders.values()]),
                batch_size=8,
            )
        return [32, 16, global_loader(trains), global_loader(tests), counts,
                trains, tests, 2, {}]

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
        with patch("algorithms_standalone.basePS.basePSmanager.load_data", synthetic_data):
            manager = FedAVGManager(device, args)
        for client in manager.client_list:
            attach_device_probe(client.trainer, device, evidence, handles)
        attach_device_probe(manager.aggregator.trainer, device, evidence, handles)
        handles.append(torch.nn.modules.module.register_module_forward_pre_hook(mmd_hook))
        manager.train()
        assert manager.server_timer.global_comm_round_idx == 2
        assert len(manager.aggregator.model_list) == 2
        assert len(manager.aggregator.dist_list) == 2
        manager.generate_personalized_data()
        manager.test_locally()
        manager.test_with_personalized_batch("index")
        scores = []
        for loader in manager.test_data_local_dict.values():
            score, choice = manager.aggregator.select_pluggable(loader, "MMD")
            assert np.isfinite(score).all() and 0 <= choice < 2
            scores.append({"score": score.tolist(), "choice": int(choice)})
        manager.test_with_personalized_batch("MMD")
        for counts in evidence[:2]:
            assert counts["shared_forwards"] == 2 and counts["head_forwards"] == 2
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
        result = dict(status="passed", synthetic=True, device=options.device, seed=args.seed,
                      shared_rounds=1, head_epochs=1, clients=2,
                      train_per_client=16, test_per_client=8, batch_size=8,
                      model=args.model, markers=2, mmd=scores, checkpoints=checkpoints,
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
