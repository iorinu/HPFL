"""CPU の明示的状態だけを受け取り、worker 内で client を再構築する。"""
import copy
import hashlib
import os
import random
import resource
import time

import numpy as np
import torch
from torch.utils.data import (DataLoader, Dataset, RandomSampler, SequentialSampler,
                              TensorDataset, BatchSampler, default_collate)


RNG_POLICY = "task-sha256-v1"


def weights_digest(weights):
    digest = hashlib.sha256()
    for name, tensor in weights.items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def task_seed(seed, client_id, round_index, phase):
    value = f"{RNG_POLICY}:{seed}:{client_id}:{round_index}:{phase}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big")


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    return copy.deepcopy(value)


def export_client(client):
    trainer = client.trainer
    return dict(model=cpu_tree(trainer.model.state_dict()),
                optimizer=cpu_tree(trainer.optimizer.state_dict()),
                scheduler=(cpu_tree({k: v for k, v in vars(trainer.lr_scheduler).items()
                                     if k != "optimizer"}) if trainer.lr_scheduler else None),
                timer=cpu_tree({k: v for k, v in vars(client.client_timer).items() if k != "args"}),
                requires_grad={k: p.requires_grad for k, p in trainer.model.named_parameters()},
                training={k: m.training for k, m in trainer.model.named_modules()},
                global_epochs_per_round=client.global_epochs_per_round)


def restore_client(client, state):
    trainer = client.trainer
    trainer.model.load_state_dict(state["model"])
    trainer.optimizer.load_state_dict(state["optimizer"])
    if trainer.lr_scheduler is not None:
        vars(trainer.lr_scheduler).update(state["scheduler"])
    vars(client.client_timer).update(state["timer"])
    client.global_epochs_per_round = state["global_epochs_per_round"]
    for name, parameter in trainer.model.named_parameters():
        parameter.requires_grad = state["requires_grad"][name]
    for name, module in trainer.model.named_modules():
        module.training = state["training"][name]


def export_loaders(loaders, bank=None):
    """canonical partition 索引と未変換画像を共有する。loader 自体は送らない。"""
    from data_preprocessing.cifar10.datasets import CIFAR10_truncated_WO_reload

    bank = {} if bank is None else bank
    specs = {}
    for client_id, loader in loaders.items():
        if (type(loader) is not DataLoader or type(loader.batch_sampler) is not BatchSampler
                or loader.collate_fn is not default_collate):
            raise ValueError("process は標準 DataLoader/batch/collate のみ対応します")
        if type(loader.sampler) not in (SequentialSampler, RandomSampler):
            raise ValueError("process 非対応 sampler です")
        if isinstance(loader.sampler, RandomSampler) and (
                loader.sampler.replacement or loader.sampler.num_samples != len(loader.dataset)
                or loader.sampler.generator is not None):
            raise ValueError("process は標準の非復元 RandomSampler のみ対応します")
        if loader.generator is not None or loader.worker_init_fn is not None:
            raise ValueError("process は独自 loader RNG/worker 初期化に非対応です")
        dataset = loader.dataset
        if type(dataset) is TensorDataset:
            if len(dataset.tensors) != 2 or any(t.device.type != "cpu" for t in dataset.tensors):
                raise ValueError("process の合成データは CPU の画像・ラベル対が必要です")
            key = id(dataset)
            if key not in bank:
                bank[key] = tuple(t.detach().clone().share_memory_() for t in dataset.tensors)
            data, targets = bank[key]
            indices, transform, target_transform, kind = None, None, None, "tensor"
        elif type(dataset) is CIFAR10_truncated_WO_reload:
            full = dataset.full_dataset
            key = id(full)
            if key not in bank:
                bank[key] = (torch.from_numpy(np.asarray(full.data).copy()).share_memory_(),
                             torch.as_tensor(np.asarray(full.targets).copy()).share_memory_())
            data, targets = bank[key]
            indices = None if dataset.dataidxs is None else np.asarray(dataset.dataidxs).copy()
            transform, target_transform, kind = dataset.transform, dataset.target_transform, "cifar"
        else:
            raise ValueError(f"process 非対応 dataset: {type(dataset).__name__}")
        specs[client_id] = dict(data=data, targets=targets, indices=indices,
                                transform=transform, target_transform=target_transform, kind=kind,
                                batch_size=loader.batch_size, drop_last=loader.drop_last,
                                shuffle=type(loader.sampler) is RandomSampler)
    return specs


class PartitionDataset(Dataset):
    def __init__(self, spec):
        self.spec = spec

    def __len__(self):
        indices = self.spec["indices"]
        return len(self.spec["targets"]) if indices is None else len(indices)

    def __getitem__(self, index):
        spec = self.spec
        if spec["indices"] is not None:
            index = int(spec["indices"][index])
        image, target = spec["data"][index], spec["targets"][index]
        if spec["kind"] == "cifar":
            image, target = image.numpy(), target.numpy()
        if spec["transform"] is not None:
            image = spec["transform"](image)
        if spec["target_transform"] is not None:
            target = spec["target_transform"](target)
        return image, target


def make_loader(spec):
    return DataLoader(PartitionDataset(spec), batch_size=spec["batch_size"],
                      shuffle=spec["shuffle"], drop_last=spec["drop_last"], num_workers=0)


def initialize_worker(train_specs, test_specs, counts):
    global _TRAIN, _TEST, _COUNTS
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _TRAIN, _TEST, _COUNTS = train_specs, test_specs, counts


def train_task(task):
    from configs.config import CN
    from model.build import create_model
    from trainers.build import create_trainer
    from utils.metrics import Metrics
    from utils.data_utils import get_avg_num_iterations
    from .client import FedAVGClient

    start = time.monotonic()
    args = CN(task["config"])
    client_id = task["client_id"]
    seed = task_seed(args.seed, client_id, task["round"], task["phase"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(task["device"])
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("要求した CUDA は利用できません")
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    # worker は checkpoint/telemetry を所有しない。学習条件は変更しない。
    args.checkpoint_save = False
    args.data_save_memory_mode = True
    args.wandb_record = False
    trains = {client_id: make_loader(_TRAIN[client_id])}
    tests = {client_id: make_loader(_TEST[client_id])}
    model = create_model(args, model_name=args.model, output_dim=args.model_output_dim,
                         device=torch.device("cpu"), pretrained=False)
    trainer = create_trainer(args, device, model,
                             num_iterations=get_avg_num_iterations(_COUNTS, args.batch_size),
                             client_index=client_id, role="client")
    client = FedAVGClient(client_id, trains, _COUNTS, tests, sum(_COUNTS.values()),
                          device, args, trainer, metrics=Metrics([1], task=args.task))
    restore_client(client, task["state"])
    forwards, losses = 0, 0
    loaded_digest = None

    def before_forward(module, inputs):
        nonlocal loaded_digest
        if loaded_digest is None:
            loaded_digest = weights_digest(module.state_dict())
            if loaded_digest != task["weights_sha256"]:
                raise RuntimeError("round 開始 weights が一致しません")

    def probe(module, inputs, output):
        nonlocal forwards
        if inputs[0].device != device or any(p.device != device for p in module.parameters()):
            raise RuntimeError("worker の学習 device が要求と一致しません")
        forwards += 1

    def loss_probe(module, inputs):
        nonlocal losses
        if any(t.device != device for t in inputs):
            raise RuntimeError("worker の logits/target device が一致しません")
        losses += 1

    handle = model.register_forward_hook(probe)
    before_handle = model.register_forward_pre_hook(before_forward)
    loss_handle = trainer.criterion.register_forward_pre_hook(loss_probe)
    built = time.monotonic()
    result = client.train(task["update"], task["time"], client_id, task["weights"], "model",
                          {}, traininig_start=task["round"] == 0, shared_params_for_simulation={})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    trained = time.monotonic()
    marker = client.cal_dist(client_id) if task["phase"] == "head" else None
    client.move_to_cpu()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    end = time.monotonic()
    handle.remove()
    before_handle.remove()
    loss_handle.remove()
    observation = dict(event="client", client_id=client_id, round=task["round"], phase=task["phase"],
                       pid=os.getpid(), device=str(device), start=start, end=end,
                       build_seconds=built-start, train_seconds=trained-built,
                       marker_seconds=end-trained, forward_calls=forwards,
                       loss_calls=losses, target_device=str(device),
                       start_weights_sha256=task["weights_sha256"], loaded_weights_sha256=loaded_digest,
                       seed=seed, rng_policy=RNG_POLICY,
                       host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       cuda_peak_allocated_bytes=(torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0),
                       cuda_peak_reserved_bytes=(torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0))
    return dict(client_id=client_id, result=cpu_tree(result), state=export_client(client),
                marker=marker, observation=observation)
