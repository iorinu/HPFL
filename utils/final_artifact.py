"""HPFL 推論用 v1 成果物。学習再開用 checkpoint ではない。"""
import copy
import hashlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import signal
import json

_PUBLICATIONS = ContextVar("hpfl_publications", default=None)


@contextmanager
def publication_scope():
    """保存後のログ失敗・SIGTERM も成功とせず、自分の成果物だけ取り消す。"""
    if _PUBLICATIONS.get() is not None:
        yield
        return
    paths = []
    token = _PUBLICATIONS.set(paths)
    previous = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame):
        raise RuntimeError("終了要求により実行を中止しました")
    signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    except BaseException as exc:
        for path, inode in paths:
            if path.exists() and path.stat().st_ino == inode:
                path.unlink()
            status = path.parent / "status.json"
            if status.exists():
                status.write_text(json.dumps(dict(status="failed", error=str(exc)), ensure_ascii=False))
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)
        _PUBLICATIONS.reset(token)


def exclusive_run(method):
    @wraps(method)
    def run(cfg, device, output):
        output = Path(output).resolve()
        if any(p.name != "effective.yaml" for p in output.iterdir()):
            raise FileExistsError("空の固有出力ディレクトリが必要です")
        try:
            with publication_scope():
                return method(cfg, device, output)
        except BaseException as exc:
            (output / "summary.json").write_text(json.dumps(dict(
                status="failed", synthetic=cfg.runner.mode == "synthetic", device=device,
                error=str(exc)), ensure_ascii=False) + "\n")
            raise
    return run


def atomic_save(value, target):
    """同一 filesystem の hard link で、既存出力を置換せず公開する。"""
    import torch

    target = Path(target)
    if target.exists():
        raise FileExistsError(target)
    fd, name = tempfile.mkstemp(prefix=".final-", suffix=".tmp", dir=target.parent)
    temporary = Path(name)
    inode = os.fstat(fd).st_ino
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        publications = _PUBLICATIONS.get()
        if publications is not None:
            publications.append((target, inode))
        os.link(temporary, target)
        temporary.unlink()
    except BaseException:
        if target.exists() and target.stat().st_ino == inode:
            target.unlink()
        raise
    finally:
        temporary.unlink(missing_ok=True)


def publish_final(manager, output):
    """呼び出し側の全評価成功後だけ呼ぶ。tensor はここで CPU へ複製する。"""
    import torch
    from algorithms_standalone.fedavg.process_worker import cpu_tree
    import yaml

    aggregator = manager.aggregator
    ids = getattr(manager, "head_client_ids", list(range(len(aggregator.model_list))))
    if len(ids) != manager.args.client_num_in_total or len(ids) != len(aggregator.dist_list):
        raise RuntimeError("全 client の head/marker が必要です")
    # 認証・外部記録の設定を成果物に含めない。
    config = {k: v for k, v in dict(manager.args).items()
              if not any(word in k.lower() for word in ("key", "token", "secret", "password", "credential", "auth"))}
    artifact = dict(format="hpfl-inference", version=1, resume_supported=False,
                    config_yaml=yaml.safe_dump(config), client_ids=ids,
                    client_to_head={str(client): index for index, client in enumerate(ids)},
                    shared=cpu_tree(aggregator.trainer.model.state_dict()),
                    heads=[[cpu_tree(m.state_dict()) for m in head] for head in aggregator.model_list],
                    markers=[[torch.as_tensor(v).clone() for v in marker] for marker in aggregator.dist_list],
                    metadata=dict(model_residency=manager.args.model_residency,
                                  capacity_policy=manager.args.capacity_policy,
                                  rng_policy=manager.args.client_rng_policy))
    target = Path(output) / "final-model.pt"
    with publication_scope():
        atomic_save(artifact, target)
        stat = target.stat()
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return dict(path=target.name, bytes=stat.st_size, allocated_bytes=stat.st_blocks * 512,
                    sha256=digest.hexdigest(), model_write_count=1)


class InferenceArtifact:
    def logits(self, x, head_index=None):
        import torch

        model = self.aggregator.trainer.model
        if head_index is not None:
            model = copy.deepcopy(model)
            for module, head in zip(list(model.children())[-self.args.freeze_layer:],
                                    self.aggregator.model_list[head_index], strict=True):
                module.load_state_dict(head.state_dict())
        model.eval()
        with torch.no_grad():
            return model(x.to(self.device))

    def select(self, loader):
        return self.aggregator.select_pluggable(loader, "MMD")


def load_final(path, device="cpu"):
    """安全な tensor/primitive 読込で共有・個別推論と既存 MMD 選択を復元する。"""
    import torch
    import yaml
    from configs.config import CN
    from model.build import create_model
    from algorithms_standalone.fedavg.aggregator import FedAVGAggregator

    value = torch.load(path, map_location="cpu", weights_only=True)
    if value["format"] != "hpfl-inference" or value["version"] != 1:
        raise ValueError("非対応の推論形式です")
    result = InferenceArtifact()
    result.args = CN(yaml.safe_load(value["config_yaml"]))
    result.device = torch.device(device)
    result.client_ids = value["client_ids"]
    model = create_model(result.args, model_name=result.args.model,
                         output_dim=result.args.model_output_dim, device=result.device, pretrained=False)
    model.load_state_dict(value["shared"])
    model.to(result.device).eval()
    aggregator = FedAVGAggregator.__new__(FedAVGAggregator)
    aggregator.args, aggregator.device = result.args, result.device
    aggregator.trainer = SimpleNamespace(model=model)
    aggregator.model_list = []
    for states in value["heads"]:
        head = copy.deepcopy(list(model.children())[-result.args.freeze_layer:])
        for module, state in zip(head, states, strict=True):
            module.load_state_dict(state)
        aggregator.model_list.append(head)
    aggregator.dist_list = [[v.numpy() for v in marker] for marker in value["markers"]]
    result.aggregator = aggregator
    return result
