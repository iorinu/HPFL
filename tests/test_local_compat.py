"""今回の実障害だけを確認する回帰テスト。"""
from types import SimpleNamespace

import torch
import pytest


def test_numpy2_recorder_import():
    # NumPy の非公開モジュールに依存せず読み込めること。
    import recorders.model_dif_track


def test_dataloader_iterator_and_restart():
    from trainers.normal_trainer import NormalTrainer

    trainer = NormalTrainer.__new__(NormalTrainer)
    trainer.args = SimpleNamespace(batch_size=2)
    loader = torch.utils.data.DataLoader(torch.arange(4).reshape(4, 1), batch_size=2)
    first = trainer.get_train_batch_data(loader)
    second = trainer.get_train_batch_data(loader)
    restart = trainer.get_train_batch_data(loader)
    assert first.flatten().tolist() == [0, 1]
    assert second.flatten().tolist() == [2, 3]
    assert torch.equal(first, restart)


@pytest.mark.parametrize("name", ["auto", "cuda:1", "invalid"])
def test_invalid_device(name):
    from tools.local_smoke import select_device

    with pytest.raises(ValueError):
        select_device(name)


def test_cuda_unavailable_has_no_cpu_fallback(monkeypatch):
    from tools.local_smoke import select_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU"):
        select_device("cuda")
    assert select_device("cpu") == torch.device("cpu")


def test_cuda_request_selects_cuda(monkeypatch):
    from tools.local_smoke import select_device

    # GPU のないテスト環境でも選択処理だけを確認し、context は作らない。
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert select_device("cuda") == torch.device("cuda:0")
