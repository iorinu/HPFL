"""既存 YACS 設定に実験 YAML と明示 CLI 上書きを順に適用する。"""
from pathlib import Path
import math
import yaml

from configs import get_cfg
from configs.config import CN

ROOT = Path(__file__).resolve().parents[1]


def validate_supplied(values, prefix=""):
    for key, value in values.items():
        if not isinstance(key, str):
            raise ValueError("設定キーは文字列で指定してください")
        name = prefix + key
        if any(word in key.lower() for word in ("key", "token", "secret", "password", "credential", "auth")):
            raise ValueError(f"{name}: 認証用設定は禁止です")
        if isinstance(value, dict):
            validate_supplied(value, name + ".")
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name}: 有限の非負数が必要です")
        elif isinstance(value, (list, tuple)):
            validate_supplied({str(i): v for i, v in enumerate(value)}, name + ".")


def load_experiment(path, overrides=()):
    cfg = get_cfg()
    cfg.runner = CN(dict(mode="real", train_per_client=16, test_per_client=8))
    cfg.wandb_record = False
    cfg.wandb_key = None
    cfg.gpu_index = 0
    cfg.data_dir = ".local-data/cifar10"
    required_values = {key for key, value in cfg.items() if value is not None}
    try:
        supplied = yaml.safe_load(Path(path).read_text())
        if not isinstance(supplied, dict):
            raise ValueError("YAML の最上位は mapping が必要です")
        validate_supplied(supplied)
        cfg.merge_from_other_cfg(CN(supplied))
        if len(overrides) % 2:
            raise ValueError("CLI 上書きは KEY VALUE の組が必要です")
        for key, value in zip(overrides[::2], overrides[1::2]):
            validate_supplied({key: yaml.safe_load(value)})
        cfg.merge_from_list(list(overrides))
    except (KeyError, AssertionError, TypeError, yaml.YAMLError) as exc:
        raise ValueError(f"設定のキー・型・構文が不正です: {exc}") from exc
    for key in required_values:
        if cfg[key] is None:
            raise ValueError(f"{key}: null は指定できません")
    # YACS が文字列から数値へ変換した値も再検証する。
    for key in [*supplied, *overrides[::2]]:
        value = cfg
        for part in key.split("."):
            value = value[part]
        validate_supplied({key: value})
    if cfg.runner.mode not in ("synthetic", "real"):
        raise ValueError("runner.mode: synthetic または real が必要です")
    for key in ("client_num_in_total", "client_num_per_round", "batch_size", "comm_round",
                "max_epochs", "global_epochs_per_round", "HPFL_local_iteration", "num_classes",
                "model_output_dim", "frequency_of_the_test"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{key}: 正の整数が必要です")
    # 明示指定は保持し、省略時だけ最終的な round/local epoch から算出する。
    if "max_epochs" not in supplied and "max_epochs" not in overrides[::2]:
        cfg.max_epochs = cfg.comm_round * cfg.global_epochs_per_round
    if cfg.client_num_per_round > cfg.client_num_in_total:
        raise ValueError("client_num_per_round は client_num_in_total 以下が必要です")
    if cfg.num_classes != cfg.model_output_dim:
        raise ValueError("num_classes と model_output_dim は一致が必要です")
    if cfg.wandb_record or not cfg.wandb_offline or cfg.wandb_save_record_dataframe:
        raise ValueError("W&B は disabled のみ対応します")
    for key, expected in dict(algorithm="FedAvg", HPFL=True, mode="standalone",
                              pretrained=False, pretrained_submodel=False,
                              load_backbone_from=None, gpu_index=0).items():
        if cfg[key] != expected:
            raise ValueError(f"{key}: この runner は {expected} のみ対応します（resume なし）")
    for key in ("checkpoint_root_path", "image_save_path"):
        path_value = Path(cfg[key])
        if path_value.is_absolute() or ".." in path_value.parts:
            raise ValueError(f"{key}: 固有出力内の相対パスが必要です")
    if cfg.runner.mode == "real":
        if any(key in supplied.get("runner", {}) for key in ("train_per_client", "test_per_client")) or any(
                key.startswith("runner.") and key != "runner.mode" for key in overrides[::2]):
            raise ValueError("runner の合成件数設定は synthetic 専用です")
    if cfg.comm_round * cfg.global_epochs_per_round != cfg.max_epochs:
        raise ValueError("max_epochs = comm_round * global_epochs_per_round が必要です（上流の round 算出）")
    if cfg.runner.mode == "synthetic":
        for key, low, high in (("client_num_in_total", 2, 8), ("num_classes", 2, 10),
                               ("batch_size", 2, 64), ("comm_round", 1, 5),
                               ("global_epochs_per_round", 1, 5), ("HPFL_local_iteration", 1, 5)):
            if not low <= cfg[key] <= high:
                raise ValueError(f"synthetic {key}: {low}..{high} が必要です")
        for key in ("train_per_client", "test_per_client"):
            count = cfg.runner[key]
            if type(count) is not int or not max(cfg.num_classes, cfg.batch_size) <= count <= 128:
                raise ValueError(f"runner.{key}: 全クラスと非空 batch を含む 128 件以下が必要です")
            if count % cfg.batch_size:
                raise ValueError(f"runner.{key}: MMD/BatchNorm のため完全 batch が必要です")
        if cfg.client_num_per_round != cfg.client_num_in_total:
            raise ValueError("synthetic は全 client 参加が必要です")
        for key, expected in dict(model="resnet18_v2", dataset="cifar10", HPFL=True,
                                  freeze_layer=1, noise_on_feat=False, noise_on_test_feat=False,
                                  group_feat=False, if_OOD=False, fedavg_cfl=False,
                                  finetune_FedAvg=False, instantiate_all=True,
                                  fedavg_local_step_type="whole", pretrained=False).items():
            if cfg[key] != expected:
                raise ValueError(f"synthetic {key}: {expected} のみ対応します")
    cfg.data_dir = str((ROOT / cfg.data_dir).resolve())
    if not Path(cfg.data_dir).is_relative_to(ROOT / ".local-data"):
        raise ValueError("data_dir: データはリポジトリ内 .local-data 以下に保存してください")
    return cfg


def hpfl_config(cfg):
    resolved = cfg.clone()
    del resolved["runner"]
    return resolved
