# Hot Plgguable Federated Learning

Official code for paper "Hot-pluggable Federated Learning: Bridging General and Personalized FL via Dynamic Selection" (ICLR 2025).


![](./pipeline.png)

## Code Structure.

```
.
├── algorithms_standalone    # multiple round training and testing for algorithms
├── configs    # some default configurations
├── data_preprocessing    # loading datasets
├── experiments
│   ├── main_args.conf    # includes the default hyper-parameters.
│   ├── configs_algorithm    # includes the configuration hyper-parameters of the specific algorithms.
│   ├── configs_system    # includes the configuration hyper-parameters related to the running environment. This is designed for users' convenience of no need to specifying Python Path and Data Dir when launching experiments every time.
│   └── standalone    # includes scripts for HPFL and all algorithms, together with some default configuration.
├── requirements.yml    # dependency requirements
├── trainers
│   └── normal_trainer.py    # single round training for specific algorithms, like FedRoD, FedTHE, FedSAM, etc.
└── utils    # implementation of small algorithm components like CCA, CKA, MMD
```


## Launch Experiments.

このforkの推奨入口は **[YAML実験設定・共通Nix runner](docs/experiment-config.md)**。
Nixは依存だけを管理し、実験条件は `configs/experiments/*.yaml` に保存する。

```bash
bash tools/run-experiment.sh --config configs/experiments/smoke.yaml --check
bash tools/run-local-smoke.sh                 # 合成画像のみ・CPU
bash tools/run-local-smoke.sh --device cuda   # 合成画像のみ・GPU
bash tools/run-experiment.sh --config configs/experiments/smoke.yaml comm_round 2
bash tools/run-experiment.sh --config configs/experiments/cifar10-10clients.yaml --check
bash tools/run-experiment.sh --config configs/experiments/cifar10-100clients.yaml --check
```

実データpresetは今回は検証のみ。`--check` を外すとダウンロード・長時間学習を開始し得る。
出力は毎回固有の `.local-experiments/`、データは `.local-data/`。外部実験記録は無効。
編集先は各YAML。`max_epochs` は省略時にround×local epochから自動算出する。
共通runnerはGit flake（`.` / `.#gpu`）の依存環境とライブ作業ツリーを使い、git addは不要。

### モデル常駐と最終保存（issue #2）

既定は `model_residency=device`（CUDA は VRAM、CPU は RAM）、`capacity_policy=error`。
未選択 client を含む現在のモデル・optimizer を保持し、容量不足では自動 CPU/disk 退避せず失敗する。
8GB GPU に10/100-client presetが収まる保証はない。CPU保持は `model_residency cpu` の明示指定だけ。
共有・head・marker・選択・全最終評価の成功後にだけ **`final-model.pt`**（共有モデル＋個別head＋marker＋対応表）を公開する。
途中 `.pth`、比較contract、全roundのtensor履歴は既定で保存しない。resume/checkpoint_saveは非対応。
比較専用の大きなdumpは合成runで `debug_tensor_dumps True` を明示する。
旧issue1の保存・CPU往復・VRAM上限説明は今回の仕様で置き換わる。形式・再読込・障害時・容量予算は
[詳細](docs/experiment-config.md#issue2-model-lifecycle)を参照。今回のCUDA常駐実機検証はHermesに委ねる。

### client並行学習（明示opt-in）

既定のlegacy `sequential / 1` は維持する。標準FedAvg+HPFLの共有roundと最終headを、
spawn workerで最大2clients並行実行できる（単一 `cuda:0`、またはCPU）。

```bash
bash tools/run-experiment.sh --config configs/experiments/smoke.yaml --device cuda runner.execution_backend process runner.max_concurrent_clients 2 comm_round 2 momentum 0.9
```

CPU確認は `--device cpu`、比較参照は `runner.max_concurrent_clients 1`。
`runner.client_timeout_seconds` は既定120秒。全task成功後にsampling順で集約し、失敗時はworkerを回収して中止する。
processは独立した`task-sha256-v1`乱数ポリシーであり、historical RNGとの一致は保証しない。
CPU suiteに加え、[RTX 5060実機検証](docs/experiment-config.md#gpu-verification)でprocess/1対2の状態一致を確認。安定した高速化は未確認、kernel同時実行traceは未実施。
特殊algorithm/loader等の未対応経路、設定、観測ログ、host/VRAMの限界は [詳細](docs/experiment-config.md#opt-in-の-bounded-process-client-学習) を参照。

以下は上流の旧起動方法で、共通runnerはこれらのshell設定を読み込まない。

1. Download and specific the data directory with parameter ``data_dir`` in command (For most dataset, the downloading will begin when runnning the script)

2. To launch HPFL and baselines, you can ``cd experiments/standlone``. And type ``./auto_pipeline.sh`` to run the code for 10 clients, ``./auto_pipeline_100client.sh`` for 100 clients. You can type ``dataset=cifar10 partition_alpha=0.1 partition_method=hetero algorithm=FedSAM ./launch_standalone.sh`` to set the dataset as CIFAR-10 and the a=0.1 of  partition method, while for FedSAM algorithm. 


## Wandb Usage

If you do not want to use wandb for recording, you can add ``wandb_record=False`` in to the command (such as in experiments/standalone/auto_pipeline.sh and experiments/standalone/auto_pipeline_100client.sh). Then you don't need to install and config wandb.

If else, please replace WANDB_KEY_EXAMPLE in configs/default.py, standalone/auto_pipeline.sh and experiments/standalone/auto_pipeline_100client.sh








