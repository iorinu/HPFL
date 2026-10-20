# YAML 実験設定と共通 Nix runner

承認範囲・受入条件は [仕様](specs/experiment-config.md)。Nixは依存管理のみ、実験条件はYAMLに置く。
リポジトリのルートで実行する。実データ学習は今回未実行。
共通runnerはGit flakeの `nix develop .` / `nix develop .#gpu` を使う。
追跡済みの `flake.nix` / `flake.lock` で依存環境だけを用意し、Python/YAMLはライブ作業ツリーを参照する。
新規実験ファイルのgit addは不要。出力を含むディレクトリ全体をNixへコピーしないため、path flakeは使わない。

## コマンド

```bash
# 検証・表示のみ。学習、データ取得、run作成なし。CUDA実使用可否までは検査しない。
bash tools/run-experiment.sh --config configs/experiments/smoke.yaml --check
bash tools/run-experiment.sh --config configs/experiments/cifar10-10clients.yaml --check
bash tools/run-experiment.sh --config configs/experiments/cifar10-100clients.yaml --device cuda --dry-run

# 合成画像だけの確認。既定CPU、学習プロセス上限180秒。
bash tools/run-experiment.sh --config configs/experiments/smoke.yaml
bash tools/run-experiment.sh --config configs/experiments/smoke.yaml --device cuda
# 既存入口も同じYAML・共通runnerに委譲。
bash tools/run-local-smoke.sh --device cuda

# オプションを先に、YACS KEY VALUEの組を最後に指定。boolはTrue/False。
bash tools/run-experiment.sh --config configs/experiments/smoke.yaml comm_round 2
```

将来、実データを**明示的に開始**する場合だけ `--check` を外す。
その場合、公式loaderがCIFAR10をダウンロードする可能性があり、1000 roundsの長時間学習になる。
実データには既定180秒制限を適用しない。必要なら `--timeout 3600` など有限の正数で上限秒を指定する。
CUDAは1台目（`cuda:0`）のみ。利用不可ならエラーとし、自動CPU切替はしない。
CPUは公式mainのCUDA自動判定をwrapper内で無効にし、CPUモデル・データ配置を強制する。
複数GPU・分散通信・実データの精度や性能は未検証。

Nix shell内では `python tools/run_experiment.py --config ... --check` も使える。
旧 `python tools/local_smoke.py --output .local-smoke/新規名 [--device cuda]` は維持するが、
180秒制限・共通の失敗ログ管理が必要ならshellの共通入口を使う。

## スキーマと優先順位

HPFL設定は `configs/default.py` と同じ**平坦なキー名**。予約部分は次だけ。

```yaml
runner:
  mode: synthetic  # synthetic または real
  train_per_client: 16  # synthetic専用
  test_per_client: 8    # synthetic専用
```

YACS defaultをcloneし、runnerの安全基準を加え、YAML、明示CLIの順でmerge/type checkする。
安全基準は `wandb_record=False`、`wandb_key=None`、`gpu_index=0`、
`data_dir=.local-data/cifar10`。
YAML/CLIのどちらにも `max_epochs` がなければ、merge後の `comm_round * global_epochs_per_round` から自動算出する。
提供presetでは省略しているため、roundやlocal epochの変更時に二重編集は不要。
`max_epochs` を明示した場合はその値を保持し、積との不一致をエラーにする。黙って上書きはしない。
これは上流がround数をepochsから算出するための整合性確認で、default→YAML→CLIの優先順位は維持する。
runner部分は公式コードへ渡す前に除去する。legacy `main_args.conf` は一切sourceしない。

- 未知キー、型違い、負数・非有限の明示数値、非正の件数/epoch、参加数>総数、クラス数≠出力数は早期エラー。
- この入口はstandalone FedAvg + HPFLを対象とし、pretrained / load_backbone_from / resumeは拒否する。
- syntheticはCIFAR10形状のResNet18、全client参加、noise無効、freeze_layer=1。
  client 2〜8、class 2〜10、batch 2〜64、round/local/head epoch各1〜5。
  各train/test件数はクラス数・batch以上、128以下、batchの倍数（端数1件のBatchNorm/MMDを避ける）。
  dataset名がcifar10でも、公式データ入口を合成TensorDatasetに置換し、実データを取得しない。
- realの合成件数指定は無視せず拒否する。W&B有効化・オンライン設定・認証用キーも拒否する。
  disabled runは上流の無条件 `wandb.run.summary` 書き込みのためだけに初期化する。

## 実データpresetの条件

10/100 clients、毎round5参加、CIFAR10/`resnet18_v2`、class/output=10、hetero Dirichlet alpha=0.1。
`comm_round=1000`（実効 `max_epochs=1000`）、共有local epoch=1、`HPFL_local_iteration=1`（明示的に10などへ変更可能）、
batch128、SGD lr0.1 / momentum0.0 / wd0.0001、sched=`no`、HPFL有効、freeze_layer=1。
`noise_on_feat=True`、`noise_on_test_feat=False`、`noisy_coefficient=1`、`noise_type=stat`、`group_feat=False` を明示した。
**test noise無効は上流defaultに合わせたpresetの選択であり、論文の完全再現を保証しない。**
その他のHPFL設定は既存defaultを継承し、実効設定に記録する。

## データ・出力・障害時の扱い

- データはignore対象 `.local-data/` 以下に限定し、`data_dir` は **repo基準でchdir前に絶対化**する。
  出力と分離し、trackedデータ資産には保存しない。
- shellは毎回 `.local-experiments/launch-日時-乱数/` を確保する。
  `command.log`、Nixを含む `launcher.log`、`pipeline-status`（Nix/Pythonとtee）、`exit-status` を保存する。
  Nix起動前には未検証引数の値を記録しない。Nix自体が起動できない場合もここに失敗が残る。
- Pythonはさらに新規 `experiment/` を排他的に作成する。
  `original.yaml`、`effective.yaml`（全HPFL+runner）、`execution.json`（device・上限秒）、
  `command.json`（検証済みの明示引数のみ、環境変数なし）、`console.log`、`exit-status` を保存する。
  直接Python起動の場合は `.local-experiments/run-乱数/` に同じ内容を保存する。
- 公式コードのcwdは新規 `artifacts/`。相対checkpoint/image出力はこの中、絶対パスや `..` は拒否する。
  realの `hpfl.yaml`、合成の `summary.json`、公式checkpoint・`result/` もここに残る。
  過去checkpointの自動検出を避け、既存出力は再利用・上書きしない。
- timeoutは終了値124、その他も実終了値を保存する。Nix/Pythonまたはtee/ログ保存の失敗を成功扱いしない。
  保存先自体が書けない場合は記録を保証できないが、非ゼロ終了する。
  `.local-smoke/` の過去ログは変更しない。[旧smoke結果](local-smoke.md)は履歴として保持する。

## メモリ懸念と起動方式の修正

一時停止時の報告値はWindows空き約1GiB、vmmemWSL約23GB、WSL cache約20GiB。
独立レビューで、旧path flakeのstore sourceにignore対象のlaunch出力6件が含まれていた。
これをコピー増大・キャッシュ圧迫・入力ロック待ちの原因候補とし、Git flake方式へ変更した。
Hermesによるsudo drop_cachesの試行は権限不足で失敗。その後Windows空き20.8GiB、vmmem1.76GiB、
WSL cache449MiBへの回復が確認されたが、直接の解放原因は未確認であり、修正による効果とは断定しない。
以後のNix検証は逐次実行する。グローバルメモリ設定・Nix profile・WSL永続設定は変更していない。

## Git flake切替後の確認結果

ログ一式: `.local-experiments/finish-f20WVrUq/`（RED/GREEN、各検証ログ・終了値、`results.json`）。
CPU/GPU環境の `tests/` は各 **68 passed**。CPU/GPU build、flake check、bash構文、pycompile、diff検査は成功。
合成CPUは2.627秒・peak RSS 956632KiB、RTX 5060は4.314秒・2303240KiB、
VRAM peak allocated/reservedは221455360/251658240 bytes。いずれもモデル・入力・target配置とMMDを確認した。
`comm_round 2` だけの変更でも成功し、各clientの共有forwardが2→4、headは2のままになった。
成果物は順に `.local-experiments/launch-20261020T163710Z-ExjChwsd/`、
`launch-20261020T163725Z-ZoGDWJAc/`、`launch-20261020T163740Z-y7tqGOx5/` の `experiment/artifacts/`。
実データ10/100-client presetはcheckのみ成功。実データ取得・学習は実行していない。
