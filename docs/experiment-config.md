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
  execution_backend: sequential  # sequential（既定）または process
  max_concurrent_clients: 1      # sequential は1、process は1または2
  client_timeout_seconds: 120.0  # worker 起動・各 task の上限秒（有限の正数）
```

YACS defaultをcloneし、runnerの安全基準を加え、YAML、明示CLIの順でmerge/type checkする。
安全基準は `wandb_record=False`、`wandb_key=None`、`gpu_index=0`、
`data_dir=.local-data/cifar10`。
YAML/CLIのどちらにも `max_epochs` がなければ、merge後の `comm_round * global_epochs_per_round` から自動算出する。
提供presetでは省略しているため、roundやlocal epochの変更時に二重編集は不要。
`max_epochs` を明示した場合はその値を保持し、積との不一致をエラーにする。黙って上書きはしない。
これは上流がround数をepochsから算出するための整合性確認で、default→YAML→CLIの優先順位は維持する。
runner部分は公式コードへ渡す前に除去する。ただし実行設定は先に平坦な内部キー
`client_execution_backend` / `max_concurrent_clients` / `client_timeout_seconds` / `client_rng_policy`
へ明示転送する。これらの内部キーを共通runnerのYAML/CLIから直接指定することは禁止する。
realでもrunnerの実行設定は利用できる（合成件数だけがsynthetic専用）。legacy `main_args.conf` は一切sourceしない。

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

## 現在の仕様と過去記録の区別

この節と後続のissue1検証記録は当時の履歴である。**現在の保存・常駐仕様は次節が優先**し、
当時のCPU往復、worker数だけのCUDAモデル保持、暗黙 `.pth`/contract出力は廃止した。

<a id="issue2-model-lifecycle"></a>

## issue #2: 現在の常駐・保存契約

対象は共通runnerの通常FedAvg+HPFL（sequential/process）と旧standalone mainの同経路。
データ、sampling、batch、epoch、LR、metric、CPU MMDは変更しない。バックエンド既定はsequential/1のまま。
v1ではCFL/OOD/plug削減/finetune/既存backbone読込の特殊推論・復旧経路はデータ取得前に拒否する。

### 設定・所有権・容量

平坦なHPFLキーとして、実効YAMLとexecution/summaryに次を記録する。

```yaml
model_residency: device  # device（既定）または明示的な cpu
capacity_policy: error   # error のみ。自動CPU/disk spillなし
debug_tensor_dumps: false
checkpoint_save: false  # Trueは拒否。resumeは実装していない
```

- `device` は計算先と同じ常駐先（CUDAならVRAM、CPUならRAM）。sequentialはclientのmodel/BNとoptimizerを
  round境界でCPUへ戻さない。`get_model_params`等の観測も生モデルをCPUへ移動しない。
  `cpu` はclient model/optimizerをtask間RAMに保持し、計算時だけ指定deviceへ移す明示的な低VRAM選択。
  sequentialの返却集約snapshotもCPUに置き、live model退避後にCUDAストレージを保持しない。
  serverの計算モデルと最良共有snapshotは計算deviceに置く。
- processは `client_id % max_concurrent_clients` の固定所有権でspawn workerへ割り当てる。
  初回round前に**未選択clientを含む全client**を初期化し、model/optimizer/scheduler/timer/freeze/BN状態を
  各workerでhead遷移まで保持する。並行**計算**は最大1または2だが、常駐モデル数はworker数ではなく全client数。
  head終了後にworkerを回収する。失敗時もterminate/joinし、再試行しない。
- 親は初期CPU状態を一度だけ渡し、以後clientの構造/freezeフラグをmeta tensorで保持する。
  共有roundではoptimizer等のauthoritative状態を送受信せず、開始共有weightsと集約結果だけCPU IPC snapshotにする。
  当該roundの集約snapshotは集約後に解放する。CPU IPC snapshotはVRAMからの状態退避でもdisk transportでもない。
  最終head終了時だけ、既存の親所有・逐次評価に必要な最終clientモデルをCPUに復元する（optimizerミラーは不要）。
  親がsampling順の集約、server/最良共有snapshot、head/markerの対応、選択と評価を所有する。
- `task-sha256-v1` は維持する。旧実装がtaskごとにモデル初期化で消費したCPU RNGを保つため、
  各taskで捨てるCPU一時モデルを一個構築する。live trainer/optimizerは再構築しない。
  このCPU構築費用、初期転送、spawn/context、集約/評価IPC費用は残り、高速化の保証はない。
  生trainerやCUDA tensorをspawnでpickleしない。途中iteratorのresumeは引き続き非対応。
- VRAM予算にはserverと最良snapshot、**inactiveを含む全client**、momentum/optimizer、BN/freeze/scheduler状態、
  gradient/activation、集約用一時状態、head/評価モデル、CUDA allocator予約と各process/contextを含める。
  `cpu`でも計算とserverのVRAMは必要。host側も初期snapshot、最終評価ミラー、共有画像、markerとIPCを予算化する。
  全歴史roundのモデルを保持しない（小さなmetric/hashログは保持）。
- 信頼できる事前ピーク推定は未実装。実際のPyTorch OOMは明示的失敗となり、成功成果物を公開しない。
  合成2/3-client CPU検証は容量保証ではない。RTX 5060約8GiBにResNetの10/100-client presetが収まると仮定しない。
  実CUDAの小規模常駐確認は下記で成功したが、OOM限界・性能検証は未実施。MPS/WSL/GC/driver等は変更しない。

### 最終成果物 v1・再読込・失敗

通常のモデル成果物は一個の **`final-model.pt`**。合成は`artifacts/final-model.pt`、
実データと旧mainは排他的に新設した`hpfl-run-ランダム/final-model.pt`（共通runnerなら`artifacts/`配下）。
`summary.json`の`artifact.path`に実際の相対パス、SHA256、最終ファイルbytes/allocated bytes、最終モデルwrite数を記録する。

- `format=hpfl-inference`, `version=1`, `resume_supported=False`。
  `shared`は最良共有モデルのstate_dict（BNを含む）、`heads`は全個別headのstate_dict、
  `markers`は既存選択に必要なmean/covariance/feature tensor。
  `client_ids`と`client_to_head`はhead/marker slot対応、`config_yaml`は再構築条件、`metadata`は常駐/容量/RNG方針。
  認証設定を含めず、tensorと安全なprimitiveだけを保存する。optimizerや学習履歴は含めず、学習再開用ではない。
- `train()`は保存しない。共有学習、最良モデル選択、全head学習とmarker生成、personalized-data生成、
  選択と**呼び出し側の全最終評価**の成功後にのみ保存する。旧mainも同じ境界を使う。
  暗黙の既存`.pth`検出による学習skip/resumeは通常FedAvg+HPFLでは行わない。
- 同一run内の新規`.final-*.tmp`へ一度serialize/fsyncし、同一filesystemのatomic hard-linkで
  **既存宛先を置換せず**公開、temp名を除去する。renameの置換動作は使わない。
  serialize/link失敗、OOM、head/選択/評価失敗、timeoutのSIGTERM、KeyboardInterruptでは自分の公開ファイルを取り消す。
  成功summary/statusは公開後だけ。公開後のsummary/終了処理失敗も成功扱いせず取り消す。
  既存出力は再利用・移動・削除しない。保存先故障時の失敗ログ、SIGKILL/電源断後のcleanupやdurabilityは保証しない。
  確認できた例外では小さなfailed summary/statusを残す。外側launcherの非ゼロ終了値も確認する。

```python
from utils.final_artifact import load_final
inference = load_final(".../final-model.pt", device="cpu")  # weights_only=True
shared_logits = inference.logits(images)
scores, head_slot = inference.select(loader)  # 既存CPU MMD
personalized_logits = inference.logits(images, head_slot)
client_id = inference.client_ids[head_slot]
```

個別logits計算では一時的なモデル複製が一個必要。GPU推論でもその予算を含める。
test noise有効時の選択は従来同様にRNG依存で、実験のseed設定が必要。

### 通常の小出力と比較opt-in

通常はeffective config、command/status/log、metric/hash、最終一個だけ。
自動tensor履歴はfalse。合成比較に限り `debug_tensor_dumps True` でprocessの全round共有tensor履歴と
`process-contract.pt`/`legacy-contract.pt`を追加する。比較dumpも全評価後に作るが、**非常に大きい**RAM/diskを要し、
全client optimizer等も含む。通常検証には使わない。これらは推論/復旧形式ではなく、resume非対応。
旧issue1の比較テストはこのopt-inを明示するよう変更した。

`client-execution.jsonl`にはsequentialの`residency`、processの初期化所有権とclient task境界の
model/optimizer device、常駐client数、既存hash/RNG/forward数、peak RSS/VRAMと親観測を残す。
CPUへのhash観測コピーをoffloadとは扱わない。各worker peakの和は同時VRAM peakではない。

### CPU回帰・証拠

証拠は新設 `.local-experiments/issue2-iWM35e0C/`。縦断動作ごとの`01`〜`13`のRED/GREENログを保持する。
早期head失敗で旧 `.pth` 保存が実際にREDになることを確認し、保存境界、atomic公開、再読込logits/MMD、
固定routing/subset、inactive client、永続状態と旧再構築のRNG、summary/終了失敗まで順に実装した。
process/1対2、明示CPU保持対device保持は共有2round＋head、momentum0.9/StepLRの状態を比較する。
実データ入口はdownloadせず合成loader置換でライフサイクルを検証する。過去のverified結果は変更しない。
最終CPU suiteの件数はHermesの後続確認に委ね、ここでは確定値を記載しない。

### Hermesによるissue2実CUDA検証（2026-10-21）

根拠は `.local-experiments/hermes-issue2-cuda-8t5ix6ww/runs.json` と `comparison.json`。
上記CPU回帰とは別に、単一RTX 5060で合成データ・共有2rounds＋head、momentum=0.9、StepLRを実行した。

- sequential/device・process/1/device・process/2/device・process/2/明示cpu（各2clients）、
  process/2/device（3clients）の計5runが成功。既定のモデル出力は各run `final-model.pt` 一個だけ。
  2client成果物は約46,938,615 bytes（sequentialは46,938,679）、allocated bytesは46,940,160。
- process/1対2、process/device対cpuは各132tensorの最大差0.0。historical baselineとの差は
  sequentialで1.4551915228366852e-11、processで1.2732925824820995e-11で、rtol=1e-6 / atol=1e-7内。
  対応するMMD値・選択も一致した。sequentialとprocessを同一RNG条件とは扱わない。
- device保持workerの境界・保持model/optimizerはCUDA（初回optimizerは空）、明示cpu保持ではCPUを記録。
  client所有PIDは安定し、3clients/2workersでも全clientの常駐を確認した。
- driver全体のサンプル最大はprocess/1が1912MiB、process/2が2233MiB、3clientsが2367MiB。
  他contextを含み、真のpeakや安全な容量上限を示さない。確認の一部でCPUテストが並行していたため、
  throughput・高速化は主張しない。10/100clientsのGPU容量、実データ精度、kernel同時実行は未検証。
- 既存モデル73ファイルのsize/mtime/inodeは保存された。旧issue1の検証結果は履歴として維持する。

上記5runにはsequential/明示cpuは含まれない。その返却集約snapshotのCUDA保持不具合は、
`tests/test_model_lifecycle.py::test_sequential_cuda_aggregation_snapshot_residency` の実CUDA回帰で
device/cpu両保持を確認する（CPUのみの環境ではskip）。モデル・optimizerと集約器に渡るsnapshotを直接検査し、
科学条件・RNGやdevice保持のlive stateは変更しない。

## 履歴: メモリ懸念と起動方式の修正

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

## opt-in の bounded process client 学習

**以下はissue1時点の説明と検証記録。保存・CPU往復・常駐モデル数は上記issue2仕様で置換済み。**

既定は従来の **sequential / 1 / historical-global** のまま。GPU選択と並行数指定は独立している。
単一 `cuda:0` またはCPU上で、spawnした永続workerを最大2個使い、共有roundと最終headを実行する。

```bash
# 合成画像2clients・共有2rounds＋head。CUDA実機での確認用（長時間実験ではない）。
bash tools/run-experiment.sh --config configs/experiments/smoke.yaml --device cuda runner.execution_backend process runner.max_concurrent_clients 2 comm_round 2 momentum 0.9
# 同条件の isolated serial 参照。worker数だけを1に変える。
bash tools/run-experiment.sh --config configs/experiments/smoke.yaml --device cuda runner.execution_backend process runner.max_concurrent_clients 1 comm_round 2 momentum 0.9
# CPUで同じ契約を確認するには --device cpu に変更する。
# 実データpresetは検証のみ。合成件数指定は不要・禁止。
bash tools/run-experiment.sh --config configs/experiments/cifar10-10clients.yaml --check runner.execution_backend process runner.max_concurrent_clients 2

# Nix操作は逐次実行する。CPU属性は default（cpuという属性はない）。
nix develop . --command python -m pytest -q tests/
nix develop . --command python -m compileall -q algorithms_standalone configs tools trainers tests
nix develop .#gpu --command python -m pytest -q tests/
nix build .#default .#gpu --no-link
nix flake check .
```

### 同期・状態・乱数

- sampling、round開始weights、集約、最良共有モデルの選択、評価、checkpoint、telemetryは親が所有する。
  全task成功のバリア後に元のsampling順で結果を取り込み、従来どおり件数加重集約する。
  head phaseは共有集約せず、clientに対応するheadとmarkerを親へ戻す。
- clientのmodel（BatchNorm bufferを含む）、optimizer/momentum、scheduler（optimizer参照を除く値）、
  timer、requires_grad/trainフラグをCPU状態として往復する。親のclientモデルも更新する。
  最良共有モデルへ戻すときもoptimizerをリセットせず、既存のhead学習とtimerの進み方を維持する。
- worker内でmodel/trainer/loaderを生成する。manager、生trainer、CUDA tensor、DataLoader/iteratorは送らない。
  親で確定したCIFARのfull画像・ラベルをCPU共有tensor化し、canonical partition索引を参照する。
  workerで再分割/downloadせず、clientごとにfull datasetを複製しない。
  workerのPyTorch intra/inter-op threadは各1、DataLoader workerは0。
  whole epochだけを支持し、task境界では使い切ったiteratorを再生成する（途中cursorのresumeは非対応）。
- processの乱数ポリシーは **`task-sha256-v1`**。
  `task-sha256-v1:{seed}:{client_id}:{round_index}:{shared|head}` のUTF-8 SHA256先頭4bytesを
  big-endian整数にし、taskごとにPython/NumPy/PyTorch/CUDAをseedする。
  初期化・shuffle・augmentation・marker noiseもそのworkerの隔離されたストリームを使う。
  worker割当・完了順では変わらない。CUDAはdeterministic algorithmsを要求し、非対応演算は失敗させる。
  **historicalなglobal RNGとは別条件**であり、legacyとのbitwise一致を主張しない。
  親のclient iteratorを起動しないため、初期共有モデルのRNG消費もlegacyとは異なり得る。
  比較の参照は必ずprocess/1とprocess/2で揃える。

### 対応範囲と失敗時

最初の対応範囲はstandalone FedAvg+HPFL、CIFAR10/`resnet18_v2`、通常CrossEntropy trainer、
全clientのCPUミラー保持（instantiate_all）、whole epoch、標準Random/SequentialSampler、freeze_layer=1。
SGDの非ゼロmomentum、no/StepLR/MultiStepLR/CosineAnnealingLR、通常augmentation、既存feature noise/group化は保持する。
batch、epoch、LR、参加client、partition、head選択規則を並行化のために変更しない。

Scaffold/control variate、FedProx、VHL、FedAlign、OOD/CFL、finetune、中央事前学習、
compression/diff、途中step方式、特殊loss/optimizer（SAM等）、独自sampler/collate/loader RNG、
特殊model/transform、pruning、モデル追跡/dataframe、plug削減、評価間隔の変更（frequency_of_the_test≠1）、resume、複数GPUは未対応。
該当する設定は黙って無視せず拒否する。設定上検出できるものはデータ取得・出力作成前に検証する。
独自dataset/loaderの実オブジェクトはworker起動前のexport時に検証する。

例外、OOM、異常終了、timeout時は全workerをterminate/joinし、未完roundの取り込み・集約・成功checkpointを行わない。
自動retry/CPU fallbackはない。過去の完了round成果物は保持する。runは非ゼロ終了し、失敗roundをJSONLに記録する。
SIGTERMでも親はcleanupする（SIGKILLやOS自体の停止ではPython cleanupを保証できない）。W&Bは常にdisabled。
worker timeoutは起動と各taskに適用し、初回roundにはspawn/import/IPC費用も含む。

### 観測と性能の限界

`artifacts/client-execution.jsonl` は親だけが書く。

- `client`: client/round/phase、PID、device・target配置、forward/loss数、seed/policy、開始weightsと
  初回forward直前weightsのSHA256、monotonic開始/終了、構築・学習（data/transferを含む）・marker時間、
  worker lifetimeのpeak RSS、CUDA allocated/reserved peak（task開始時reset）を記録する。
  worker時間は結果のCPUシリアライズ・返送を含まないので、roundバリア時間も併読する。
- `round`: 親の開始/バリア完了、sampling順、バリアの成功/失敗。`parent_phases`: 取り込み、集約、round評価、checkpoint時間。
- `resources`: 約250ms間隔の親＋workerの現在RSS合計（共有ページの重複を含むため物理使用量ではない）。
  CUDA時は取得可能なら `nvidia-smi` のGPU全体utilization（%）/memory.used（MiB）も残す。
  driver値は他プロセス/contextを含み、短いtaskやkernel重複を保証しない。
- 合成の `summary.json` にはstartupを含む全体時間、MMD score/選択、追加評価・選択時間を記録する。
  `process-contract.pt` / `legacy-contract.pt` は合成確認専用のCPU状態と比較用履歴（resume形式ではない）。
  `effective.yaml` / `execution.json` にbackend/上限/RNG policyが残る。

CUDAにresidentな**学習clientモデル数**はworker数以下。親のserver評価モデル/contextと各workerのCUDA allocator予約は別に存在する。
親のinactive clientはCPUに保持し、親でのclient評価後もCPUへ戻す。
CPU側の既存clientモデル/optimizer、共有データ領域、IPCコピー、spawn/importの費用は残るため、
総client数増大時のhost RAM上限を解消するものではない。全workerのpeakを足して同時VRAM peakとは呼ばない。
複数CUDA processはtime-sliceやcontext重複により遅くなり得る。MPS/AMP/GC/システム設定の変更は行わない。
task区間の重なりはCUDA kernel同時実行の証拠ではない。GPU trace・実データ性能・安全なVRAM上限は別途実機検証が必要。
tiny syntheticの時間はspeedupの根拠にしない。

CPU自動テストは2clients/共有2rounds＋head、momentum=0.9/StepLRをprocess/1とprocess/2で比較し、
model/buffer/optimizer/scheduler/timer/head/marker/履歴/MMDを検証する（float32: rtol=1e-6, atol=1e-7、ID/件数/構造/seedは一致）。
canonical索引＋ランダムaugmentation＋marker noise、逆順sampling、worker異常終了/OOM相当例外/timeout、
partial aggregation禁止、cleanup、特殊設定の拒否もカバーする。legacyとの差は別に記録する。

### 今回のCPU検証結果（2026-10-21）

ログ一式は `.local-experiments/issue1-20261021-opencode/`、要約は `results.json`。
変更前68件、変更後はCPU環境・GPU依存環境とも **119 passed**（いずれも学習deviceはCPU）。
CPU/GPU環境のcompile、`nix build .#default .#gpu --no-link`、`nix flake check .` も成功。
上流のNumPy private namespace使用に関する既存DeprecationWarningが1件ある。

2clients/共有2rounds＋head、momentum=0.9/StepLRのprocess/1対2は、比較tensor最大差0.0、MMD score/選択も一致。
legacyとの共有モデル最大差は0.3709418で、同一のhistorical条件とは扱わない。
CPU process/2のtask区間重複は各round約0.795/0.693/0.377秒。
観測したRSS合計最大はprocess/1で約2.45GB、process/2で約3.25GB（共有ページ重複込み）。
単発の全体時間はprocess/1が13.27秒、process/2が10.89秒、legacyが3.70秒であり、
spawn/IPC/検査・状態出力の費用が大きい小規模合成確認である。warm-up/反復性能比較ではなく、速度向上の根拠にはしない。
このCPU検証時点ではCUDA学習実機・GPU kernel重複・VRAM実測・実データ性能は未検証だった。後続のGPU確認は次節に記録する。

<a id="gpu-verification"></a>

### HermesによるGPU実機検証（2026-10-21）

記録更新日時: 2026-10-21 12:11 JST。根拠は `.local-experiments/hermes-issue1-verify-3n83k6f_/` の
`runs.json`、`cuda-state-comparison.json`、`gpu-observations.json`（既存ファイルは変更していない）。

- 単一RTX 5060、合成2clients、train 16件/client、batch 8、共有2rounds＋head 1epoch、momentum=0.9、StepLR。
  legacy 1run、process/1・process/2各2runの全5runが成功。全clientの共有・head phaseで入力・parameter・targetの`cuda:0`配置hookが成功した。
- process全4runのCPU保存状態（共有/clientモデル、optimizer、BN、scheduler、timer、freeze状態、head/marker、履歴）を
  計2598tensor比較し、rtol=1e-6 / atol=1e-7で実測最大差 **0.0**。MMD値・選択も一致した。
  process/2は両runとも全3roundでtask区間が重複したが、**CUDA kernel同時実行traceは未実施**。

| 実行方式 | harness時間（秒） | 平均（秒） |
| --- | --- | --- |
| process/1 | 12.9308 / 12.8106 | 12.8707 |
| process/2 | 11.2535 / 17.1296 | 14.1916 |
| legacy | 4.2062 | — |

安定した高速化は確認できなかった。小さいworkloadではimport/IPC等の費用が目立つ。
表はNix起動等を含む外側wall時間ではなくharness時間であり、legacyは別RNG条件なので精度同値参照ではない。

worker allocator peakは各workerでallocated約238.35MiB / reserved 270MiB。
driver全体のサンプル最大VRAMはprocess/1が1756MiB、process/2が2227MiBで、他contextを含み、真のpeakを保証せず、worker peakの合計でもない。
外部`nvidia-smi`サンプルの最大utilizationはprocess/1が32% / 16%、process/2が38% / 34%で、kernel重複の証拠にはならない。

Hermes報告ではCPU suite再実行も119 passed・既存`numpy.core` deprecation warning 1件、独立read-only reviewでも重大なlogic/security不備なし。
実データ性能・安全なVRAM上限・論文全体の再現は未検証。MMDはCPU、既定はsequentialのまま維持する。
