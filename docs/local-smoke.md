# CPU ローカル synthetic smoke

## 目的と確認範囲

上流 SHA: `c21ebf2a7b03a03262afae218a98a427fb89fa9a`。
2026-10-20 に WSL 上で **CPU の最小動作確認に成功**した。
合成画像による実行確認であり、論文再現・CIFAR の実データ評価・精度比較ではない。

- 公式 `FedAVGManager` → `FedAVGClient` → `NormalTrainer` → `FedAVGAggregator` を使用。
  `HPFL=True` が上流の HPFL 経路。FedRep は別の比較手法であり今回使用しない。
- データ入口 `basePSmanager.load_data` だけを固定 seed の TensorDataset に置換。
- `resnet18_v2`、32×32 RGB、2 クライアント、各 train 16 / test 8、batch 8。
- 共有 1 round（各 1 epoch）、個別ヘッド 1 epoch、freeze_layer=1。
- 公式 `cal_dist` の marker 2 個、local 推論、公式 personalized index / MMD 選択推論。
  `utils/mmd_loss.py` を含め、MMD 数式・集約・学習アルゴリズムは変更していない。
- 上流の共有モデル `.pth` 保存と disabled W&B run の `finish()` まで完了。
  通常の checkpoint オプションは無効。ヘッドや marker の resume は実装していない。
- クラス数を 2 に縮小。実データの subset ではなく、全件が意図的な合成画像。
  `noise_on_feat=False` / `noise_on_test_feat=False`、seed=7、CPU threads=2。

## 固定環境

`flake.lock` は nixpkgs 25.05 の
`ac62194c3917d5f474c1a844b6fd6da2db95077d` に固定。
`config.cudaSupport=false` の `python312.withPackages` を使用する。
CPU の `torch` / `torchvision` は cache から取得できた。
uv、venv、conda、pip による環境管理は使用していない。

| パッケージ | 実行時バージョン |
| --- | --- |
| Nix | 2.35.2 |
| Python | 3.12.12 |
| torch | 2.7.0 |
| torchvision | 0.22.0 |
| NumPy | 2.2.5 |
| SciPy | 1.15.3 |
| scikit-learn | 1.6.1 |
| W&B | 0.19.10 |
| pytest | 8.3.5 |

W&B は `WANDB_MODE=disabled`、`WANDB_DISABLED=true`、
`wandb_record=False`、`wandb.init(mode="disabled")`。
上流が無条件に `wandb.run.summary` に書くため disabled run が必要。
実験データの外部送信・login は行わない。
smoke 内の HOME とキャッシュ先も固有出力ディレクトリに設定する。

## 実行コマンド

このリポジトリのルートで実行する。未追跡 flake を使うため `path:.` が必要。
Git staging は不要。

```bash
nix flake check path:. --no-build
bash -n tools/run-local-smoke.sh
nix develop path:. --command python -m py_compile tools/local_smoke.py tests/test_local_compat.py recorders/model_dif_track.py trainers/normal_trainer.py
nix develop path:. --command env MPLCONFIGDIR="$PWD/.local-smoke/pytest-mpl-20261020" python -m pytest -q tests/test_local_compat.py
git diff --check
bash tools/run-local-smoke.sh
```

runner は `mktemp -d` で毎回別の `.local-smoke/run-日時-XXXXXX/` を作り、
`command.log`、`console.log`、`exit-status` を保存する。
内部コマンドは次の形式（学習側に 180 秒の上限）。

```bash
nix develop path:. --command timeout 180s python tools/local_smoke.py --output "$run/artifacts"
```

出力先が既にある場合 Python は失敗するため、以前の実行を流用しない。
作業ディレクトリを固有 artifacts に変えてから上流処理を呼ぶことで、
上流が自動で探す checkpoint も以前の実行から読み込まれない。

## 実測結果と終了値

成功実行: `.local-smoke/run-20261020T075503Z-a2MAx4/`

- `bash tools/run-local-smoke.sh`: **exit 0**。
- `artifacts/summary.json`: `status=passed`、marker 2 個、round counter=2
  （共有 round 1 回＋個別化フェーズ 1 回を assert）。
- Python 内計測（import 後、manager 構築から推論完了）: **2.5514 秒**。
- `resource.getrusage(RUSAGE_SELF).ru_maxrss`: **967656 KiB（約945 MiB）**。
  Python プロセスの peak RSS であり、Nix 取得処理やシステム全体の使用量ではない。
- **VRAM は未使用（CPU 実行）**。
- local/index/personalized MMD 推論の表示値はいずれも 50%。
  合成 2 クラスの実行ログに過ぎず、モデル品質の根拠にはならない。
- local test loader に対する追加の公式 MMD 呼び出しは、有限スコア
  `[0.49182105, 0.72340560]` → head 0、
  `[0.68965244, 0.53496337]` → head 1。
  公式 `generate_personalized_data()` が再配分した test batch では head 1 / 0 を選択。
  この二つの入力集合は異なるため、選択結果も分けて記録している。
- checkpoint:
  `artifacts/2client_FedAvg_resnet18_v2_cifar10_1_round_alpha=0.1.pth`。
  `artifacts/result/` に公式選択スコア `.npy` も保存。

構文・flake 評価・pytest・diff 検証は上記の順で全て **exit 0**。
pytest は **2 passed**。記録は `.local-smoke/checks-20261020-5oaoot.log`。

### 失敗履歴と上流互換性

| 実行/確認 | 実終了値 | 原因・対応 |
| --- | --- | --- |
| CPU import 確認 | 1 | NumPy 2 で `numpy.lib.function_base` が存在しない。未使用の `kaiser` import を削除 |
| `.local-smoke/cpu-20261020-attempt01` | 1 | `h5py` 不足。Nix closure に追加（この初回の標準出力は会話内のみ） |
| `run-20261020T075353Z-HsgBtc` | 1 | `tqdm` 不足。Nix closure に追加 |
| `run-20261020T075414Z-r5iCfm` | 1 | harness の `server_index` 設定不足。公式 main と同じ -1 を設定 |
| `run-20261020T075437Z-GNG57h` | 1 | 現行 DataLoader iterator に `.next()` がない。標準の `next(iterator)` に変更 |
| `run-20261020T075503Z-a2MAx4` | 0 | end-to-end 完了 |

互換修正は `recorders/model_dif_track.py` の未使用 import 削除と
`trainers/normal_trainer.py` の 2 箇所の iterator 呼び出しだけ。
回帰テストで NumPy 2 import、および batch 取得・末尾での再開を確認する。

上流 `PSTrainer.update_dataset` は通常の SequentialSampler に `set_epoch` を呼び、
例外を捕捉して traceback を出力する。今回もそのログが出るが、学習・集約は継続し完了した。
この挙動や未使用コードの SyntaxWarning は未修正。

### GPU の状態

最初に torch-bin / torchvision-bin + CUDA 12.8 closure を試したが、
Nix 起動が 120 秒・300 秒の上限に達し、その後の試行も中断した。
後続の NCCL ソースビルドは、Hermes が `ps` で多数の `nvcc` / `cicc` / `ptxas`
プロセスを直接確認して検出し、関連プロセスの停止も Hermes が実行・確認した。
GPU 起動の正常終了値は得られていない。
ユーザーの依頼は最小動作確認と DRAM 使用への懸念であり、Hermes の軌道修正で
GPU 環境構築を打ち切り、CPU cache 環境へ切り替えた。
**RTX 5060 / CUDA 実行は未検証**。現在の flake は CPU 専用で WSL の driver パスは不要。

global 設定・system profile・Home Manager・認証情報の変更、commit/push は行っていない。
`.local-smoke/` 全体は Git ignore 対象。今回作成したソースと flake は未ステージのまま。

## レビュー追補（2026-10-20）

`tools/local_smoke.py` で manager 構築直前に `random.seed(args.seed)`、
`np.random.seed(args.seed)`、`torch.manual_seed(args.seed)` を明示した。
合成画像専用 generator に加え、モデル初期化・personalized 再配分に使う RNG も
記録した seed=7 に合わせるため。上流 `utils.set_seed.set_seed` は CUDA API も呼ぶので、
harness の追加初期化は CPU 向けの呼び出しだけにした。
なお上流 manager 自体にも `set_seed` 呼び出しは既にあり、旧版で全経路が
未 seed だったという意味ではない。今回の変更は harness 自身の初期化を明示するもの。
単純な seed 呼び出しをなぞるだけのテストは追加せず、既存の狭い回帰テスト 2 件と
CPU end-to-end smoke を再実行した。

以前の結果・成果物は保持している。Hermes の独立実行
`.local-smoke/run-20261020T075744Z-zLMuIq/artifacts/summary.json` も
**旧 seed 初期化版**の成功結果として保持：exit 0（Hermes 報告）、
2.6482 秒、peak RSS 943696 KiB。summary の `status=passed` と数値も確認した。

### seed 明示後の再実行

- コマンド：`bash tools/run-local-smoke.sh` → **exit 0**。
- 新しい出力：`.local-smoke/run-20261020T075923Z-p4NLOz/`
  （`console.log` / `command.log` / `exit-status` / `artifacts/summary.json`）。
- 共有 1 round、ヘッド 1 epoch、marker 2 個、local/index/MMD 推論、checkpoint 保存まで完了。
- Python 内計測 **2.4550 秒**、peak RSS **950568 KiB**。VRAM 未使用。
- local-loader MMD スコア・選択と personalized 選択は前回と同じ。
  実行時バージョンも上記表から変更なし。
- 以下も各 **exit 0**、pytest は **2 passed in 4.40s**。
  ログ：`.local-smoke/review-checks-20261020-4CGE5i.log`。

```bash
nix develop path:. --command python -m py_compile tools/local_smoke.py
nix develop path:. --command env MPLCONFIGDIR="$PWD/.local-smoke/pytest-mpl-20261020" python -m pytest -q tests/test_local_compat.py
git diff --check
```

NCCL 検出・停止の帰属を Hermes に訂正した。
`recorders/model_dif_track.py` と `trainers/normal_trainer.py` の末尾空行数も上流へ戻し、
これらの差分は互換修正だけにした。GPU 環境構築は再開していない。

## GPU 最小 smoke の追加（2026-10-20、上記実行とは別）

その後の明示的な GPU 確認依頼に対応した追補。上記の「未検証」「再開していない」は
当時の状態として保持する。論文再現やデータ取得は行わず、同じ合成データと設定を使う。
編集前の task ファイルは `.local-smoke/gpu-backup-20261020T151401Z-icAonA/` に保存した。
既存の CPU 成果物や Hermes の独立実行は変更していない。

### GPU 環境とビルド前の確認

- `flake.lock` と CPU default の Python closure は変更なし。
  `devShells.x86_64-linux.gpu` / `packages.x86_64-linux.gpu` を追加した。
- GPU のみ `torch-bin` 2.7.0 と `torchvision-bin` 0.22.0 を使用。
  torchvision が参照する torch も同じ override に揃えている。
- ソース版 NCCL の代わりに公式 PyPI wheel `nvidia-nccl-cu12==2.26.2` の
  `libnccl.so.2` とヘッダーを `fetchurl` → unzip → copy → autoPatchelf で導入する。
  Python package manager は使用せず、NCCL のコンパイルも行わない。
- wheel URL は `flake.nix` に固定、SHA256 hex は
  `694cf3879a206553cc9d7dbda76b13efaf610fdb70a50cba303de1b0d1530ac6`。
- unfree 許可は GPU 用 nixpkgs import の CUDA/torch 関連名だけ。
  WSL の `LD_LIBRARY_PATH=/usr/lib/wsl/lib` は GPU shell だけに追加。
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` は torch import / CUDA context 作成前に設定。
- `--device cuda` は CUDA がなければ非ゼロ終了し、自動 CPU 切替はしない。
  モデル forward hook と criterion hook で入力・パラメータ・出力・target のデバイスを
  assert。各 client の共有学習2 batch、ヘッド学習2 batch を別々に確認する。
  MMD は公式処理のまま CPU 上で距離計算し、hook で実際の入力デバイスを記録する。
- CUDA タイミングは前後で synchronize。VRAM peak allocated/reserved は PyTorch allocator、
  RAM は Python プロセス peak RSS。driver/context や他プロセスの VRAM を含む総量ではない。

```bash
nix build path:.#gpu --dry-run --no-link
nix derivation show --recursive path:.#gpu
nix build path:.#devShells.x86_64-linux.gpu --dry-run --no-link
nix build path:.#gpu --no-link -L
bash tools/run-local-smoke.sh --device cuda
bash tools/run-local-smoke.sh
```

初回 dry-run は構文の dynamic attribute 重複で exit 1、属性をまとめて修正した後は exit 0。
成功した GPU package dry-run は未実現9 derivation（CUDA binary 展開4件、NCCL wheel取得と
展開、torch/torchvision wheel、Python環境）だけ。recursive derivation と CUDA の
`dontBuild=true` / wheel source を確認してから build を開始した。
graph 中に既に cache 済みの CPU torch 等のソース derivation が含まれることと、
今回それが実際にビルドされることは区別している。ソース版 NCCL は graph にない。
記録：`.local-smoke/gpu-build-20261020T151444Z-VpGxsd/` の `dry-run.log`、
`shell-dry-run.log`、`derivations.json`、`derivation-review.log`、`cuda-derivation-review.json`。

### CPU 維持・デバイス検証

- `bash tools/run-local-smoke.sh`：**exit 0**。
  `.local-smoke/run-20261020T152109Z-PmpWEC/`。
  共有・ヘッド・marker・index/MMD 推論・checkpoint まで成功。
  **2.6837 秒、peak RSS 938108 KiB、VRAM 未使用**。
  各 client は共有2 / ヘッド2 forward、モデル・入力・target は CPU、MMD 入力も CPU。
- CPU shell から `python tools/local_smoke.py --device cuda --output ...` を実行すると
  **exit 1**（意図した失敗）。CUDA 利用不可の明示エラーで学習は開始しない。
  `.local-smoke/no-cuda-fallback-20261020T152201Z-8f6OON/console.log` と `exit-status`。
- `bash -n`、`py_compile`、pytest、`git diff --check`：**全て exit 0**。
  既存2テスト＋無効デバイス3ケース＋CUDAなしの非fallback＋CUDA選択の計7件で
  **7 passed in 39.04s**。
  `.local-smoke/gpu-review-checks-20261020T151936Z-e6LeKO/checks.log`。
- `nix flake check path:. --no-build` は CPU/GPU shell と全 package の評価を完了し
  `all checks passed!`。その後の連結検証コマンド全体は120秒制限で中断されたため、
  上記の構文・pytest等は別実行で取り直した。
  `.local-smoke/gpu-review-checks-20261020T151705Z-9aNisX/checks.log`。

### GPU 実行結果

**GPU end-to-end は exit 0 で完了した**。
`bash tools/run-local-smoke.sh --device cuda` の出力先は
`.local-smoke/run-20261020T152511Z-rYL15G/`。
`command.log`、`console.log`、`exit-status`、`artifacts/summary.json` を保存している。
実際の内部コマンド：

```bash
nix develop path:.#gpu --command timeout 180s python tools/local_smoke.py --device cuda --output /home/iori/src/github.com/iorinu/HPFL/.local-smoke/run-20261020T152511Z-rYL15G/artifacts
```

| 項目 | 実測 |
| --- | --- |
| device | cuda:0 / NVIDIA GeForce RTX 5060 |
| capability | 12.0（sm120） |
| torch / torchvision | 2.7.0+cu128 / 0.22.0+cu128 |
| torch CUDA / NCCL | 12.8 / 2.26.2 |
| ホスト NVIDIA-SMI | 610.57.01、KMD 610.88、CUDA UMD 13.3 |
| GPU 容量 | 8151 MiB（表示使用分を含む総容量） |
| 同期して計測した Python 内処理時間 | 3.1578 秒 |
| Python process peak RSS | 2503124 KiB（約2.39 GiB） |
| peak allocated VRAM | 221455360 bytes（約211.20 MiB） |
| peak reserved VRAM | 251658240 bytes（240 MiB） |

計測区間は import と CUDA context 初期化の後、manager 構築から学習・推論完了まで。
RSS はプロセス全体の最高値。VRAM は PyTorch allocator の最高値であり、
driver / context / 他プロセスを含む `nvidia-smi` の総使用量とは異なる。
`nvidia-smi-before.log` では実行前から表示用等に1286 MiBが使われていた。

- 公式 manager / client / trainer / aggregator の共有1 round＋ヘッド1 epochが完了。
  client 0 / 1 とも各 forward 7 回、うち学習4回（共有2 / ヘッド2）。
  サーバーモデル forward は8回。入力・全パラメータ・教師ラベルは `cuda:0`、
  出力も hook 内で同じデバイスであることを assert した。
- marker 2個、local / index / personalized MMD 選択推論、共有モデル `.pth` 保存に成功。
  いずれの表示精度も50%で、合成2クラスの runtime 確認に過ぎない。
- MMDLoss の実呼び出し8回、入力は全て `cpu`。公式処理が特徴をCPUへ戻すためであり、
  科学的アルゴリズムを変更してGPU計算へ置き換えてはいない。
- local-loader の MMD スコアは `[0.49185133, 0.72304988]` → head 0、
  `[0.68843007, 0.53420639]` → head 1。公式 personalized batch では head 1 / 0。
  CPU値との小さな数値差は残っており、CPU/GPU間のビット一致は主張しない。
- checkpoint は同実行の
  `artifacts/2client_FedAvg_resnet18_v2_cifar10_1_round_alpha=0.1.pth`。
  resume、長時間学習、実データ評価、複数GPUや実際のNCCL collective通信は試していない。

### バイナリ検証と最終チェック

最初の `nix build path:.#gpu --no-link -L` はバイナリ展開・fixupが600秒制限内に
完了せず、harness が中断した（正常なプロセス終了値は取得できず、`build.exit-status` はない）。
`build.log` は保持。その後のGPU runnerのshell起動が残りを実現して終了値0となり、
後から実行した同じ `nix build path:.#gpu --no-link` も **exit 0** を確認した。
ソース版 NCCL / CUDA / PyTorch のコンパイルは起動していない。
追加の上流コード変更や学習チェック緩和も不要だった。

NCCL の ELF 確認を最初に試した時点では未実現だったためファイル不存在で exit 1
（`nccl-elf.log`）。構築後の再確認は **exit 0**：

- `.local-smoke/gpu-build-20261020T151444Z-VpGxsd/nccl-elf-final-01wdcl.log`：
  wheel の `sha256sum` が固定値と一致。`readelf -d` / `patchelf --print-rpath` / `ldd` で
  `libnccl.so.2` の DT_NEEDED と RUNPATH、全依存解決を確認。
  NEEDED は libc / pthread / rt / dl / libstdc++ / libm / libgcc_s / loader、
  RUNPATH は Nix の gcc runtime 2パス。
- 同ディレクトリの `gpu-runtime-links-HAHLHn.log`：
  `libtorch_cuda.so` の DT_NEEDED は `libnccl.so.2` を含み、RUNPATH に
  `/nix/store/r9kk5y97x6qm1hrsxc66aar69gdcamaf-nccl-binary-2.26.2/lib` がある。
  実際のロード先も `/proc/self/maps` で同じバイナリNCCLであることを確認。
  libcuda は `/usr/lib/wsl/lib/libcuda.so.1`（およびWSL driver実体）からロードされている。

最終検証も **全て exit 0**：

```bash
nix build path:.#gpu --no-link
nix flake check path:. --no-build
bash -n tools/run-local-smoke.sh
nix develop path:.#gpu --command python -m py_compile tools/local_smoke.py tests/test_local_compat.py
nix develop path:.#gpu --command env MPLCONFIGDIR="$PWD/.local-smoke/gpu-final-checks-20261020T153211Z-N1ulWU/mpl" python -m pytest -q tests/test_local_compat.py
git diff --check
```

GPU shell でも **7 passed in 6.57s**。
ログ：`.local-smoke/gpu-final-checks-20261020T153211Z-N1ulWU/checks.log` と `exit-status`。
CPU package build と flake check の独立確認も exit 0：
`.local-smoke/gpu-flake-check-20261020T152307Z-pxzui4/`。
lock・CPU default closureは維持、commit/staging/push、global設定変更、外部実験送信はなし。
