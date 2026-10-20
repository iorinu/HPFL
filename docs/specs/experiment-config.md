# 実験 YAML・共通 runner の承認仕様

対象はこの HPFL fork のみ。科学的アルゴリズム・集約・MMD 数式は変更しない。

| ID | 要件・受入条件 | 検証 |
| --- | --- | --- |
| EC-01 | 既存 YACS default → 実験 YAML → 明示 KEY VALUE。max_epochs省略時はmerge後に算出、明示時は整合性検証。未知キー・型違い・不正値を拒否 | loader・自動算出・不正設定・null回帰テスト |
| EC-02 | 合成専用入口、件数・クラス・client・batch・round・epoch が実際に変化 | 合成データテスト、変更条件の合成実行 |
| EC-03 | CPU 既定、CUDA 要求時の CPU 切替なし。モデル・入力・target・MMD・メモリの証拠 | 既存deviceテスト、CPU/GPU smoke |
| EC-04 | 実データ10/100-client preset、公式 main を薄い wrapper から呼ぶ。legacy conf は読まない | preset検証、公式mainのモックテスト |
| EC-05 | check/dry-run は検証・表示のみ。学習・取得・run作成なし | 副作用禁止テスト、real presetのcheck |
| EC-06 | 固有出力、原本・実効設定・明示コマンド・console・終了値。過去checkpointを使わない | 出力排他・失敗ログテスト、smoke成果物 |
| EC-07 | Nix/Python失敗とtee/ログ保存失敗を非ゼロで伝播 | shell・Pythonログ保存障害テスト |
| EC-08 | W&B disabledのみ、認証設定拒否、データと出力の分離 | 不正設定・危険なパス・無視される設定の拒否テスト |
| EC-09 | NixはGit flake（`.` / `.#gpu`）で逐次起動し、ライブPython/YAMLを使う。出力を含むpath flakeコピーを避ける | shell選択の回帰テスト |

各縦切りは RED を実行・記録してから最小実装を GREEN にする。
実データ学習・ダウンロードは今回実行しない。resume・継承・grid・scheduler基盤・並列実行・APIは対象外。
Nixは既存の依存環境のみ。global環境変更・新規主要依存・commit/pushは行わない。
