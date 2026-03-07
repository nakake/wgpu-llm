# タスク表

目標: GPT-2 Smallが動く → LLaMA系に拡張

## フェーズ1: GPU基盤 + 最初のカーネル群

| # | タスク | 状態 | 備考 |
|---|--------|------|------|
| 1 | GPU初期化ユーティリティ + テストヘルパー | 完了 | `init_gpu()` + `create_device()` |
| 2 | WGSLテンプレートエンジン | 完了 | `template::render()` — `{{PLACEHOLDER}}`置換 |
| 3 | LayerNormカーネル | 完了 | naive実装（1スレッド=1トークン）、CPU参照比較テスト済み |
| 4 | GEMMカーネル | 完了 | naive実装（1スレッド=1出力要素）、CPU参照比較テスト済み |
| 5 | Embedding lookup | 完了 | token_embed + position_embed、CPU参照比較テスト済み |

## フェーズ2: グラフ骨格（カーネルと並行）

| # | タスク | 状態 | 備考 |
|---|--------|------|------|
| 6 | DecoderConfig + GPT-2 config.json解析 | 完了 | 14フィールド + 4 enum、GPT-2 config.jsonパーステスト済み |
| 7 | グラフ構築基盤 | 未着手 | 名前付きスロット、バッファ割り当て |
| 8 | 実行エンジン | 未着手 | CommandEncoder、submit、logits読み出し |
| 9 | 縦スライス検証 | 未着手 | Embedding→LayerNorm→Linearでend-to-end確認 |

## フェーズ3: 残りのカーネル + Attention

| # | タスク | 状態 | 備考 |
|---|--------|------|------|
| 10 | Softmaxカーネル | 未着手 | |
| 11 | GELU活性化関数 | 未着手 | |
| 12 | Attention | 未着手 | GEMM + Softmax組み合わせ方式 |

## フェーズ4: GPT-2完成

| # | タスク | 状態 | 備考 |
|---|--------|------|------|
| 13 | WeightMapping（GPT-2） | 未着手 | Conv1D転置、fused QKV分割 |
| 14 | InferenceState / KVCacheManager | 未着手 | KVキャッシュの確保・追記・破棄 |
| 15 | PrefillGraph / DecodeGraph | 未着手 | 2グラフ方式 |
| 16 | PyTorch fixtureテスト | 未着手 | forward pass全体の数値検証 |

## フェーズ5: CLI

| # | タスク | 状態 | 備考 |
|---|--------|------|------|
| 17 | wgpu-llm-cli | 未着手 | トークナイザー、サンプリング、生成ループ |
