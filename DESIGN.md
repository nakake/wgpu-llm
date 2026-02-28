# wgpu-llm 設計書

## 概要

wgpuベースのLLM推論エンジン。HuggingFace config.jsonからGPUパイプラインを自動構築する。

### 目的

- 対応モデルを増やしたい（ブラウザの制約を超えるためネイティブ化）
- モデル定義からパイプライン構築を自動化する仕組みを作りたい
- 既存プロジェクト（WebGPU GPT-2推論エンジン）のWGSLカーネル資産を活用

### スコープ

decoder-only causal LMに特化する。GPT-2, LLaMA, Mistral, Qwen等が対象。
encoder-decoder、画像、音声モデルは対象外。

### 非目標

以下は明示的にスコープ外とする:

- バッチ推論（同時に複数プロンプトを処理）
- テンソル並列（複数GPU間での分散推論）
- 学習・ファインチューニング
- 動的バッチング
- 投機的デコーディング

### マイルストーン

GPT-2 Smallが動く → LLaMA系に拡張。

---

## リポジトリ構成

モノリポ。2 crateに分割する。

| crate | 責務 |
|---|---|
| wgpu-llm | 推論エンジン本体。カーネル・パイプライン構築・実行。トークンIDを入力とし、logitsを出力する |
| wgpu-llm-cli | CLIツール（薄いラッパー）。トークナイザー・サンプリング・ユーザーインターフェース |

wgpu-llm内部はモジュールで論理分離する:

| モジュール | 責務 |
|---|---|
| kernels | GPUカーネル（GEMM, Softmax, LayerNorm, RMSNorm, Attention等） |
| graph | パイプライン構築・グラフ実行 |
| model | config解析・重みロード・DecoderConfig |

カーネルを独立crateにする必要が生じた場合は後から切り出す。

### 層の責務境界

- wgpu-llm: モデル定義→グラフ構築→カーネル実行。入力はトークンID列、出力はlogits（f32配列）。サンプリングは含まない。
- wgpu-llm-cli: トークナイザー（`tokenizers` crate）、サンプリング（temperature, top-k, top-p）、ユーザーインターフェース。

---

## データ型・精度戦略

### 計算精度

- カーネル内部の演算はf32で行う
- wgpuの `shader-f16` は普及率が不十分なため、当面はf32のみ

### 重みの格納精度

| フォーマット | 格納 | 計算時 |
|---|---|---|
| safetensors (f32) | f32のままGPUBufferへ | f32 |
| safetensors (f16) | f16でGPUBufferへ、カーネル内でf32に変換 | f32 |
| 量子化 (Q4等) | パック済みバイナリでGPUBuffer | カーネル内でdequantize→f32で演算 |

既存プロジェクトのgemm_q4カーネル（Q4重み + f32アキュムレータ）と同じ方式。

### カーネルインターフェースへの影響

各カーネルは入力バッファのデータ型を型パラメータまたは列挙で受け取る。
量子化GEMMは専用カーネル（GemmQ4等）として分離し、通常GEMMとインターフェースを揃える。

---

## wgpu-kernels

### カーネル設計

各カーネル（Gemm, GemmQ4, Softmax, LayerNorm, RMSNorm, Attention等）は構造体として実装する。

### サイズ戦略: ハイブリッド

- モデルロード時に確定する値（hidden_dim, n_heads等）→ WGSLに定数として埋め込む
- 実行時に変わる値（seq_len, batch_size）→ uniformバッファで渡す

定数確定値はコンパイラの最適化が効き、可変値はパイプライン再生成を避ける。
KVキャッシュ使用時にseq_lenがステップごとに変わるため、これを動的にすることが必須。

### WGSL定数埋め込み方法: 文字列テンプレート

WGSLソース内の `{{HIDDEN_DIM}}` 等をRust側で置換してからコンパイルする。
`override` キーワードは `var<workgroup>` の配列サイズに使えないため、テンプレート方式を採用する。
ワークグループサイズもテンプレートパラメータとする（デフォルト: 1Dは256、2Dは16x16）。

### パイプライン生成の最適化

同一パラメータのカーネルは1つのComputePipelineを共有する。
例: 24層モデルのLayerNormは全層同じdimなので、パイプラインは1つだけ生成する。
これにより生成数は「層数×Op種類」から「ユニークなパラメータ組み合わせ数」に削減される。

---

## wgpu-llm

### パイプライン構築: config駆動（案E）

HuggingFace config.jsonだけでパイプラインを構築する。
decoder-only causal LMに絞ることで、これが実現可能になる。

ほぼ全てのdecoder-onlyモデルは同一構造を持つ:

```
Embedding → [Norm → Attention → Residual → Norm → FFN → Residual] × N → Final Norm → LM Head
```

モデル間の差分はパラメータレベルに収まる:

| 差分 | 選択肢 | 判定方法 |
|---|---|---|
| 正規化 | LayerNorm / RMSNorm | model_typeまたは明示フィールド |
| Attention | MHA / GQA | n_heads vs n_kv_heads |
| 位置エンコーディング | 学習済み / RoPE | rope_thetaの有無 |
| FFN | 2層(標準) / 3層(gated) | model_typeまたはintermediate_sizeの使われ方 |
| 活性化関数 | GELU / SiLU | hidden_act |
| バイアス | あり / なし | use_bias等 |
| LM Head重み共有 | tied / untied | tie_word_embeddings |

これらを統一構造体 DecoderConfig に集約し、1つの関数 build_decoder でグラフを構築する。

### DecoderConfig 定義

```
DecoderConfig:
  vocab_size: usize
  hidden_size: usize
  num_layers: usize
  num_attention_heads: usize
  num_kv_heads: usize             # MHAなら == num_attention_heads
  intermediate_size: usize
  max_position_embeddings: usize
  norm_type: LayerNorm | RMSNorm
  norm_eps: f64
  position_encoding: Learned | RoPE { theta: f64 }
  ffn_type: Standard | Gated
  activation: Gelu | Silu
  use_bias: bool
  tie_word_embeddings: bool
```

### 案Eと案Dの境界基準

案Eで対応: グラフトポロジーが共通構造と同一で、カーネル選択のみ異なるモデル。
案Dにフォールバック: グラフトポロジー自体が異なるモデル。

具体例:
- GPT-2, LLaMA, Mistral, Qwen → 案E（共通トポロジー、パラメータ差分のみ）
- GPT-J, Phi（Attention+FFN並列）→ 案D（トポロジーが異なる）
- Mixtral（MoE）→ 案D（FFNがルーティング付き複数エキスパート）

要件が増えて案Eの分岐が複雑になりすぎた場合は、案D主体に移行する。

### フォールバック: 遅延命令型（案D）

案Eで表現できない例外的なアーキテクチャは、
モデル固有のグラフ構築関数を手書きする（遅延命令型グラフ構築）。
命令的に見えるRustコードで計算グラフを構築する方式。llama.cppで実証済みのパターン。
案Eの統一関数と同じGraphBuilder APIを使うため、カーネル層への影響はない。

### 重み名マッピング

safetensorsの重み名はモデルごとに異なる。マッピングは以下を処理する:

- 名前解決（h.0.attn.c_attn.weight → layers.0.self_attn.q_proj.weight等）
- 転置（GPT-2のConv1DはLinearと転置関係にある）
- 分割/結合（GPT-2のfused QKVを3つに分解等）
- 省略可能テンソル（バイアスの有無）

model_typeごとにWeightMapping定義を持つ。
新モデル追加時はDecoderConfigのフィールド名マッピングとWeightMappingの追加が必要。

### グラフ実行: 2グラフ方式（案1）

| グラフ | 用途 |
|---|---|
| PrefillGraph | 初回入力（複数トークン）。KVキャッシュを埋める |
| DecodeGraph | 自己回帰生成（1トークンずつ）。KVキャッシュを参照＋追記 |

Prefill/Decodeの区別はグラフ層の関心事。カーネル層はこの区別を知らない。
最初は両グラフが同じカーネルを使う。後でDecodeGraphのみ特化カーネルに差し替え可能。
この移行はカーネルの追加であり、グラフ設計の変更ではない。

### キャッシュ管理: 外部マネージャ（案3）

グラフはステートレスに保つ。KVキャッシュはグラフ外の InferenceState で管理する。

| 要素 | 責務 |
|---|---|
| KVCacheManager | KVバッファの確保・追記・破棄 |
| position | 現在の書き込み位置 |

グラフ実行時に `&mut InferenceState` を注入する。

この設計を選んだ理由:
- グラフに状態を持たせる（案2）と、バッファのライフタイム管理が複雑化する
- Transient（中間バッファ）とPersistent（キャッシュ）の混在がバッファアロケータを複雑にする
- グラフのリセット/破棄の意味論が曖昧になる
- 外部マネージャなら関心の分離がきれい。キャッシュマネージャの作り直しでリセットできる

### KVキャッシュのメモリ制約

KVキャッシュのメモリ量: `num_layers × 2(K,V) × max_seq_len × hidden_size × sizeof(f32)`

LLaMA 7Bの場合（32層、4096 hidden_size、max_seq_len=4096）: 約4GB。
初期実装ではmax_seq_lenを低めに設定して対応する。
将来的にはPaged KV Cache（vLLM方式）の導入を検討する。

### バッファ割り当て戦略

グラフ構築時に全バッファを事前に割り当てる（静的割り当て）。

- グラフの構造は静的に確定するため、各Opの出力バッファサイズが事前に分かる
- 生存期間分析により、使い終わったバッファを後続のOpに再利用する
- 同一形状のバッファはプールから割り当てる

KVキャッシュバッファはInferenceStateが所有し、グラフは借用する。

### バッファの識別: 名前付きスロット

Op間のバッファ受け渡しは名前付きスロット（例: `"h.0.attn_output"`）で参照する。
グラフ構築時（GPU実行前）に全スロットの接続を検証し、未接続・タイポを早期検出する。

### GPU-CPU同期モデル

1回のforward passにつき1つのCommandEncoderを使用する。
全Opのdispatchを積んだ後、queue.submit()を1回呼ぶ。
logitsの読み出しはbuffer.map_async()でCPUに転送する。
サンプリング（argmax/top-k/top-p）はCPU上で実行する（vocab_size分のf32は~200KBなので転送コストは無視できる）。

### デバイス要件と検証

モデルロード時に以下を検証する:

- adapter要求時にmax_buffer_size, max_storage_buffer_binding_sizeを最大値で要求
- モデルの重みサイズ + KVキャッシュサイズ + 中間バッファサイズがデバイス制限内に収まるか確認
- 制限超過時はロード前に明確なエラーメッセージで失敗する（wgpu validation errorにしない）

---

## 入力フォーマット

| 種類 | フォーマット | 備考 |
|---|---|---|
| モデル定義 | HuggingFace config.json | DecoderConfigに変換して使用 |
| 重み | safetensors | Rust crateあり。後でGGUF対応を追加 |

### GGUF対応方針

GGUF対応は「フルローディング」とする（config + 量子化済み重みの両方をGGUFから取得）。
GGUFメタデータからDecoderConfigを構築するパスを追加する。
DecoderConfigは入力ソース（config.json / GGUF）に依存しない統一構造体として機能する。

---

## テスト戦略

| 対象 | 方式 |
|---|---|
| カーネル単体 | Rust CPU参照実装と数値比較 |
| Forward Pass全体 | PyTorchで生成したfixtureと比較 |
| config解析 | 実際のHuggingFace config.jsonをfixtureとして使用し、DecoderConfigへの変換を検証 |

---

## API方針

低レベルAPI（カーネル単位で使える）を先行実装する。
高レベルAPI（モデルパスを渡すだけで推論できる）は後から追加する。

wgpu-llmの出力はlogits。サンプリングはwgpu-llm-cli側の責務。

---

## 調査に基づく判断根拠

主要フレームワークの調査結果:

| フレームワーク | 方式 | モデルあたりのコード |
|---|---|---|
| candle (Rust) | 1モデル1ファイル手書き | 400-800行 |
| burn (Rust) | derive(Module) + 手書きforward | 400-800行 |
| llama.cpp (C++) | 遅延命令型グラフ構築 | 200-500行 |
| vLLM (Python) | レジストリ + クラス手書き | 500-1500行 |
| ONNX Runtime | 外部グラフ読み込み | 0行 |

全フレームワークが「1モデル = 1実装」方式を採用している。
config.jsonだけでモデルを動かすアプローチは既存にない。
これはフレームワークが全アーキテクチャ対応を目指しているためであり、
decoder-only causal LMに特化すれば構造の共通性からconfig駆動が成立する。
