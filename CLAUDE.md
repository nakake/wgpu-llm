# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Claudeの役割

このプロジェクトはLLMの理解を深める学習目的のプロジェクト。Claudeはペアプログラミングのナビゲーター（指示役）に徹すること。コードは書かず、方針・設計・ヒント・レビューを提供する。実装はユーザーが行う。

## プロジェクト概要

wgpuベースのLLM推論エンジン。HuggingFace config.jsonからGPUパイプラインを自動構築する。decoder-only causal LM（GPT-2, LLaMA, Mistral, Qwen等）が対象。詳細な設計はDESIGN.md参照。

## ビルド・テストコマンド

```bash
cargo build                          # 全crateビルド
cargo test                           # 全テスト実行
cargo test -p wgpu-kernels           # 単一crateのテスト
cargo test -p wgpu-kernels test_name # 単一テスト実行
cargo clippy --workspace             # lint
cargo fmt --all                      # フォーマット
cargo fmt --all -- --check           # フォーマットチェック
```

GPU必要なテストは`tokio::test`と`wgpu`デバイス初期化を使用する。

## 開発環境

Nixフレークのdevshell（`nix develop`またはdirenv）。Rustツールチェイン（stable + rust-analyzer + clippy）、Vulkanローダー、開発ツールを提供。WSL2環境でホストGPUパススルー（`/usr/lib/wsl/lib`、`dzn_icd` Vulkan ICD）。

## ワークスペース構成

Cargoワークスペース、2 crate構成:

- **wgpu-llm-core** — 推論エンジン本体。GPUカーネル（GEMM, Softmax, LayerNorm, RMSNorm, Attention等）、config解析（HuggingFace config.json → `DecoderConfig`）、グラフ構築（`build_decoder`）、重みロード（safetensors + モデルごとの`WeightMapping`）、実行。カーネルは`kernels`モジュールとして内包。WGSLシェーダーは`{{PLACEHOLDER}}`文字列テンプレートでコンパイル時定数を埋め込み、ランタイム値はuniformバッファで渡す。
- **wgpu-llm-cli** — 薄いCLIラッパー。トークナイザー（`tokenizers` crate）、サンプリング（temperature/top-k/top-p）、UI。wgpu-llm-coreに依存。

依存関係: `wgpu-llm-cli → wgpu-llm-core`

## 主要アーキテクチャ判断

- **config駆動パイプライン（案E）**: 単一の`DecoderConfig`構造体 + `build_decoder`関数で全対応モデルを処理。モデル間の差分はパラメータのみ（正規化種別、Attention種別、位置エンコーディング、FFN種別、活性化関数、バイアス有無）。構造が異なるモデル（GPT-J, Phi, Mixtral）は手書きグラフ構築関数（案D）にフォールバック。
- **2グラフ実行**: `PrefillGraph`（複数トークン、KVキャッシュ充填）と`DecodeGraph`（1トークン自己回帰生成）。初期実装では両方同じカーネルを使用。
- **ステートレスグラフ**: KVキャッシュは外部の`InferenceState`で管理し、`&mut InferenceState`で注入。グラフは永続状態を持たない。
- **静的バッファ割り当て**: グラフ構築時に全バッファを事前割り当て。名前付きスロット（例: `"h.0.attn_output"`）でOp間のバッファ受け渡し、構築時に接続検証。
- **f32演算**: カーネル内の演算はすべてf32。重みはf16やQ4で格納可能、カーネル内でdequantize。
- **パイプライン共有**: 同一パラメータのカーネルは単一の`ComputePipeline`を共有。
- **forward pass毎に単一CommandEncoder**: 全Opをdispatch後、`queue.submit()`を1回。logitsは`buffer.map_async()`でCPUに転送、サンプリングはCPU上で実行。

## 新モデル追加手順

1. config.jsonフィールドマッピングを追加し`DecoderConfig`を生成
2. モデルのsafetensors命名規則に対応する`WeightMapping`を追加（名前解決、転置、分割/結合、省略可能テンソル）
