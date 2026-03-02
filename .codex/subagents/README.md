# Codex Subagents (Issue-Driven)

Claude team 運用に合わせて、Codex 側でも役割分担して進めるためのテンプレート。

## 目的

- Issue を起点に、役割ごとに作業を分離する
- レビューしやすい粒度で PR を作る
- 仕様変更や前提は Issue/PR コメントに残す

## 役割

- `team-lead`: Issue選定、分割、受け入れ条件定義、進行管理
- `researcher`: モデルのコア設計と論文詳細（数式、学習目的、アーキテクチャ差分、実装上の論点）
- `implementer`: 実装
- `tester`: テスト追加、回帰確認、CI整備
- `mlops`: 依存関係、実験・パイプライン、実行環境
- `reviewer`: 仕様整合、品質レビュー、リスク指摘

## 最小フロー

1. `team-lead` が Issue を選び、受け入れ条件を明確化
2. 必要なら `researcher` が前提調査を先行
3. `implementer` が最小差分で実装
4. `tester` がテストと検証を実施
5. `reviewer` が差分レビュー
6. 条件を満たしたら PR をマージ

## 運用ルール

- 1 Issue = 1 ブランチ = 1 PR
- ブランチ: `issue-<id>-<topic>`
- コミット: `<type>: <summary> (#<id>)`
- 必須検証: `make lint` と `make test`
- 必要時のみ `dvc repro` を追加

## 使い方

各 role の具体的な作業指示テンプレートは `prompts/` 配下を使う。
