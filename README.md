# mock_lightgcn

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

LightGCN (Light Graph Convolutional Network) を用いた推薦システムの実装プロジェクトです。

## プロジェクト構成

```
├── LICENSE            <- オープンソースライセンス (選択した場合)
├── Makefile           <- `make data` や `make train` などの便利コマンド集
├── README.md          <- 開発者向けトップレベル README
├── data
│   ├── external       <- 外部サードパーティのデータ
│   ├── interim        <- 変換途中の中間データ
│   ├── processed      <- モデリング用の最終データセット
│   └── raw            <- 元の不変データ
│
├── docs               <- mkdocs プロジェクト (詳細は www.mkdocs.org を参照)
│
├── models             <- 学習済み・シリアライズ済みモデル、予測結果、モデルサマリ
│
├── notebooks          <- Jupyter ノートブック。命名規則: 番号 (順序付け)、
│                         作成者イニシャル、ハイフン区切りの短い説明
│                         例: `1.0-jqp-initial-data-exploration`
│
├── pyproject.toml     <- mock_lightgcn のパッケージメタデータおよびツール設定ファイル
│
├── references         <- データ辞書、マニュアル、その他の説明資料
│
├── reports            <- HTML、PDF、LaTeX 等で生成された分析レポート
│   └── figures        <- レポートに使用する図・グラフ
│
├── requirements.txt   <- 分析環境を再現するための要件ファイル
│                         (例: `pip freeze > requirements.txt` で生成)
│
├── setup.cfg          <- flake8 の設定ファイル
│
└── mock_lightgcn   <- プロジェクトのソースコード
    │
    ├── __init__.py             <- mock_lightgcn を Python モジュールとして認識させる
    │
    ├── config.py               <- 変数・設定の一元管理
    │
    ├── dataset.py              <- データのダウンロード・生成スクリプト
    │
    ├── features.py             <- モデリング用の特徴量生成コード
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- 学習済みモデルを用いた推論コード
    │   └── train.py            <- モデルの学習コード
    │
    └── plots.py                <- 可視化コード
```

--------

