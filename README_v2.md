# Football League Predictor v2.0 

⚽ 多リーグ対応・継続学習可能なサッカーリーグ予測システム

## 🎯 新機能

### v2.0の主な改善点
- 🌍 **多リーグ対応**: プレミアリーグ、Jリーグ、その他リーグのデータを統合学習
- 🧠 **深層学習**: PyTorchベースのニューラルネットワーク
- 🔄 **継続学習**: 新しいデータで既存モデルをファインチューニング
- 📊 **オッズ非依存**: PPG、xGベースの純粋な統計学習
- ⚡ **GPU対応**: Google Colab環境での高速学習
- 🔮 **リアルタイム予測**: 試合結果更新→即座に順位予測

## 📁 新しいプロジェクト構造

```
football-predictor-v2/
├── data/
│   ├── raw/                     # 生データ
│   │   ├── premier_league/
│   │   ├── j_league/
│   │   └── other_leagues/
│   ├── processed/               # 前処理済みデータ
│   └── fixtures/                # 試合予定・結果
│       ├── 2024_season.csv     # 現シーズン全試合予定
│       └── latest_results.csv   # 最新結果（手動更新）
├── models/
│   ├── saved/                   # 保存済みモデル
│   │   ├── league_predictor.pth
│   │   └── model_config.json
│   └── checkpoints/             # 学習チェックポイント
├── src/
│   ├── training/                # 学習用コード
│   │   ├── train_model.py       # メイン学習スクリプト
│   │   ├── data_loader.py       # データローダー
│   │   └── model_architecture.py # モデル定義
│   ├── prediction/              # 予測用コード
│   │   ├── predict_matches.py   # 試合予測
│   │   ├── predict_standings.py # 順位予測
│   │   └── update_results.py    # 結果更新
│   └── utils/
│       ├── data_preprocessing.py
│       └── evaluation.py
├── notebooks/
│   ├── colab_training.ipynb     # Google Colab学習用
│   └── analysis.ipynb           # データ分析用
├── config/
│   ├── model_config.yaml        # モデル設定
│   └── league_config.yaml       # リーグ設定
└── requirements_v2.txt
```

## 🔧 技術スタック

- **深層学習**: PyTorch, PyTorch Lightning
- **データ処理**: pandas, numpy
- **GPU学習**: CUDA (Google Colab)
- **モデル管理**: MLflow
- **設定管理**: YAML, Hydra

## 🚀 使用方法

### 1. 学習フェーズ (Google Colab)
```python
# colab_training.ipynb で実行
from src.training.train_model import train_league_predictor

# 新しいリーグデータで学習
model = train_league_predictor(
    data_paths=['data/raw/premier_league/', 'data/raw/j_league/'],
    pretrained_model='models/saved/league_predictor.pth',  # 継続学習
    epochs=50,
    use_gpu=True
)
```

### 2. 予測フェーズ (ローカル)
```python
# 週次予測の実行
from src.prediction.predict_matches import predict_weekly_matches
from src.prediction.predict_standings import predict_final_standings

# 次週の試合予測
weekly_predictions = predict_weekly_matches('data/fixtures/2024_season.csv')

# 最終順位予測
final_standings = predict_final_standings('data/fixtures/latest_results.csv')
```

## 📊 特徴量設計 (オッズ非依存)

### 主要特徴量
1. **チーム実力指標**
   - PPG (Points Per Game): 直近10試合、シーズン累計
   - xG (Expected Goals): 攻撃力指標
   - xGA (Expected Goals Against): 守備力指標

2. **フォーム指標**
   - 直近5試合の結果
   - ホーム/アウェイ別パフォーマンス
   - 対戦相手との過去成績

3. **コンテキスト特徴量**
   - シーズン進行度
   - 残り試合数
   - リーグ順位差

## 🎯 ワークフロー

1. **データ収集**: 新しいリーグデータを `data/raw/` に追加
2. **学習**: Google Colabで既存モデルをファインチューニング
3. **予測**: 試合結果更新後、自動で順位予測
4. **評価**: 予測精度を継続的にモニタリング

---

**次のステップ**: 詳細な実装を開始しましょう！
