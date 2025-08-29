# Premier League Position Predictor

プレミアリーグのシーズン途中の順位や最終順位を予測する機械学習モデルです。

## プロジェクト概要

このプロジェクトは、サッカーのプレミアリーグで消化済みの試合結果とそのスタッツ、残りの試合（対戦相手も考慮）から、最終的な順位もしくは勝ち点を予測するモデルを作成します。

## データ

- **使用データ**: プレミアリーグ 2018-19シーズンの試合結果
- **データファイル**: `stats-csv/england-premier-league-matches-2018-to-2019-stats.csv`
- **試合数**: 380試合（20チーム × 38節）
- **特徴量**: 66の項目（試合前統計、試合結果、オッズなど）

### 主要な特徴量

1. **試合前情報**
   - PPG (Points Per Game): ホーム・アウェイ別
   - xG (Expected Goals): 試合前期待得点
   - 過去の統計データ（平均得点、BTTS確率など）

2. **ブックメーカーオッズ**
   - 勝利・ドロー・敗北のオッズ
   - Over/Under得点オッズ
   - Both Teams to Score (BTTS) オッズ

3. **試合結果・統計**
   - 得点、シュート数、ポゼッション
   - コーナーキック、カード数
   - 実際のxG値

## 実装済み機能

### 1. データ分析 (`data_analysis.py`)
- 基本的なデータ探索と統計
- 欠損値チェック
- チーム別試合数確認

### 2. 順位表作成 (`create_league_table.py`)
- 試合結果から最終順位表を生成
- 実際の2018-19シーズン結果と照合

### 3. 特徴量分析 (`feature_analysis.py`)
- 利用可能な特徴量の詳細分析
- PPGの進化過程確認
- 予測に有効な特徴量の特定

### 4. 基本予測モデル (`basic_predictor.py`)
- ランダムフォレストを使用した試合結果予測
- ホーム・アウェイ別得点予測
- 単一試合の勝敗・スコア予測

### 5. 順位予測シミュレーター (`position_predictor.py`)
- 指定されたゲームウィークからの最終順位予測
- モンテカルロシミュレーション
- チャンピオンズリーグ出場確率・降格確率計算

## インストール・セットアップ

### GitHubからクローン
```bash
git clone https://github.com/{ユーザー名}/premier-league-predictor.git
cd premier-league-predictor
```

### 環境構築
```bash
# 仮想環境作成・有効化
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# または
.venv\Scripts\activate  # Windows

# 必要パッケージインストール
pip install -r requirements.txt
```

## 使用方法

1. **データ分析の実行**
```bash
python data_analysis.py
```

2. **順位表の作成**
```bash
python create_league_table.py
```

3. **単一試合の予測**
```python
from basic_predictor import PremierLeaguePredictor

predictor = PremierLeaguePredictor()
predictor.load_data('stats-csv/england-premier-league-matches-2018-to-2019-stats.csv')
predictor.train_models()

# 試合予測
result = predictor.predict_match(
    home_team="Manchester City",
    away_team="Liverpool", 
    game_week=20,
    home_ppg=2.8, away_ppg=2.7,
    home_xg=2.2, away_xg=2.0,
    home_odds=1.8, draw_odds=4.0, away_odds=4.5
)
print(result)
```

4. **シーズン順位予測**
```python
from position_predictor import LeaguePositionPredictor

position_predictor = LeaguePositionPredictor(predictor)
results = position_predictor.simulate_season_from_gameweek(
    predictor.df, from_gameweek=20, n_simulations=1000
)
print(results)
```

## モデル性能

現在の基本モデル（プレミアリーグ2018-19データ）:
- **試合結果予測精度**: 63.2%
- **ホーム得点予測RMSE**: 1.227
- **アウェイ得点予測RMSE**: 1.068

### 重要な特徴量（上位5位）
1. away_win_odds (アウェイ勝利オッズ)
2. away_win_prob (アウェイ勝利確率)
3. home_win_prob (ホーム勝利確率)
4. home_win_odds (ホーム勝利オッズ)
5. home_xg (ホームチームxG)

## プロジェクト構造

```
league-predictor/
├── stats-csv/
│   └── england-premier-league-matches-2018-to-2019-stats.csv
├── data_analysis.py              # データ探索・分析
├── create_league_table.py        # 順位表作成
├── feature_analysis.py           # 特徴量分析
├── basic_predictor.py            # 基本予測モデル
├── position_predictor.py         # 順位予測シミュレーター
├── premier_league_2018_19_final_table.csv  # 最終順位表
└── README.md                     # このファイル
```

## 今後の改善案

### 1. モデルの改善
- より高度なアルゴリズム（XGBoost、Neural Networks）
- 時系列モデルの導入
- アンサンブル学習

### 2. 特徴量エンジニアリング
- チーム間の直接対戦成績
- 最近のフォーム（直近5試合など）
- 負傷者・出場停止情報
- 移籍期間の影響

### 3. データの拡張
- 複数シーズンのデータ
- 他リーグのデータ
- より詳細な選手データ

### 4. 評価指標の改善
- ランクド確率スコア (RPS)
- ブックメーカーオッズとの比較
- 長期的な予測精度評価

## 実際の結果との比較

**プレミアリーグ 2018-19シーズン最終順位（上位5チーム）:**
1. Manchester City - 98点
2. Liverpool - 97点
3. Chelsea - 72点
4. Tottenham Hotspur - 71点
5. Arsenal - 70点

マンチェスター・シティとリバプールの激しいタイトル争いが特徴的なシーズンでした。

## 貢献・開発

### ブランチ戦略
- `main`: 安定版リリース
- `develop`: 開発版
- `feature/機能名`: 新機能開発

### コントリビューション
1. このリポジトリをフォーク
2. 新しいブランチを作成 (`git checkout -b feature/amazing-feature`)
3. 変更をコミット (`git commit -m 'Add some amazing feature'`)
4. ブランチにプッシュ (`git push origin feature/amazing-feature`)
5. プルリクエストを作成

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。

## 免責事項

このモデルは教育・研究目的で作成されており、ギャンブルや商業目的での使用は推奨されません。予測結果の正確性は保証されません。
