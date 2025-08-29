# ⚡ クイックスタート

Premier League Position Predictorをすぐに試したい方向けの簡単ガイドです。

## 🚀 30秒で始める

```bash
# 1. リポジトリをクローン
git clone https://github.com/{your-username}/premier-league-predictor.git
cd premier-league-predictor

# 2. 依存関係をインストール
pip install -r requirements.txt

# 3. 基本的なデータ分析を実行
python data_analysis.py
```

## 📊 主要な機能を試す

### 1. プレミアリーグ2018-19の最終順位表を見る
```bash
python create_league_table.py
```

### 2. 試合予測を試す
```python
python -c "
from basic_predictor import PremierLeaguePredictor

predictor = PremierLeaguePredictor()
predictor.load_data('stats-csv/england-premier-league-matches-2018-to-2019-stats.csv')
predictor.train_models()

result = predictor.predict_match(
    'Manchester City', 'Liverpool', 20, 
    2.8, 2.7, 2.2, 2.0, 1.8, 4.0, 4.5
)
print('予測結果:', result)
"
```

### 3. プロジェクトサマリーを確認
```bash
python project_summary.py
```

## 🎯 期待される結果

- **データ分析**: 380試合、20チーム、66特徴量の詳細
- **最終順位**: Manchester City (98点) が優勝
- **試合予測**: 63.2%の精度で勝敗予測
- **特徴量**: オッズが最重要な予測因子

## 📖 詳細ガイド

より詳しい情報は [README.md](README.md) と [GITHUB_SETUP.md](GITHUB_SETUP.md) をご確認ください。

## 🤝 問題が発生した場合

1. Python 3.8以上がインストールされているか確認
2. 仮想環境の使用を推奨
3. エラーが発生した場合はIssueを作成してください

---
🏆 **楽しいサッカーデータ分析を！**
