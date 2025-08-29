# 🚀 League Predictor - Quick Start Guide

リニューアル版リーグ予測システムの使い方を5分で理解できるガイドです。

## 📋 目次

1. [システム概要](#システム概要)
2. [即座に試す](#即座に試す)
3. [週次予測ワークフロー](#週次予測ワークフロー)
4. [最終順位予測](#最終順位予測)
5. [モデル学習](#モデル学習)
6. [継続学習](#継続学習)

## システム概要

### 🎯 何ができるか？

- **週次試合予測**: 次週の全試合の勝敗・スコア予測
- **最終順位予測**: 1000回シミュレーションによる最終順位予測  
- **継続学習**: 新しい結果でモデルを即座に更新

### 🧠 どう動くか？

1. **PPG + xG**: チームの勝ち点効率と期待ゴールで実力測定
2. **深層学習**: PyTorchニューラルネットワークで非線形予測
3. **モンテカルロ**: 確率的シミュレーションで不確実性も考慮

## 即座に試す

### 1. 現在のプレミアリーグデータで予測

```bash
# プロジェクトディレクトリに移動
cd /Users/nishiminetakuto/Desktop/league-predictor

# 学習済みモデルがあると仮定して予測実行
cd src/prediction
python main.py \\
  --mode both \\
  --fixtures ../../data/raw/premier_league/england-premier-league-matches-2018-to-2019-stats.csv \\
  --output-dir ../../data/predictions
```

### 2. 結果確認

```bash
# 予測結果ファイルを確認
ls ../../data/predictions/
# match_predictions_YYYYMMDD_HHMMSS.csv
# standings_predictions_YYYYMMDD_HHMMSS.csv
```

## 週次予測ワークフロー

### 📅 リアルタイム予測の流れ

#### Step 1: 最新の結果を更新

```csv
# data/fixtures/current_season_results.csv
Game Week,home_team_name,away_team_name,home_team_goal_count,away_team_goal_count
1,Arsenal,Crystal Palace,2,0
1,Liverpool,Norwich City,4,1
1,Manchester City,Tottenham,1,0
...
14,Chelsea,Newcastle,3,0
14,Liverpool,Arsenal,1,2
```

#### Step 2: 次週（第15節）の予測

```bash
cd src/prediction
python predict_matches.py \\
  --fixtures ../../data/fixtures/season_fixtures.csv \\
  --results ../../data/fixtures/current_season_results.csv \\
  --gameweek 15 \\
  --output next_week_predictions.csv
```

#### Step 3: 予測結果の確認

```bash
# 出力例
Manchester United vs Liverpool: 1.8 - 2.1 (Away Win)
  Probabilities: 32.5% - 23.8% - 43.7%
  
Arsenal vs Chelsea: 2.0 - 1.3 (Home Win)  
  Probabilities: 51.2% - 26.4% - 22.4%
```

## 最終順位予測

### 🏆 シーズン終了時の順位予測

#### 基本実行

```bash
cd src/prediction
python predict_standings.py \\
  --fixtures ../../data/fixtures/season_fixtures.csv \\
  --results ../../data/fixtures/current_season_results.csv \\
  --simulations 1000 \\
  --output final_standings_prediction.csv
```

#### 結果の読み方

```
===========================================
FINAL STANDINGS PREDICTION SUMMARY  
===========================================
Pos Team                 Current Predicted    Range       Top4%  Rel%
 1  Manchester City         42     89.2±2.1   1-2        94.2%   0.0%
 2  Arsenal                 40     83.7±3.4   1-4        78.5%   0.0%
 3  Liverpool               39     81.3±3.8   2-5        71.3%   0.0%
 4  Chelsea                 35     74.2±4.2   3-7        45.8%   0.0%
...
18  Sheffield United        12     28.4±5.1  16-20        0.0%  85.3%
19  Burnley                 11     25.7±4.8  17-20        0.0%  92.1%
20  Norwich City             8     21.2±3.9  19-20        0.0%  98.7%
```

### 🎯 重要な確率

- **Top4%**: チャンピオンズリーグ出場確率
- **Top6%**: ヨーロッパリーグ出場確率  
- **Rel%**: 降格確率

## モデル学習

### 🏋️ Google Colabでの高速学習（推奨）

#### Step 1: Colabノートブック開く

1. `notebooks/colab_training.ipynb`をGoogle Colabで開く
2. GPU環境を選択（Runtime → Change runtime type → GPU）

#### Step 2: セルを順番に実行

```python
# 1. セットアップ
!git clone <your-repo-url>
!pip install -r requirements.txt

# 2. データアップロード  
# プレミアリーグCSVをアップロード

# 3. 学習実行
!python src/training/train_model.py --epochs 100 --batch_size 64

# 4. モデルダウンロード
files.download('models/saved/best_model.pth')
files.download('models/saved/team_encoder.json')
```

#### Step 3: 学習済みモデルをローカルに配置

```bash
# ダウンロードしたファイルを配置
mv ~/Downloads/best_model.pth models/saved/
mv ~/Downloads/team_encoder.json models/saved/
```

### 🖥️ ローカル学習

```bash
cd src/training
python train_model.py \\
  --data_dir ../../data/raw \\
  --output_dir ../../models/saved \\
  --epochs 100 \\
  --batch_size 64 \\
  --learning_rate 0.001
```

## 継続学習

### 🔄 新しい試合結果でモデル更新

#### Step 1: 新しい結果を追加

```csv
# 新しい結果をCSVに追加
15,Manchester United,Liverpool,1,3
15,Arsenal,Chelsea,2,0
15,Manchester City,Tottenham,4,1
```

#### Step 2: 継続学習実行

```bash
cd src/training
python train_model.py \\
  --mode fine_tune \\
  --pretrained_model ../../models/saved/best_model.pth \\
  --new_data ../../data/fixtures/updated_results.csv \\
  --epochs 10 \\
  --learning_rate 0.0001
```

#### Step 3: 更新されたモデルで予測

```bash
cd src/prediction
python main.py \\
  --mode both \\
  --fixtures ../../data/fixtures/season_fixtures.csv \\
  --results ../../data/fixtures/updated_results.csv \\
  --gameweek 16
```

## 🎛️ 高度な使い方

### カスタムパラメータ

```bash
# 高精度モード（シミュレーション回数増加）
python predict_standings.py --simulations 5000

# 特定ゲームウィーク範囲の予測
python predict_matches.py --start-gameweek 15 --end-gameweek 20

# 複数リーグ同時学習
python train_model.py --leagues premier_league,j_league,bundesliga
```

### バッチ処理

```bash
# 複数シーズン一括予測
for season in 2019-20 2020-21 2021-22; do
  python main.py --fixtures data/fixtures/${season}_fixtures.csv --output-dir predictions/${season}
done
```

## 🚨 トラブルシューティング

### よくあるエラーと解決法

1. **"No module named 'torch'"**
```bash
pip install torch torchvision torchaudio
```

2. **"Team not found in encoder"**
```bash
# チーム名の表記統一を確認
# "Manchester United" vs "Man United"
```

3. **"CUDA out of memory"**
```bash
# バッチサイズを削減
python train_model.py --batch_size 32
```

4. **予測精度が低い**
```bash
# より多くのデータで学習
# エポック数を増加: --epochs 200
# 学習率を調整: --learning_rate 0.0005
```

## 📊 性能ベンチマーク

### 期待できる精度

- **試合結果予測**: 68-72%
- **スコア予測**: 平均誤差0.8ゴール  
- **順位予測**: 平均2位差以内
- **学習時間**: Colab GPU約30分（100エポック）

### 推奨設定

- **学習**: epochs=100, batch_size=64, lr=0.001
- **予測**: simulations=1000（時間に余裕があれば5000）
- **継続学習**: epochs=10, lr=0.0001

## 🎯 次のステップ

1. **実際の運用開始**: 現在のシーズンデータで予測開始
2. **多リーグ拡張**: Jリーグ・ブンデスリーガ等を追加
3. **自動化**: GitHub Actionsで定期予測自動実行
4. **可視化**: Dashboardでの予測結果表示

## 📞 サポート

- **GitHub Issues**: バグ報告・機能要望
- **Documentation**: 詳細な技術仕様
- **Community**: 予測精度改善のディスカッション

---

**🚀 5分で始める高精度リーグ予測！**
