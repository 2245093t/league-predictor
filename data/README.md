# Data Directory

このディレクトリはサッカーリーグのデータを管理します。

## 📁 ディレクトリ構造

```
data/
├── raw/                      # 生データ（Google Driveで管理）
│   ├── premier_league/       # プレミアリーグのCSVファイル
│   ├── j_league/            # Jリーグのデータ
│   └── other_leagues/       # その他のリーグ
├── processed/               # 前処理済みデータ
├── fixtures/                # 試合予定・結果データ
└── predictions/             # 予測結果の出力
```

## 🔒 プライバシー保護

データファイル（CSV）は**.gitignore**で除外され、GitHubには含まれません。
データは**Google Drive**で安全に管理されます。

## 📊 必要なデータ形式

### 基本的な試合データ
- `Game Week`: ゲームウィーク番号
- `home_team_name`: ホームチーム名
- `away_team_name`: アウェーチーム名  
- `home_team_goal_count`: ホームチームのゴール数
- `away_team_goal_count`: アウェーチームのゴール数

### 推奨データ（高精度予測用）
- `home_team_goal_count_expected`: ホームチームのxG
- `away_team_goal_count_expected`: アウェーチームのxG

## 🚀 データの使用方法

### 1. Google Driveからデータをアップロード
Google Colabの学習ノートブックでデータをアップロードしてください。

### 2. 学習用データの配置
```python
# Google Colab内で
for filename in uploaded.keys():
    if 'premier_league' in filename.lower():
        !mv "{filename}" data/raw/premier_league/
    elif 'j_league' in filename.lower():
        !mv "{filename}" data/raw/j_league/
```

### 3. 予測用の現在シーズンデータ
現在進行中のシーズンのデータは`data/fixtures/`に配置：
- `current_season_fixtures.csv`: 全試合予定
- `current_season_results.csv`: 完了済み試合結果

## ⚠️ 注意事項

- データファイルはGitHubにコミットしないでください
- 新しいリーグを追加する場合は適切なディレクトリを作成
- データの更新後は継続学習でモデルを更新

---

**データのプライバシーを守りながら、高精度な予測を実現しましょう！**
