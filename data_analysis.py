import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# データの読み込み
df = pd.read_csv('stats-csv/england-premier-league-matches-2018-to-2019-stats.csv')

print("=== データの基本情報 ===")
print(f"データの形状: {df.shape}")
print(f"試合数: {len(df)}")
print(f"列数: {len(df.columns)}")

print("\n=== 列名一覧 ===")
for i, col in enumerate(df.columns):
    print(f"{i+1:2d}. {col}")

print("\n=== 基本統計 ===")
print(df.describe())

print("\n=== 欠損値の確認 ===")
missing_data = df.isnull().sum()
print(missing_data[missing_data > 0])

print("\n=== ユニークなチーム数 ===")
home_teams = set(df['home_team_name'].unique())
away_teams = set(df['away_team_name'].unique())
all_teams = home_teams.union(away_teams)
print(f"ホームチーム数: {len(home_teams)}")
print(f"アウェイチーム数: {len(away_teams)}")
print(f"全チーム数: {len(all_teams)}")

print("\n=== 全チーム一覧 ===")
for team in sorted(all_teams):
    print(f"- {team}")

print("\n=== ゲームウィーク情報 ===")
print(f"最小ゲームウィーク: {df['Game Week'].min()}")
print(f"最大ゲームウィーク: {df['Game Week'].max()}")
print(f"ゲームウィーク別試合数:")
print(df['Game Week'].value_counts().sort_index())

print("\n=== 得点情報 ===")
print(f"ホームチーム平均得点: {df['home_team_goal_count'].mean():.2f}")
print(f"アウェイチーム平均得点: {df['away_team_goal_count'].mean():.2f}")
print(f"試合あたり総得点: {df['total_goal_count'].mean():.2f}")

print("\n=== データの期間 ===")
df['date'] = pd.to_datetime(df['date_GMT'])
print(f"開始日: {df['date'].min()}")
print(f"終了日: {df['date'].max()}")

# 各チームの試合数を確認
print("\n=== 各チームの試合数 ===")
home_games = df['home_team_name'].value_counts()
away_games = df['away_team_name'].value_counts()
total_games = home_games.add(away_games, fill_value=0)
print(total_games.sort_values(ascending=False))
