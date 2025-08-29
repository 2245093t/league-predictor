import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# データの読み込み
df = pd.read_csv('stats-csv/england-premier-league-matches-2018-to-2019-stats.csv')

print("=== 利用可能な特徴量の分析 ===")

print("\n1. 試合前情報 (Pre-Match Features):")
pre_match_features = [
    'Pre-Match PPG (Home)', 'Pre-Match PPG (Away)',
    'Home Team Pre-Match xG', 'Away Team Pre-Match xG',
    'average_goals_per_match_pre_match',
    'btts_percentage_pre_match',
    'over_15_percentage_pre_match', 'over_25_percentage_pre_match',
    'over_35_percentage_pre_match', 'over_45_percentage_pre_match',
    'average_corners_per_match_pre_match',
    'average_cards_per_match_pre_match'
]

for feature in pre_match_features:
    print(f"  - {feature}")

print("\n2. ブックメーカーオッズ:")
odds_features = [
    'odds_ft_home_team_win', 'odds_ft_draw', 'odds_ft_away_team_win',
    'odds_ft_over15', 'odds_ft_over25', 'odds_ft_over35', 'odds_ft_over45',
    'odds_btts_yes', 'odds_btts_no'
]

for feature in odds_features:
    print(f"  - {feature}")

print("\n3. 試合結果・統計:")
match_stats = [
    'home_team_goal_count', 'away_team_goal_count',
    'home_team_shots', 'away_team_shots',
    'home_team_shots_on_target', 'away_team_shots_on_target',
    'home_team_possession', 'away_team_possession',
    'home_team_corner_count', 'away_team_corner_count',
    'home_team_yellow_cards', 'home_team_red_cards',
    'away_team_yellow_cards', 'away_team_red_cards',
    'team_a_xg', 'team_b_xg'
]

for feature in match_stats:
    print(f"  - {feature}")

print("\n=== PPG (Points Per Game) の進化分析 ===")

# 各ゲームウィークでのPPGの変化を分析
def analyze_ppg_evolution():
    teams = df['home_team_name'].unique()
    
    for team in ['Manchester City', 'Liverpool', 'Chelsea'][:3]:  # 上位3チームのサンプル
        team_home = df[df['home_team_name'] == team].sort_values('Game Week')
        team_away = df[df['away_team_name'] == team].sort_values('Game Week')
        
        print(f"\n{team} のPPG進化:")
        print("Game Week | Home PPG | Away PPG")
        
        # ホーム戦のPPG推移（最初の5試合）
        for i, (_, match) in enumerate(team_home.head().iterrows()):
            print(f"    {match['Game Week']:2d}    |   {match['home_ppg']:.2f}   |    -   ")
        
        # アウェイ戦のPPG推移（最初の5試合）
        for i, (_, match) in enumerate(team_away.head().iterrows()):
            print(f"    {match['Game Week']:2d}    |    -     |  {match['away_ppg']:.2f}  ")

analyze_ppg_evolution()

print("\n=== 予測に使えそうな特徴量の組み合わせ ===")

# 予測で重要そうな特徴量
prediction_features = {
    'チーム実力指標': [
        'Pre-Match PPG (Home)', 'Pre-Match PPG (Away)',
        'Home Team Pre-Match xG', 'Away Team Pre-Match xG'
    ],
    'ホーム・アウェイ効果': [
        'home_team_name', 'away_team_name'
    ],
    'シーズン進行度': [
        'Game Week'
    ],
    'マーケット予想': [
        'odds_ft_home_team_win', 'odds_ft_draw', 'odds_ft_away_team_win'
    ],
    '過去の統計': [
        'average_goals_per_match_pre_match',
        'btts_percentage_pre_match'
    ]
}

for category, features in prediction_features.items():
    print(f"\n{category}:")
    for feature in features:
        print(f"  - {feature}")

print("\n=== データ品質チェック ===")

# ゲームウィーク1での特徴量の状況
print("\nゲームウィーク1の特徴量 (初期状態):")
week1_data = df[df['Game Week'] == 1]
print(f"Pre-Match PPG (Home) - 範囲: {week1_data['Pre-Match PPG (Home)'].min()} - {week1_data['Pre-Match PPG (Home)'].max()}")
print(f"Pre-Match PPG (Away) - 範囲: {week1_data['Pre-Match PPG (Away)'].min()} - {week1_data['Pre-Match PPG (Away)'].max()}")

print("\nゲームウィーク38の特徴量 (最終状態):")
week38_data = df[df['Game Week'] == 38]
print(f"Pre-Match PPG (Home) - 範囲: {week38_data['Pre-Match PPG (Home)'].min():.2f} - {week38_data['Pre-Match PPG (Home)'].max():.2f}")
print(f"Pre-Match PPG (Away) - 範囲: {week38_data['Pre-Match PPG (Away)'].min():.2f} - {week38_data['Pre-Match PPG (Away)'].max():.2f}")

# 残り試合数の概念
print("\n=== 残り試合数の重要性 ===")
print("各ゲームウィークでの残り試合数:")
for week in [1, 10, 20, 30, 38]:
    remaining_games = 38 - week
    print(f"ゲームウィーク{week:2d}: 残り{remaining_games:2d}試合")

print("\n=== 予測モデルのアプローチ案 ===")
print("""
1. 時系列アプローチ:
   - 各ゲームウィークでの状況を基に、残り試合の結果を予測
   - PPG、xG、オッズなどを特徴量として使用

2. 対戦相手を考慮したアプローチ:
   - 各チーム同士の相性（過去の対戦成績）
   - ホーム・アウェイでの強さの違い
   - 直接対戦での期待値計算

3. シミュレーションアプローチ:
   - モンテカルロシミュレーション
   - 各試合の結果をポアソン分布などでモデル化
   - 最終順位の確率分布を計算
""")
