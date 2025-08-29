import pandas as pd
import numpy as np

# データの読み込み
df = pd.read_csv('stats-csv/england-premier-league-matches-2018-to-2019-stats.csv')

def create_league_table(df):
    """
    試合結果から最終順位表を作成
    """
    teams = list(set(df['home_team_name'].unique()) | set(df['away_team_name'].unique()))
    
    # 各チームの統計を初期化
    table = []
    
    for team in teams:
        # ホームゲーム
        home_games = df[df['home_team_name'] == team]
        # アウェイゲーム
        away_games = df[df['away_team_name'] == team]
        
        # 試合数
        games_played = len(home_games) + len(away_games)
        
        # 得失点
        goals_for = home_games['home_team_goal_count'].sum() + away_games['away_team_goal_count'].sum()
        goals_against = home_games['away_team_goal_count'].sum() + away_games['home_team_goal_count'].sum()
        goal_difference = goals_for - goals_against
        
        # 勝ち点計算
        wins = 0
        draws = 0
        losses = 0
        
        # ホームゲームの結果
        for _, match in home_games.iterrows():
            home_goals = match['home_team_goal_count']
            away_goals = match['away_team_goal_count']
            if home_goals > away_goals:
                wins += 1
            elif home_goals == away_goals:
                draws += 1
            else:
                losses += 1
        
        # アウェイゲームの結果
        for _, match in away_games.iterrows():
            home_goals = match['home_team_goal_count']
            away_goals = match['away_team_goal_count']
            if away_goals > home_goals:
                wins += 1
            elif home_goals == away_goals:
                draws += 1
            else:
                losses += 1
        
        points = wins * 3 + draws * 1
        
        table.append({
            'Team': team,
            'GP': games_played,
            'W': wins,
            'D': draws,
            'L': losses,
            'GF': goals_for,
            'GA': goals_against,
            'GD': goal_difference,
            'Pts': points
        })
    
    # データフレームに変換し、勝ち点でソート
    table_df = pd.DataFrame(table)
    table_df = table_df.sort_values(['Pts', 'GD', 'GF'], ascending=[False, False, False])
    table_df = table_df.reset_index(drop=True)
    table_df.index = table_df.index + 1  # 順位を1から開始
    
    return table_df

# 最終順位表を作成
final_table = create_league_table(df)

print("=== プレミアリーグ 2018-19シーズン 最終順位表 ===")
print(final_table.to_string())

# チャンピオンズリーグ、ヨーロッパリーグ、降格圏の確認
print("\n=== ポジション別分析 ===")
print("チャンピオンズリーグ出場権 (1-4位):")
print(final_table.iloc[0:4]['Team'].tolist())

print("\nヨーロッパリーグ出場権 (5-6位):")
print(final_table.iloc[4:6]['Team'].tolist())

print("\n降格圏 (18-20位):")
print(final_table.iloc[17:20]['Team'].tolist())

# 勝ち点の分布を確認
print(f"\n=== 勝ち点分布 ===")
print(f"優勝チーム勝ち点: {final_table.iloc[0]['Pts']}")
print(f"4位チーム勝ち点: {final_table.iloc[3]['Pts']}")
print(f"17位チーム勝ち点: {final_table.iloc[16]['Pts']}")
print(f"最下位チーム勝ち点: {final_table.iloc[19]['Pts']}")

print(f"\n優勝と2位の差: {final_table.iloc[0]['Pts'] - final_table.iloc[1]['Pts']}点")
print(f"4位と5位の差: {final_table.iloc[3]['Pts'] - final_table.iloc[4]['Pts']}点")
print(f"17位と18位の差: {final_table.iloc[16]['Pts'] - final_table.iloc[17]['Pts']}点")

# CSVとして保存
final_table.to_csv('premier_league_2018_19_final_table.csv', index_label='Position')
print(f"\n最終順位表を 'premier_league_2018_19_final_table.csv' として保存しました。")
