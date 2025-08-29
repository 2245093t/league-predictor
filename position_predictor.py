import pandas as pd
import numpy as np
from collections import defaultdict
import itertools

class LeaguePositionPredictor:
    def __init__(self, predictor):
        self.predictor = predictor
        self.teams = None
        
    def get_remaining_fixtures(self, df, current_gameweek):
        """指定されたゲームウィーク以降の残り試合を生成"""
        # 実際のフィクスチャデータから残り試合を抽出
        remaining_matches = df[df['Game Week'] > current_gameweek].copy()
        return remaining_matches
    
    def create_full_season_fixtures(self, teams):
        """全シーズンのフィクスチャを生成（総当たり戦）"""
        fixtures = []
        gameweek = 1
        
        # 各チームが19回ずつホーム・アウェイで戦う
        team_list = list(teams)
        n_teams = len(team_list)
        
        # ラウンドロビン方式でフィクスチャ生成
        for round_num in range(n_teams - 1):
            for match_num in range(n_teams // 2):
                if round_num % 2 == 0:  # 前半戦
                    home_idx = match_num
                    away_idx = (n_teams - 1 - match_num + round_num) % n_teams
                else:  # 後半戦
                    away_idx = match_num
                    home_idx = (n_teams - 1 - match_num + round_num) % n_teams
                
                fixtures.append({
                    'Game Week': gameweek,
                    'home_team_name': team_list[home_idx],
                    'away_team_name': team_list[away_idx]
                })
            
            gameweek += 1
            if gameweek > 19:  # 前半戦終了後、後半戦開始
                if round_num == n_teams // 2 - 1:
                    gameweek = 20
        
        return pd.DataFrame(fixtures)
    
    def simulate_match(self, home_team, away_team, game_week, team_stats):
        """単一試合をシミュレート"""
        # チーム統計から特徴量を取得
        home_ppg = team_stats.get(home_team, {}).get('ppg', 1.5)
        away_ppg = team_stats.get(away_team, {}).get('ppg', 1.5)
        home_xg = team_stats.get(home_team, {}).get('xg', 1.2)
        away_xg = team_stats.get(away_team, {}).get('xg', 1.2)
        
        # 簡易的なオッズ計算（実際のデータがない場合）
        strength_diff = home_ppg - away_ppg + 0.3  # ホームアドバンテージ
        
        if strength_diff > 0.5:
            home_odds, draw_odds, away_odds = 1.8, 3.5, 4.0
        elif strength_diff > 0:
            home_odds, draw_odds, away_odds = 2.2, 3.2, 3.4
        elif strength_diff > -0.5:
            home_odds, draw_odds, away_odds = 2.8, 3.3, 2.6
        else:
            home_odds, draw_odds, away_odds = 3.8, 3.5, 2.0
        
        try:
            prediction = self.predictor.predict_match(
                home_team, away_team, game_week,
                home_ppg, away_ppg, home_xg, away_xg,
                home_odds, draw_odds, away_odds
            )
            
            # 確率的な結果決定
            probs = prediction['probabilities']
            result_choice = np.random.choice(
                ['ホーム勝利', 'ドロー', 'アウェイ勝利'],
                p=[probs['ホーム勝利'], probs['ドロー'], probs['アウェイ勝利']]
            )
            
            # スコア生成（予測値をベースにポアソン分布で変動）
            home_goals = max(0, int(np.random.poisson(max(0.1, prediction['home_goals']))))
            away_goals = max(0, int(np.random.poisson(max(0.1, prediction['away_goals']))))
            
            # 結果と整合性を保つ
            if result_choice == 'ホーム勝利' and home_goals <= away_goals:
                home_goals = away_goals + 1
            elif result_choice == 'アウェイ勝利' and away_goals <= home_goals:
                away_goals = home_goals + 1
            elif result_choice == 'ドロー':
                if abs(home_goals - away_goals) > 1:
                    away_goals = home_goals
            
            return home_goals, away_goals
            
        except Exception as e:
            # フォールバック：簡易予測
            home_expected = max(0.1, home_ppg * 0.6 + 0.3)  # ホームアドバンテージ
            away_expected = max(0.1, away_ppg * 0.6)
            
            home_goals = max(0, int(np.random.poisson(home_expected)))
            away_goals = max(0, int(np.random.poisson(away_expected)))
            
            return home_goals, away_goals
    
    def calculate_team_stats(self, df, gameweek):
        """指定ゲームウィークまでのチーム統計を計算"""
        completed_matches = df[df['Game Week'] < gameweek]
        team_stats = {}
        
        teams = set(df['home_team_name'].unique()) | set(df['away_team_name'].unique())
        
        for team in teams:
            home_matches = completed_matches[completed_matches['home_team_name'] == team]
            away_matches = completed_matches[completed_matches['away_team_name'] == team]
            
            total_matches = len(home_matches) + len(away_matches)
            
            if total_matches == 0:
                team_stats[team] = {'ppg': 1.5, 'xg': 1.2}
                continue
            
            # 勝ち点計算
            points = 0
            goals_for = 0
            goals_against = 0
            
            for _, match in home_matches.iterrows():
                home_goals = match['home_team_goal_count']
                away_goals = match['away_team_goal_count']
                goals_for += home_goals
                goals_against += away_goals
                
                if home_goals > away_goals:
                    points += 3
                elif home_goals == away_goals:
                    points += 1
            
            for _, match in away_matches.iterrows():
                home_goals = match['home_team_goal_count']
                away_goals = match['away_team_goal_count']
                goals_for += away_goals
                goals_against += home_goals
                
                if away_goals > home_goals:
                    points += 3
                elif home_goals == away_goals:
                    points += 1
            
            ppg = points / total_matches if total_matches > 0 else 1.5
            avg_goals = goals_for / total_matches if total_matches > 0 else 1.2
            
            team_stats[team] = {'ppg': ppg, 'xg': avg_goals}
        
        return team_stats
    
    def simulate_season_from_gameweek(self, df, from_gameweek, n_simulations=1000):
        """指定ゲームウィークから最終順位をシミュレート"""
        print(f"第{from_gameweek}節からシーズン終了までをシミュレート...")
        
        # 既に完了している試合から現在の順位表を作成
        current_table = self.create_current_table(df, from_gameweek)
        print(f"\\n第{from_gameweek-1}節終了時点での順位:")
        print(current_table.head(10).to_string())
        
        # 残り試合を取得
        remaining_matches = df[df['Game Week'] >= from_gameweek].copy()
        print(f"\\n残り試合数: {len(remaining_matches)}試合")
        
        final_positions = defaultdict(list)
        final_points = defaultdict(list)
        
        for sim in range(n_simulations):
            if sim % 200 == 0:
                print(f"シミュレーション進捗: {sim}/{n_simulations}")
            
            # 現在の順位表をコピー
            sim_table = current_table.copy()
            
            # 各ゲームウィークをシミュレート
            for week in range(from_gameweek, 39):
                week_matches = remaining_matches[remaining_matches['Game Week'] == week]
                team_stats = self.calculate_team_stats(df, week)
                
                for _, match in week_matches.iterrows():
                    home_team = match['home_team_name']
                    away_team = match['away_team_name']
                    
                    home_goals, away_goals = self.simulate_match(
                        home_team, away_team, week, team_stats
                    )
                    
                    # 順位表を更新
                    self.update_table(sim_table, home_team, away_team, home_goals, away_goals)
            
            # 最終順位を記録
            final_table = sim_table.sort_values(['Pts', 'GD', 'GF'], ascending=[False, False, False])
            final_table = final_table.reset_index(drop=True)
            
            for idx, (_, row) in enumerate(final_table.iterrows()):
                final_positions[row['Team']].append(idx + 1)
                final_points[row['Team']].append(row['Pts'])
        
        return self.analyze_simulation_results(final_positions, final_points)
    
    def create_current_table(self, df, up_to_gameweek):
        """指定ゲームウィークまでの現在の順位表を作成"""
        completed_matches = df[df['Game Week'] < up_to_gameweek]
        teams = set(df['home_team_name'].unique()) | set(df['away_team_name'].unique())
        
        table = []
        for team in teams:
            home_matches = completed_matches[completed_matches['home_team_name'] == team]
            away_matches = completed_matches[completed_matches['away_team_name'] == team]
            
            games_played = len(home_matches) + len(away_matches)
            wins = draws = losses = 0
            goals_for = goals_against = 0
            
            # ホーム戦の結果
            for _, match in home_matches.iterrows():
                home_goals = match['home_team_goal_count']
                away_goals = match['away_team_goal_count']
                goals_for += home_goals
                goals_against += away_goals
                
                if home_goals > away_goals:
                    wins += 1
                elif home_goals == away_goals:
                    draws += 1
                else:
                    losses += 1
            
            # アウェイ戦の結果
            for _, match in away_matches.iterrows():
                home_goals = match['home_team_goal_count']
                away_goals = match['away_team_goal_count']
                goals_for += away_goals
                goals_against += home_goals
                
                if away_goals > home_goals:
                    wins += 1
                elif home_goals == away_goals:
                    draws += 1
                else:
                    losses += 1
            
            points = wins * 3 + draws
            goal_difference = goals_for - goals_against
            
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
        
        table_df = pd.DataFrame(table)
        return table_df.sort_values(['Pts', 'GD', 'GF'], ascending=[False, False, False])
    
    def update_table(self, table, home_team, away_team, home_goals, away_goals):
        """試合結果で順位表を更新"""
        # ホームチーム更新
        home_idx = table[table['Team'] == home_team].index[0]
        away_idx = table[table['Team'] == away_team].index[0]
        
        table.loc[home_idx, 'GP'] += 1
        table.loc[away_idx, 'GP'] += 1
        
        table.loc[home_idx, 'GF'] += home_goals
        table.loc[home_idx, 'GA'] += away_goals
        table.loc[away_idx, 'GF'] += away_goals
        table.loc[away_idx, 'GA'] += home_goals
        
        table.loc[home_idx, 'GD'] = table.loc[home_idx, 'GF'] - table.loc[home_idx, 'GA']
        table.loc[away_idx, 'GD'] = table.loc[away_idx, 'GF'] - table.loc[away_idx, 'GA']
        
        if home_goals > away_goals:  # ホーム勝利
            table.loc[home_idx, 'W'] += 1
            table.loc[home_idx, 'Pts'] += 3
            table.loc[away_idx, 'L'] += 1
        elif home_goals < away_goals:  # アウェイ勝利
            table.loc[away_idx, 'W'] += 1
            table.loc[away_idx, 'Pts'] += 3
            table.loc[home_idx, 'L'] += 1
        else:  # ドロー
            table.loc[home_idx, 'D'] += 1
            table.loc[home_idx, 'Pts'] += 1
            table.loc[away_idx, 'D'] += 1
            table.loc[away_idx, 'Pts'] += 1
    
    def analyze_simulation_results(self, final_positions, final_points):
        """シミュレーション結果を分析"""
        results = []
        
        for team in final_positions.keys():
            positions = final_positions[team]
            points = final_points[team]
            
            avg_position = np.mean(positions)
            avg_points = np.mean(points)
            
            # 順位分布
            top4_prob = sum(1 for pos in positions if pos <= 4) / len(positions)
            top6_prob = sum(1 for pos in positions if pos <= 6) / len(positions)
            relegation_prob = sum(1 for pos in positions if pos >= 18) / len(positions)
            
            results.append({
                'Team': team,
                'Avg_Position': round(avg_position, 2),
                'Avg_Points': round(avg_points, 1),
                'Top4_Prob': round(top4_prob * 100, 1),
                'Top6_Prob': round(top6_prob * 100, 1),
                'Relegation_Prob': round(relegation_prob * 100, 1)
            })
        
        results_df = pd.DataFrame(results)
        return results_df.sort_values('Avg_Position')

# テスト実行
if __name__ == "__main__":
    from basic_predictor import PremierLeaguePredictor
    
    # 基本予測モデルを初期化
    predictor = PremierLeaguePredictor()
    predictor.load_data('stats-csv/england-premier-league-matches-2018-to-2019-stats.csv')
    predictor.train_models()
    
    # 順位予測シミュレーター初期化
    position_predictor = LeaguePositionPredictor(predictor)
    
    # 第20節終了時点からシーズン最終順位を予測
    results = position_predictor.simulate_season_from_gameweek(
        predictor.df, from_gameweek=20, n_simulations=500
    )
    
    print(f"\\n=== 最終順位予測結果 ===")
    print(results.to_string(index=False))
    
    print(f"\\n=== 主要な確率 ===")
    top_teams = results.head(6)
    bottom_teams = results.tail(4)
    
    print(f"\\nチャンピオンズリーグ出場権争い:")
    for _, team in top_teams.iterrows():
        print(f"{team['Team']:20s}: Top4確率 {team['Top4_Prob']:5.1f}%, 平均順位 {team['Avg_Position']:4.1f}位")
    
    print(f"\\n降格圏争い:")
    for _, team in bottom_teams.iterrows():
        print(f"{team['Team']:20s}: 降格確率 {team['Relegation_Prob']:5.1f}%, 平均順位 {team['Avg_Position']:4.1f}位")
