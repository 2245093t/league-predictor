"""
Final Standings Prediction System
最終順位・勝ち点予測システム
"""

import pandas as pd
import numpy as np
import torch
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 相対インポート
try:
    from ..training.model_architecture import FootballMatchPredictor, FeatureExtractor
    from ..utils.data_preprocessing import TeamStatsCalculator
    from .predict_matches import WeeklyMatchPredictor
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from training.model_architecture import FootballMatchPredictor, FeatureExtractor
    from utils.data_preprocessing import TeamStatsCalculator
    from prediction.predict_matches import WeeklyMatchPredictor


class FinalStandingsPredictor:
    """
    最終順位・勝ち点予測システム
    モンテカルロシミュレーション対応
    """
    
    def __init__(self, model_path: str, team_encoder_path: str):
        """
        Args:
            model_path: 学習済みモデルのパス
            team_encoder_path: チームエンコーダーのパス
        """
        
        self.match_predictor = WeeklyMatchPredictor(model_path, team_encoder_path)
        self.team_decoder = {v: k for k, v in self.match_predictor.team_encoder.items()}
        
        print("Final Standings Predictor initialized")
        
    def calculate_current_standings(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        現在の順位表を計算
        
        Args:
            results_df: 完了済み試合結果
            
        Returns:
            standings_df: 順位表DataFrame
        """
        
        teams = set(results_df['home_team_name'].unique()) | set(results_df['away_team_name'].unique())
        
        standings = []
        
        for team in teams:
            # ホーム戦績
            home_matches = results_df[results_df['home_team_name'] == team]
            home_wins = len(home_matches[home_matches['home_team_goal_count'] > home_matches['away_team_goal_count']])
            home_draws = len(home_matches[home_matches['home_team_goal_count'] == home_matches['away_team_goal_count']])
            home_losses = len(home_matches[home_matches['home_team_goal_count'] < home_matches['away_team_goal_count']])
            home_goals_for = home_matches['home_team_goal_count'].sum()
            home_goals_against = home_matches['away_team_goal_count'].sum()
            
            # アウェー戦績
            away_matches = results_df[results_df['away_team_name'] == team]
            away_wins = len(away_matches[away_matches['away_team_goal_count'] > away_matches['home_team_goal_count']])
            away_draws = len(away_matches[away_matches['away_team_goal_count'] == away_matches['home_team_goal_count']])
            away_losses = len(away_matches[away_matches['away_team_goal_count'] < away_matches['home_team_goal_count']])
            away_goals_for = away_matches['away_team_goal_count'].sum()
            away_goals_against = away_matches['home_team_goal_count'].sum()
            
            # 合計
            matches_played = len(home_matches) + len(away_matches)
            wins = home_wins + away_wins
            draws = home_draws + away_draws
            losses = home_losses + away_losses
            goals_for = home_goals_for + away_goals_for
            goals_against = home_goals_against + away_goals_against
            goal_difference = goals_for - goals_against
            points = wins * 3 + draws
            
            standings.append({
                'team': team,
                'matches_played': matches_played,
                'wins': wins,
                'draws': draws,
                'losses': losses,
                'goals_for': goals_for,
                'goals_against': goals_against,
                'goal_difference': goal_difference,
                'points': points
            })
            
        standings_df = pd.DataFrame(standings)
        standings_df = standings_df.sort_values(['points', 'goal_difference', 'goals_for'], ascending=[False, False, False])
        standings_df.reset_index(drop=True, inplace=True)
        standings_df['position'] = standings_df.index + 1
        
        return standings_df
        
    def simulate_remaining_matches(self, fixtures_df: pd.DataFrame, results_df: pd.DataFrame,
                                 num_simulations: int = 1000) -> Dict[str, Dict]:
        """
        残り試合をシミュレーション
        
        Args:
            fixtures_df: 全試合予定
            results_df: 完了済み結果
            num_simulations: シミュレーション回数
            
        Returns:
            simulation_results: シミュレーション結果
        """
        
        print(f"Running {num_simulations} simulations...")
        
        # 残り試合抽出
        completed_matches = set()
        for _, match in results_df.iterrows():
            key = (match['home_team_name'], match['away_team_name'], match['Game Week'])
            completed_matches.add(key)
            
        remaining_fixtures = []
        for _, fixture in fixtures_df.iterrows():
            key = (fixture['home_team_name'], fixture['away_team_name'], fixture['Game Week'])
            if key not in completed_matches:
                remaining_fixtures.append(fixture)
                
        print(f"Remaining matches: {len(remaining_fixtures)}")
        
        if len(remaining_fixtures) == 0:
            print("No remaining matches. Using current standings.")
            return self._single_simulation_result(results_df, [])
            
        # 現在の順位表
        current_standings = self.calculate_current_standings(results_df)
        teams = current_standings['team'].tolist()
        
        # シミュレーション結果格納
        final_positions = {team: [] for team in teams}
        final_points = {team: [] for team in teams}
        
        for sim in range(num_simulations):
            if sim % 100 == 0:
                print(f"Simulation {sim}/{num_simulations}")
                
            # 残り試合を予測
            simulated_results = self._simulate_remaining_season(remaining_fixtures, results_df)
            
            # 全試合結果を合成
            all_results = pd.concat([results_df, simulated_results], ignore_index=True)
            
            # 最終順位表計算
            final_standings = self.calculate_current_standings(all_results)
            
            # 結果記録
            for _, row in final_standings.iterrows():
                team = row['team']
                final_positions[team].append(row['position'])
                final_points[team].append(row['points'])
                
        # 統計計算
        simulation_results = {}
        for team in teams:
            positions = final_positions[team]
            points = final_points[team]
            
            simulation_results[team] = {
                'current_position': int(current_standings[current_standings['team'] == team]['position'].iloc[0]),
                'current_points': int(current_standings[current_standings['team'] == team]['points'].iloc[0]),
                'avg_final_position': np.mean(positions),
                'std_final_position': np.std(positions),
                'avg_final_points': np.mean(points),
                'std_final_points': np.std(points),
                'min_position': min(positions),
                'max_position': max(positions),
                'min_points': min(points),
                'max_points': max(points),
                'top_4_probability': sum(1 for p in positions if p <= 4) / num_simulations,
                'top_6_probability': sum(1 for p in positions if p <= 6) / num_simulations,
                'relegation_probability': sum(1 for p in positions if p >= len(teams) - 2) / num_simulations,
                'position_probabilities': self._calculate_position_probabilities(positions, len(teams))
            }
            
        return simulation_results
        
    def _simulate_remaining_season(self, remaining_fixtures: List, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        残りシーズンの1回分シミュレーション
        
        Args:
            remaining_fixtures: 残り試合リスト
            results_df: 完了済み結果
            
        Returns:
            simulated_results: シミュレーション結果DataFrame
        """
        
        simulated_results = []
        current_results = results_df.copy()
        
        # ゲームウィーク順にソート
        remaining_fixtures_sorted = sorted(remaining_fixtures, key=lambda x: x['Game Week'])
        
        for fixture in remaining_fixtures_sorted:
            gameweek = fixture['Game Week']
            
            # 現在時点での統計計算
            team_stats = self.match_predictor.calculate_current_stats(current_results, gameweek)
            
            # 試合予測
            try:
                prediction = self.match_predictor._predict_single_match(fixture, team_stats)
                
                # 予測結果をサンプリング
                home_goals, away_goals = self._sample_match_result(prediction)
                
                # 結果をDataFrameに追加
                match_result = {
                    'Game Week': gameweek,
                    'home_team_name': fixture['home_team_name'],
                    'away_team_name': fixture['away_team_name'],
                    'home_team_goal_count': home_goals,
                    'away_team_goal_count': away_goals
                }
                
                simulated_results.append(match_result)
                
                # 現在結果に追加（次の予測で使用）
                current_results = pd.concat([current_results, pd.DataFrame([match_result])], ignore_index=True)
                
            except Exception as e:
                print(f"Error simulating match {fixture['home_team_name']} vs {fixture['away_team_name']}: {e}")
                # エラーの場合はドロー
                match_result = {
                    'Game Week': gameweek,
                    'home_team_name': fixture['home_team_name'],
                    'away_team_name': fixture['away_team_name'],
                    'home_team_goal_count': 1,
                    'away_team_goal_count': 1
                }
                simulated_results.append(match_result)
                current_results = pd.concat([current_results, pd.DataFrame([match_result])], ignore_index=True)
                
        return pd.DataFrame(simulated_results)
        
    def _sample_match_result(self, prediction: Dict) -> Tuple[int, int]:
        """
        予測結果から実際のスコアをサンプリング
        
        Args:
            prediction: 予測結果辞書
            
        Returns:
            home_goals, away_goals: サンプリングされたゴール数
        """
        
        # ポアソン分布からサンプリング
        home_goals = np.random.poisson(max(0, prediction['predicted_home_goals']))
        away_goals = np.random.poisson(max(0, prediction['predicted_away_goals']))
        
        return int(home_goals), int(away_goals)
        
    def _calculate_position_probabilities(self, positions: List[int], num_teams: int) -> Dict[int, float]:
        """
        各順位の確率を計算
        
        Args:
            positions: シミュレーション順位リスト
            num_teams: チーム数
            
        Returns:
            position_probs: 各順位の確率辞書
        """
        
        position_probs = {}
        for pos in range(1, num_teams + 1):
            position_probs[pos] = sum(1 for p in positions if p == pos) / len(positions)
            
        return position_probs
        
    def _single_simulation_result(self, results_df: pd.DataFrame, simulated_results: List) -> Dict[str, Dict]:
        """
        シミュレーション無しの場合の結果作成
        
        Args:
            results_df: 現在の結果
            simulated_results: 空リスト
            
        Returns:
            results: 結果辞書
        """
        
        current_standings = self.calculate_current_standings(results_df)
        results = {}
        
        for _, row in current_standings.iterrows():
            team = row['team']
            results[team] = {
                'current_position': int(row['position']),
                'current_points': int(row['points']),
                'avg_final_position': float(row['position']),
                'std_final_position': 0.0,
                'avg_final_points': float(row['points']),
                'std_final_points': 0.0,
                'min_position': int(row['position']),
                'max_position': int(row['position']),
                'min_points': int(row['points']),
                'max_points': int(row['points']),
                'top_4_probability': 1.0 if row['position'] <= 4 else 0.0,
                'top_6_probability': 1.0 if row['position'] <= 6 else 0.0,
                'relegation_probability': 1.0 if row['position'] >= len(current_standings) - 2 else 0.0,
                'position_probabilities': {int(row['position']): 1.0}
            }
            
        return results
        
    def export_predictions(self, simulation_results: Dict[str, Dict], 
                         current_standings: pd.DataFrame, output_file: str):
        """
        予測結果をCSVにエクスポート
        
        Args:
            simulation_results: シミュレーション結果
            current_standings: 現在の順位表
            output_file: 出力ファイルパス
        """
        
        export_data = []
        
        for team, results in simulation_results.items():
            row = {
                'team': team,
                'current_position': results['current_position'],
                'current_points': results['current_points'],
                'predicted_final_position': round(results['avg_final_position'], 1),
                'position_std': round(results['std_final_position'], 1),
                'predicted_final_points': round(results['avg_final_points'], 1),
                'points_std': round(results['std_final_points'], 1),
                'best_case_position': results['min_position'],
                'worst_case_position': results['max_position'],
                'max_possible_points': results['max_points'],
                'min_possible_points': results['min_points'],
                'top_4_probability': round(results['top_4_probability'] * 100, 1),
                'top_6_probability': round(results['top_6_probability'] * 100, 1),
                'relegation_probability': round(results['relegation_probability'] * 100, 1)
            }
            export_data.append(row)
            
        df = pd.DataFrame(export_data)
        df = df.sort_values('predicted_final_position')
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"Final standings predictions exported to {output_file}")
        
    def print_prediction_summary(self, simulation_results: Dict[str, Dict]):
        """
        予測結果のサマリーを表示
        
        Args:
            simulation_results: シミュレーション結果
        """
        
        print(f"\\n{'='*80}")
        print("FINAL STANDINGS PREDICTION SUMMARY")
        print(f"{'='*80}")
        
        # 順位順にソート
        sorted_teams = sorted(simulation_results.items(), key=lambda x: x[1]['avg_final_position'])
        
        print(f"{'Pos':>3} {'Team':20} {'Current':>8} {'Predicted':>10} {'Range':>15} {'Top4%':>7} {'Rel%':>6}")
        print("-" * 80)
        
        for team, results in sorted_teams:
            pos_range = f"{results['min_position']}-{results['max_position']}"
            print(f"{results['current_position']:>3} {team:20} "
                  f"{results['current_points']:>8} "
                  f"{results['avg_final_points']:>7.1f}±{results['std_final_points']:>2.1f} "
                  f"{pos_range:>15} "
                  f"{results['top_4_probability']*100:>6.1f}% "
                  f"{results['relegation_probability']*100:>5.1f}%")
                  
        print("-" * 80)
        print("Legend: Current=Current points, Predicted=Predicted final points±std")
        print("        Top4%=Top 4 probability, Rel%=Relegation probability")


def predict_final_standings(fixture_file: str, results_file: str = None,
                          num_simulations: int = 1000,
                          model_path: str = "models/saved/best_model.pth",
                          team_encoder_path: str = "models/saved/team_encoder.json") -> Dict[str, Dict]:
    """
    エントリーポイント関数：最終順位予測
    
    Args:
        fixture_file: 試合予定ファイル
        results_file: 結果ファイル（オプション）
        num_simulations: シミュレーション回数
        model_path: モデルファイルパス
        team_encoder_path: チームエンコーダーパス
        
    Returns:
        simulation_results: シミュレーション結果
    """
    
    # 予測器初期化
    predictor = FinalStandingsPredictor(model_path, team_encoder_path)
    
    # データ読み込み
    fixtures_df, results_df = predictor.match_predictor.load_season_data(fixture_file, results_file)
    
    # 現在の順位表
    current_standings = predictor.calculate_current_standings(results_df)
    print(f"\\nCurrent standings after {len(results_df)} matches:")
    print(current_standings[['position', 'team', 'matches_played', 'points', 'goal_difference']].to_string(index=False))
    
    # シミュレーション実行
    simulation_results = predictor.simulate_remaining_matches(fixtures_df, results_df, num_simulations)
    
    # 結果表示
    predictor.print_prediction_summary(simulation_results)
    
    return simulation_results


if __name__ == "__main__":
    # テスト実行
    import argparse
    
    parser = argparse.ArgumentParser(description='Football Final Standings Predictor')
    parser.add_argument('--fixtures', required=True, help='Fixtures CSV file')
    parser.add_argument('--results', help='Results CSV file (optional)')
    parser.add_argument('--simulations', type=int, default=1000, help='Number of simulations')
    parser.add_argument('--model', default='models/saved/best_model.pth', help='Model file path')
    parser.add_argument('--encoder', default='models/saved/team_encoder.json', help='Team encoder file path')
    parser.add_argument('--output', help='Output CSV file (optional)')
    
    args = parser.parse_args()
    
    try:
        # 予測実行
        simulation_results = predict_final_standings(
            fixture_file=args.fixtures,
            results_file=args.results,
            num_simulations=args.simulations,
            model_path=args.model,
            team_encoder_path=args.encoder
        )
        
        # 結果出力
        if args.output:
            predictor = FinalStandingsPredictor(args.model, args.encoder)
            fixtures_df, results_df = predictor.match_predictor.load_season_data(args.fixtures, args.results)
            current_standings = predictor.calculate_current_standings(results_df)
            predictor.export_predictions(simulation_results, current_standings, args.output)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
