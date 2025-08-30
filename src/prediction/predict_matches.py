"""
Weekly Match Prediction System
週次の試合予測を行うメインスクリプト
"""

import pandas as pd
import numpy as np
import torch
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 相対インポート
try:
    from ..training.model_architecture import FootballMatchPredictor, FeatureExtractor
    from ..utils.data_preprocessing import TeamStatsCalculator
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from training.model_architecture import FootballMatchPredictor, FeatureExtractor
    from utils.data_preprocessing import TeamStatsCalculator


class WeeklyMatchPredictor:
    """
    週次試合予測システム
    リアルタイムでの試合結果更新に対応
    """
    
    def __init__(self, model_path: str, team_encoder_path: str):
        """
        Args:
            model_path: 学習済みモデルのパス
            team_encoder_path: チームエンコーダーのパス
        """
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # モデル読み込み
        print(f"Loading model from {model_path}...")
        self.model = FootballMatchPredictor.load_model(model_path, self.device)
        self.model.eval()
        
        # チームエンコーダー読み込み
        with open(team_encoder_path, 'r', encoding='utf-8') as f:
            self.team_encoder = json.load(f)
            
        # デコーダー作成
        self.team_decoder = {v: k for k, v in self.team_encoder.items()}
        
        # 特徴量抽出器
        self.feature_extractor = FeatureExtractor()
        
        # チーム統計計算器
        self.stats_calculator = TeamStatsCalculator()
        
        print(f"Model loaded successfully. Device: {self.device}")
        print(f"Teams in encoder: {len(self.team_encoder)}")
        
    def load_season_data(self, fixture_file: str, results_file: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        シーズンデータ読み込み
        
        Args:
            fixture_file: 全試合予定ファイル
            results_file: 結果更新ファイル（オプション）
            
        Returns:
            fixtures_df: 試合予定データ
            results_df: 結果データ
        """
        
        print(f"Loading season data from {fixture_file}...")
        
        # 試合予定読み込み
        fixtures_df = pd.read_csv(fixture_file)
        
        # 結果データ読み込み（あれば）
        if results_file and os.path.exists(results_file):
            results_df = pd.read_csv(results_file)
            # statusカラムがある場合は、'complete'のみを対象とする
            if 'status' in results_df.columns:
                before_count = len(results_df)
                results_df = results_df[results_df['status'] == 'complete'].copy()
                print(f"Results filtered by status: {before_count} -> {len(results_df)} matches")
            else:
                print(f"Results loaded: {len(results_df)} completed matches")
        else:
            # 結果データがない場合は、試合予定から完了済みを抽出
            if 'status' in fixtures_df.columns:
                # statusカラムがある場合
                completed_mask = fixtures_df['status'] == 'complete'
            else:
                # 従来の方法（ゴール数による判定）
                completed_mask = (
                    fixtures_df['home_team_goal_count'].notna() & 
                    fixtures_df['away_team_goal_count'].notna()
                )
            results_df = fixtures_df[completed_mask].copy()
            print(f"Completed matches from fixtures: {len(results_df)}")
            
        print(f"Total fixtures: {len(fixtures_df)}")
        
        return fixtures_df, results_df
        
    def calculate_current_stats(self, results_df: pd.DataFrame, current_gameweek: int) -> Dict:
        """
        現在のゲームウィークまでのチーム統計を計算
        
        Args:
            results_df: 完了済み試合結果
            current_gameweek: 現在のゲームウィーク
            
        Returns:
            team_stats: チーム統計辞書
        """
        
        print(f"Calculating team stats up to gameweek {current_gameweek}...")
        
        # 指定ゲームウィークまでの結果を抽出
        completed_results = results_df[results_df['Game Week'] < current_gameweek].copy()
        
        return self.stats_calculator.calculate_team_stats(completed_results)
        
    def predict_weekly_matches(self, fixtures_df: pd.DataFrame, team_stats: Dict, 
                             target_gameweek: int) -> List[Dict]:
        """
        指定ゲームウィークの試合を予測
        
        Args:
            fixtures_df: 試合予定データ
            team_stats: チーム統計
            target_gameweek: 予測対象ゲームウィーク
            
        Returns:
            predictions: 予測結果リスト
        """
        
        print(f"Predicting matches for gameweek {target_gameweek}...")
        
        # 対象ゲームウィークの試合を抽出
        week_matches = fixtures_df[fixtures_df['Game Week'] == target_gameweek].copy()
        
        if len(week_matches) == 0:
            print(f"No matches found for gameweek {target_gameweek}")
            return []
            
        predictions = []
        
        for _, match in week_matches.iterrows():
            try:
                prediction = self._predict_single_match(match, team_stats)
                predictions.append(prediction)
                
                print(f"  {prediction['home_team']} vs {prediction['away_team']}: "
                      f"{prediction['predicted_score']} ({prediction['predicted_result']})")
                      
            except Exception as e:
                print(f"Error predicting match {match['home_team_name']} vs {match['away_team_name']}: {e}")
                continue
                
        return predictions
        
    def _predict_single_match(self, match_data: pd.Series, team_stats: Dict) -> Dict:
        """
        単一試合の予測
        
        Args:
            match_data: 試合データ
            team_stats: チーム統計
            
        Returns:
            prediction: 予測結果辞書
        """
        
        home_team = match_data['home_team_name']
        away_team = match_data['away_team_name']
        
        # チームIDを取得
        home_team_id = self.team_encoder.get(home_team)
        away_team_id = self.team_encoder.get(away_team)
        
        if home_team_id is None or away_team_id is None:
            raise ValueError(f"Team not found in encoder: {home_team} or {away_team}")
            
        # 特徴量抽出
        match_info = {
            'home_team': home_team,
            'away_team': away_team,
            'game_week': match_data.get('Game Week', 1)
        }
        
        features = self.feature_extractor.extract_features(match_info, team_stats)
        
        # モデル予測
        prediction = self.model.predict_match(home_team_id, away_team_id, features)
        
        # 結果整理
        result = {
            'gameweek': match_info['game_week'],
            'home_team': home_team,
            'away_team': away_team,
            'predicted_home_goals': round(prediction['home_goals'], 1),
            'predicted_away_goals': round(prediction['away_goals'], 1),
            'predicted_score': f"{prediction['home_goals']:.1f} - {prediction['away_goals']:.1f}",
            'home_win_prob': round(prediction['home_win_prob'] * 100, 1),
            'draw_prob': round(prediction['draw_prob'] * 100, 1),
            'away_win_prob': round(prediction['away_win_prob'] * 100, 1),
            'predicted_result': prediction['predicted_result'],
            'confidence': max(prediction['home_win_prob'], prediction['draw_prob'], prediction['away_win_prob'])
        }
        
        return result
        
    def predict_multiple_gameweeks(self, fixtures_df: pd.DataFrame, results_df: pd.DataFrame,
                                 start_gameweek: int, end_gameweek: int) -> Dict[int, List[Dict]]:
        """
        複数ゲームウィークの予測
        
        Args:
            fixtures_df: 試合予定データ
            results_df: 完了済み結果データ
            start_gameweek: 開始ゲームウィーク
            end_gameweek: 終了ゲームウィーク
            
        Returns:
            all_predictions: ゲームウィーク別予測結果
        """
        
        all_predictions = {}
        
        for gameweek in range(start_gameweek, end_gameweek + 1):
            # 各ゲームウィーク時点での統計を計算
            team_stats = self.calculate_current_stats(results_df, gameweek)
            
            # 予測実行
            predictions = self.predict_weekly_matches(fixtures_df, team_stats, gameweek)
            all_predictions[gameweek] = predictions
            
        return all_predictions
        
    def export_predictions(self, predictions: Dict[int, List[Dict]], output_file: str):
        """
        予測結果をCSVにエクスポート
        
        Args:
            predictions: 予測結果
            output_file: 出力ファイルパス
        """
        
        all_predictions = []
        
        for gameweek, week_predictions in predictions.items():
            for pred in week_predictions:
                pred['gameweek'] = gameweek
                all_predictions.append(pred)
                
        df = pd.DataFrame(all_predictions)
        df.to_csv(output_file, index=False, encoding='utf-8')
        
        print(f"Predictions exported to {output_file}")
        
    def get_prediction_summary(self, predictions: List[Dict]) -> Dict:
        """
        予測結果のサマリー統計
        
        Args:
            predictions: 予測結果リスト
            
        Returns:
            summary: サマリー辞書
        """
        
        if not predictions:
            return {}
            
        home_wins = sum(1 for p in predictions if p['predicted_result'] == 'Home Win')
        draws = sum(1 for p in predictions if p['predicted_result'] == 'Draw')
        away_wins = sum(1 for p in predictions if p['predicted_result'] == 'Away Win')
        
        avg_goals = np.mean([p['predicted_home_goals'] + p['predicted_away_goals'] for p in predictions])
        avg_confidence = np.mean([p['confidence'] for p in predictions])
        
        return {
            'total_matches': len(predictions),
            'home_wins': home_wins,
            'draws': draws, 
            'away_wins': away_wins,
            'home_win_rate': home_wins / len(predictions),
            'draw_rate': draws / len(predictions),
            'away_win_rate': away_wins / len(predictions),
            'average_goals_per_match': round(avg_goals, 2),
            'average_confidence': round(avg_confidence, 3)
        }


def predict_weekly_matches(fixture_file: str, results_file: str = None, 
                         target_gameweek: int = None,
                         model_path: str = "models/saved/best_model.pth",
                         team_encoder_path: str = "models/saved/team_encoder.json") -> List[Dict]:
    """
    エントリーポイント関数：週次試合予測
    
    Args:
        fixture_file: 試合予定ファイル
        results_file: 結果ファイル（オプション）
        target_gameweek: 予測対象ゲームウィーク（Noneの場合は次週）
        model_path: モデルファイルパス
        team_encoder_path: チームエンコーダーパス
        
    Returns:
        predictions: 予測結果リスト
    """
    
    # 予測器初期化
    predictor = WeeklyMatchPredictor(model_path, team_encoder_path)
    
    # データ読み込み
    fixtures_df, results_df = predictor.load_season_data(fixture_file, results_file)
    
    # 対象ゲームウィーク決定
    if target_gameweek is None:
        completed_gameweeks = results_df['Game Week'].max() if len(results_df) > 0 else 0
        target_gameweek = completed_gameweeks + 1
        
    print(f"Predicting for gameweek {target_gameweek}")
    
    # チーム統計計算
    team_stats = predictor.calculate_current_stats(results_df, target_gameweek)
    
    # 予測実行
    predictions = predictor.predict_weekly_matches(fixtures_df, team_stats, target_gameweek)
    
    # サマリー表示
    summary = predictor.get_prediction_summary(predictions)
    print(f"\\nPrediction Summary:")
    print(f"  Total matches: {summary.get('total_matches', 0)}")
    print(f"  Expected results: {summary.get('home_wins', 0)}H - {summary.get('draws', 0)}D - {summary.get('away_wins', 0)}A")
    print(f"  Average goals per match: {summary.get('average_goals_per_match', 0)}")
    print(f"  Average confidence: {summary.get('average_confidence', 0):.1%}")
    
    return predictions


if __name__ == "__main__":
    # テスト実行
    import argparse
    
    parser = argparse.ArgumentParser(description='Football Weekly Match Predictor')
    parser.add_argument('--fixtures', required=True, help='Fixtures CSV file')
    parser.add_argument('--results', help='Results CSV file (optional)')
    parser.add_argument('--gameweek', type=int, help='Target gameweek (optional)')
    parser.add_argument('--model', default='models/saved/best_model.pth', help='Model file path')
    parser.add_argument('--encoder', default='models/saved/team_encoder.json', help='Team encoder file path')
    parser.add_argument('--output', help='Output CSV file (optional)')
    
    args = parser.parse_args()
    
    try:
        # 予測実行
        predictions = predict_weekly_matches(
            fixture_file=args.fixtures,
            results_file=args.results,
            target_gameweek=args.gameweek,
            model_path=args.model,
            team_encoder_path=args.encoder
        )
        
        # 結果出力
        if args.output:
            df = pd.DataFrame(predictions)
            df.to_csv(args.output, index=False, encoding='utf-8')
            print(f"Results saved to {args.output}")
        
        # コンソール出力
        print(f"\\n{'='*60}")
        print("WEEKLY MATCH PREDICTIONS")
        print(f"{'='*60}")
        
        for pred in predictions:
            print(f"{pred['home_team']:20s} vs {pred['away_team']:20s}")
            print(f"  Score: {pred['predicted_score']:>10s}")
            print(f"  Probabilities: {pred['home_win_prob']:>5.1f}% - {pred['draw_prob']:>5.1f}% - {pred['away_win_prob']:>5.1f}%")
            print(f"  Result: {pred['predicted_result']}")
            print()
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
