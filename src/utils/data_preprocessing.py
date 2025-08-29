"""
Data Preprocessing Utilities
データ前処理・統計計算ユーティリティ
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class TeamStatsCalculator:
    """
    チーム統計計算クラス
    PPG, xG, xGA, 最近の成績など
    """
    
    def __init__(self):
        self.recent_matches = 5  # 最近の試合数
        
    def calculate_team_stats(self, results_df: pd.DataFrame, gameweek: int = None) -> Dict:
        """
        チーム統計を計算
        
        Args:
            results_df: 試合結果DataFrame
            gameweek: 指定ゲームウィーク（Noneの場合は全期間）
            
        Returns:
            team_stats: チーム統計辞書
        """
        
        if gameweek is not None:
            # 指定ゲームウィークまでの結果
            filtered_df = results_df[results_df['Game Week'] < gameweek].copy()
        else:
            filtered_df = results_df.copy()
            
        if len(filtered_df) == 0:
            return {}
            
        # 全チームリスト
        teams = set(filtered_df['home_team_name'].unique()) | set(filtered_df['away_team_name'].unique())
        
        team_stats = {}
        
        for team in teams:
            stats = self._calculate_single_team_stats(team, filtered_df)
            team_stats[team] = stats
            
        return team_stats
        
    def _calculate_single_team_stats(self, team: str, results_df: pd.DataFrame) -> Dict:
        """
        単一チームの統計計算
        
        Args:
            team: チーム名
            results_df: 試合結果DataFrame
            
        Returns:
            stats: チーム統計辞書
        """
        
        # ホーム試合
        home_matches = results_df[results_df['home_team_name'] == team].copy()
        
        # アウェー試合
        away_matches = results_df[results_df['away_team_name'] == team].copy()
        
        # 全試合
        total_matches = len(home_matches) + len(away_matches)
        
        if total_matches == 0:
            return self._empty_stats()
            
        # 基本統計
        stats = {
            'matches_played': total_matches,
            'ppg': 0.0,  # Points per game
            'goals_for_per_game': 0.0,
            'goals_against_per_game': 0.0,
            'goal_difference_per_game': 0.0,
            'xg_for_per_game': 0.0,
            'xg_against_per_game': 0.0,
            'xg_difference_per_game': 0.0,
            'win_rate': 0.0,
            'draw_rate': 0.0,
            'loss_rate': 0.0,
            'home_advantage': 0.0,
            'recent_form': 0.0,  # 最近5試合の勝ち点率
            'recent_goals_for': 0.0,
            'recent_goals_against': 0.0
        }
        
        # ホーム成績
        home_stats = self._calculate_match_stats(home_matches, team, 'home')
        
        # アウェー成績
        away_stats = self._calculate_match_stats(away_matches, team, 'away')
        
        # 全体統計
        total_points = home_stats['points'] + away_stats['points']
        total_goals_for = home_stats['goals_for'] + away_stats['goals_for']
        total_goals_against = home_stats['goals_against'] + away_stats['goals_against']
        total_xg_for = home_stats['xg_for'] + away_stats['xg_for']
        total_xg_against = home_stats['xg_against'] + away_stats['xg_against']
        
        total_wins = home_stats['wins'] + away_stats['wins']
        total_draws = home_stats['draws'] + away_stats['draws']
        total_losses = home_stats['losses'] + away_stats['losses']
        
        # PPG計算
        stats['ppg'] = total_points / total_matches
        
        # ゴール統計
        stats['goals_for_per_game'] = total_goals_for / total_matches
        stats['goals_against_per_game'] = total_goals_against / total_matches
        stats['goal_difference_per_game'] = stats['goals_for_per_game'] - stats['goals_against_per_game']
        
        # xG統計
        stats['xg_for_per_game'] = total_xg_for / total_matches
        stats['xg_against_per_game'] = total_xg_against / total_matches
        stats['xg_difference_per_game'] = stats['xg_for_per_game'] - stats['xg_against_per_game']
        
        # 勝敗率
        stats['win_rate'] = total_wins / total_matches
        stats['draw_rate'] = total_draws / total_matches
        stats['loss_rate'] = total_losses / total_matches
        
        # ホームアドバンテージ
        if len(home_matches) > 0 and len(away_matches) > 0:
            home_ppg = home_stats['points'] / len(home_matches)
            away_ppg = away_stats['points'] / len(away_matches)
            stats['home_advantage'] = home_ppg - away_ppg
        
        # 最近の成績
        recent_stats = self._calculate_recent_form(team, results_df)
        stats['recent_form'] = recent_stats['recent_ppg']
        stats['recent_goals_for'] = recent_stats['recent_goals_for']
        stats['recent_goals_against'] = recent_stats['recent_goals_against']
        
        return stats
        
    def _calculate_match_stats(self, matches_df: pd.DataFrame, team: str, venue: str) -> Dict:
        """
        試合統計計算（ホームまたはアウェー）
        
        Args:
            matches_df: 試合DataFrame
            team: チーム名
            venue: 'home' または 'away'
            
        Returns:
            stats: 統計辞書
        """
        
        if len(matches_df) == 0:
            return {
                'matches': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'points': 0,
                'goals_for': 0, 'goals_against': 0, 'xg_for': 0.0, 'xg_against': 0.0
            }
            
        if venue == 'home':
            goals_for_col = 'home_team_goal_count'
            goals_against_col = 'away_team_goal_count'
            xg_for_col = 'home_team_goal_count_expected'
            xg_against_col = 'away_team_goal_count_expected'
        else:
            goals_for_col = 'away_team_goal_count'
            goals_against_col = 'home_team_goal_count'
            xg_for_col = 'away_team_goal_count_expected'
            xg_against_col = 'home_team_goal_count_expected'
            
        # ゴール計算
        goals_for = matches_df[goals_for_col].sum() if goals_for_col in matches_df.columns else 0
        goals_against = matches_df[goals_against_col].sum() if goals_against_col in matches_df.columns else 0
        
        # xG計算（データがない場合はゴール数で代用）
        if xg_for_col in matches_df.columns and matches_df[xg_for_col].notna().any():
            xg_for = matches_df[xg_for_col].fillna(matches_df[goals_for_col]).sum()
            xg_against = matches_df[xg_against_col].fillna(matches_df[goals_against_col]).sum()
        else:
            xg_for = float(goals_for)
            xg_against = float(goals_against)
            
        # 勝敗計算
        wins = len(matches_df[matches_df[goals_for_col] > matches_df[goals_against_col]])
        draws = len(matches_df[matches_df[goals_for_col] == matches_df[goals_against_col]])
        losses = len(matches_df[matches_df[goals_for_col] < matches_df[goals_against_col]])
        
        points = wins * 3 + draws
        
        return {
            'matches': len(matches_df),
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'points': points,
            'goals_for': goals_for,
            'goals_against': goals_against,
            'xg_for': xg_for,
            'xg_against': xg_against
        }
        
    def _calculate_recent_form(self, team: str, results_df: pd.DataFrame) -> Dict:
        """
        最近の成績計算
        
        Args:
            team: チーム名
            results_df: 試合結果DataFrame
            
        Returns:
            recent_stats: 最近の統計
        """
        
        # チームの全試合を時系列順で取得
        team_matches = []
        
        for _, match in results_df.iterrows():
            if match['home_team_name'] == team:
                match_info = {
                    'game_week': match['Game Week'],
                    'venue': 'home',
                    'goals_for': match['home_team_goal_count'],
                    'goals_against': match['away_team_goal_count']
                }
            elif match['away_team_name'] == team:
                match_info = {
                    'game_week': match['Game Week'],
                    'venue': 'away',
                    'goals_for': match['away_team_goal_count'],
                    'goals_against': match['home_team_goal_count']
                }
            else:
                continue
                
            team_matches.append(match_info)
            
        # ゲームウィーク順でソート
        team_matches.sort(key=lambda x: x['game_week'])
        
        # 最近の試合
        recent_matches = team_matches[-self.recent_matches:] if len(team_matches) >= self.recent_matches else team_matches
        
        if len(recent_matches) == 0:
            return {'recent_ppg': 0.0, 'recent_goals_for': 0.0, 'recent_goals_against': 0.0}
            
        # 最近の統計計算
        recent_points = 0
        recent_goals_for = 0
        recent_goals_against = 0
        
        for match in recent_matches:
            goals_for = match['goals_for']
            goals_against = match['goals_against']
            
            recent_goals_for += goals_for
            recent_goals_against += goals_against
            
            if goals_for > goals_against:
                recent_points += 3
            elif goals_for == goals_against:
                recent_points += 1
                
        recent_ppg = recent_points / len(recent_matches)
        recent_goals_for_avg = recent_goals_for / len(recent_matches)
        recent_goals_against_avg = recent_goals_against / len(recent_matches)
        
        return {
            'recent_ppg': recent_ppg,
            'recent_goals_for': recent_goals_for_avg,
            'recent_goals_against': recent_goals_against_avg
        }
        
    def _empty_stats(self) -> Dict:
        """
        空の統計辞書を返す
        
        Returns:
            stats: 空の統計辞書
        """
        
        return {
            'matches_played': 0,
            'ppg': 0.0,
            'goals_for_per_game': 0.0,
            'goals_against_per_game': 0.0,
            'goal_difference_per_game': 0.0,
            'xg_for_per_game': 0.0,
            'xg_against_per_game': 0.0,
            'xg_difference_per_game': 0.0,
            'win_rate': 0.0,
            'draw_rate': 0.0,
            'loss_rate': 0.0,
            'home_advantage': 0.0,
            'recent_form': 0.0,
            'recent_goals_for': 0.0,
            'recent_goals_against': 0.0
        }
        
    def get_head_to_head_stats(self, team1: str, team2: str, results_df: pd.DataFrame, 
                              last_n_matches: int = 10) -> Dict:
        """
        直接対戦成績を計算
        
        Args:
            team1: チーム1
            team2: チーム2
            results_df: 試合結果DataFrame
            last_n_matches: 過去N試合
            
        Returns:
            h2h_stats: 直接対戦統計
        """
        
        # 直接対戦試合を抽出
        h2h_matches = results_df[
            ((results_df['home_team_name'] == team1) & (results_df['away_team_name'] == team2)) |
            ((results_df['home_team_name'] == team2) & (results_df['away_team_name'] == team1))
        ].copy()
        
        # 最新N試合に限定
        h2h_matches = h2h_matches.sort_values('Game Week').tail(last_n_matches)
        
        if len(h2h_matches) == 0:
            return {
                'matches_played': 0,
                'team1_wins': 0,
                'draws': 0,
                'team2_wins': 0,
                'avg_goals_team1': 0.0,
                'avg_goals_team2': 0.0
            }
            
        team1_wins = 0
        team2_wins = 0
        draws = 0
        team1_goals = 0
        team2_goals = 0
        
        for _, match in h2h_matches.iterrows():
            if match['home_team_name'] == team1:
                goals1 = match['home_team_goal_count']
                goals2 = match['away_team_goal_count']
            else:
                goals1 = match['away_team_goal_count']
                goals2 = match['home_team_goal_count']
                
            team1_goals += goals1
            team2_goals += goals2
            
            if goals1 > goals2:
                team1_wins += 1
            elif goals2 > goals1:
                team2_wins += 1
            else:
                draws += 1
                
        return {
            'matches_played': len(h2h_matches),
            'team1_wins': team1_wins,
            'draws': draws,
            'team2_wins': team2_wins,
            'avg_goals_team1': team1_goals / len(h2h_matches),
            'avg_goals_team2': team2_goals / len(h2h_matches)
        }


class DataNormalizer:
    """
    データ正規化クラス
    特徴量のスケーリングと前処理
    """
    
    def __init__(self):
        self.scalers = {}
        
    def fit_normalize_features(self, features_df: pd.DataFrame, 
                              target_columns: List[str] = None) -> pd.DataFrame:
        """
        特徴量の正規化とフィッティング
        
        Args:
            features_df: 特徴量DataFrame
            target_columns: 正規化対象カラム（Noneの場合は数値カラム全て）
            
        Returns:
            normalized_df: 正規化済みDataFrame
        """
        
        normalized_df = features_df.copy()
        
        if target_columns is None:
            target_columns = features_df.select_dtypes(include=[np.number]).columns.tolist()
            
        for col in target_columns:
            if col in features_df.columns:
                mean_val = features_df[col].mean()
                std_val = features_df[col].std()
                
                # 標準偏差が0の場合は正規化をスキップ
                if std_val > 0:
                    normalized_df[col] = (features_df[col] - mean_val) / std_val
                    self.scalers[col] = {'mean': mean_val, 'std': std_val}
                else:
                    self.scalers[col] = {'mean': mean_val, 'std': 1.0}
                    
        return normalized_df
        
    def transform_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        既存のスケーラーで特徴量を変換
        
        Args:
            features_df: 特徴量DataFrame
            
        Returns:
            normalized_df: 正規化済みDataFrame
        """
        
        normalized_df = features_df.copy()
        
        for col, scaler in self.scalers.items():
            if col in features_df.columns:
                normalized_df[col] = (features_df[col] - scaler['mean']) / scaler['std']
                
        return normalized_df


class FixtureProcessor:
    """
    試合予定処理クラス
    """
    
    @staticmethod
    def load_and_validate_fixtures(filepath: str) -> pd.DataFrame:
        """
        試合予定ファイル読み込みと検証
        
        Args:
            filepath: ファイルパス
            
        Returns:
            fixtures_df: 検証済み試合予定DataFrame
        """
        
        fixtures_df = pd.read_csv(filepath)
        
        # 必須カラムチェック
        required_columns = ['Game Week', 'home_team_name', 'away_team_name']
        missing_columns = [col for col in required_columns if col not in fixtures_df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # データ型変換
        fixtures_df['Game Week'] = fixtures_df['Game Week'].astype(int)
        
        # 重複チェック
        duplicates = fixtures_df.duplicated(subset=['Game Week', 'home_team_name', 'away_team_name'])
        if duplicates.any():
            print(f"Warning: Found {duplicates.sum()} duplicate fixtures")
            
        return fixtures_df
        
    @staticmethod
    def merge_results_with_fixtures(fixtures_df: pd.DataFrame, 
                                  results_df: pd.DataFrame) -> pd.DataFrame:
        """
        試合予定と結果をマージ
        
        Args:
            fixtures_df: 試合予定DataFrame
            results_df: 結果DataFrame
            
        Returns:
            merged_df: マージ済みDataFrame
        """
        
        # マージキー作成
        fixtures_df['merge_key'] = (fixtures_df['Game Week'].astype(str) + '_' + 
                                  fixtures_df['home_team_name'] + '_' + 
                                  fixtures_df['away_team_name'])
        
        results_df['merge_key'] = (results_df['Game Week'].astype(str) + '_' + 
                                 results_df['home_team_name'] + '_' + 
                                 results_df['away_team_name'])
        
        # マージ実行
        merged_df = fixtures_df.merge(
            results_df[['merge_key', 'home_team_goal_count', 'away_team_goal_count']],
            on='merge_key',
            how='left',
            suffixes=('', '_result')
        )
        
        # 不要カラム削除
        merged_df.drop('merge_key', axis=1, inplace=True)
        
        return merged_df


if __name__ == "__main__":
    # テスト用
    print("Data preprocessing utilities loaded successfully")
