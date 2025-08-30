"""
Unified Football Data Loader for Multi-League Prediction
全リーグ統合データローダー - Google Drive対応
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from collections import defaultdict


class UnifiedFootballDataset(Dataset):
    """
    統合サッカー試合データセット（PyTorch Dataset）
    全リーグのデータを対等に扱い、オッズ非依存の特徴量でモデル学習
    """
    
    def __init__(self, data_dir: str, feature_config: Dict, 
                 team_encoder: Optional[Dict] = None):
        """
        Args:
            data_dir: CSVファイルが格納されているディレクトリパス
            feature_config: 特徴量設定
            team_encoder: チーム名→IDエンコーダー（Noneの場合は自動生成）
        """
        
        self.feature_config = feature_config
        self.data = []
        self.team_encoder = team_encoder or {}
        self.team_decoder = {}
        
        # データ読み込み・前処理
        self._load_all_csv_files(data_dir)
        self._create_team_mapping()
        self._extract_features()
        
    def _load_all_csv_files(self, data_dir: str):
        """指定ディレクトリから全CSVファイルを読み込み"""
        
        print(f"Loading all CSV files from: {data_dir}")
        
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {data_dir}")
            
        all_matches = []
        
        for csv_file in csv_files:
            try:
                print(f"  Loading: {os.path.basename(csv_file)}")
                df = pd.read_csv(csv_file)
                
                if len(df) > 0:
                    # statusカラムがある場合は、'complete'のみを対象とする
                    if 'status' in df.columns:
                        before_count = len(df)
                        df = df[df['status'] == 'complete'].copy()
                        print(f"    Filtered by status: {before_count} -> {len(df)} matches")
                    
                    # リーグ名を推定（ファイル名から）
                    league_name = self._infer_league_from_filename(os.path.basename(csv_file))
                    df['league'] = league_name
                    all_matches.append(df)
                    print(f"    {len(df)} matches loaded from {league_name}")
                    
            except Exception as e:
                print(f"    Error loading {csv_file}: {e}")
                continue
                
        if all_matches:
            self.raw_data = pd.concat(all_matches, ignore_index=True)
            print(f"Total matches loaded: {len(self.raw_data)}")
            
            # リーグ別統計表示
            league_stats = self.raw_data['league'].value_counts()
            print("League distribution:")
            for league, count in league_stats.items():
                print(f"  {league}: {count} matches")
        else:
            raise ValueError("No data could be loaded from any CSV files")
            
    def _infer_league_from_filename(self, filename: str) -> str:
        """ファイル名からリーグ名を推定"""
        
        filename_lower = filename.lower()
        
        if 'premier' in filename_lower or 'england' in filename_lower:
            return 'Premier League'
        elif 'j_league' in filename_lower or 'j-league' in filename_lower or 'jleague' in filename_lower:
            return 'J-League'
        elif 'bundesliga' in filename_lower or 'germany' in filename_lower:
            return 'Bundesliga'
        elif 'serie_a' in filename_lower or 'italy' in filename_lower:
            return 'Serie A'
        elif 'ligue_1' in filename_lower or 'france' in filename_lower:
            return 'Ligue 1'
        elif 'laliga' in filename_lower or 'spain' in filename_lower:
            return 'La Liga'
        else:
            return 'Other League'
            
    def _create_team_mapping(self):
        """チーム名をIDにマッピング"""
        
        # 全チーム名を収集
        all_teams = set()
        all_teams.update(self.raw_data['home_team_name'].unique())
        all_teams.update(self.raw_data['away_team_name'].unique())
        
        # 既存のエンコーダーがない場合は新規作成
        if not self.team_encoder:
            for i, team in enumerate(sorted(all_teams)):
                self.team_encoder[team] = i
                
        # デコーダー作成
        self.team_decoder = {v: k for k, v in self.team_encoder.items()}
        
        print(f"Total teams: {len(self.team_encoder)}")
        
    def _extract_features(self):
        """オッズ非依存特徴量の抽出"""
        
        print("Extracting features...")
        
        for idx, row in self.raw_data.iterrows():
            try:
                # 基本情報
                home_team = row['home_team_name']
                away_team = row['away_team_name']
                home_team_id = self.team_encoder.get(home_team, 0)
                away_team_id = self.team_encoder.get(away_team, 0)
                
                # PPG特徴量（既存データから）
                home_ppg = row.get('Pre-Match PPG (Home)', 0.0)
                away_ppg = row.get('Pre-Match PPG (Away)', 0.0)
                
                # xG特徴量
                home_xg = row.get('Home Team Pre-Match xG', 0.0)
                away_xg = row.get('Away Team Pre-Match xG', 0.0)
                
                # 実際のxG（試合後）
                home_xg_actual = row.get('team_a_xg', home_xg)
                away_xg_actual = row.get('team_b_xg', away_xg)
                
                # ゲームウィーク
                game_week = row.get('Game Week', 1)
                
                # 特徴量ベクトル構築
                features = np.array([
                    home_ppg,
                    away_ppg,
                    home_ppg,  # フォームPPG（暫定的に同じ値）
                    away_ppg,  # フォームPPG（暫定的に同じ値）
                    home_xg,
                    away_xg,
                    1.0,  # home_xg_against（デフォルト値）
                    1.0,  # away_xg_against（デフォルト値）
                    game_week / 38.0,  # 正規化
                    0.3,  # ホームアドバンテージ
                    0.0   # head_to_head（デフォルト値）
                ], dtype=np.float32)
                
                # ターゲット（実際の結果）
                home_goals = int(row['home_team_goal_count'])
                away_goals = int(row['away_team_goal_count'])
                
                # 結果分類
                if home_goals > away_goals:
                    result = 0  # ホーム勝利
                elif home_goals < away_goals:
                    result = 2  # アウェイ勝利
                else:
                    result = 1  # ドロー
                
                # データポイント追加
                self.data.append({
                    'home_team_id': home_team_id,
                    'away_team_id': away_team_id,
                    'features': features,
                    'home_goals': home_goals,
                    'away_goals': away_goals,
                    'result': result,
                    'home_team_name': home_team,
                    'away_team_name': away_team
                })
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
                
        print(f"Processed {len(self.data)} matches successfully")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """データポイントを取得"""
        sample = self.data[idx]
        
        return {
            'home_team_id': torch.tensor(sample['home_team_id'], dtype=torch.long),
            'away_team_id': torch.tensor(sample['away_team_id'], dtype=torch.long),
            'features': torch.tensor(sample['features'], dtype=torch.float32),
            'home_goals': torch.tensor(sample['home_goals'], dtype=torch.float32),
            'away_goals': torch.tensor(sample['away_goals'], dtype=torch.float32),
            'result': torch.tensor(sample['result'], dtype=torch.long)
        }


class UnifiedDataLoader:
    """
    統合データローダー
    Google Driveの単一フォルダから全リーグデータを読み込み
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.team_encoder = {}
        self.team_stats = defaultdict(dict)
        
    def load_from_drive(self, csv_dir: str) -> DataLoader:
        """
        Google Driveのフォルダから全CSVファイルを読み込み統合DataLoaderを作成
        
        Args:
            csv_dir: CSVファイルが格納されているディレクトリパス
            例: "/content/drive/MyDrive/league-predictor/stats-csv"
                
        Returns:
            data_loader: 統合されたDataLoader
        """
        
        print(f"Loading data from Google Drive: {csv_dir}")
        
        # 統合データセット作成
        dataset = UnifiedFootballDataset(
            data_dir=csv_dir,
            feature_config=self.config,
            team_encoder=self.team_encoder
        )
        
        # チームエンコーダーを更新
        self.team_encoder = dataset.team_encoder
        
        # DataLoader作成
        data_loader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0  # Google Colab対応
        )
        
        print(f"DataLoader created with {len(dataset)} samples")
        return data_loader
    
    def save_team_encoder(self, filepath: str):
        """チームエンコーダーを保存"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.team_encoder, f, ensure_ascii=False, indent=2)
            
    def load_team_encoder(self, filepath: str):
        """チームエンコーダーを読み込み"""
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                self.team_encoder = json.load(f)
                print(f"Team encoder loaded: {len(self.team_encoder)} teams")
        else:
            print(f"Team encoder file not found: {filepath}")
            
    def get_league_statistics(self, csv_dir: str) -> Dict:
        """リーグ別統計情報を取得"""
        
        dataset = UnifiedFootballDataset(
            data_dir=csv_dir,
            feature_config=self.config
        )
        
        # リーグ別統計
        league_stats = dataset.raw_data['league'].value_counts().to_dict()
        
        # チーム数統計
        total_teams = len(dataset.team_encoder)
        
        # 試合数統計
        total_matches = len(dataset.raw_data)
        
        return {
            'league_distribution': league_stats,
            'total_teams': total_teams,
            'total_matches': total_matches,
            'teams_per_league': {
                league: len(dataset.raw_data[dataset.raw_data['league'] == league]['home_team_name'].unique())
                for league in league_stats.keys()
            }
        }


# Google Drive設定例
GOOGLE_DRIVE_CONFIG = {
    'csv_directory': '/content/drive/MyDrive/league-predictor/stats-csv',
    'model_directory': '/content/drive/MyDrive/league-predictor/models'
}

# 学習設定例
TRAINING_CONFIG = {
    'batch_size': 64,
    'feature_dim': 11,
    'learning_rate': 0.001,
    'epochs': 100
}


if __name__ == "__main__":
    # テスト用設定
    config = {
        'batch_size': 32,
        'feature_dim': 11
    }
    
    # データローダーテスト
    loader = UnifiedDataLoader(config)
    
    # ローカルテスト用（実際はGoogle Driveのstats-csv/を使用）
    test_csv_dir = "../stats-csv"  # ローカルテスト用
    
    try:
        # 統計情報を取得
        stats = loader.get_league_statistics(test_csv_dir)
        print("League Statistics:")
        print(f"  Total matches: {stats['total_matches']}")
        print(f"  Total teams: {stats['total_teams']}")
        print("  League distribution:")
        for league, count in stats['league_distribution'].items():
            print(f"    {league}: {count} matches")
        print("  Teams per league:")
        for league, count in stats['teams_per_league'].items():
            print(f"    {league}: {count} teams")
        
        # DataLoader作成テスト
        data_loader = loader.load_from_drive(test_csv_dir)
        print(f"\nDataLoader created successfully!")
        
        # サンプルバッチを確認
        for batch in data_loader:
            print(f"Batch shapes:")
            print(f"  Home team IDs: {batch['home_team_id'].shape}")
            print(f"  Features: {batch['features'].shape}")
            print(f"  Goals: {batch['home_goals'].shape}")
            break
            
    except Exception as e:
        print(f"Test failed: {e}")
        print("Note: This test requires actual CSV data files")
