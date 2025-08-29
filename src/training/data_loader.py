"""
Multi-League Data Loader for Football Prediction
複数リーグのデータを統合して学習用データセットを構築
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from collections import defaultdict


class FootballDataset(Dataset):
    """
    サッカー試合データセット（PyTorch Dataset）
    オッズ非依存の特徴量でモデル学習
    """
    
    def __init__(self, data_paths: List[str], feature_config: Dict, 
                 team_encoder: Optional[Dict] = None):
        """
        Args:
            data_paths: データファイルのパスリスト
            feature_config: 特徴量設定
            team_encoder: チーム名→IDエンコーダー（Noneの場合は自動生成）
        """
        
        self.feature_config = feature_config
        self.data = []
        self.team_encoder = team_encoder or {}
        self.team_decoder = {}
        
        # データ読み込み・前処理
        self._load_and_process_data(data_paths)
        self._create_team_mapping()
        self._extract_features()
        
    def _load_and_process_data(self, data_paths: List[str]):
        """複数ファイルからデータを読み込み"""
        
        all_matches = []
        
        for path in data_paths:
            if os.path.exists(path):
                print(f"Loading data from: {path}")
                
                if path.endswith('.csv'):
                    df = pd.read_csv(path)
                else:
                    # ディレクトリの場合、全CSVファイルを読み込み
                    for csv_file in Path(path).glob('*.csv'):
                        df = pd.read_csv(csv_file)
                        all_matches.append(df)
                        
                if len(df) > 0:
                    all_matches.append(df)
                    
        if all_matches:
            self.raw_data = pd.concat(all_matches, ignore_index=True)
            print(f"Total matches loaded: {len(self.raw_data)}")
        else:
            raise ValueError("No data found in specified paths")
            
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


class MultiLeagueDataLoader:
    """
    複数リーグデータの統合ローダー
    継続学習とファインチューニングに対応
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.datasets = {}
        self.team_encoder = {}
        self.team_stats = defaultdict(dict)
        
    def load_leagues(self, league_configs: Dict[str, Dict]) -> Dict[str, DataLoader]:
        """
        複数リーグのデータを読み込み
        
        Args:
            league_configs: リーグ設定辞書
                {
                    'premier_league': {'data_path': 'path/to/data', 'weight': 1.0},
                    'j_league': {'data_path': 'path/to/data', 'weight': 0.8}
                }
                
        Returns:
            data_loaders: リーグ別DataLoader辞書
        """
        
        all_data_paths = []
        
        # 全リーグのチーム情報を事前収集
        print("Building global team mapping...")
        for league_name, config in league_configs.items():
            data_path = config['data_path']
            if os.path.exists(data_path):
                all_data_paths.append(data_path)
                
        # グローバルチームエンコーダー作成
        self._build_global_team_encoder(all_data_paths)
        
        # リーグ別データセット作成
        data_loaders = {}
        
        for league_name, config in league_configs.items():
            print(f"Loading {league_name}...")
            
            dataset = FootballDataset(
                data_paths=[config['data_path']],
                feature_config=self.config,
                team_encoder=self.team_encoder
            )
            
            # サンプリング重み適用
            weight = config.get('weight', 1.0)
            batch_size = int(self.config['batch_size'] * weight)
            
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0  # Google Colab対応
            )
            
            data_loaders[league_name] = data_loader
            self.datasets[league_name] = dataset
            
        return data_loaders
    
    def _build_global_team_encoder(self, data_paths: List[str]):
        """全リーグ共通のチームエンコーダーを構築"""
        
        all_teams = set()
        
        for path in data_paths:
            if os.path.exists(path):
                if path.endswith('.csv'):
                    df = pd.read_csv(path)
                    all_teams.update(df['home_team_name'].unique())
                    all_teams.update(df['away_team_name'].unique())
                else:
                    # ディレクトリの場合
                    for csv_file in Path(path).glob('*.csv'):
                        df = pd.read_csv(csv_file)
                        all_teams.update(df['home_team_name'].unique())
                        all_teams.update(df['away_team_name'].unique())
        
        # チームIDアサイン
        for i, team in enumerate(sorted(all_teams)):
            self.team_encoder[team] = i
            
        print(f"Global team encoder built: {len(self.team_encoder)} teams")
        
    def save_team_encoder(self, filepath: str):
        """チームエンコーダーを保存"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.team_encoder, f, ensure_ascii=False, indent=2)
            
    def load_team_encoder(self, filepath: str):
        """チームエンコーダーを読み込み"""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.team_encoder = json.load(f)
            
    def get_combined_dataloader(self, league_configs: Dict[str, Dict]) -> DataLoader:
        """
        全リーグを統合した単一DataLoaderを作成
        """
        
        all_data_paths = [config['data_path'] for config in league_configs.values()]
        
        combined_dataset = FootballDataset(
            data_paths=all_data_paths,
            feature_config=self.config,
            team_encoder=self.team_encoder
        )
        
        return DataLoader(
            combined_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0
        )


# 設定例
LEAGUE_CONFIGS = {
    'premier_league': {
        'data_path': 'data/raw/premier_league/',
        'weight': 1.0,
        'priority': 'high'
    },
    'j_league': {
        'data_path': 'data/raw/j_league/',
        'weight': 0.8,
        'priority': 'high'  
    },
    'bundesliga': {
        'data_path': 'data/raw/other_leagues/bundesliga/',
        'weight': 0.9,
        'priority': 'medium'
    }
}


if __name__ == "__main__":
    # テスト用設定
    config = {
        'batch_size': 32,
        'feature_dim': 11
    }
    
    # データローダーテスト
    loader = MultiLeagueDataLoader(config)
    
    # 既存のプレミアリーグデータでテスト
    test_config = {
        'premier_league': {
            'data_path': 'data/raw/premier_league/england-premier-league-matches-2018-to-2019-stats.csv',
            'weight': 1.0
        }
    }
    
    try:
        data_loaders = loader.load_leagues(test_config)
        print("Data loading test successful!")
        
        # サンプルバッチを確認
        for league_name, data_loader in data_loaders.items():
            for batch in data_loader:
                print(f"{league_name} - Batch shapes:")
                print(f"  Home team IDs: {batch['home_team_id'].shape}")
                print(f"  Features: {batch['features'].shape}")
                print(f"  Goals: {batch['home_goals'].shape}")
                break
            break
            
    except Exception as e:
        print(f"Data loading test failed: {e}")
