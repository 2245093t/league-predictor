"""
PyTorch Neural Network for Football Match Prediction
オッズに依存しない、PPGとxGベースの予測モデル
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import json


class FootballMatchPredictor(nn.Module):
    """
    サッカー試合結果予測のためのニューラルネットワーク
    
    Features:
    - PPG (Points Per Game) ベースの実力評価
    - xG (Expected Goals) による攻撃・守備力評価
    - チーム埋め込み（Transfer Learning対応）
    - 時系列コンテキスト（シーズン進行度、フォームなど）
    """
    
    def __init__(self, config: Dict):
        super(FootballMatchPredictor, self).__init__()
        
        self.config = config
        self.num_teams = config.get('num_teams', 100)  # 多リーグ対応
        self.embedding_dim = config.get('embedding_dim', 32)
        self.hidden_dim = config.get('hidden_dim', 128)
        
        # チーム埋め込み（Transfer Learning対応）
        self.team_embedding = nn.Embedding(self.num_teams, self.embedding_dim)
        
        # 特徴量次元
        # PPG関連: home_ppg, away_ppg, home_ppg_form, away_ppg_form (4)
        # xG関連: home_xg, away_xg, home_xga, away_xga (4)  
        # コンテキスト: game_week, home_advantage, head_to_head (3)
        self.feature_dim = 4 + 4 + 3  # 11次元
        
        # 入力層: チーム埋め込み + 統計特徴量
        self.input_dim = self.embedding_dim * 2 + self.feature_dim
        
        # ニューラルネットワーク
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc3 = nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4)
        
        # 出力層
        self.goal_predictor = nn.Linear(self.hidden_dim // 4, 2)  # [home_goals, away_goals]
        self.result_predictor = nn.Linear(self.hidden_dim // 4, 3)  # [home_win, draw, away_win]
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.get('dropout', 0.2))
        
        # Batch Normalization
        self.bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.bn2 = nn.BatchNorm1d(self.hidden_dim // 2)
        
    def forward(self, home_team_ids: torch.Tensor, away_team_ids: torch.Tensor, 
                features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            home_team_ids: ホームチームID [batch_size]
            away_team_ids: アウェイチームID [batch_size]  
            features: 統計特徴量 [batch_size, feature_dim]
            
        Returns:
            goals: 予測得点 [batch_size, 2] (home, away)
            result_probs: 結果確率 [batch_size, 3] (home_win, draw, away_win)
        """
        
        # チーム埋め込み
        home_embed = self.team_embedding(home_team_ids)  # [batch_size, embedding_dim]
        away_embed = self.team_embedding(away_team_ids)  # [batch_size, embedding_dim]
        
        # 特徴量結合
        x = torch.cat([home_embed, away_embed, features], dim=1)  # [batch_size, input_dim]
        
        # Hidden layers
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.fc3(x))
        
        # 出力
        goals = F.relu(self.goal_predictor(x))  # 非負制約
        result_logits = self.result_predictor(x)
        result_probs = F.softmax(result_logits, dim=1)
        
        return goals, result_probs
    
    def predict_match(self, home_team_id: int, away_team_id: int, 
                     features: np.ndarray) -> Dict:
        """
        単一試合の予測
        
        Args:
            home_team_id: ホームチームID
            away_team_id: アウェイチームID
            features: 統計特徴量 [feature_dim]
            
        Returns:
            prediction: 予測結果辞書
        """
        
        self.eval()
        with torch.no_grad():
            # テンソル変換
            home_ids = torch.tensor([home_team_id], dtype=torch.long)
            away_ids = torch.tensor([away_team_id], dtype=torch.long)
            feat_tensor = torch.tensor(features.reshape(1, -1), dtype=torch.float32)
            
            # 予測
            goals, result_probs = self.forward(home_ids, away_ids, feat_tensor)
            
            # 結果整理
            pred_goals = goals[0].numpy()
            pred_probs = result_probs[0].numpy()
            
            return {
                'home_goals': float(pred_goals[0]),
                'away_goals': float(pred_goals[1]),
                'home_win_prob': float(pred_probs[0]),
                'draw_prob': float(pred_probs[1]),
                'away_win_prob': float(pred_probs[2]),
                'predicted_result': ['Home Win', 'Draw', 'Away Win'][np.argmax(pred_probs)]
            }
    
    def save_model(self, filepath: str, metadata: Optional[Dict] = None):
        """
        モデルの保存（重みとメタデータ）
        
        Args:
            filepath: 保存先パス(.pth)
            metadata: メタデータ（設定、学習履歴など）
        """
        
        save_dict = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'model_class': self.__class__.__name__
        }
        
        if metadata:
            save_dict['metadata'] = metadata
            
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str, device: str = 'cpu') -> 'FootballMatchPredictor':
        """
        保存されたモデルの読み込み
        
        Args:
            filepath: モデルファイルパス
            device: 実行デバイス ('cpu' or 'cuda')
            
        Returns:
            loaded_model: 読み込まれたモデル
        """
        
        checkpoint = torch.load(filepath, map_location=device)
        config = checkpoint['config']
        
        # モデル再構築
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        print(f"Model loaded from {filepath}")
        if 'metadata' in checkpoint:
            print(f"Metadata: {checkpoint['metadata']}")
            
        return model


class FeatureExtractor:
    """
    オッズ非依存の特徴量抽出器
    PPG、xG、フォーム、コンテキスト情報を統合
    """
    
    def __init__(self):
        self.feature_names = [
            'home_ppg', 'away_ppg', 'home_ppg_form', 'away_ppg_form',
            'home_xg', 'away_xg', 'home_xga', 'away_xga',
            'game_week', 'home_advantage', 'head_to_head'
        ]
    
    def extract_features(self, match_data: Dict, team_stats: Dict) -> np.ndarray:
        """
        試合データからオッズ非依存特徴量を抽出
        
        Args:
            match_data: 試合情報
            team_stats: チーム統計情報
            
        Returns:
            features: 特徴量ベクトル [feature_dim]
        """
        
        home_team = match_data['home_team']
        away_team = match_data['away_team']
        
        # PPG特徴量
        home_ppg = team_stats[home_team].get('ppg', 1.5)
        away_ppg = team_stats[away_team].get('ppg', 1.5)
        home_ppg_form = team_stats[home_team].get('ppg_form_5', 1.5)  # 直近5試合
        away_ppg_form = team_stats[away_team].get('ppg_form_5', 1.5)
        
        # xG特徴量
        home_xg = team_stats[home_team].get('xg_for', 1.2)
        away_xg = team_stats[away_team].get('xg_for', 1.2)
        home_xga = team_stats[home_team].get('xg_against', 1.2)
        away_xga = team_stats[away_team].get('xg_against', 1.2)
        
        # コンテキスト特徴量
        game_week = match_data.get('game_week', 1) / 38.0  # 正規化
        home_advantage = 0.3  # ホームアドバンテージ定数
        head_to_head = team_stats.get('h2h', {}).get(f"{home_team}_vs_{away_team}", 0.0)
        
        features = np.array([
            home_ppg, away_ppg, home_ppg_form, away_ppg_form,
            home_xg, away_xg, home_xga, away_xga,
            game_week, home_advantage, head_to_head
        ], dtype=np.float32)
        
        return features
    
    def get_feature_dim(self) -> int:
        """特徴量次元数を返す"""
        return len(self.feature_names)


# モデル設定例
DEFAULT_CONFIG = {
    'num_teams': 100,        # 多リーグ対応のため大きめに設定
    'embedding_dim': 32,     # チーム埋め込み次元
    'hidden_dim': 128,       # 隠れ層次元
    'dropout': 0.2,          # ドロップアウト率
    'learning_rate': 0.001,  # 学習率
    'batch_size': 64,        # バッチサイズ
    'epochs': 100,           # エポック数
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}


if __name__ == "__main__":
    # モデルテスト
    model = FootballMatchPredictor(DEFAULT_CONFIG)
    print(f"Model architecture: {model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    
    # ダミーデータでテスト
    batch_size = 4
    home_ids = torch.randint(0, 20, (batch_size,))
    away_ids = torch.randint(0, 20, (batch_size,))
    features = torch.randn(batch_size, 11)
    
    goals, probs = model(home_ids, away_ids, features)
    print(f"Goals shape: {goals.shape}")
    print(f"Probabilities shape: {probs.shape}")
    
    # 特徴量抽出器テスト
    extractor = FeatureExtractor()
    print(f"Feature dimension: {extractor.get_feature_dim()}")
    print(f"Feature names: {extractor.feature_names}")
