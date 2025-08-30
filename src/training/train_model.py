"""
Main Training Script for Football League Predictor
Google Colab GPU対応・継続学習可能な学習スクリプト
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

# 相対インポート
try:
    from .model_architecture import FootballMatchPredictor, DEFAULT_CONFIG
    from .data_loader import UnifiedDataLoader
except ImportError:
    # 直接実行時
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from training.model_architecture import FootballMatchPredictor, DEFAULT_CONFIG
    from training.data_loader import UnifiedDataLoader


class FootballTrainer:
    """
    サッカー予測モデルの学習クラス
    継続学習・ファインチューニング対応
    """
    
    def __init__(self, config: Dict, device: str = None):
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # ログ設定
        self._setup_logging()
        
        # モデル・オプティマイザー初期化
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # 学習履歴
        self.train_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'goal_loss': [],
            'result_loss': [],
            'accuracy': []
        }
        
    def _setup_logging(self):
        """ログ設定"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def initialize_model(self, pretrained_path: Optional[str] = None):
        """
        モデル初期化（新規または継続学習）
        
        Args:
            pretrained_path: 事前学習済みモデルのパス
        """
        
        if pretrained_path and os.path.exists(pretrained_path):
            # 継続学習: 既存モデルを読み込み
            self.logger.info(f"Loading pretrained model from {pretrained_path}")
            self.model = FootballMatchPredictor.load_model(pretrained_path, self.device)
            
            # 設定をマージ（新しい設定を優先）
            loaded_config = self.model.config.copy()
            loaded_config.update(self.config)
            self.config = loaded_config
            
        else:
            # 新規学習: モデル新規作成
            self.logger.info("Initializing new model")
            self.model = FootballMatchPredictor(self.config)
            self.model.to(self.device)
            
        # オプティマイザー設定
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        # 学習率スケジューラー
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
    def train_epoch(self, data_loader: DataLoader) -> Tuple[float, float, float]:
        """
        1エポックの学習
        
        Returns:
            epoch_loss, goal_loss, result_loss
        """
        
        self.model.train()
        total_loss = 0.0
        total_goal_loss = 0.0
        total_result_loss = 0.0
        
        goal_criterion = nn.MSELoss()
        result_criterion = nn.CrossEntropyLoss()
        
        for batch_idx, batch in enumerate(data_loader):
            # データをデバイスに移動
            home_ids = batch['home_team_id'].to(self.device)
            away_ids = batch['away_team_id'].to(self.device)
            features = batch['features'].to(self.device)
            
            target_home_goals = batch['home_goals'].to(self.device)
            target_away_goals = batch['away_goals'].to(self.device)
            target_results = batch['result'].to(self.device)
            
            # 勾配リセット
            self.optimizer.zero_grad()
            
            # フォワードパス
            pred_goals, pred_results = self.model(home_ids, away_ids, features)
            
            # 損失計算
            goal_targets = torch.stack([target_home_goals, target_away_goals], dim=1)
            goal_loss = goal_criterion(pred_goals, goal_targets)
            result_loss = result_criterion(pred_results, target_results)
            
            # 総損失（重み付き）
            total_loss_batch = (
                self.config.get('goal_loss_weight', 1.0) * goal_loss +
                self.config.get('result_loss_weight', 2.0) * result_loss
            )
            
            # バックプロパゲーション
            total_loss_batch.backward()
            
            # 勾配クリッピング
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # パラメータ更新
            self.optimizer.step()
            
            # 損失累積
            total_loss += total_loss_batch.item()
            total_goal_loss += goal_loss.item()
            total_result_loss += result_loss.item()
            
            # 進捗表示
            if batch_idx % 50 == 0:
                self.logger.info(
                    f"Batch {batch_idx}/{len(data_loader)}, "
                    f"Loss: {total_loss_batch.item():.4f}"
                )
                
        return (
            total_loss / len(data_loader),
            total_goal_loss / len(data_loader),
            total_result_loss / len(data_loader)
        )
    
    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        """
        モデル評価
        
        Returns:
            val_loss, accuracy
        """
        
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        goal_criterion = nn.MSELoss()
        result_criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for batch in data_loader:
                # データをデバイスに移動
                home_ids = batch['home_team_id'].to(self.device)
                away_ids = batch['away_team_id'].to(self.device)
                features = batch['features'].to(self.device)
                
                target_home_goals = batch['home_goals'].to(self.device)
                target_away_goals = batch['away_goals'].to(self.device)
                target_results = batch['result'].to(self.device)
                
                # 予測
                pred_goals, pred_results = self.model(home_ids, away_ids, features)
                
                # 損失計算
                goal_targets = torch.stack([target_home_goals, target_away_goals], dim=1)
                goal_loss = goal_criterion(pred_goals, goal_targets)
                result_loss = result_criterion(pred_results, target_results)
                
                total_loss_batch = (
                    self.config.get('goal_loss_weight', 1.0) * goal_loss +
                    self.config.get('result_loss_weight', 2.0) * result_loss
                )
                
                total_loss += total_loss_batch.item()
                
                # 精度計算
                pred_classes = torch.argmax(pred_results, dim=1)
                correct_predictions += (pred_classes == target_results).sum().item()
                total_predictions += target_results.size(0)
                
        avg_loss = total_loss / len(data_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def train(self, league_configs: Dict[str, Dict], 
              validation_split: float = 0.2,
              save_path: str = "models/saved/",
              checkpoint_interval: int = 10):
        """
        メイン学習ループ
        
        Args:
            league_configs: リーグ設定
            validation_split: 検証用データの割合
            save_path: モデル保存パス
            checkpoint_interval: チェックポイント保存間隔
        """
        
        self.logger.info("Starting training...")
        
        # 統合データローダー作成
        data_loader = UnifiedDataLoader(self.config)
        
        # 統合データローダー（従来の互換性維持）
        combined_loader = data_loader.load_from_drive(league_configs) if isinstance(league_configs, str) else None
        
        # データセット分割
        if combined_loader:
            dataset = combined_loader.dataset
        else:
            # 従来のリーグ設定形式の場合
            self.logger.warning("古いリーグ設定形式が使用されています。統合データローダーをお使いください。")
            return
            
        dataset_size = len(dataset)
        val_size = int(validation_split * dataset_size)
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False
        )
        
        # チームエンコーダー保存
        os.makedirs(save_path, exist_ok=True)
        data_loader.save_team_encoder(f"{save_path}/team_encoder.json")
        
        # 学習ループ
        best_val_loss = float('inf')
        epochs = self.config.get('epochs', 100)
        
        for epoch in range(epochs):
            self.logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # 学習
            train_loss, goal_loss, result_loss = self.train_epoch(train_loader)
            
            # 検証
            val_loss, accuracy = self.evaluate(val_loader)
            
            # 学習率更新
            self.scheduler.step(val_loss)
            
            # 履歴記録
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['goal_loss'].append(goal_loss)
            self.train_history['result_loss'].append(result_loss)
            self.train_history['accuracy'].append(accuracy)
            
            # ログ出力
            self.logger.info(
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Goal Loss: {goal_loss:.4f}, Result Loss: {result_loss:.4f}, "
                f"Accuracy: {accuracy:.4f}"
            )
            
            # ベストモデル保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(
                    f"{save_path}/best_model.pth",
                    metadata={
                        'epoch': epoch + 1,
                        'val_loss': val_loss,
                        'accuracy': accuracy,
                        'train_history': self.train_history
                    }
                )
                self.logger.info(f"New best model saved (val_loss: {val_loss:.4f})")
            
            # 定期チェックポイント
            if (epoch + 1) % checkpoint_interval == 0:
                self.save_model(
                    f"{save_path}/checkpoint_epoch_{epoch+1}.pth",
                    metadata={'epoch': epoch + 1, 'train_history': self.train_history}
                )
                
        self.logger.info("Training completed!")
        
    def save_model(self, filepath: str, metadata: Optional[Dict] = None):
        """モデル保存（履歴付き）"""
        self.model.save_model(filepath, metadata)
    
    def train_with_unified_data(self, train_dataloader: DataLoader, 
                              validation_split: float = 0.2,
                              save_path: str = "models/saved",
                              checkpoint_interval: int = 10):
        """
        統合データローダーでの学習（Google Colab対応）
        
        Args:
            train_dataloader: 統合データローダー
            validation_split: 検証用データの割合
            save_path: モデル保存パス
            checkpoint_interval: チェックポイント保存間隔
        """
        
        self.logger.info("Starting unified training...")
        
        # データセット分割
        dataset = train_dataloader.dataset
        dataset_size = len(dataset)
        val_size = int(validation_split * dataset_size)
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True,
            num_workers=0  # Google Colab対応
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=0  # Google Colab対応
        )
        
        # 保存ディレクトリ作成
        os.makedirs(save_path, exist_ok=True)
        
        # 学習ループ
        best_val_loss = float('inf')
        
        for epoch in range(self.config['epochs']):
            # 学習フェーズ
            train_loss, goal_loss, result_loss = self._train_epoch(train_loader)
            
            # 検証フェーズ
            val_loss, accuracy = self._validate_epoch(val_loader)
            
            # 学習率調整
            self.scheduler.step(val_loss)
            
            # 履歴記録
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['goal_loss'].append(goal_loss)
            self.train_history['result_loss'].append(result_loss)
            self.train_history['accuracy'].append(accuracy)
            
            # ログ出力
            self.logger.info(
                f"Epoch {epoch+1}/{self.config['epochs']}: "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Goal Loss: {goal_loss:.4f}, Result Loss: {result_loss:.4f}, "
                f"Accuracy: {accuracy:.4f}"
            )
            
            # ベストモデル保存
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(
                    f"{save_path}/best_model.pth",
                    metadata={
                        'epoch': epoch + 1,
                        'val_loss': val_loss,
                        'accuracy': accuracy,
                        'train_history': self.train_history
                    }
                )
                self.logger.info(f"New best model saved (val_loss: {val_loss:.4f})")
            
            # 定期チェックポイント
            if (epoch + 1) % checkpoint_interval == 0:
                self.save_model(
                    f"{save_path}/checkpoint_epoch_{epoch+1}.pth",
                    metadata={'epoch': epoch + 1, 'train_history': self.train_history}
                )
                
        self.logger.info("Unified training completed!")


def train_league_predictor(
    data_paths: List[str] = None,
    pretrained_model: str = None,
    config: Dict = None,
    epochs: int = 100,
    use_gpu: bool = True
) -> FootballMatchPredictor:
    """
    エントリーポイント関数（Google Colab用）
    
    Args:
        data_paths: データパスリスト
        pretrained_model: 事前学習モデルパス
        config: 学習設定
        epochs: エポック数
        use_gpu: GPU使用フラグ
        
    Returns:
        trained_model: 学習済みモデル
    """
    
    # デフォルト設定
    if config is None:
        config = DEFAULT_CONFIG.copy()
        
    config['epochs'] = epochs
    config['device'] = 'cuda' if (use_gpu and torch.cuda.is_available()) else 'cpu'
    
    # データパス設定
    if data_paths is None:
        data_paths = ['data/raw/premier_league/']
        
    league_configs = {}
    for i, path in enumerate(data_paths):
        league_name = f"league_{i}"
        league_configs[league_name] = {
            'data_path': path,
            'weight': 1.0
        }
    # トレーナー初期化
    trainer = FootballTrainer(config)
    trainer.initialize_model(pretrained_model)
    
    # 学習実行
    trainer.train(league_configs)
    
    return trainer.model


if __name__ == "__main__":
    # ローカルテスト用
    print("Football League Predictor Training")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # 設定
    config = DEFAULT_CONFIG.copy()
    config.update({
        'epochs': 20,  # テスト用に短く
        'batch_size': 16,
        'learning_rate': 0.001
    })
    
    # 学習実行（既存データで）
    try:
        model = train_league_predictor(
            data_paths=['data/raw/premier_league/england-premier-league-matches-2018-to-2019-stats.csv'],
            config=config,
            epochs=5,
            use_gpu=False  # ローカルテスト用
        )
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
