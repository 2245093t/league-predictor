import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class PremierLeaguePredictor:
    def __init__(self):
        self.match_result_model = None
        self.goals_model_home = None
        self.goals_model_away = None
        self.team_encoder = LabelEncoder()
        self.df = None
        
    def load_data(self, filepath):
        """データを読み込み、前処理を行う"""
        self.df = pd.read_csv(filepath)
        print(f"データ読み込み完了: {len(self.df)}試合")
        
    def create_match_result(self, home_goals, away_goals):
        """試合結果をエンコード (0: アウェイ勝利, 1: ドロー, 2: ホーム勝利)"""
        if home_goals > away_goals:
            return 2  # ホーム勝利
        elif home_goals < away_goals:
            return 0  # アウェイ勝利
        else:
            return 1  # ドロー
    
    def prepare_features(self, df):
        """機械学習用の特徴量を準備"""
        features = []
        
        for _, row in df.iterrows():
            feature_row = {
                # チーム情報（エンコード済み）
                'home_team': self.team_encoder.transform([row['home_team_name']])[0],
                'away_team': self.team_encoder.transform([row['away_team_name']])[0],
                
                # ゲームウィーク
                'game_week': row['Game Week'],
                
                # PPG（シーズン開始時は0なので、小さな値で置換）
                'home_ppg': max(row['Pre-Match PPG (Home)'], 0.01),
                'away_ppg': max(row['Pre-Match PPG (Away)'], 0.01),
                
                # xG
                'home_xg': row['Home Team Pre-Match xG'],
                'away_xg': row['Away Team Pre-Match xG'],
                
                # オッズ（勝利確率の指標）
                'home_win_odds': row['odds_ft_home_team_win'],
                'draw_odds': row['odds_ft_draw'],
                'away_win_odds': row['odds_ft_away_team_win'],
                
                # オッズから算出される暗黙の勝利確率
                'home_win_prob': 1 / row['odds_ft_home_team_win'],
                'draw_prob': 1 / row['odds_ft_draw'], 
                'away_win_prob': 1 / row['odds_ft_away_team_win'],
                
                # その他統計
                'avg_goals_pre': row['average_goals_per_match_pre_match'],
                'btts_percentage': row['btts_percentage_pre_match'],
            }
            features.append(feature_row)
            
        return pd.DataFrame(features)
    
    def train_models(self):
        """モデルを訓練"""
        print("モデル訓練開始...")
        
        # チーム名のエンコーディング
        all_teams = list(set(self.df['home_team_name'].unique()) | set(self.df['away_team_name'].unique()))
        self.team_encoder.fit(all_teams)
        
        # 特徴量準備
        X = self.prepare_features(self.df)
        
        # ターゲット変数
        y_goals_home = self.df['home_team_goal_count']
        y_goals_away = self.df['away_team_goal_count']
        y_result = [self.create_match_result(h, a) for h, a in zip(y_goals_home, y_goals_away)]
        
        print(f"特徴量数: {X.shape[1]}")
        print(f"サンプル数: {X.shape[0]}")
        
        # モデル訓練
        self.goals_model_home = RandomForestRegressor(n_estimators=100, random_state=42)
        self.goals_model_away = RandomForestRegressor(n_estimators=100, random_state=42)
        self.match_result_model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # 訓練・テストデータ分割
        X_train, X_test, y_goals_home_train, y_goals_home_test = train_test_split(
            X, y_goals_home, test_size=0.2, random_state=42
        )
        _, _, y_goals_away_train, y_goals_away_test = train_test_split(
            X, y_goals_away, test_size=0.2, random_state=42
        )
        _, _, y_result_train, y_result_test = train_test_split(
            X, y_result, test_size=0.2, random_state=42
        )
        
        # モデル訓練
        self.goals_model_home.fit(X_train, y_goals_home_train)
        self.goals_model_away.fit(X_train, y_goals_away_train)
        self.match_result_model.fit(X_train, y_result_train)
        
        # 性能評価
        home_pred = self.goals_model_home.predict(X_test)
        away_pred = self.goals_model_away.predict(X_test)
        result_pred = self.match_result_model.predict(X_test)
        
        print(f"\\nモデル性能:")
        print(f"ホーム得点予測 RMSE: {np.sqrt(mean_squared_error(y_goals_home_test, home_pred)):.3f}")
        print(f"アウェイ得点予測 RMSE: {np.sqrt(mean_squared_error(y_goals_away_test, away_pred)):.3f}")
        print(f"試合結果予測精度: {accuracy_score(y_result_test, result_pred):.3f}")
        
        # 特徴量重要度
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.match_result_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\\n特徴量重要度（上位5）:")
        print(feature_importance.head().to_string(index=False))
        
    def predict_match(self, home_team, away_team, game_week, 
                     home_ppg=1.5, away_ppg=1.5, home_xg=1.2, away_xg=1.2,
                     home_odds=2.0, draw_odds=3.5, away_odds=3.0):
        """単一試合の結果を予測"""
        
        # 特徴量作成
        feature_row = pd.DataFrame([{
            'home_team': self.team_encoder.transform([home_team])[0],
            'away_team': self.team_encoder.transform([away_team])[0],
            'game_week': game_week,
            'home_ppg': home_ppg,
            'away_ppg': away_ppg,
            'home_xg': home_xg,
            'away_xg': away_xg,
            'home_win_odds': home_odds,
            'draw_odds': draw_odds,
            'away_win_odds': away_odds,
            'home_win_prob': 1 / home_odds,
            'draw_prob': 1 / draw_odds,
            'away_win_prob': 1 / away_odds,
            'avg_goals_pre': 2.5,
            'btts_percentage': 50,
        }])
        
        # 予測
        home_goals_pred = self.goals_model_home.predict(feature_row)[0]
        away_goals_pred = self.goals_model_away.predict(feature_row)[0]
        result_prob = self.match_result_model.predict_proba(feature_row)[0]
        
        result_labels = ['アウェイ勝利', 'ドロー', 'ホーム勝利']
        predicted_result = result_labels[np.argmax(result_prob)]
        
        return {
            'home_goals': round(home_goals_pred, 2),
            'away_goals': round(away_goals_pred, 2),
            'predicted_result': predicted_result,
            'probabilities': {
                'ホーム勝利': round(result_prob[2], 3),
                'ドロー': round(result_prob[1], 3),
                'アウェイ勝利': round(result_prob[0], 3)
            }
        }

# モデルのテスト実行
if __name__ == "__main__":
    # モデル初期化・訓練
    predictor = PremierLeaguePredictor()
    predictor.load_data('stats-csv/england-premier-league-matches-2018-to-2019-stats.csv')
    predictor.train_models()
    
    print(f"\\n=== サンプル予測 ===")
    
    # サンプル予測
    sample_predictions = [
        ("Manchester City", "Liverpool", 20, 2.8, 2.7, 2.2, 2.0, 1.8, 4.0, 4.5),
        ("Arsenal", "Chelsea", 25, 2.1, 2.0, 1.8, 1.7, 2.2, 3.4, 3.6),
        ("Burnley", "Brighton & Hove Albion", 30, 1.2, 1.1, 1.0, 0.9, 2.8, 3.2, 2.7)
    ]
    
    for home, away, week, h_ppg, a_ppg, h_xg, a_xg, h_odds, d_odds, a_odds in sample_predictions:
        result = predictor.predict_match(home, away, week, h_ppg, a_ppg, h_xg, a_xg, h_odds, d_odds, a_odds)
        print(f"\\n{home} vs {away} (第{week}節)")
        print(f"予測スコア: {result['home_goals']} - {result['away_goals']}")
        print(f"予測結果: {result['predicted_result']}")
        print(f"確率: {result['probabilities']}")
