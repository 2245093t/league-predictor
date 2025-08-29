"""
Main Prediction Entry Point
予測システムのメインエントリーポイント
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime
from typing import Optional

# パス設定
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.prediction.predict_matches import predict_weekly_matches
from src.prediction.predict_standings import predict_final_standings


def main():
    """
    メイン関数
    """
    
    parser = argparse.ArgumentParser(description='Football League Prediction System')
    parser.add_argument('--mode', choices=['matches', 'standings', 'both'], 
                       default='both', help='Prediction mode')
    parser.add_argument('--fixtures', required=True, 
                       help='Fixtures CSV file path')
    parser.add_argument('--results', 
                       help='Results CSV file path (optional)')
    parser.add_argument('--gameweek', type=int, 
                       help='Target gameweek for match prediction (optional)')
    parser.add_argument('--simulations', type=int, default=1000,
                       help='Number of simulations for standings prediction')
    parser.add_argument('--model', default='models/saved/best_model.pth',
                       help='Model file path')
    parser.add_argument('--encoder', default='models/saved/team_encoder.json',
                       help='Team encoder file path')
    parser.add_argument('--output-dir', default='predictions',
                       help='Output directory for predictions')
    
    args = parser.parse_args()
    
    # 出力ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # タイムスタンプ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        if args.mode in ['matches', 'both']:
            print("=" * 60)
            print("WEEKLY MATCH PREDICTIONS")
            print("=" * 60)
            
            # 週次試合予測
            match_predictions = predict_weekly_matches(
                fixture_file=args.fixtures,
                results_file=args.results,
                target_gameweek=args.gameweek,
                model_path=args.model,
                team_encoder_path=args.encoder
            )
            
            # 結果保存
            if match_predictions:
                match_output_file = os.path.join(args.output_dir, f'match_predictions_{timestamp}.csv')
                match_df = pd.DataFrame(match_predictions)
                match_df.to_csv(match_output_file, index=False, encoding='utf-8')
                print(f"Match predictions saved to {match_output_file}")
            
        if args.mode in ['standings', 'both']:
            print("\\n" + "=" * 60)
            print("FINAL STANDINGS PREDICTIONS")
            print("=" * 60)
            
            # 最終順位予測
            standings_predictions = predict_final_standings(
                fixture_file=args.fixtures,
                results_file=args.results,
                num_simulations=args.simulations,
                model_path=args.model,
                team_encoder_path=args.encoder
            )
            
            # 結果保存
            standings_output_file = os.path.join(args.output_dir, f'standings_predictions_{timestamp}.csv')
            
            export_data = []
            for team, results in standings_predictions.items():
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
                
            standings_df = pd.DataFrame(export_data)
            standings_df = standings_df.sort_values('predicted_final_position')
            standings_df.to_csv(standings_output_file, index=False, encoding='utf-8')
            print(f"Standings predictions saved to {standings_output_file}")
        
        print(f"\\nAll predictions completed successfully!")
        print(f"Results saved to {args.output_dir}/")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
