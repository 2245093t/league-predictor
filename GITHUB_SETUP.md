# GitHub アップロード手順

このプロジェクトをGitHubにアップロードするための完全ガイドです。

## 前提条件

✅ Git がインストール済み（確認済み）
✅ ローカルリポジトリが初期化済み（完了）
✅ 初回コミット完了

## GitHubでの操作

### 1. GitHubアカウントにログイン
[GitHub.com](https://github.com) にアクセスしてログインしてください。

### 2. 新しいリポジトリの作成

1. 右上の「+」ボタンをクリック → 「New repository」を選択
2. 以下の設定を入力：

```
Repository name: premier-league-predictor
Description: 🏆 Premier League position prediction using machine learning
Visibility: Public (または Private お好みで)

❌ Add a README file (チェックしない - 既にローカルにあるため)
❌ Add .gitignore (チェックしない - 既にローカルにあるため)  
❌ Choose a license (チェックしない - 既にローカルにあるため)
```

3. 「Create repository」ボタンをクリック

### 3. ローカルリポジトリとGitHubの接続

GitHubでリポジトリを作成後、表示されるURLをコピーして以下のコマンドを実行：

```bash
# GitHubリポジトリをリモートとして追加
git remote add origin https://github.com/{あなたのユーザー名}/premier-league-predictor.git

# メインブランチの設定確認
git branch -M main

# GitHubにプッシュ
git push -u origin main
```

## 推奨されるリポジトリ名と説明

### リポジトリ名の候補：
- `premier-league-predictor` (推奨)
- `football-league-position-predictor`
- `soccer-league-ml-predictor`
- `epl-position-predictor`

### 説明文の例：
```
🏆 Machine learning system for predicting Premier League final positions using match statistics, team performance metrics, and Monte Carlo simulation
```

### トピック（タグ）の提案：
```
machine-learning
football
soccer
premier-league
prediction
python
scikit-learn
monte-carlo
sports-analytics
data-science
```

## 次のステップ

1. **GitHubのIssueを活用**：
   - 今後の改善項目をIssueとして登録
   - マイルストーンの設定

2. **ブランチ戦略**：
   - `main`: 安定版
   - `develop`: 開発版
   - `feature/機能名`: 新機能開発

3. **継続的な開発**：
   - 定期的なコミット
   - 意味のあるコミットメッセージ
   - リリースタグの活用

## コミットメッセージの推奨フォーマット

```
🎯 type: 簡潔な説明

詳細な説明（必要に応じて）

影響範囲:
- 変更した機能
- 追加した機能
- 修正したバグ
```

### タイプの例：
- `🚀 feat:` 新機能
- `🐛 fix:` バグ修正  
- `📚 docs:` ドキュメント更新
- `♻️ refactor:` リファクタリング
- `✨ style:` コードスタイル修正
- `🧪 test:` テスト追加・修正
- `⚡ perf:` パフォーマンス改善

## トラブルシューティング

### よくある問題と解決方法

1. **認証エラー**：
   ```bash
   # Personal Access Tokenの設定が必要
   # GitHub Settings > Developer settings > Personal access tokens
   ```

2. **リモートURL変更**：
   ```bash
   git remote set-url origin https://github.com/username/repository.git
   ```

3. **プッシュエラー**：
   ```bash
   # 最新の変更を取得してからプッシュ
   git pull origin main --rebase
   git push origin main
   ```

完了したら、GitHubリポジトリのURLを共有して、他の開発者が簡単にクローンできるようになります！
