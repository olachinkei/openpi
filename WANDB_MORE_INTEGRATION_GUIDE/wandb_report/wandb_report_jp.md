# OpenPI x W&B: Physical AIワークフローの統合ガイド

---

## はじめに

[OpenPI](https://github.com/Physical-Intelligence/openpi) は、Physical AI 領域で急速に注目を集めているオープンソースの汎用ロボット制御フレームワークです。PyTorch ベースの学習ループを持ち、W&B との基本的な連携（loss や learning rate のロギング）は既に備わっています。

しかし、実際の研究・開発ワークフローでは「学習メトリクスの記録」だけでは不十分です。複数の実験を効率的に比較し、評価結果をシミュレーション動画と共に可視化し、チェックポイントを体系的に管理する――こうした一連のワークフローが求められます。

本レポートでは、OpenPI に対して追加実装した W&B 統合の全体像を紹介します。IsaacSim のようなシミュレータにはネイティブな可視化連携が存在しますが、OpenPI のコードベースにはそれがありません。本プロジェクトでは、その不足を補い、Physical AI 開発における W&B の活用パターンを実証しました。

技術的な実装の詳細については、[GitHub リポジトリ](https://github.com/olachinkei/openpi)を参照してください。

---

## 参考リンク

- **GitHub（拡張 W&B 統合版）**: https://github.com/olachinkei/openpi
- **YouTube（Physical AI x W&B デモ）**: https://www.youtube.com/watch?v=45Beo0ZkJJA
- **W&B Project（OpenPI 統合）**: <!-- プロジェクト URL を挿入 -->
- **参考: W&B Project（IsaacLab）**: https://wandb.ai/wandb-smle/isaaclab-wandb-crwv?nw=nwuseranushravvatsa

---

## 概要: 何を実装したか

### 題材

ALOHA Sim 環境の **Transfer Cube** タスクを題材に、`pi0_aloha_sim` 設定を使用してエンドツーエンドの学習・評価・管理ワークフローを構築しました。ALOHA Sim は再現性の高いシミュレーション環境であり、繰り返し評価に適しているため、W&B 統合のデモ対象として最適です。

この題材で解いている課題は、**2 本腕の ALOHA ロボットがキューブを把持し、受け渡し、安定して保持できるようにすること**です。OpenPI では `lerobot/aloha_sim_transfer_cube_human` のデモデータを使って、模倣学習ベースで方策を学習します。

データセットは役割ごとに 3 つに分かれます。

- **training dataset**: 学習に使うデモデータ本体。最終評価に使う held-out episode は除外しています
- **中間 eval 用 dataset**: 学習中の checkpoint を軽く比較するための小さな固定サブセット
- **最終 eval 用 dataset**: 学習に使わない held-out episode 群。最終比較用の評価セット

今回の W&B 連携では、このタスクに対して次のような metric を記録しています。

- **`primary_score`**: そのタスクで最も重要な代表スコアです。**Transfer Cube では `success_rate` と同じ値**を使っています。つまり「何本のエピソードが成功したか」を、run 間比較の代表値として使っています
- **`success_rate`**: 評価エピソードのうち、成功条件を満たした割合です。Transfer Cube では最も分かりやすい主指標です
- **`mean_max_reward`**: 各エピソードで到達した最大報酬の平均です。成功まで届かなくても、どれだけ目標に近づいたかを滑らかに見られます
- **`num_examples`**: その評価で実際に回したエピソード数です
- **`checkpoint_step`**: どの checkpoint を評価した結果かを示します
- **`train/loss`**: 模倣学習の学習損失です。低いほど、デモ行動の再現誤差が小さいことを示します
- **`train/grad_norm`**: 勾配の大きさです。学習が不安定になっていないかを見るための補助指標です
- **`train/param_norm`**: モデル重み全体の大きさです。学習の発散や異常を検知する補助指標です

要するに、この題材では **「最終的にタスクを成功できるか」** を `success_rate` / `primary_score` で見つつ、**「そこに向かってどれだけ近づいているか」** を `mean_max_reward` で補完しています。

### 追加した W&B 統合

本プロジェクトでは、以下の統合を OpenPI に追加しました。

1. **学習メトリクスの標準化と実験比較基盤** — config、tags、メタデータの構造化により、Workspace 上での複数実験の効率的な比較を実現
2. **評価パイプライン** — 評価データセットに対するシミュレーション実行とメトリクス・動画の自動ロギング（`scripts/eval_aloha_dataset.py`）
3. **W&B Table による Leaderboard** — 1 run = 1 行の評価結果テーブルで、複数実験のスコアを一覧比較
4. **シミュレーション動画の可視化** — ロールアウト動画を W&B Media Panel にロギングし、定性的な評価を可能に
5. **Artifacts によるチェックポイント管理** — マイルストーンチェックポイントのバージョン管理とリネージ追跡

汎用的な W&B ユーティリティレイヤーとして `src/openpi/utils/wandb/` を構築し、タスク固有のロジックとの分離を実現しています。

---

## 学習: 複数実験の効率的な比較

大量の実験を行った際に、それらを容易かつ体系的に比較できることは、研究の生産性に直結します。W&B では、`wandb.init` 時に適切なメタデータを記録しておくことで、後から Workspace 上で柔軟な分析が可能になります。

### Config の記録

本プロジェクトでは、`src/openpi/utils/wandb/run_context.py` の `WandbRunContext` クラスが、config name、task name、dataset name、backend、Git SHA、ホスト情報などを自動的に `wandb.config` に記録します。これにより、すべての run が一貫したメタデータを持ち、後からのフィルタリング・比較が容易になります。

<details>
<summary>parameter / config の詳細を見る</summary>

```python
import wandb

with wandb.init(
    project="openpi-aloha",
    notes="ALOHA Sim Transfer Cube baseline",
    tags=["baseline", "aloha_sim"],
    config={
        "config_name": "pi0_aloha_sim",
        "task_name": "transfer_cube",
        "backend": "pytorch",
        "learning_rate": 2.5e-5,
        "batch_size": 64,
    },
) as run:
    ...
```

- `config_name`: どの学習設定を使った run か
- `task_name`: 今回解いているロボットタスク名
- `backend`: JAX / PyTorch などの実装系
- `learning_rate`: 最適化のステップ幅
- `batch_size`: 1 step でまとめて使うデータ数
- `tags`: baseline / ablation / test など、人間が run を整理しやすくするラベル

</details>

config の詳細: https://docs.wandb.ai/models/track/config

---

### Parallel Coordinates Chart

Parallel Coordinates Chart は、ハイパーパラメータと結果メトリクスの関係を一つのチャートで俯瞰できるパネルです。

**主な活用方法:**
- 各軸にハイパーパラメータ（学習率、バッチサイズ、モデルファミリーなど）と結果メトリクス（eval loss、success rate など）を配置
- 複数の run がライン（線）として描画され、パラメータの組み合わせと結果の相関を直感的に把握可能
- 軸上の範囲をドラッグして選択すると、該当する run のみがハイライトされ、インタラクティブに絞り込みが可能

<!-- 実際の run の Parallel Coordinates Chart のスクリーンショットを挿入 -->

詳細: https://docs.wandb.ai/models/app/features/panels/parallel-coordinates

#### W&B Sweeps によるハイパーパラメータ探索の自動化

手動で大量の run を回してハイパーパラメータを探索することも可能ですが、**W&B Sweeps** を使うことで、このプロセスを自動化できます。

**手動実行の場合:**
- 実行方法: 各パラメータを手動で設定し、個別に run を起動
- 探索戦略: 研究者の経験と直感に依存
- スケーラビリティ: 実験数が増えると管理が困難
- 早期終了: 手動判断

**W&B Sweeps の場合:**
- 実行方法: YAML で探索空間を定義し、`wandb agent` がパラメータの選択と run の起動を自動実行
- 探索戦略: Grid / Random / Bayesian（ベイズ最適化）から選択可能
- スケーラビリティ: Controller が複数の Agent を協調させ、大規模探索を効率的に実行
- 早期終了: Hyperband 等のアルゴリズムによる自動早期終了をサポート

Sweeps は特に、探索すべきパラメータの組み合わせが多い場合に有効です。

---

### Pinned Runs & Baseline Comparison

ベースラインとなる run を設定し、最大 5 つの run をピン留めすることで、フィルタ条件に関係なく、常に Run Selector の上部に表示させることができます。

**主な利点:**
- ベースライン run がライングラフ上でハイライト表示され、新しい実験との差分が一目で分かる
- フィルタを変更しても比較対象が維持されるため、分析の文脈が失われない

デモ動画: https://www.loom.com/share/b8a5352c01594778ac38ff0ad5fa18d8

---

### Semantic Colors by Config Values

Config 値（ハイパーパラメータ）に基づいて run の色を自動で割り当てる機能です。

**主な利点:**
- 学習率やバッチサイズ、モデルファミリーなどの Config 値に応じて run の色が自動設定される
- 集約（Aggregation）を必要とせず、個々の run レベルでConfig の影響を視覚的に把握できる
- 他のメンバーの Workspace でも Semantic Legend が利用可能で、チーム内での共有・議論が容易

大量の run を比較する際に、パラメータごとの傾向やパターンを素早く発見でき、洞察のスピードが大幅に向上します。

デモ動画: https://www.loom.com/share/640c6d2c04ec4c328c92b530516778bd

---

### Saved Views

W&B Workspace の **Saved Views** 機能を使うと、パネルの配置・フィルタ条件・表示設定を「ビュー」として保存し、いつでも再利用できます。

**主な利点:**
- 分析の目的（学習曲線の比較、評価スコアの確認、動画レビューなど）に応じた専用ビューを作成
- チームメンバーとビューを共有し、同じ視点での議論が可能
- 個人用のビューとチーム共有のビューを分離して管理

![Saved Views メニュー](https://mintcdn.com/wb-21fd5541/4kbs1cW6PdjDOqU3/images/app_ui/Menu_Views.jpg?w=2500&fit=max&auto=format&n=4kbs1cW6PdjDOqU3&q=85&s=9b6ce8a5be6b812d6d3520af75e75bbc)

詳細: https://docs.wandb.ai/models/track/workspaces

---

## 評価体系の充実

Physical AI の開発において、評価は学習と同等以上に重要です。モデルの改善を定量的に追跡し、シミュレーション結果を定性的に確認できる体制がなければ、実験の良し悪しを正しく判断できません。

本プロジェクトでは、以下の 2 つの評価ニーズに対応しました。

- **定量評価**: 評価データセットに対する成功率やスコアを一括で確認
- **定性評価**: シミュレーションのロールアウト動画を視覚的に確認

### 実装した評価パイプライン

`scripts/eval_aloha_dataset.py` として、ALOHA Sim 環境での自動評価スクリプトを構築しました。

- **評価マニフェスト**に基づき、固定された評価データセット（subsample / full）に対してシミュレーションを実行
- 各エピソードの **成功/失敗、最大報酬、ステップ数** を記録
- ロールアウト動画を自動生成し、W&B にロギング
- 評価結果を **W&B Table**（Leaderboard 行 + Example 詳細行）として保存
- すべての結果が同時に **`run.summary`** にも反映され、Workspace 上でのフィルタ・比較に利用可能

---

### 評価ベンチマークに対する結果の確認

#### Runs Table による確認

W&B の Runs Table を使えば、すべての run の summary メトリクス（`eval_full/success_rate`、`eval_full/primary_score` など）を一覧で確認できます。カラムのソートやフィルタにより、最も性能の高い run を素早く特定できます。

#### W&B Table を活用した Leaderboard

W&B Table を利用することで、より構造化された比較が可能になります。

**仕組み:**
- 各評価 run が **1 行の Leaderboard 行**（`eval_full/leaderboard`）を W&B Table として保存
- Workspace 上で複数の run の Table パネルを表示すると、各 run の結果が自動的に集約され、実験横断の比較テーブルが構成される

Leaderboard の各行には以下の情報が含まれます:

- `eval_name` — 評価名
- `config_name` — 使用した設定名
- `checkpoint_step` — チェックポイントのステップ数
- `primary_score` — 主要スコア。Transfer Cube では成功率と同義
- `success_rate` — 成功率
- `mean_max_reward` — 平均最大報酬
- `num_examples` — 評価したエピソード数

---

### シミュレーション動画の可視化

IsaacSim を使用する場合はシミュレーション結果の可視化がネイティブにサポートされていますが、OpenPI のコードベースにはその機能がありませんでした。本プロジェクトでは、ALOHA Sim のロールアウト動画を W&B に自動ロギングする仕組みを追加しました。

実装の詳細は、[GitHub リポジトリの `scripts/eval_aloha_dataset.py`](https://github.com/olachinkei/openpi) と `src/openpi/utils/wandb/videos.py` を参照してください。

<!-- ここに実際の OpenPI ロールアウト動画のスクリーンショットまたは W&B Panel を挿入 -->

#### Media Panel の活用

W&B の Media Panel には、動画・画像の評価を効率化する多くの機能があります。以下に、Physical AI の評価ワークフローに特に有用な機能を紹介します。

詳細: https://docs.wandb.ai/models/app/features/panels/media#media-panels

---

#### Synchronized Video Playback（動画の同期再生）

マルチモーダルモデルの評価では、定性的な確認が不可欠です。Synchronized Video Playback を使うと、複数の動画を同期して再生・一時停止・スクラブ・速度調整でき、視覚的な差異を即座に把握できます。

**主な利点:**
- **正確な比較**: 一度スクラブすれば全動画が追従。時間的な一貫性やモーションのアーティファクトを検出しやすい
- **Workspace 全体での同期**: Workspace、Section、または Panel レベルで一度設定すれば、すべての Media Panel の動画が同期
- **自動再生 & ループ**: 動画の自動再生やループ再生を設定可能

**Physical AI での活用例:**
- 異なる学習 run 間でのロールアウト動画の比較
- スタイル・タイミング・動作の品質確認（Visual QA）
- 回帰・修正のフレーム単位での確認

設定方法: Settings → Media → Sync で「Sync video playback」を有効化

デモ動画: https://www.loom.com/share/244cb3be1de04ad8a4ee22654d730b0f

---

#### Synced Media Sliders（メディアスライダーの同期）

複数パネル間でメディアスライダーを同期してステップ実行でき、関連する画像群を効率的に比較できます。Section 単位または Workspace 全体で同期を設定可能です。

デモ動画: https://app.getbeamer.com/pictures?id=519192-Ge-_vXnesu-_vTUj77-977-977-977-9Pu-_vVtqbXDWhu-_vX0X77-91r7vv70uRzRd77-9&v=4

---

#### Media Panel の一括制御

ライングラフと同様に、Media Panel の設定を Workspace 全体または Section 単位で一括管理できます。

- Epoch ごとの表示やカスタムグリッドパターンへの配置を、パネルごとに個別設定する必要なく一括で適用
- 必要に応じて個別パネルのオーバーライドも可能

設定時間を削減し、分析により多くの時間を割くことができます。

デモ動画: https://app.getbeamer.com/pictures?id=504954-Y17vv73vv73vv73vv71_Ie-_vUI1zZg4PO-_ve-_vWHvv73vv70oC--_ve-_ve-_ve-_vXta3ZZz&v=4

---

## アセット管理

### Artifacts: 学習中のアセットのバージョン管理

W&B Artifacts は、データセット・モデルチェックポイント・評価結果などの成果物をバージョン管理し、実験間のリネージ（系譜）を追跡するための仕組みです。

**主な価値:**
- **バージョン管理**: すべてのアーティファクトに自動的にバージョン番号が付与され、任意の過去バージョンに遡れる
- **リネージ追跡**: どの run がどのデータセットを使い、どのチェックポイントを生成したかの依存関係を自動的に記録
- **エイリアス**: `latest`、`best-eval`、`final` などのエイリアスを設定し、意味のあるラベルで参照可能
- **ストレージ効率**: 変更がないファイルは重複保存されず、差分のみが記録される

![Artifacts ページ](https://mintcdn.com/wb-21fd5541/wKCrMJZKG3PxyJhv/images/artifacts/artifacts_landing_page2.png?w=2500&fit=max&auto=format&n=wKCrMJZKG3PxyJhv&q=85&s=3f9dfc871cf363d168542c779d61a6c6)

#### 本プロジェクトでの実装

`src/openpi/utils/wandb/artifacts.py` に `WandbArtifactManager` を実装し、以下のアーティファクトを管理しています。

- **`model-checkpoint`** — 学習中のマイルストーンチェックポイント（`step-N`、`latest`、`final` エイリアス付き）
- **`dataset-manifest`** — 評価データセットのマニフェスト（subsample / full）
- **`eval-results`** — 評価結果（メトリクス + 詳細テーブル）
- **`eval-video-bundle`** — 評価ロールアウト動画のバンドル

チェックポイントの公開ポリシーとして、すべてのステップを保存するのではなく、マイルストーンステップ（設定した間隔）と最終ステップのみを W&B に公開することで、ストレージコストを抑えています。

---

### Registry: 組織横断のアセット共有と検索

W&B Registry は、Artifacts の上位レイヤーとして、組織全体でモデルやデータセットを共有・検索・管理するための集中型カタログです。

**主な価値:**
- **集中管理**: 組織内のすべてのモデル・データセットを一元的に管理し、検索可能にする
- **ガバナンス**: アクセス制御やバージョン管理のポリシーを組織レベルで適用
- **Automation との連携**: Registry にリンクされたアーティファクトのバージョン変更をトリガーとして、デプロイやテストの自動化が可能
- **カスタムコレクション**: チーム固有の分類やタグ付けで、目的に応じたコレクションを作成

![Registry ページ](https://mintcdn.com/wb-21fd5541/AXlwJe6YUBax3n2I/images/registry/registry_landing_page.png?w=2500&fit=max&auto=format&n=AXlwJe6YUBax3n2I&q=85&s=88562e36bd19c3d5a7e492a6cabb604c)

Physical AI の開発では、多数のチェックポイントが生成されます。Registry を活用することで、優れた性能を示したモデルをチーム内で共有し、本番デプロイや追加学習のベースとして再利用する体制を構築できます。

---

## おわりに

本レポートでは、OpenPI に対して実装した W&B 統合の全体像を紹介しました。

- **学習フェーズ**: 構造化された Config とメタデータにより、複数実験の効率的な比較を実現
- **評価フェーズ**: 自動評価パイプラインと W&B Table による Leaderboard で、定量・定性の両面から評価
- **アセット管理**: Artifacts と Registry によるチェックポイント・データセットの体系的な管理

これらの統合により、Physical AI の開発ワークフロー全体を通じて、実験の再現性・追跡可能性・チーム内共有を向上させることができます。

実装の詳細やコードは [GitHub リポジトリ](https://github.com/olachinkei/openpi) を、Physical AI ワークフローにおける W&B の一般的な活用デモは [YouTube 動画](https://www.youtube.com/watch?v=45Beo0ZkJJA) をご確認ください。
