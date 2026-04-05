# OpenPI x W&B: Physical AI ワークフローをつなぐ統合ガイド

## はじめに

[OpenPI](https://github.com/Physical-Intelligence/openpi) は、[Physical Intelligence](https://www.physicalintelligence.company/) が公開しているオープンソースのロボティクス向けフレームワークで、Physical AI 領域で急速に注目を集めています。本レポートでは Physical AI の開発でより便利に W&B を使う方法を OpenPI を例に解説していきます。OpenPI は JAX と PyTorch の両方に対応しており、W&B との基本的な連携（loss や learning rate のロギング）は既に備わっていますが、追加でシミュレーション結果の可視化などの integration 機能も実装しています。

- [integrationを追加したgithub**](https://github.com/olachinkei/openpi) : 
- W&B Project（OpenPI integration 例）: 

OpenPI を例に、Physical AI における W&B の便利な使い方について、順番にみていきましょう。なお、追加 integration の技術的な実装の詳細については、[GitHub リポジトリのReadme](https://github.com/olachinkei/openpi)を参照してください。

なお、Physical AIにおけるW&Bの価値を解説したWhite Paperや一般的なデモ動画（こちらはIssacLabの例）もありますので、あわせてご参照ください。

- [White Paper : Advancing Physical AI: From learning to embodied intelligence](https://wandb.ai/site/resources/whitepapers/advancing-physical-ai/)
- [W&B デモ動画 （IsaacLab）](https://www.youtube.com/watch?v=45Beo0ZkJJA)
- [W&B Projectの例（IsaacLab）](https://wandb.ai/wandb-smle/isaaclab-wandb-crwv?nw=nwuseranushravvatsa)

AI の開発では、1 本の学習 run から大量の情報が生まれます。train loss だけでなく、評価用の episode 一覧、checkpoint、成功率、最大報酬まで見ないと、モデルの良し悪しを判断できません。ところが、これらが別々のスクリプト、別々の job、別々の保存先に散らばると、比較は一気に難しくなります。また、特にPhysical AIでは、metricsだけではなく、シミュレーションの可視化も重要になります。

そこで、OpenPIのコードに以下の観点で追加のintegrationを開発しました。
1. 実験比較基盤の強化
2. 可視化を伴う評価パイプラインの強化
3. Artifacts / Registry を使ったアセット管理
本レポートでは、追加integrationと合わせて、それを活用したUI上の便利な操作も解説していきます。

---

## 今回の題材: ALOHA Sim Transfer Cube

今回のデモでは、2 本腕の ALOHA ロボットがキューブを把持し、受け渡しし、安定して保持する **Transfer Cube** タスクを使いました。学習データは `lerobot/aloha_sim_transfer_cube_human`、設定は `pi0_aloha_sim` です。

このデータセットは全 50 episode で、この設定ではそのうち 10 episode を held-out の final eval 用に切り出し、学習には残り 40 episode を使います。さらに periodic eval では、その held-out 10 episode のうち固定の 4 episode を使います。学習は 20,000 step まで行います。

今回は W&B 上で学習・評価・アセット管理をどうつなぐかを示すデモストレーションが主目的のため、データセット規模は小さめです。そのため、ここでの結果は厳密なベンチマークや統計的に頑健な性能比較というより、ワークフローを確認するための例として参考にされてください。


---

## 1. 実験比較基盤の強化

### config に実験条件を詳細に保存
W&B では、loss のような時系列データだけでなく、各 run の条件を `config` に残せます。`config_name`、`task_name`、`dataset_name`、`backend`、`git sha`、`hostname` のような情報を入れておくと、あとから Workspace 上で特定の条件だけを絞り込んだり、あるパラメータごとに run を見比べたりしやすくなります。後で紹介するように、config の値を使って表示色を分けることもできます。

もともとの OpenPI の W&B integration では、比較に使いたい情報が `config` に十分そろっていませんでした。そこで OpenPI 側では `src/openpi/utils/wandb/run_context.py` を追加し、W&B run の初期化処理を共通化しました。これにより、後で比較や整理に使いやすい項目を、`wandb.config` に残せるようにしています。JAX と PyTorch のどちらを使う場合でも、同じ考え方で metadata を残せるようにしています。

例えば、W&B では次のようなイメージで run 情報を構造化できます。

```python
import wandb

wandb.init(
    project="openpi-integration",
    config={
        "config_name": "pi0_aloha_sim",
        "task_name": "Transfer cube",
        "dataset_name": "lerobot/aloha_sim_transfer_cube_human",
        "training_backend": "jax",
    },
    tags=["aloha", "transfer-cube"],
)
```


### Workspace 上で大量の実験を比較する

上記のconfigをもとに、W&B の Workspace 上で大量の run を比較する便利な機能を順に見ていきましょう。

### 便利機能 1: Parallel Coordinates Chart

[Parallel Coordinates Chart](https://docs.wandb.ai/models/app/features/panels/parallel-coordinates) は、複数のハイパーパラメータと結果指標の関係を一枚で俯瞰できるパネルです。例えば `learning_rate`、`batch_size`、`training_backend`、`eval/success_rate`、`eval_final/primary_score` を軸に置くと、どの設定の run が高い成功率につながったかを視覚的に追えます。

follow-up 実験を何本も回す場面では、「良さそうな run の帯」が見えるだけでも有用です。loss の最終値だけでは判断しづらいときでも、成功率や reward と一緒に見ると、設定と結果の関係が読みやすくなります。


大量の探索を手動で回す場合、毎回コマンドや設定を変えて job を投げる必要があります。**W&B Sweeps** を使うと、探索空間を YAML で定義し、`wandb agent` が候補設定の選定と run 起動を自動化することができます。今回は利用していませんが、簡単に使えるので、是非ご確認ください。


### 便利機能 2: Pinned Runs と Baseline Comparison

Pinned Runs は、比較の基準にしたい run を常に上部に残しておく機能です。新しい実験を何本も追加しても、ベースライン run が見失われにくくなります。

Baseline Comparison を併用すると、ラインプロット上で基準 run が強調されるため、「新しい run は本当に改善したのか」を短時間で判断できます。例えば公開 checkpoint を基準にして fine-tuning run を比べる、といった使い方がしやすくなります。

![Pinned Runs と Baseline Comparison](https://mintcdn.com/wb-21fd5541/57wwTAGN9Q-FX-xN/images/models/pinned-and-baseline-runs/runs-table-with-pinned-and-baseline-runs.png?w=1100&fit=max&auto=format&n=57wwTAGN9Q-FX-xN&q=85&s=288f18afe190c9e11ce65f9e3b3086e1)

デモ動画: https://www.loom.com/share/b8a5352c01594778ac38ff0ad5fa18d8


### 便利機能 3: Semantic Colors by Config Values

Semantic Colors by Config Values は、Config 値に応じて run の色を自動で付ける機能です。学習率ごと、モデル family ごと、backend ごとに色が揃うだけで、グラフの読みやすさはかなり変わります。

例えば、`training_backend=jax` と `training_backend=pytorch` の色が自動で分かれると、同じ Workspace 上で backend 差の傾向を直感的に追えます。集約前の個々の run が見やすいので、follow-up 実験の判断も速くなります。

![Semantic Colors by Config Values](https://mintcdn.com/wb-21fd5541/_OEDykSS2PIumrEw/images/track/color-code-runs-plot.png?w=1100&fit=max&auto=format&n=_OEDykSS2PIumrEw&q=85&s=93b9f741937503187baa665f41568973)

デモ動画: https://www.loom.com/share/640c6d2c04ec4c328c92b530516778bd

### 便利機能 4: Saved Views

W&B Workspace の [Saved Views](https://docs.wandb.ai/models/track/workspaces) は、パネル配置、filter 条件、色分け、表示対象をそのまま保存できる機能です。

これは単なるレイアウト保存ではありません。研究の観点そのものを保存できるのが重要です。例えば次のような view を分けておくと便利です。

- 学習確認用: `train/loss` と `train/grad_norm` を中心に見る view
- 中間評価確認用: `eval/success_rate` と `eval/leaderboard` を中心に見る view
- 動画レビュー用: `eval_final/videos` を中心に見る view

チームで同じ Saved View を共有しておくと、「どの画面を見ながら議論しているか」が揃います。これだけでもレビューの速度はかなり変わります。

![Saved Views メニュー](https://mintcdn.com/wb-21fd5541/4kbs1cW6PdjDOqU3/images/app_ui/Menu_No_views.jpg?w=1100&fit=max&auto=format&n=4kbs1cW6PdjDOqU3&q=85&s=7ee0771a9d10880e774d04deff43ed01)


### 次の実験を自動で決めて実行する: W&B Skills

W&B は [Skills](https://wandb.ai/site/skills/) も提供しています。これは coding agent が W&B の実験管理機能を使いやすくするための仕組みで、学習結果や run の比較結果を参照しながら、次に試す条件を検討したり、必要に応じて実験コードの更新まで含めて作業を進めたりできます。

たとえば、いくつかの学習が終わった段階で run を見直し、どの条件を次に試すかを agent に考えさせる、といった使い方ができます。これをうまく組み込むと、結果を見ながら次の実験を継続的に改善していくワークフローを作りやすくなります。

今回の実験でも、後半の条件検討では W&B Skills を補助的に使いました。W&B Skillsはまだ新しいツールになります。是非、W&B Skillsを使い、たくさんfeedbackをいただければと思います！

---

## 2. 評価パイプラインの強化

Physical AI ではシミュレーションの確認がとても重要です。成功率や報酬のような定量指標だけでは、ロボットが実際にどう動いたのか、どこで不安定だったのか、どのように失敗したのかまでは分かりません。そのため、評価ではシミュレーションを可視化して挙動を確かめることが欠かせません。

つまり、Physical AI の評価では次の 2 つがどちらも重要です。

- 定量評価: 成功率、最大報酬、ステップ数など、run 間で比較しやすい指標
- 定性評価: シミュレーションのロールアウト動画を見て、動き方や失敗パターンを確認すること


### 定量評価: W&B Table で leaderboard を作る

今回追加した `scripts/eval_aloha_dataset.py` は、評価結果を JSON / CSV に保存するだけでなく、W&B Table としても記録します。W&B Table は、列構造を持ったデータをそのまま UI 上で扱える仕組みです。

今回の工夫は 2 つあります。1 つ目は、checkpoint ごとの評価結果を俯瞰できる leaderboard を用意したことです。これにより、複数 run を Workspace 上で並べたときに、どの checkpoint が良かったかを比較しやすくなります。

2 つ目は、集約結果だけで終わらせず、サンプルごとの個別評価も残せるようにしたことです。これにより、「全体スコアは近いが、どのサンプルで差が出ているか」「どの episode で失敗したか」といった点まで追えるようになります。

Table にどの情報を持たせているかの詳細は `scripts/eval_aloha_dataset.py` と `src/openpi/utils/wandb/leaderboard.py` に実装しています。主要な評価値は `run.summary` にも反映しているため、Table を開かなくても Workspace 上の filter や sort に使えます。

<!-- ここに OpenPI の leaderboard table のスクリーンショットを挿入 -->

### 定性評価: ロールアウト動画を W&B Media Panel で見る

今回の integration では、ALOHA Sim の rollout を `wandb.Video` として記録し、W&B Media Panel で step ごとに見られるようにしました。評価サンプルごとのロールアウト動画を残しているので、各 checkpoint でロボットがどう動いたかを後から追えます。

これにより、学習が進むにつれてロボットが本当にタスクを達成できるようになっているのか、あるいは途中で不安定さが残っていないかを、動画を見ながら確認できるようになりました。数値だけでは見えにくい改善や失敗の仕方を、step ごとのシミュレーション動画として追えるようにしたことがポイントです。


実装の詳細は、`scripts/eval_aloha_dataset.py`、`src/openpi/training/aloha_eval.py`、`src/openpi/utils/wandb/videos.py` を参照してください。

<!-- ここに OpenPI ロールアウト動画のスクリーンショット、または W&B Media Panel のキャプチャを挿入 -->

### Media Panel で特に便利な機能

[Media Panels の詳細](https://docs.wandb.ai/models/app/features/panels/media#media-panels) にもある通り、W&B の Media Panel は 2025 年にも機能拡充が進みました。Physical AI への注目が高まる中で、シミュレーションやロールアウト動画を扱いやすくするための改善も進んでいます。

ここでは、その中でも Physical AI の文脈で特に便利な Media Panel の機能を紹介します。

#### Synchronized Video Playback

複数動画を同期再生できる機能です。ある checkpoint の成功例と失敗例、あるいは JAX run と PyTorch run の同じ prompt を並べて見るときに便利です。一度スクラブすると全動画が同じ時刻にそろうので、把持タイミングや受け渡し動作の違いをフレーム単位で比較できます。

デモ動画: https://www.loom.com/share/244cb3be1de04ad8a4ee22654d730b0f

#### Synced Media Sliders

複数 panel のメディアスライダーを同期できる機能です。動画だけでなく、関連画像や派生メディアを同じ step で見比べたいときに役立ちます。

デモ動画: https://app.getbeamer.com/pictures?id=519192-Ge-_vXnesu-_vTUj77-977-977-977-9Pu-_vVtqbXDWhu-_vX0X77-91r7vv70uRzRd77-9&v=4

#### Media Panel の一括制御

Workspace 全体や Section 単位で panel 設定をそろえる機能です。動画 panel が増えるほど、個別設定の手間は無視できません。一括制御を使うと、表示方式やグリッド設定をまとめて揃えられるため、分析に集中しやすくなります。

デモ動画: https://app.getbeamer.com/pictures?id=504954-Y17vv73vv73vv73vv71_Ie-_vUI1zZg4PO-_ve-_vWHvv73vv70oC--_ve-_ve-_ve-_vXta3ZZz&v=4

---

## 3. アセット管理

### なぜアセット管理が重要なのか

Physical AI では、あとから「どのデータで学習し、どの checkpoint を評価し、その結果として何が得られたか」を辿れることが重要です。特に checkpoint が増えてくると、ローカルのファイル名だけで管理するのはすぐに難しくなります。

そこで今回の integration では、checkpoint のバージョン管理と、データからモデル、評価結果までのリネージを保つことを重視しました。

### Artifacts: checkpoint のバージョン管理を強化する

W&B Artifacts は、実験の中で大量に発生するcheckpointや結果ファイルをバージョン付きで管理するための仕組みです。今回特に強化したのは、学習で出てくるモデル checkpoint の扱いです。

`pi0_aloha_sim` では 5,000 step ごとのマイルストーンと final を公開しており、`step-5000`、`step-10000`、`step-15000`、`final` のような alias で辿れます。これにより、保存コストを抑えつつ、学習の進み方を checkpoint 単位で比較しやすくしています。ここのパラメータは容易に変更できるようになっています。

あわせて、lineage機能を使い、各 checkpoint が、どのデータセットとパラメータを利用して作られたのかを追いやすくし、モデルのバージョン管理が単なる保存ではなく、後から比較と検証に使える形になるようにしました。



### Registry: データとモデルのリネージを保つ

W&B Registry は、チームで使う dataset や model を参照しやすく保つための仕組みです。今回の連携では、まず学習用 dataset と評価用 dataset を Registry に載せ、それを参照して学習と評価を行う形にしました。Artifacts が実験中に出てくる大量のアセットを管理するための仕組みであるのに対して、Registry はチームで共有したい重要なアセットを参照しやすく保つための仕組みです。

学習データセットや評価データセットをRegistryに登録し、そのデータを利用することで学習と評価を開始するようにしています。

今回参照している artifact ref の例は次の 3 つです。

- train: `wandb32/wandb-registry-Physical AI - openpi/training dataset:v0`
- eval: `wandb32/wandb-registry-Physical AI - openpi/evaluation dataset for openpi:v0`
- holdout (`eval_final`): `wandb32/wandb-registry-Physical AI - openpi/evaluation dataset for openpi:v1`

実際には、今回の実装では次のように `use_artifact` で train / eval / holdout (`eval_final`) の artifact を参照しています。train は学習に使った dataset version を記録するために参照し、eval と holdout は manifest を解決して実際の評価に使っています。

```python
dataset_artifact_refs = _dataset_artifacts.configured_dataset_artifact_refs(config)
if train_ref := dataset_artifact_refs.get("train"):
    wandb.run.use_artifact(train_ref)

if eval_ref := dataset_artifact_refs.get("eval"):
    wandb.run.use_artifact(eval_ref)
    resolved_eval_manifest = _download_artifact_payload(eval_ref, artifact_root / "eval")
    if resolved_eval_manifest is not None:
        config = dataclasses.replace(config, eval_manifest_path=str(resolved_eval_manifest))

if holdout_ref := dataset_artifact_refs.get("eval_final"):
    wandb.run.use_artifact(holdout_ref)
    resolved_holdout_manifest = _download_artifact_payload(holdout_ref, artifact_root / "eval_final")
    if resolved_holdout_manifest is not None:
        config = dataclasses.replace(config, final_eval_manifest_path=str(resolved_holdout_manifest))
```


run ごとに同じ dataset を再 upload しなくてよいので、運用も整理しやすくなりますし、後からモデルを見直すときにも前提条件を追いやすくなります。

![Registry ページ](https://mintcdn.com/wb-21fd5541/AXlwJe6YUBax3n2I/images/registry/registry_landing_page.png?w=2500&fit=max&auto=format&n=AXlwJe6YUBax3n2I&q=85&s=88562e36bd19c3d5a7e492a6cabb604c)

---

## おわりに

今回の OpenPI x W&B integration では、1 本の run から、使った設定、checkpoint ごとの評価結果、ロールアウト動画、そして学習・評価データとのつながりまで追える状態を目指しました。

また、Physical AI では、loss や成功率だけでは十分ではありません。シミュレーションの結果をわかりやすく可視化することで、パフォーマンスをより深く理解することができるようにしました。

今回の integration や本レポートで紹介した機能が、Physical AI の実験基盤を整えるときの参考になればと思います。
