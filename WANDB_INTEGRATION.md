# W&B Integration

このリポジトリの W&B 連携は、OpenPI の学習・評価・データセット管理を W&B 上で再現できるようにするためのものです。主な役割は次の 4 つです。

1. 学習 run を W&B に作成し、loss や checkpoint step を安定した key で記録する。
2. dataset manifest や checkpoint を W&B Artifacts として登録・取得する。
3. ALOHA Sim の eval 結果を table、leaderboard、summary metric として記録する。
4. eval 動画を W&B media と video artifact の両方に保存する。

実装の共通部品は `src/openpi/utils/wandb/` にあります。`scripts/train.py`、`scripts/train_pytorch.py`、`scripts/eval_aloha_dataset.py` はこの utility を使って W&B と接続します。

## 有効化の考え方

W&B 連携は `TrainConfig.wandb_enabled` が true のときに有効になります。通常のオンライン同期では、環境変数も明示しておくと挙動が分かりやすくなります。

```bash
export WANDB_MODE=online
export WANDB_ENTITY=wandb-smle
export WANDB_PROJECT=openpi-aloha-wandb-integration
```

project 名は `WANDB_PROJECT` が最優先です。指定がない場合は config の `project_name` を使います。entity は `TrainConfig.wandb_entity`、または `WANDB_ENTITY` から解決されます。

run id は checkpoint directory 配下の `wandb_id.txt` に保存されます。これにより、評価 script が同じ学習 run に resume して動画や eval 結果を追記できます。

## 記録される Run 情報

学習 run には config 全体に加えて、次のような追跡用 metadata が入ります。

- backend: `jax` または `pytorch`
- config name: 例 `pi0_aloha_sim`
- task name / dataset name
- eval manifest path / final eval manifest path
- git sha / branch
- hostname、Slurm job id、node name、GPU 情報

tags は明示的なものだけを使います。config の `wandb_tags` と環境変数 `WANDB_TAGS` が対象です。task name や dataset name のような機械的 metadata は tags ではなく config/summary に入ります。

## Metric Key

学習時は `train/*` に集約されます。

- `train/loss`
- `train/grad_norm`
- `train/param_norm`
- `train/learning_rate`
- `train/step_time_sec`
- `train/checkpoint_step`

評価時は split ごとに prefix が分かれます。

- periodic eval: `eval_subsample/*`
- final eval: `eval_full/*`

代表的な eval key は次の通りです。

- `primary_score`
- `success_rate`
- `mean_max_reward`
- `num_examples`
- `checkpoint_step`
- `videos/*`

## Artifacts

この連携では、W&B Artifacts を「入力データの固定」と「出力の保存」の両方に使います。

### 入力 Dataset Artifacts

config で次を指定すると、学習開始時に W&B から artifact を取得して使います。

- `train_dataset_artifact_ref`
- `eval_dataset_artifact_ref`
- `eval_final_dataset_artifact_ref`

実際に検証した artifact refs は次です。

- `pi0_aloha_sim-aloha-final-8gpu-20260331-train-dataset:v0`
- `pi0_aloha_sim-aloha-final-8gpu-20260331-eval-dataset:v0`
- `pi0_aloha_sim-aloha-final-8gpu-20260331-eval-final-dataset:v0`

`eval_dataset_artifact_ref` と `eval_final_dataset_artifact_ref` は、artifact 内の JSON manifest を checkpoint directory の `_wandb_inputs/resolved/` に download し、eval manifest path として使います。

### Dataset Artifact の作成

`publish_dataset_artifacts=True` にすると、現在の config から dataset specification と eval manifest を W&B Artifacts として登録できます。登録される artifact は次の種類です。

- train dataset spec
- periodic eval manifest
- final eval manifest

この仕組みにより、あとから同じ artifact ref を指定して同じ dataset split を再利用できます。

### Checkpoint Artifact

checkpoint は `model-checkpoint` artifact として保存できます。

- `wandb_checkpoint_artifact_interval=None`: final checkpoint のみ保存
- `wandb_checkpoint_artifact_interval=5000`: 5000 step ごとに保存
- `wandb_checkpoint_artifact_interval<0`: checkpoint artifact 保存を無効化

alias は `latest`、`step-N`、final checkpoint では `final` が付与されます。

### Eval Output Artifacts

ALOHA Sim eval 後は、結果 JSON と動画 bundle が artifact として保存されます。

- `eval-results`
- `eval-video-bundle`

動画は W&B の media panel にも出ますが、artifact としても残るため、run 画面以外からも再取得できます。

## 動画連携

評価 script は rollout 動画を mp4 として保存し、W&B に次の 2 系統で記録します。

1. `wandb.Video` による run media
2. `eval-video-bundle` artifact

主な入口は `scripts/eval_aloha_dataset.py` です。学習 run に追記したい場合は、checkpoint directory の `wandb_id.txt` を使って既存 run に resume します。

CoreWeave 上の MuJoCo headless rendering では、次の環境変数を使って検証しました。

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export OPENPI_VIDEO_BACKEND=pyav
```

`OPENPI_VIDEO_BACKEND` を指定しない場合も、現在は `pyav` が既定になります。これは CoreWeave 環境で `torchcodec` が FFmpeg 共有ライブラリ不足により失敗するケースを避けるためです。

## Stable Run Groups

run group は W&B 上で用途別に見やすくするためのものです。

- `inputs`: dataset artifact 登録
- `train`: 学習
- `baseline_eval`: baseline checkpoint の評価
- `periodic_eval`: checkpoint ごとの評価
- `final_eval`: 最終評価

## Table / Leaderboard

eval 結果は W&B table にも保存されます。

- `eval_subsample/examples`
- `eval_subsample/leaderboard`
- `eval_full/examples`
- `eval_full/leaderboard`

examples table には example id、score、checkpoint step、動画参照などが入ります。leaderboard table には run id、source train run、checkpoint alias、success rate、artifact ref などが入ります。

## 検証済み Run

CoreWeave GPU 上で、artifact 取得、学習、eval 動画保存まで確認した run です。

- W&B run: https://wandb.ai/wandb-smle/openpi-aloha-wandb-integration/runs/km2jnrc2
- state: `finished`
- 使用 artifact: 上記 3 つの dataset artifact refs
- 学習: 10 step smoke training
- eval 動画: W&B media と `eval-video-bundle` artifact の両方に保存済み

動画保存の確認では、学習 checkpoint step 9 に対して ALOHA Sim eval を実行し、run に resume して動画を追加しました。

## 実装ファイル

- `src/openpi/utils/wandb/run_context.py`: run 作成、resume、summary、config metadata
- `src/openpi/utils/wandb/artifacts.py`: artifact 登録、checkpoint alias、artifact ref 生成
- `src/openpi/utils/wandb/videos.py`: mp4 を `wandb.Video` として logging
- `src/openpi/utils/wandb/tables.py`: typed record から `wandb.Table` を生成
- `src/openpi/utils/wandb/leaderboard.py`: leaderboard table と summary 更新
- `src/openpi/training/dataset_artifacts.py`: dataset artifact の作成・取得・manifest binding
- `scripts/train.py`: JAX training と W&B logging
- `scripts/train_pytorch.py`: PyTorch training と W&B logging
- `scripts/eval_aloha_dataset.py`: ALOHA Sim eval、動画、table、artifact logging

## よく見る場所

W&B 画面で確認する場合は、次の順に見ると分かりやすいです。

1. run の Overview で state が `finished` になっていること。
2. Charts で `train/loss` と `train/checkpoint_step` が出ていること。
3. Artifacts で dataset refs、checkpoint、eval results、eval video bundle が紐づいていること。
4. Media または Files で eval mp4 が保存されていること。
5. Tables で `eval_subsample/examples` と `eval_subsample/leaderboard` が作られていること。
