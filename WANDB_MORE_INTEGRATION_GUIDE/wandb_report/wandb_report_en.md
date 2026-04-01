# This file is for developing W&B Report for further explanation

# Title of Report: Openpi with W&B
# Purpose of this report:
- Openpi　（https://github.com/Physical-Intelligence/openpi） is getting popular in Physical AI. It has an integration with W&B as it uses pytorch, that has W&B integration.
- But, it is just for simple experiment tracking. Kei expanded the integaration and created more how tos and tips more to demonstrate how your workflow will be better with W&B.
- This reports show the overall of the integrations. For technical details, please refer:
- Isaac simだとシミュレーションの可視化のintegrationはあるが、Openpiのコードにはない。それをdemonstrateしてみた


# Asset
- W&B Report URL (EN): 
- W&B Report URL (JP): 
- github (more integaration with W&B are implemented): 
- Youtube: Physical AIのWorkflowで使える一般的なW&Bの機能デモについては以下を確認してください。https://www.youtube.com/watch?v=45Beo0ZkJJA


# Overall

題材: PLAN.md やWANDB_INTEGRATION.mdからALOHAのやつをなんかしたことを書いて
W&B values
- こんなことを追加した
- こんなことができる




# 学習: 複数実験の比較
大量に実験をした際に、容易に比較することができる。
wandb.initをする際に、いくつかの情報を残すと、後でWorkspaceで便利にみることができる。

import wandb
config = 
with wandb.init(
    project="cat-classification",
    notes="My first experiment",
    tags=["baseline", "paper1"],
    config={"epochs": 100, "learning_rate": 0.001, "batch_size": 128},
) as run:
    ...
    
今回のスクリプトでは、XXでそれを行っている
configの詳細こちら https://docs.wandb.ai/models/track/config

- Parallel coordinates chart
    - 以下のdocからvalueを書いて。使い方は、docmentに渡したら良い。
    https://docs.wandb.ai/models/app/features/panels/parallel-coordinates
    - <今回のrunの実際の結果をここに>
    - W&B Sweepsというものがあることも説明して。自分で手動で大量にrunを回すのと、W&B Sweepsがagent, controlerを使って勝手にやってくることの違いを簡潔に書いて

- Pinned Runs and Baseline Comparison
    - You can now set a baseline run and pin up to five runs to keep them on top of the run selector—regardless of filters. 

The baseline run is highlighted in line plots for easy comparison, with more analysis features on the way.
        - demo video; https://www.loom.com/share/b8a5352c01594778ac38ff0ad5fa18d8
- Semantic Colors by Config Values
    - You can now color runs based on config (e.g. hyperparameter) values, without requiring aggregation. This is a powerful new way to visually explore and analyze your AI experiments.

Here’s what’s new:

Set run colors automatically based on a chosen hyperparameter configuration (e.g., learning rate, batch size, model family).

Visually spot config-driven effects immediately.

Now works in other people’s workspaces. You can now use Semantic legends when exploring shared or workspaces.
Why this matters

This update makes it much easier to spot trends, patterns, and config-driven effects in complex experiments, helping you find insights faster,  even across large, multi-run comparisons.


    - demo video: https://www.loom.com/share/640c6d2c04ec4c328c92b530516778bd
- Saved View
    - fill here by refering https://docs.wandb.ai/models/track/workspaces
    - get images from https://mintcdn.com/wb-21fd5541/4kbs1cW6PdjDOqU3/images/app_ui/Menu_Views.jpg?w=2500&fit=max&auto=format&n=4kbs1cW6PdjDOqU3&q=85&s=9b6ce8a5be6b812d6d3520af75e75bbc

# 評価体系の充実
評価は大事。
- 評価データセットに対する精度を一気に確認したい。
- Simulationの結果を確認したい

今回、Scriptに以下を追加した
- 評価を入れた(PLAN.md やWANDB_INTEGRATION.mdからALOHAのデータを使ってどうやったかを入れて)
- 評価を


- 評価ベンチマークに対する結果の確認
    - Runs tableで確認する方法
        ...
    - W&B Table機能を使ったleaderboard
        - W&B Tableを利用することで、それができる。一run 一行でtableを保存して、workspaceで表示すると、複数実験の結果が自動的にaggregateされ、比較テーブルができる
    - 


- 一つ一つの可視化
    - Isaac Simを使うとSimulationの結果を簡単に可視化することができるが、Openpiのコードではそれがneativeにできないので追加をした。追加の方法については、githubのXXを見て欲しい
    - 今回の
        <ここに実際のopenpiの可視化結果を入れる reportでは、pannelを入れると良い>
    - Media pannelについては、細かくできることが増えており、以下のdocを参考にして欲しい。いくつかの機能を以下で紹介する
        - https://docs.wandb.ai/models/app/features/panels/media#media-panels
        - Synchronized Video playback for video evaluations
            - When evaluating multimodal models, qualitative evaluation is crucial. With synchronized video playback, you can play, pause, scrub, and adjust the speed of multiple videos in sync, allowing visual differences to be immediately apparent.
            - Why you’ll love it
Accurate evaluation: Scrub once; every video follows. Great for spotting temporal consistency, motion artifacts, flicker, and style shifts.

Works for all media panels with videos in your workspace: Configure it once at Workspace, Section, or Panel level, and all videos in media panels across sections or panels in your workspace are synced. 

Auto-play & Auto-loop: Configure videos to auto-play or on loop.

Perfect for
Comparing videos across training runs 

Quick visual QA on style, timing, motion, and overall vibe check

Reviewing regressions/fixes by aligning moments frame-for-frame

Getting started
Go to Settings → Media → Sync and toggle Sync video playback.

Tell us what you think and what would make image and video evaluations even better. 


            - demo video; https://www.loom.com/share/244cb3be1de04ad8a4ee22654d730b0f
        - Synced media sliders
            - you can now step through media sliders in sync, across multiple panels, letting you easily compare across several related images.  You can sync panels at either the section or the entire workspace level. 
            - demo video; https://app.getbeamer.com/pictures?id=519192-Ge-_vXnesu-_vTUj77-977-977-977-9Pu-_vVtqbXDWhu-_vX0X77-91r7vv70uRzRd77-9&v=4

        - Control media pannnels in bulk
            - Just like line plots, you can now manage all your media panel settings at once—across an entire workspace or a specific section.

Easily configure media panels to display by epoch or arrange them into customized grid patterns, without having to adjust each panel separately.  (You can always override the global settings for individual panels, though.)  We're aiming to reduce setup time, letting you focus more on analysis and less on configuration.
            - demo video; https://app.getbeamer.com/pictures?id=504954-Y17vv73vv73vv73vv71_Ie-_vUI1zZg4PO-_ve-_vWHvv73vv70oC--_ve-_ve-_ve-_vXta3ZZz&v=4


# アセット管理
## Artifacts: 学習中の大量のアセット管理
- Artifactsのvalueをdocから調べて書いて
- 使い方のイメージは以下を使って
    https://mintcdn.com/wb-21fd5541/wKCrMJZKG3PxyJhv/images/artifacts/artifacts_landing_page2.png?w=2500&fit=max&auto=format&n=wKCrMJZKG3PxyJhv&q=85&s=3f9dfc871cf363d168542c779d61a6c6

- 今回 implementationした内容を書いて


## Registry: 優れたものを組織で共有し、検索可能にするアセット管理
- registryのvalueをdocから調べて簡潔に書いて
- https://mintcdn.com/wb-21fd5541/AXlwJe6YUBax3n2I/images/registry/registry_landing_page.png?w=2500&fit=max&auto=format&n=AXlwJe6YUBax3n2I&q=85&s=88562e36bd19c3d5a7e492a6cabb604c


# 終わりに
- 便利に使ってねということ