# 12 th place solution

## 概要

- 以下のモデルのアンサンブル
    1. deberta-v3-large (exp002, CV: 0.9716400324509147, Public: 0.9714, Private: 0.9762)
        - auxiliary task for `Rating` classfication
        - max_length: 256
        - CrossEntropyLoss
    2. deberta-v3-large (exp004, CV: 0.9712126630716782, Public: 0.9703, Private: 0.9759)
        - auxiliary task for `Rating` classfication
        - max_length: 256
        - SmoothFocalLoss(reduction="mean", smoothing=0.1)
    3. deberta-v3-large (exp006, CV: 0.9673735264656412, Public: 0.9713, Private: 0.9767)
        - auxiliary task for `Rating` classfication
        - max_length: 512
        - CrossEntropyLoss

- Submission (ensemble)
    - Sub1:
        - 0.5 * `1.` + 0.5 * `2.`
        - Public: 0.9715, Private: 0.9768
    - Sub2:
        - 0.4 * `1.` + 0.4 * `2.` + 0.2 * `3.`
        - Public: 0.9712, Private: 0.9768

## うまくいかなかったこと

- hidden_states[-1]と生成した集約特徴量を使ってFFN Headに入れる
- CNN+MaxPooling Head

## 参考資料

[1] takaito, [atmaCup17] Tutorial Notebook① DeBERTa Large Modelの学習 (少し手を加えるとLB: 0.9718), https://www.guruguru.science/competitions/24/discussions/4aee3312-ba40-4934-9035-22c7d2b51c09/

[2] nishimoto, 2023-24年のKaggleコンペから学ぶ、NLPコンペの精度の上げ方, https://zenn.dev/nishimoto/articles/974f2a445f9d74

[3] AI SHIFT, 【AI Shift/Kaggle Advent Calendar 2022】Kaggleで学んだBERTをfine-tuningする際のTips④〜Adversarial Training編〜, https://www.ai-shift.co.jp/techblog/2985