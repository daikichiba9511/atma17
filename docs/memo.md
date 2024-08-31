- baselineで、CV_mean: 0.9668424424548234, LB: 0.9702
  - エラー分析するとまだ伸ばせそう
  - RatingをAuxiliary Taskにしてみる
  - 入力にそのIDの数を入れる `group_by("Clothing ID").count()`
  - Age, Positive Feedback Countも入れる

- clothing_masterをうまく使えないだろうか？
    - とりあえず後回し

- label smoothingはpublicにはきいてないけどPrivateにはきいてた
  - exp006(label_smoothing=0.1): CV: 0.9673735264656412, Public: 0.9713, Private: 0.9767
  - exp007(label_smoothing=0.0): CV: 0.9724848032127599, Public: 0.9722, Private: 0.9761