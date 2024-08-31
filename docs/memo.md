- baselineで、CV_mean: 0.9668424424548234, LB: 0.9702
  - エラー分析するとまだ伸ばせそう
  - RatingをAuxiliary Taskにしてみる
  - 入力にそのIDの数を入れる `group_by("Clothing ID").count()`
  - Age, Positive Feedback Countも入れる

- clothing_masterをうまく使えないだろうか？
    - とりあえず後回し