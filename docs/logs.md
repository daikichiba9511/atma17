# Log

## Exp000

- simple baseline
- CV_mean: 0.9668424424548234
- LB: 0.9702

## Exp001

- simple baseline
    + promptの作り方を変える Title: <title> [SEP] Review_Text: <review> text>
    + fill_null("none")で埋める

- CV_mean: 0.9676224890480876
- LB: 0.9697

- Tokenの仕方で少し変わる
    - Review_Text => ['_Review', '_', '_Text']
    - Review Text => ['_Review', '_Text']
    - 下の方がトークン数を節約できる

## Exp002

- simple baseline
    + aux task

- CV_mean: 0.9716400324509147
- LB: 0.9714

## Exp003

- FFNで他の特徴量と合わせる
- paddingは"max_length"に変更

## Exp004

- loss: SmoothFocalLoss
- CV_mean: 0.9712126630716782
- LB:

## Exp005

- base: exp002
- CNN+MaxPooling Head, lr=1e-5
- CV_mean: 0.9494960601323804


## Exp006

- base: exp005
- max_length: 256*3, base, lr=2e-5, label_smoothing=0.1
- CV_mean: 0.9283955404132418

- base: exp005
- max_length: 256*2, large, lr=2e-5, label_smoothing=0.1
- CV_mean: 0.9673735264656412

## Exp007

CV_mean: 0.9724848032127599