# AtmaCup 17

competition page: https://www.guruguru.science/competitions/24

## Description

- 12 th place solution for [AtmaCup 17](https://www.guruguru.science/competitions/24)
- Task: predict the item's recommendation from data including review text.
- solution
  - please see solution.md.
  - Discussion post version (Ja): https://www.guruguru.science/competitions/24/discussions/2a97a005-de13-4e2b-87f4-1b2b35d8011e/

## Usage

1. Clone this repository
2. Download datesets from [the competition page](https://www.guruguru.science/competitions/24/)
3. unzip the datasets and put them in the `input` directory
4. Docker build and Run

   ```bash
   docker compose up -d
   # to interact with the container
   docker compose local-dev exec bash
   ```
5. run the following command

   ```bash
   make setup
   python -m src.exp.exp002
   python -m src.exp.exp004
   python -m src.exp.exp006
   python -m ensemble
   ```

## References

[1] takaito, [atmaCup17] Tutorial Notebook① DeBERTa Large Modelの学習 (少し手を加えるとLB: 0.9718), https://www.guruguru.science/competitions/24/discussions/4aee3312-ba40-4934-9035-22c7d2b51c09/

[2] nishimoto, 2023-24年のKaggleコンペから学ぶ、NLPコンペの精度の上げ方, https://zenn.dev/nishimoto/articles/974f2a445f9d74

[3] AI SHIFT, 【AI Shift/Kaggle Advent Calendar 2022】Kaggleで学んだBERTをfine-tuningする際のTips④〜Adversarial Training編〜, https://www.ai-shift.co.jp/techblog/2985