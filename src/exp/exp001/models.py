import torch
import torch.nn as nn
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput


class Atma17CustomModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, model_path: str) -> None:
        model_config = self.config_class.from_pretrained(model_path)
        model_config.num_labels = 2
        # model_config.num_labels = 1
        # model_config.attention_probs_dropout_prob = 0.0
        # model_config.hidden_dropout_prob = 0.0
        super().__init__(config=model_config)
        self.model_config = model_config
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, config=model_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> SequenceClassifierOutput:
        out = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        return out


def _test_model() -> None:
    m = Atma17CustomModel(model_path="microsoft/deberta-v3-large")
    i = torch.randint(0, 1000, (8, 256))
    out = m(i, torch.randint(0, 2, (8,)))
    print(out)


if __name__ == "__main__":
    _test_model()
