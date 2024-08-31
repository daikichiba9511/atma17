import torch
import torch.nn as nn
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput


class AuxHead(nn.Module):
    def __init__(self, out_features: int) -> None:
        super().__init__()
        self.fc = nn.LazyLinear(out_features)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        out = self.fc(logits)
        return out


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

        # Rating for 5 classes
        self.aux_head = AuxHead(out_features=6)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> SequenceClassifierOutput:
        out = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        aux_out = self.aux_head(out.logits)
        output = SequenceClassifierOutput(
            logits=out.logits,
            hidden_states=out.hidden_states,
            attentions=out.attentions,
            loss=out.loss,
        )
        output["aux_logits"] = aux_out
        return output


def _test_model() -> None:
    m = Atma17CustomModel(model_path="microsoft/deberta-v3-large")
    i = torch.randint(0, 1000, (8, 256))
    out = m(i, torch.randint(0, 2, (8,)))
    print(out)
    print(out.aux_logits.shape)


if __name__ == "__main__":
    _test_model()
