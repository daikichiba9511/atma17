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


class FFN(nn.Module):
    def __init__(self, out_features: int, dropout_rate: float = 0.2, hidden_size: int | None = None):
        super().__init__()
        if hidden_size is None:
            hidden_size = out_features * 4
        self.fc1 = nn.LazyLinear(hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Atma17CustomModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, model_path: str) -> None:
        model_config = self.config_class.from_pretrained(model_path)
        model_config.num_labels = 2
        print(model_config)
        # model_config.num_labels = 1
        # model_config.attention_probs_dropout_prob = 0.0
        # model_config.hidden_dropout_prob = 0.0
        super().__init__(config=model_config)
        self.model_config = model_config
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, config=model_config)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.ffn = FFN(out_features=2, hidden_size=256)

        # Rating for 5 classes
        self.aux_head = AuxHead(out_features=6)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        more_features: torch.Tensor | None = None,
    ) -> SequenceClassifierOutput:
        bs = input_ids.size(0)
        out = self.model(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # shape: (bs, seq_len, hidden_size)
        last_hidden_state = out.hidden_states[-1]
        hidden_state = self.pool(last_hidden_state).view(bs, -1)
        if more_features is not None:
            hidden_state = torch.cat([hidden_state, more_features], dim=1)
        logits = self.ffn(hidden_state)
        aux_out = self.aux_head(out.logits)
        output = SequenceClassifierOutput(
            logits=logits,
            # hidden_states=out.hidden_states,
            # attentions=out.attentions,
            loss=out.loss,
        )
        output["aux_logits"] = aux_out
        return output


def _test_model() -> None:
    m = Atma17CustomModel(model_path="microsoft/deberta-v3-large")
    i = torch.randint(0, 1000, (8, 256))
    out = m(i, torch.randint(0, 2, (8,)), more_features=torch.randn(8, 10))
    print(f"{out.aux_logits.shape = }")
    print(f"{out.logits.shape = }")
    print(f"{out.hidden_states[-1].shape = }")
    print(f"{len(out.hidden_states) = }")


if __name__ == "__main__":
    _test_model()
