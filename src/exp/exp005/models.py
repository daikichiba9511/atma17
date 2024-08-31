import torch
import torch.nn as nn
import torch.nn.functional as F
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


class CNNHead(nn.Module):
    def __init__(
        self, out_channels: int = 1, hidden_channels: int = 256, kernel_size: int = 3, padding: int = 1
    ) -> None:
        super().__init__()

        self.cnn1 = nn.LazyConv1d(hidden_channels, kernel_size=kernel_size, padding=padding)
        self.cnn2 = nn.LazyConv1d(out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn1(x)
        x = F.relu(x)
        x = self.cnn2(x)
        return x


class MaxPooling(nn.Module):
    def __init__(self, out_dim: int = 2) -> None:
        super().__init__()
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_max_pool1d(x, self.out_dim)


class Atma17CustomModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, model_path: str) -> None:
        model_config = self.config_class.from_pretrained(model_path)
        model_config.num_labels = 2
        # model_config.num_labels = 1
        # model_config.attention_probs_dropout_prob = 0.0
        # model_config.hidden_dropout_prob = 0.0
        print(model_config)
        super().__init__(config=model_config)
        self.model_config = model_config
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, config=model_config)

        self.cnn_head = CNNHead(hidden_channels=128, kernel_size=2)
        self.pool = MaxPooling()
        # self.fc = nn.LazyLinear(2)
        # Rating for 5 classes
        self.aux_head = AuxHead(out_features=6)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> SequenceClassifierOutput:
        out = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask, output_hidden_states=True)
        hidden = self.cnn_head(out.hidden_states[-1].permute(0, 2, 1))
        logits = self.pool(hidden).view(hidden.size(0), -1)
        # logits = self.fc(self.pool(out.hidden_states[-1].permute(0, 2, 1)).reshape(bs, -1))

        aux_out = self.aux_head(out.logits)
        output = SequenceClassifierOutput(
            logits=logits,
            hidden_states=None,
            attentions=out.attentions,
            loss=None,
        )
        output["aux_logits"] = aux_out
        return output


def _test_model() -> None:
    m = Atma17CustomModel(model_path="microsoft/deberta-v3-large")
    i = torch.randint(0, 1000, (8, 256))
    out = m(i, torch.randint(0, 2, (8,)))
    print(out.logits.shape)
    print(out.aux_logits.shape)


if __name__ == "__main__":
    _test_model()
