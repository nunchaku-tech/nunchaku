"""huggingface model adapter

Wraps HuggingFace transformers (https://github.com/huggingface/transformers) models for use as a text tower in CLIP model.
"""

import re

import torch
import torch.nn as nn
from torch import TensorType

try:
    import transformers
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM, AutoTokenizer, PretrainedConfig
except ImportError:
    transformers = None

    class PretrainedConfig:
        pass


from .hf_configs import arch_dict


# utils
def _camel2snake(s):
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()


# TODO: ?last - for gpt-like models
_POOLERS = {}


class HFTextEncoder(nn.Module):
    """HuggingFace model adapter"""

    def __init__(
        self,
        model_name_or_path: str,
        output_dim: int,
        tokenizer_name: str = None,
        config: PretrainedConfig = None,
        pooler_type: str = None,
        proj: str = None,
        pretrained: bool = True,
        masked_language_modeling: bool = False,
    ):
        super().__init__()

        self.output_dim = output_dim

        # TODO: find better way to get this information
        uses_transformer_pooler = pooler_type == "cls_pooler"

        if transformers is None:
            raise RuntimeError("Please `pip install transformers` to use pre-trained HuggingFace models")
        if config is None:
            self.config = AutoConfig.from_pretrained(model_name_or_path)
            if masked_language_modeling:
                create_func, model_args = (
                    (AutoModelForMaskedLM.from_pretrained, model_name_or_path)
                    if pretrained
                    else (AutoModelForMaskedLM.from_config, self.config)
                )
            else:
                create_func, model_args = (
                    (AutoModel.from_pretrained, model_name_or_path)
                    if pretrained
                    else (AutoModel.from_config, self.config)
                )
            # TODO: do all model configs have this attribute? PretrainedConfig does so yes??
            if hasattr(self.config, "is_encoder_decoder") and self.config.is_encoder_decoder:
                self.transformer = create_func(model_args)
                self.transformer = self.transformer.encoder
            else:
                self.transformer = create_func(model_args, add_pooling_layer=uses_transformer_pooler)
        else:
            self.config = config
            if masked_language_modeling:
                self.transformer = AutoModelForMaskedLM.from_config(config)
            else:
                self.transformer = AutoModel.from_config(config)

        if pooler_type is None:  # get default arch pooler
            self.pooler = _POOLERS[(arch_dict[self.config.model_type]["pooler"])]()
        else:
            self.pooler = _POOLERS[pooler_type]()

        d_model = getattr(self.config, arch_dict[self.config.model_type]["config_names"]["width"])
        if (d_model == output_dim) and (proj is None):  # do we always need a proj?
            self.proj = nn.Identity()
        elif proj == "linear":
            self.proj = nn.Linear(d_model, output_dim, bias=False)
        elif proj == "mlp":
            hidden_size = (d_model + output_dim) // 2
            self.proj = nn.Sequential(
                nn.Linear(d_model, hidden_size, bias=False),
                nn.GELU(),
                nn.Linear(hidden_size, output_dim, bias=False),
            )

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids

    def forward(self, x: TensorType) -> TensorType:
        attn_mask = (x != self.config.pad_token_id).long()
        out = self.transformer(input_ids=x, attention_mask=attn_mask)
        pooled_out = self.pooler(out, attn_mask)

        return self.proj(pooled_out)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.gradient_checkpointing_enable()

    def init_parameters(self):
        pass
