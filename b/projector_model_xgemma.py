import torch
import torch.nn as nn
import re
from transformers import PretrainedConfig, PreTrainedModel

class GemmaProjectorConfig(PretrainedConfig):
    model_type = "gemma_projector"

    def __init__(
        self,
        projector_type='mlp2x_gelu',
        retriever_hidden_size=128,
        hidden_size=3584,  # Default for Gemma-2-9B
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.projector_type = projector_type
        self.retriever_hidden_size = retriever_hidden_size
        self.hidden_size = hidden_size

class Projector(nn.Module):
    def __init__(self,config):
        super().__init__()
        projector_type = config.projector_type
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.retriever_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            self.projector = nn.Sequential(*modules)
        else:
            raise ValueError(f"Unsupported projector_type: {projector_type}")
    
    def forward(self,context_embedding):
        return self.projector(context_embedding)

class GemmaProjectorModel(PreTrainedModel):
    config_class = GemmaProjectorConfig

    def __init__(self, config: GemmaProjectorConfig):
        super().__init__(config)
        self.projector = Projector(config)

    def forward(self, context_embedding):
        return self.projector(context_embedding)