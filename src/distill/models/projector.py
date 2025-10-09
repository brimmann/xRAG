import torch.nn as nn
import re


class Projector(nn.Module):
    def __init__(self, config):
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
            # You might want to handle other projector types or raise an error
            raise ValueError(f"Unsupported projector_type: {projector_type}")

    def forward(self, context_embedding):
        return self.projector(context_embedding)