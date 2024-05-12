
# %%
import torch
from sae_lens import SparseAutoencoder
from sae_lens.training.config import LanguageModelSAERunnerConfig

# %%
layer = 0
config = LanguageModelSAERunnerConfig(
    model_name = "gelu-2l",
    hook_point = f"blocks.{layer}.hook_mlp_out",
    hook_point_layer = 0,
    dtype=torch.float32,
)
sae = SparseAutoencoder(config)

for name, param in sae.named_parameters():
    print(name, param.shape)
# %%
