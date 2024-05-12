# %%
import torch as t
import transformer_lens as tl

def load_tl_hooked_sae_model(name: str, device: t.device) -> tl.HookedSAETransformer:
    """
    Load a `HookedTransformer` model with the necessary config to perform edge patching
    (with separate edges to Q, K, and V). Sets `requires_grad` to `False` for all model
    weights (this does not affect Mask gradients).
    """
    tl_model = tl.HookedSAETransformer.from_pretrained(
        name,
        device=device,
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
    )
    tl_model.cfg.use_attn_result = True
    tl_model.cfg.use_attn_in = True
    tl_model.cfg.use_split_qkv_input = True
    tl_model.cfg.use_hook_mlp_in = True
    tl_model.eval()
    for param in tl_model.parameters():
        param.requires_grad = False
    return tl_model

# %%
device = t.device("cuda" if t.cuda.is_available() else "cpu")
model = load_tl_hooked_sae_model("gelu-1l", device)

# %%

cfg = tl.HookedSAEConfig(
    d_sae = 16 * model.cfg.d_model,
    d_in = model.cfg.d_model,
    hook_name = "blocks.0.hook_resid_pre",
)
hooked_sae = tl.HookedSAE(cfg)
# %%
for hook_point in model.hook_points():
    print(hook_point.name)
# %%
