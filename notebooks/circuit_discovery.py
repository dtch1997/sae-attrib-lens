# %%
import torch as t

from auto_circuit.data import load_datasets_from_json
from auto_circuit.experiment_utils import load_tl_model
from auto_circuit.prune_algos.mask_gradient import mask_gradient_prune_scores
from auto_circuit.types import PruneScores
from auto_circuit.utils.graph_utils import patchable_model
from auto_circuit.visualize import draw_seq_graph

from sae_attrib_lens.system_variables import PROJECT_DIR

# %%
device = t.device("cuda" if t.cuda.is_available() else "cpu")
model = load_tl_model("gpt2", device)

path = PROJECT_DIR / "datasets/ioi/ioi_vanilla_template_prompts.json"
train_loader, test_loader = load_datasets_from_json(
    model=model,
    path=path,
    device=device,
    prepend_bos=True,
    batch_size=16,
    train_test_size=(128, 128),
)

model = patchable_model(
    model,
    factorized=True,
    slice_output="last_seq",
    separate_qkv=True,
    device=device,
)


# %%
print(model.__class__)
print(model)
# %%
print(len(model.nodes))
for node in model.nodes:
    print(node)


# %%
print(len(model.edges))
for edge in model.edges:
    print(edge)


# %%
attribution_scores: PruneScores = mask_gradient_prune_scores(
    model=model,
    dataloader=train_loader,
    official_edges=None,
    grad_function="logit",
    answer_function="avg_diff",
    mask_val=0.0,
)

fig = draw_seq_graph(
    model, attribution_scores, 3.5, layer_spacing=True, orientation="v"
)