# %%
import torch as t

from auto_circuit.experiment_utils import load_tl_model
from auto_circuit.utils.graph_utils import patchable_model

# %%
device = t.device("cuda" if t.cuda.is_available() else "cpu")
model = load_tl_model("gelu-1l", device)

model = patchable_model(
    model,
    factorized=True,
    slice_output="last_seq",
    separate_qkv=True,
    device=device,
)


# %%
print(len(model.nodes))
print(len(model.edges))
# %%
# all_nodes = list(model.nodes)

import pandas as pd
from auto_circuit.types import Node

node: Node

rows = []
for node in model.srcs:
    rows.append(
        {
            "name": node.name,
            "module_name": node.module_name,            
        }
    )
df = pd.DataFrame(rows)
print(len(df))
df.sort_values(["module_name", "name"])

# %%
rows = []
for node in model.dests:
    rows.append(
        {
            "name": node.name,
            "module_name": node.module_name,            
        }
    )
df = pd.DataFrame(rows)
print(len(df))
df.sort_values(["module_name", "name"])

# %%
