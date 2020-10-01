from customized_torchkge.model.quantified_relation.TransE import (
    TransEModel as TransEQuantifiedRelations,
)
from customized_torchkge.data_structure import (
    KnowledgeGraph as KnowledgeGraphQuantityRelation,
)
import pandas as pd
import torch
import os
import pickle

emb_dim = 100
data_path = "/tmp/pycharm_project_583/data/uncmtrd/agg6_202005_ALL_tv.csv"
df = pd.read_csv(data_path, sep=",", header=0, names=["from", "rel", "to", "how-much"])
df = df[df['from'].notna()]
df = df[df['to'].notna()]


# min-max normalization of magnitudes
magnitudes = df["how-much"]  # returns a numpy array
df["how-much"] = (magnitudes - magnitudes.min()) / (magnitudes.max() - magnitudes.min())

model_prefix = "/tmp/pycharm_project_583/pretrained/agg6_202005_ALL_tv_emb100_lr0.0004_mgn0.5_epch1000_bsize2097152_t20200930231435"

kg = KnowledgeGraphQuantityRelation(df)

# with open(os.path.join(model_prefix, "kg.pickle"), "wb") as handle:
#     pickle.dump(
#         kg,
#         handle,
#         protocol=pickle.HIGHEST_PROTOCOL,
#     )

with open(os.path.join(model_prefix, "kg.pickle"), 'rb') as handle:
    kg = pickle.load(handle)

model = TransEQuantifiedRelations(emb_dim, kg.n_ent, kg.n_rel, dissimilarity_type="L2")

model.load_state_dict(torch.load(os.path.join(model_prefix, "model.torch")))
model.eval()

with open(os.path.join(model_prefix, "emb_idx.pickle"), "rb") as handle:
    emb_idx = pickle.load(handle)

ent2ix = emb_idx["ent2ix"]
rel2ix = emb_idx["rel2ix"]

# inv-maps
inv_ent2ix = {v: k for k, v in ent2ix.items()}
inv_rel2ix = {v: k for k, v in rel2ix.items()}

korea2ix = ent2ix['Rep. of Korea']
rel2ix = rel2ix['Imports_281111']

korea_emb = model.ent_emb(torch.tensor(korea2ix).long())
korea_emb_l2norm = korea_emb.norm(p=2, dim=0)

Import_281111_emb = model.rel_emb(torch.tensor(rel2ix).long())
Import_281111_emb_l2norm = Import_281111_emb.norm(p=2, dim=0)

h_plus_r = korea_emb + Import_281111_emb
h_plus_r_l2norm = h_plus_r.norm(p=2, dim=0)

candidates = model.ent_emb.weight.data
dists = (candidates - h_plus_r.expand(kg.n_ent, emb_dim)).norm(p=2, dim=1)

# best_match_tail = torch.argmin(dists)
# best_match_ent = inv_ent2ix[best_match_tail.item()]

ascending_sort_tail_cands = torch.argsort(dists, dim=0).tolist()
for cand in ascending_sort_tail_cands:
    print(inv_ent2ix[cand])

if __name__ == "__main__":
    pass
