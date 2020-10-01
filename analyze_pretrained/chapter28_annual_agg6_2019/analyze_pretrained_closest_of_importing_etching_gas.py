from customized_torchkge.model.quantified_relation.TransE import TransEModel as TransEQuantifiedRelations
from customized_torchkge.data_structure import KnowledgeGraph as KnowledgeGraphQuantityRelation
import pandas as pd
import torch
import os
import pickle

emb_dim = 100
data_path = "/tmp/pycharm_project_583/data/uncmtrd/agg6_2019_cc28_tv.csv"
df = pd.read_csv(data_path, sep=",", header=0, names=["from", "rel", "to", "how-much"])

# min-max normalization of magnitudes
magnitudes = df["how-much"]  # returns a numpy array
df["how-much"] = (magnitudes - magnitudes.min()) / (magnitudes.max() - magnitudes.min())

kg = KnowledgeGraphQuantityRelation(df)

model = TransEQuantifiedRelations(emb_dim, kg.n_ent, kg.n_rel, dissimilarity_type="L2")

model_prefix = "/tmp/pycharm_project_583/pretrained/agg6_2019_cc28_tv_emb100_lr0.0004_mgn0.5_epch1000_bsize32768_t20200930202317"
model.load_state_dict(torch.load(os.path.join(model_prefix, "model.torch")))
model.eval()

with open(os.path.join(model_prefix, "emb_idx.pickle"), 'rb') as handle:
    emb_idx = pickle.load(handle)

ent2ix = emb_idx['ent2ix']
rel2ix = emb_idx['rel2ix']

# inv-maps
inv_ent2ix = {v: k for k, v in ent2ix.items()}
inv_rel2ix = {v: k for k, v in rel2ix.items()}

df = pd.DataFrame(list(ent2ix.items()), columns=['Country', 'Idx'])['Country']
df.to_csv()

rel2ix = rel2ix['Import_281111']

Import_281111_emb = model.rel_emb(torch.tensor(rel2ix).long())
Import_281111_emb_l2norm = Import_281111_emb.norm(p=2, dim=0)

candidates = model.rel_emb.weight.data
dists = (candidates - Import_281111_emb.expand(kg.n_rel, emb_dim)).norm(p=2, dim=1)

ascending_sort_tail_cands = torch.argsort(dists, dim=0).tolist()
for cand in ascending_sort_tail_cands:
    print(inv_rel2ix[cand])

if __name__ == "__main__":
    pass
