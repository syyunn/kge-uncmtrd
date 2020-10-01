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

candidates = model.ent_emb.weight.data
candidates_df = pd.DataFrame(candidates).astype("float")
candidates_df.to_csv('./ent_emb.tsv', sep='\t', header=False, index=False)

with open(os.path.join(model_prefix, "emb_idx.pickle"), 'rb') as handle:
    emb_idx = pickle.load(handle)
ent2ix = emb_idx['ent2ix']
rel2ix = emb_idx['rel2ix']
series = pd.DataFrame(list(ent2ix.items()), columns=['Country', 'Idx'])['Country']
series.to_csv('./ent_emb_meta.tsv', sep='\t', header=False, index=False)

if __name__ == "__main__":
    pass
