from torch.nn.functional import normalize

# from torchkge.models.interfaces import TranslationModel
from customized_torchkge.model.quantified_relation.interfaces import TranslationModel
from torchkge.utils import init_embedding
import torch

class TransEModel(TranslationModel):
    """Implementation of TransE model detailed in 2013 paper by Bordes et al..
    This class inherits from the
    :class:`torchkge.models.interfaces.TranslationModel` interface. It then
    has its attributes as well.


    References
    ----------
    * Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and
      Oksana Yakhnenko.
      `Translating Embeddings for Modeling Multi-relational Data.
      <https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data>`_
      In Advances in Neural Information Processing Systems 26, pages 2787–2795,
      2013.

    Parameters
    ----------
    emb_dim: int
        Dimension of the embedding.
    n_ent: int
        Number of entities in the current data set.
    n_rel: int
        Number of relations in the current data set.
    dissimilarity_type: str
        Either 'L1' or 'L2'.

    Attributes
    ----------
    emb_dim: int
        Dimension of the embedding.
    ent_emb: `torch.nn.Embedding`, shape: (n_ent, emb_dim)
        Embeddings of the entities, initialized with Xavier uniform
        distribution and then normalized.
    rel_emb: `torch.nn.Embedding`, shape: (n_rel, emb_dim)
        Embeddings of the relations, initialized with Xavier uniform
        distribution and then normalized.

    """

    def __init__(self, emb_dim, n_entities, n_relations, dissimilarity_type="L2"):

        super().__init__(n_entities, n_relations, dissimilarity_type)

        self.emb_dim = emb_dim
        self.ent_emb = init_embedding(self.n_ent, self.emb_dim)
        self.rel_emb = init_embedding(self.n_rel, self.emb_dim)

        self.normalize_parameters()
        self.rel_emb.weight.data = normalize(self.rel_emb.weight.data, p=2, dim=1)

    def scoring_function(self, h_idx, t_idx, r_idx, magnitude):
        """Compute the scoring function for the triplets given as argument:
        :math:`||h + r - t||_p^p` with p being the `dissimilarity type (either
        1 or 2)`. See referenced paper for more details
        on the score. See torchkge.models.interfaces.Models for more details
        on the API.

        """
        b_size = h_idx.shape[0]

        h = normalize(self.ent_emb(h_idx), p=2, dim=1)
        t = normalize(self.ent_emb(t_idx), p=2, dim=1)
        # r = self.rel_emb(r_idx)
        r_normalized = normalize(self.rel_emb(r_idx), p=2, dim=1)

        magnitude_expanded = magnitude.reshape([b_size, 1]).expand(b_size, self.emb_dim)
        r_scaled = r_normalized * magnitude_expanded
        # return - self.dissimilarity(h + r, t)
        return -self.dissimilarity(h + r_scaled, t)

    def normalize_parameters(self):
        """Normalize the entity embeddings, as explained in original paper.
        This methods should be called at the end of each training epoch and at
        the end of training as well.

        """
        self.ent_emb.weight.data = normalize(self.ent_emb.weight.data, p=2, dim=1)

    def get_embeddings(self):
        """Return the embeddings of entities and relations.

        Returns
        -------
        ent_emb: torch.Tensor, shape: (n_ent, emb_dim), dtype: torch.float
            Embeddings of entities.
        rel_emb: torch.Tensor, shape: (n_rel, emb_dim), dtype: torch.float
            Embeddings of relations.
        """
        self.normalize_parameters()
        return self.ent_emb.weight.data, self.rel_emb.weight.data

    def lp_prep_cands(self, h_idx, t_idx, r_idx):
        """Link prediction evaluation helper function. Get entities embeddings
        and relations embeddings. The output will be fed to the
        `lp_scoring_function` method. See torchkge.models.interfaces.Models for
        more details on the API.

        """
        b_size = h_idx.shape[0]

        h_emb = self.ent_emb(h_idx)
        t_emb = self.ent_emb(t_idx)
        r_emb = self.rel_emb(r_idx)

        # r_sum_dev = torch.norm(r_emb, p=2, dim=1)

        candidates = self.ent_emb.weight.data.view(1, self.n_ent, self.emb_dim)
        candidates = candidates.expand(b_size, self.n_ent, self.emb_dim)

        return h_emb, t_emb, candidates, r_emb
