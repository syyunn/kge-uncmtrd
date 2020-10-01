import pandas as pd
from sklearn.utils import shuffle
from sklearn import preprocessing
from torchkge.data_structures import KnowledgeGraph
from customized_torchkge.data_structure import KnowledgeGraph as KnowledgeGraphQuantityRelation


def load_custom(data_path=None, split_size=[0.8, 0.1, 0.1]):
    """Load FB15k dataset. See `here
    <https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data>`__
    for paper by Bordes et al. originally presenting the dataset.

    Parameters
    ----------
    data_home: str, optional
        Path to the `torchkge_data` directory (containing data folders). If
        files are not present on disk in this directory, they are downloaded
        and then placed in the right place.

    Returns
    -------
    kg_train: torchkge.data_structures.KnowledgeGraph
    kg_val: torchkge.data_structures.KnowledgeGraph
    kg_test: torchkge.data_structures.KnowledgeGraph
    """

    df = pd.read_csv(data_path, sep=",", header=None, names=["from", "rel", "to"])
    df = shuffle(df)
    kg = KnowledgeGraph(df)
    train_size = int(len(df) * split_size[0])
    test_size = int(len(df) * split_size[1])
    valid_size = len(df) - (train_size + test_size)
    return kg.split_kg(sizes=(train_size, test_size, valid_size))


def load_custom_qr(data_path=None, split_size=[0.8, 0.1, 0.1]):
    """Load

    Parameters
    ----------
    data_home: str, optional
        Path to the `torchkge_data` directory (containing data folders). If
        files are not present on disk in this directory, they are downloaded
        and then placed in the right place.

    Returns
    -------
    kg_train: torchkge.data_structures.KnowledgeGraph
    kg_val: torchkge.data_structures.KnowledgeGraph
    kg_test: torchkge.data_structures.KnowledgeGraph
    """

    df = pd.read_csv(
        data_path, sep=",", header=0, names=["from", "rel", "to", "how-much"]
    )

    df = df[df['from'].notna()]
    df = df[df['to'].notna()]

    # min-max normalization of magnitudes
    magnitudes = df["how-much"]  # returns a numpy array
    df["how-much"] = (magnitudes - magnitudes.min()) / (
        magnitudes.max() - magnitudes.min()
    )

    df = shuffle(df)

    kg = KnowledgeGraphQuantityRelation(df)
    train_size = int(len(df) * split_size[0])
    test_size = int(len(df) * split_size[1])
    valid_size = len(df) - (train_size + test_size)
    return kg.split_kg(sizes=(train_size, test_size, valid_size))


def print_model(model):
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
