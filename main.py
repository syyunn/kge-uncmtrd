import os
from torch.optim import Adam
from datetime import datetime
from customized_torchkge.model.quantified_relation.TransE import TransEModel as TransEQuantifiedRelations

from torchkge.utils import MarginLoss
from customized_torchkge.training import Trainer

from utils import load_custom_qr, print_model


def main_quantified_TransE():
    # Define some hyper-parameters for training
    emb_dim = 100
    lr = 0.0004
    margin = 0.5
    n_epochs = 1000
    batch_size = 2097152

    # Load dataset
    data_path = "/tmp/pycharm_project_583/data/uncmtrd/agg6_202005_ALL_tv.csv"

    kg_train, kg_val, kg_test = load_custom_qr(data_path=data_path)

    model = TransEQuantifiedRelations(
        emb_dim, kg_train.n_ent, kg_train.n_rel, dissimilarity_type="L2"
    )

    print_model(model) # check we only have two embedding layers - one for entity, the other for relations

    dataset_name = data_path.split('/')[-1].replace('.csv', '')
    curr_time = datetime.now().strftime('%Y%m%d%H%M%S')
    model_prefix = os.path.join('./pretrained', f'{dataset_name}_emb{emb_dim}_lr{lr}_mgn{margin}_epch{n_epochs}_bsize{batch_size}_t{curr_time}')

    criterion = MarginLoss(margin)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    trainer = Trainer(
        model,
        criterion,
        kg_train,
        n_epochs,
        batch_size,
        optimizer=optimizer,
        sampling_type="bern",
        use_cuda=None,
    )

    trainer.run(kg_test=kg_test, model_prefix=model_prefix)


if __name__ == "__main__":
    main_quantified_TransE()
