import logging
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from pyke import models
from pyke.dataset import Dataset
from pyke.library import Library
from pyke.openke import Config
from pyke.utils import get_rank

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(module)s - %(message)s')
logger = logging.getLogger("pyke")


class Embedding:
    """
    Class is a Wrapper for an embedding. It consists of a dataset and a model and provides an interface
    for the normal embedding operations such as prediction, training, saving and restoring.

    :param optimizer: Possible values: SGD, Adagrad, Adadelta, Adam
    """

    def __init__(self, dataset: Dataset = None,
                 model_class: type = None,
                 out_path=None,
                 **kwargs):
        self.dataset = dataset or Dataset()
        self.model_class = model_class
        # self.__model = None
        self.__config = None
        self.__library = Library.get_library()
        # Training args
        self.neg_ent = 1
        self.neg_rel = 0
        self.bern = True
        self.workers = 1
        self.folds = 20
        self.epochs = 50
        self.optimizer = "SGD"
        self.per_process_gpu_memory_fraction = 0.5
        self.learning_rate = 0.01
        # Model specific parameters
        self.dimension = 50  # ComplEx, DistMult, HolE, RESCAL, TransD, TransE, TransH
        self.ent_dim = 50  # TransR
        self.rel_dim = 10  # TransR
        self.margin = 1.0  # HolE, RESCAL, TransD, TransE, TransH, TransR
        self.weight = 0.0001  # ComplEx, DistMult
        # used to provide easier access to embeddings.
        self.entity_embeddings = None
        self.relationship_embeddings = None
        self.rankings: pd.DataFrame = None
        self.out_path = out_path

        # Apply kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        if model_class is not None:
            self.__init_config()

    def __str__(self):
        return "<Embedding: {} {}>".format(self.model_class.__name__.split('.')[-1], self.get_model_parameters())

    def __init_config(self):
        """Wrapper for the config object"""
        con = Config()
        con.set_in_path(self.dataset.benchmark_dir)
        con.set_test_link_prediction(self.dataset.generate_valid_test)
        con.set_test_triple_classification(self.dataset.generate_valid_test)
        con.set_work_threads(self.workers)
        con.set_train_times(self.epochs)
        con.set_nbatches(self.folds)
        con.set_alpha(self.learning_rate)
        con.set_lmbda(self.weight)
        con.set_margin(self.margin)
        con.set_bern(int(self.bern))
        con.set_ent_dimension(self.ent_dim)
        con.set_rel_dimension(self.rel_dim)
        con.set_dimension(self.dimension)
        con.set_ent_neg_rate(self.neg_ent)
        con.set_rel_neg_rate(self.neg_rel)
        con.set_opt_method(self.optimizer)
        con.set_out_files(self.out_path)
        con.per_process_gpu_memory_fraction = self.per_process_gpu_memory_fraction
        con.init()
        con.set_model(self.model_class)
        self.__config = con

    def get_model_parameters(self):
        """
        Returns the model-specific parameters used by the constructor. These are for example dimension, weight, margin.
        The tuple can be unpacked with *args*.
        Returning

        - (dimension, weight) for ComplEx and DistMult,
        - (ent_dim, rel_dim, margin) for TransR,
        - (dimension, margin) for HolE, RESCAL, TransE, TransD, TransH.

        :return: tuple with the model specific parameters
        """
        if self.model_class in (models.ComplEx, models.DistMult):
            return self.dimension, self.weight
        elif self.model_class == models.TransR:
            return self.ent_dim, self.rel_dim, self.margin
        elif self.model_class in (models.HolE, models.RESCAL, models.TransE, models.TransD, models.TransH):
            return self.dimension, self.margin
        else:
            raise ValueError("Model class {} is not supported.".format(self.model_class.__name__))

    @property
    def batch_size(self):
        return self.dataset.size // self.folds

    @property
    def batch_size_neg(self):
        return self.batch_size * (1 + self.neg_ent + self.neg_rel)

    @property
    def variants(self):
        return self.neg_rel + self.neg_ent + 1

    def predict(self, head_id, tail_id, rel_id):

        head_id = int(head_id) if head_id is not None else None
        tail_id = int(tail_id) if tail_id is not None else None
        rel_id = int(rel_id) if rel_id is not None else None

        heads = [head_id] if isinstance(head_id, int) else head_id
        tails = [tail_id] if isinstance(tail_id, int) else tail_id
        rels = [rel_id] if isinstance(rel_id, int) else rel_id

        if head_id is None:
            if tail_id is None:
                if rel_id is None:
                    raise NotImplementedError('universal prediction')
                raise NotImplementedError('full-relation prediction')
            elif rel_id is None:
                raise NotImplementedError('full-tail prediction')
            heads, tails, rels = np.arange(self.dataset.ent_count), np.full([self.dataset.ent_count], tail_id), \
                                 np.full([self.dataset.ent_count], rel_id)
        elif tail_id is None:
            if rel_id is None:
                raise NotImplementedError('full-head prediction')
            heads, tails, rels = np.full([self.dataset.ent_count], head_id), np.arange(self.dataset.ent_count), \
                                 np.full([self.dataset.ent_count], rel_id)
        elif rel_id is None:
            heads, tails, rels = np.full([self.dataset.rel_count], head_id), np.full([self.dataset.rel_count], tail_id), \
                                 np.arange(self.dataset.rel_count)

        if isinstance(head_id, int) and isinstance(tail_id, int) and isinstance(rel_id, int):
            return self.__config.test_step(heads, tails, rels)[0]
        return self.__config.test_step(heads, tails, rels)

    def train(self, prefix='best', save_steps: int = 100, continue_training=True):
        """
        Train the embedding.

        :param prefix: Model prefix to save
        :param save_steps: Steps after which the model is saved
        :param continue_training: If true and an existing model is found, the training is resumed
        """
        if os.path.exists(prefix + ".index") and continue_training:
            print("Found model with prefix {}. Continuing training ...".format(prefix))
            self.restore(prefix)
        else:
            self.__config.set_import_files(None)
        self.__config.set_export_files(prefix, save_steps)
        self.__config.run()  # TODO: Add cross validation

    def save_to_json(self, path: str):
        """
        Save embedding to JSON.

        :param path: JSON path
        """
        self.__config.save_parameters(path)

    def get_parameters(self):
        """
        Get all parameters
        """
        return self.__config.get_parameters('list')

    def save_to_npz(self, path: str):
        """
        Saves indices and embeddings into a compressed numpy file (.npz).
        :param path: where to save the compressed archive
        """
        np.savez_compressed(path,
                            entity_embeddings=self.get_entity_embeddings(),
                            relationship_embeddings=self.get_relationship_embeddings())

    @classmethod
    def load_from_npz(cls, path: str, dataset: Dataset = None) -> 'Embedding':
        with np.load(path) as data:
            if dataset is not None:
                embs = cls(dataset, None)
            else:
                embs = cls(Dataset(), None)
            embs.entity_embeddings = data['entity_embeddings']
            try:
                embs.relationship_embeddings = data['relationship_embeddings']
            except KeyError:
                pass
            return embs

    def load_embeddings_from_npy(self, path: str):
        """
        Loads only the previously calculated embeddings.
        This can be used if only querying is needed.
        :param path: the path from where to load the embeddings
        """
        self.entity_embeddings = np.load(path)

    def restore(self, prefix: str):
        """
        Loads an existing embedding.

        :param prefix: Prefix of the model files
        """
        self.__config.set_import_files(prefix)
        self.__config.restore_tensorflow()

    def get_validation_triples(self):
        """
        Returns a list of triples used for the metrics.
        """
        return self.dataset.valid_set if self.dataset.generate_valid_test else self.dataset.train_set

    def get_test_triples(self):
        """
        Returns a list of triples used for the metrics.
        """
        return self.dataset.test_set if self.dataset.generate_valid_test else self.dataset.train_set

    def get_predictions(self, filtered=False, head=True, tail=True, label=False):
        """
        Computes the mean rank of the embedding.
        """
        if filtered:
            raise NotImplementedError("Filtered meanrank not implemented")

        table = []
        column_headers = ['head_id', 'tail_id', 'rel_id']
        if head:
            column_headers.append('head_rank')
        if tail:
            column_headers.append('tail_rank')
        if label:
            column_headers.append('rel_rank')

        triples = self.get_test_triples()

        for (head_id, tail_id, label_id) in tqdm(triples, desc='Calculating rankings'):
            row = [head_id, tail_id, label_id]

            value = self.predict(head_id, tail_id, label_id)
            if head:
                predictions = self.predict(None, tail_id, label_id)
                rank = get_rank(predictions, value)
                row.append(rank)
            if tail:
                predictions = self.predict(head_id, None, label_id)
                rank = get_rank(predictions, value)
                row.append(rank)
            if label:
                predictions = self.predict(head_id, tail_id, None)
                rank = get_rank(predictions, value)
                row.append(rank)
            table.append(row)

        self.rankings = pd.DataFrame(table, columns=column_headers)
        return self.rankings

    def hits_at_k(self, k: int, filtered: bool = False):
        """
        Calculates the hits@k metric (raw or filtered) for the embedding.

        :param k: First top k elements to look at
        :param filtered: flat for filtered hits@k (otherwise raw)
        """
        raise NotImplementedError("Hits@k is currently not implemented.")

    def get_relationship_embeddings(self):
        """
        Returns the relationship embeddings.

        :return: Relationship embeddings as numpy matrix
        """
        if self.relationship_embeddings is None:
            self.relationship_embeddings = self.__config.get_parameters_by_name("rel_embeddings")
        return self.relationship_embeddings

    def get_entity_embeddings(self):
        """
        Returns the entity embeddings.

        :return: Entity embeddings as numpy matrix
        """
        if self.entity_embeddings is None:
            self.entity_embeddings = self.__config.get_parameters_by_name("ent_embeddings")
        return self.entity_embeddings

    def get_embedding_for(self, entity: str):
        """
        Retrieves the embedding that belongs to the passed in entity.
        :param entity:
        :return: the embedding that belongs to the passed in entity, else None
        """
        if self.entity_embeddings is None:
            self.get_entity_embeddings()

        # get entity id
        try:
            entity_id = self.dataset.get_entity_id(entity)
        except KeyError:
            entity_id = None

        # use entity_id to get embedding
        if entity_id:
            return self.entity_embeddings[entity_id]
        else:
            return None

    def calc_metrics(self, rank_predictions=None, k=10):
        """
        Computes mean rank and hits@k score
        :param rank_predictions:
        :param k:
        :return:
        """
        if rank_predictions is None:
            if self.rankings is None:
                self.get_predictions()
        elif isinstance(rank_predictions, str):
            self.rankings = pd.read_csv(rank_predictions)

        results = []
        column_headers = []
        column_names = [column_name for column_name in list(self.rankings) if column_name.endswith("_rank")]

        for column_name in column_names:
            column_headers.append(f'{column_name.rstrip("_rank")}_hits_at_{k}')
            column_headers.append(f'{column_name.rstrip("_rank")}_mean_hits_at_{k}')

            hits_at_n = len(self.rankings[column_name][self.rankings[column_name] < k])
            mean_hits_at_n = hits_at_n / len(self.rankings[column_name])
            results.append(hits_at_n)
            results.append(mean_hits_at_n)

        column_headers.append('mean_rank')
        rank_sum = self.rankings[column_names].sum().sum()
        results.append(rank_sum / (2 * len(self.rankings[column_names[0]])))

        return pd.DataFrame([results], columns=column_headers)

    def get_parameters(self):
        """
        Returns all embedding parameters in dependence of the model These can be the entity embedding, relation
        embedding, transfer matrices, etc.

        :return: dictionary with parameters
        """
        return self.__config.get_parameters()

    def get_loss(self):
        return self.__config.current_loss
