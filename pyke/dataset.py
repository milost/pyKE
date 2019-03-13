# -*- coding: utf-8 -*-
import ctypes
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm

from pyke.library import Library
from pyke.parser import NTriplesParser


def parse_idx_file(path: str):
    entity_to_idx = {}
    idx_to_entity = {}

    with open(path) as f:
        f.readline()
        for line in f:
            try:
                rel, rel_id = line.rsplit(maxsplit=1)
                rel_id = int(rel_id)
                entity_to_idx[rel] = rel_id
                idx_to_entity[rel_id] = rel
            except ValueError:
                continue
    return entity_to_idx, idx_to_entity


class Dataset(object):
    """
    Manages a collection of relational data
    encoded as a set of triples, each describing a statement (or fact)
    over two objects called 'entities', one 'head' and one 'tail',
    being related in a manner that is symbolized by a relation 'label'.

    The application encodes both entities and relations as integral values,
    describing an index in an ordered table.
    """

    def __init__(self, filename: str = None, temp_dir: str = ".pyke", generate_valid_test: bool = False,
                 fail_silently: bool = True):
        """
        Creates a new dataset from a N-triples file.

        .. note:

           The N-triples file is parsed into the original OpenKE benchmark file structure containing a file for the
           entities (entity2id.txt), for the relations (relation2id.txt) and the training file (train2id.txt). These
           files are stored by default in the `.pyke` directory in a subdirectory named after the MD5-sum of the
           input file. The MD5-sum is used to prevent the tool from recreating the benchmark files for.
           If you change the N-triples file the MD5-sum changes and so the entities and relations get a new id.

        .. note:

           At the moment, no two datasets can be open at the same time!

        :param filename: Pathname to the N-triples file for training
        :param temp_dir: Directory for storing the benchmark files. Application needs write access
        :param fail_silently: If true, ignore errors in file and skip lines
        """
        self.__library = Library.get_library(temp_dir)

        if filename:
            self.name = Path(filename).stem
        else:
            self.name = filename

        self.size = 0
        self.benchmark_dir = ''
        self.ent_count = 0
        self.rel_count = 0
        self.shape = self.ent_count, self.rel_count
        self.entity2id = {}
        self._id2entity = {}
        self.relation2id = {}
        self._id2relation = {}

        if filename is not None:
            parser = NTriplesParser(filename, temp_dir, generate_valid_test, fail_silently)
            parser.parse()

            self.benchmark_dir = parser.output_dir if parser.output_dir[:-1] == "/" else parser.output_dir + "/"
            self.__library.setInPath(
                ctypes.create_string_buffer(self.benchmark_dir.encode(), len(self.benchmark_dir) * 2))
            self.__library.importTrainFiles()
            self.size = parser.train_count
            self.ent_count = parser.ent_count
            self.rel_count = parser.rel_count
            self.shape = self.ent_count, self.rel_count
            self.train_set = self.read_benchmark(parser.train_file, desc=' train set')
            self.test_set = self.read_benchmark(parser.test_file, desc=' test set') if generate_valid_test else []
            self.valid_set = self.read_benchmark(parser.valid_file, desc=' validation set') if generate_valid_test else []

            self.entity2id, self._id2entity = parse_idx_file(parser.entity_file)
            self.relation2id, self._id2relation = parse_idx_file(parser.relation_file)

        if generate_valid_test:
            self.__library.importTestFiles()
            # self.__library.importTypeFiles()

        self.generate_valid_test = generate_valid_test

    def __len__(self):
        """Returns the size of the dataset."""
        return self.size

    @property
    def entities(self):
        for entity in self.entity2id.keys():
            yield entity

    @property
    def relations(self):
        for relation in self.relation2id.keys():
            yield relation

    @property
    def id2entity(self):
        if self.entity2id and not self._id2entity:
            self._id2entity = {v: k for k, v in self.entity2id.items()}
        return self._id2entity

    @property
    def id2relation(self):
        if self.relation2id and not self._id2relation:
            self._id2relation = {v: k for k, v in self.relation2id.items()}
        return self._id2relation

    @property
    def train_triples(self):
        for triple in self.train_set:
            yield triple

    @property
    def test_triples(self):
        for triple in self.test_set:
            yield triple

    @property
    def validation_triples(self):
        for triple in self.valid_set:
            yield triple

    def get_entity_id(self, entity: str) -> Optional[int]:
        """
        Returns the id of the passed in entity
        :param entity: the entity for which the id is to be retrieved
        :return: the id of the entity or None
        """
        try:
            return self.entity2id[entity]
        except KeyError:
            return None

    def get_entity(self, entity_id: int) -> Optional[str]:
        """
        Retrieve the entity belonging to the passed in id
        :param entity_id: the entity id to be translated
        :return: the corresponding entity or None
        """
        try:
            return self.id2entity[entity_id]
        except KeyError:
            return None

    def get_relation_id(self, relation: str) -> Optional[int]:
        """
        Retrieve the id of the given relation
        :param relation: the relation for which the id is to be retrieved
        :return: the id of the relation or None
        """
        try:
            return self.relation2id[relation]
        except KeyError:
            return None

    def get_relation(self, rid: int) -> Optional[str]:
        """
        Retrieve the relationship belonging to the passed in id
        :param rid: the relationship id to be translated
        :return: the corresponding relationship or None
        """
        try:
            return self.id2relation[rid]
        except KeyError:
            return None

    def translate_int_triple(self, triple: Tuple[int, int, int]) -> Tuple[str, str, str]:
        """
        Translates an integer triple such as (3,4,5) int its entity presentation
        (i.e.<Springfield, Illinois, locatedIn>)
        :param triple: the triple to be translated
        :return: the translated triple
        """
        head = self.get_entity(triple[0])
        rel = self.get_relation(triple[2])
        tail = self.get_entity(triple[1])
        return head, tail, rel

    def translate_string_triple(self, triple: Tuple[str, str, str]) -> Tuple[int, int, int]:
        """
        Translates a string triple such as (Springfield,Illinois,locatedIn) in its id representation
        (i.e.(5,4,3))
        :param triple: the triple to be translated
        :return: the translated triple
        """
        head = self.get_entity_id(triple[0])
        rel = self.get_relation_id(triple[2])
        tail = self.get_entity_id(triple[1])
        return head, tail, rel

    def query(self, head, tail, relation):
        """
        Checks which facts are stored in the entire dataset.
        This method is overloaded for the task of link prediction,
        awaiting an incomplete statement and returning all known substitutes.

        :param head: Index of a head entity.
        :param tail: Index of a tail entity.
        :param relation: Index of a relation label.

        :return: A boolean array, deciding for each candidate whether or not the resulting statement is
            contained in the dataset.
        """
        raise NotImplementedError
        # if head is None:
        #     if tail is None:
        #         if relation is None:
        #             raise NotImplementedError('querying everything')
        #         raise NotImplementedError('querying full relation')
        #     if relation is None:
        #         raise NotImplementedError('querying full head')
        #     heads = np.zeros(self.shape[0], np.bool_)
        #     self.__library.query_head(get_array_pointer(heads), tail, relation)
        #     return heads
        # if tail is None:
        #     if relation is None:
        #         raise NotImplementedError('querying full tail')
        #     tails = np.zeros(self.shape[0], np.bool_)
        #     self.__library.query_tail(head, get_array_pointer(tails), relation)
        #     return tails
        # if relation is None:
        #     relations = np.zeros(self.shape[1], np.bool_)
        #     self.__library.query_rel(head, tail, get_array_pointer(relations))
        #     return relations
        # raise NotImplementedError('querying single facts')

    def to_npz(self, out_path):
        """
        Saves the dataset to an numpy npz format.
        :param out_path: the path to save the dataset to
        """
        np.savez_compressed(out_path,
                            entity2id=self.entity2id,
                            relation2id=self.relation2id,
                            train_set=self.train_set,
                            test_set=self.test_set,
                            valid_set=self.valid_set,
                            ent_count=self.ent_count,
                            rel_count=self.rel_count,
                            shape=self.shape,
                            size=self.size,
                            name=self.name)

    @classmethod
    def from_npz(cls, path: str) -> 'Dataset':
        with np.load(str(path)) as data:
            dataset = cls()
            dataset.train_set = data['train_set']
            dataset.entity2id = data['entity2id'].item()
            dataset.relation2id = data['relation2id'].item()
            dataset.train_set = data['train_set']
            dataset.test_set = data['test_set']
            dataset.valid_set = data['valid_set']
            dataset.ent_count = data['ent_count'].item()
            dataset.rel_count = data['rel_count'].item()
            dataset.shape = (data['shape'][0], data['shape'][1])
            dataset.size = data['size'].item()
            dataset.name = data['name'].item()
            return dataset

    @staticmethod
    def read_benchmark(filename, desc: str = ''):
        with open(filename) as f:
            count = int(f.readline())  # Skip first line containing the number of rows
            triple_list = [(int(line.split()[0]),
                            int(line.split()[1]),
                            int(line.split()[2])) for line in tqdm(f, total=count, desc=f'Reading{desc}',
                                                                   unit=' triples')]
        return triple_list
