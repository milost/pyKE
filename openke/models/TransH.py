#coding:utf-8
from tensorflow import (get_variable as var,
                        reduce_sum as sum,
                        reduce_mean as mean,
                        maximum as max,
                        nn, reshape)
at = nn.embedding_lookup
from .Base import ModelClass


def _lookup(h, t, r):

	nor = var('normal_vectors')
	ent = var('ent_embeddings')
	rel = var('rel_embeddings')

	return at(ent, h), at(ent, t), at(nor, l), at(rel, l)


def _term(h, t, n, l):

	def transfer(e):
		return e - sum(e * n, -1, keepdims=True) * n

	return transfer(h) + r - transfer(t)


class TransH(ModelClass):


	def _score(self, h, t, l):
		'''The term to score triples.'''

		return self._norm(_term(*_lookup(h, t, l)))


	def _embedding_def(self):
		'''Initializes the variables of the model.'''

		e, r, d = self.base[0], self.base[1], self.dimension[0]

		#Defining required parameters of the model, including embeddings of entities and relations, and normal vectors of planes
		ent = var("ent_embeddings", [e, d])
		rel = var("rel_embeddings", [r, d])
		nor = var("normal_vectors", [r, d])

		yield 'ent_embeddings', ent
		yield 'rel_embeddings', rel
		yield 'normal_vectors', nor

		self._entity = at(ent, self.predict_h)
		self._relation = at(rel, self.predict_l)


	def _loss_def(self):
		'''Initializes the loss function.'''

		def scores(h, t, r):
			s = self._score(h, t, r) # [b,n]
			return mean(s, 1) # [b]

		p = scores(*self._positive_instance(in_batch=True)) # [b]
		n = scores(*self._negative_instance(in_batch=True)) # [b]

		return sum(max(p - n + self.margin, 0)) # []


	def _predict_def(self):
		'''Initializes the prediction function.'''

		return self._score(*self._predict_instance()) # [b]


	def __init__(self, dimension, margin, baseshape, batchshape,\
			optimizer=None):
		self.dimension = dimension,
		self.margin = margin
		super().__init__(baseshape, batchshape, optimizer=optimizer)


	def __str__(self):
		return '{}-{}'.format(type(self).__name__, self.dimension[0])
