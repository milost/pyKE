from pyke.dataset import Dataset
from pyke.embedding import Embedding
from pyke.models import TransE


def main():
    # load dataset
    dataset = Dataset("../benchmarks/fb15k.nt")

    # load embeddings
    embeddings = Embedding(dataset, TransE)
    embeddings.restore(prefix='../checkpoints/TransE/TransE')

    # alternatively load embeddings from numpy matrix
    embs = Embedding(dataset, TransE)
    embs.load_embeddings_from_npy('../embeddings/TransE/TransE.npy')

    # query embeddings
    print(embeddings['/m/02f75t'])
    print(embs['/m/02f75t'])
    print()
    print(embeddings['foobar'])
    print(embs['foobar'])


if __name__ == '__main__':
    main()
