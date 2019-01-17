from pathlib import Path

import click

from pyke.dataset import Dataset
from pyke.embedding import Embedding
from pyke.models import TransE, HolE, ComplEx


@click.group()
def cli():
    pass


@cli.command(help='Calculate TransE embeddings for knowledge base')
@click.option('-f', '--folds', type=int, default=20)
@click.option('-e', '--epochs', type=int, default=20, help='Number of training epochs')
@click.option('-ne', '--neg_ent', type=int, default=1)
@click.option('-nr', '--neg_rel', type=int, default=0)
@click.option('-b', '--bern', type=bool, default=False)
@click.option('-w', '--workers', type=int, default=4, help='Number of workers that will be used during training')
@click.option('-d', '--dims', type=int, default=50, help='Dimensionality of the generated embeddings')
@click.option('-m', '--margin', type=float, default=1.0)
@click.option('-o', '--out', type=str, default='./embeddings/TransE',
              help='Output directory in which the generated embeddings are to be stored')
@click.argument('file_path')
def transe(folds,
           epochs,
           neg_ent,
           neg_rel,
           bern,
           workers,
           dims,
           margin,
           out,
           file_path):
    """Initializes the repository."""
    dataset = Dataset(filename=file_path)

    click.echo("Start training using the following parameters: ")
    click.echo("-----------------------------------------------")
    click.echo("Knowledge Base: {}".format(file_path))
    click.echo("Folds: {}".format(folds))
    click.echo("Epochs: {}".format(epochs))
    click.echo("Neg_Ent: {}".format(neg_ent))
    click.echo("Neg_Rel: {}".format(neg_rel))
    click.echo("bern: {}".format(bern))
    click.echo("Workers: {}".format(workers))
    click.echo("Dimensionality: {}".format(dims))
    click.echo("Margin: {}".format(margin))
    click.echo("Output directory: {}".format(out))
    click.echo("-----------------------------------------------")

    embedding = Embedding(
        dataset,
        TransE,
        folds=folds,
        epochs=epochs,
        neg_ent=neg_ent,
        neg_rel=neg_rel,
        bern=bern,
        workers=workers,
        dimension=dims,  # TransE-specific
        margin=margin,  # TransE-specific
    )

    checkpoint_path = Path('./checkpoints/TransE')
    out_path = Path(out)

    # Train the model. It is saved in the process.
    if not checkpoint_path.exists():
        click.echo('Creating checkpoint directory: {}'.format(checkpoint_path))
        checkpoint_path.mkdir(parents=True)

    embedding.train(prefix='{}/TransE'.format(checkpoint_path))

    if not out_path.exists():
        out_path.mkdir(parents=True)

    # Save the embedding to a JSON file
    embedding.save_to_json("{}/TransE.json".format(out_path))


@cli.command(help='Calculate TransE embeddings for knowledge base')
@click.option('-f', '--folds', type=int, default=20)
@click.option('-e', '--epochs', type=int, default=20, help='Number of training epochs')
@click.option('-ne', '--neg_ent', type=int, default=1)
@click.option('-nr', '--neg_rel', type=int, default=0)
@click.option('-b', '--bern', type=bool, default=False)
@click.option('-w', '--workers', type=int, default=4, help='Number of workers that will be used during training')
@click.option('-d', '--dims', type=int, default=50, help='Dimensionality of the generated embeddings')
@click.option('-we', '--weight', type=float, default=0.0001)
@click.option('-o', '--out', type=str, default='./embeddings/ComplEx',
              help='Output directory in which the generated embeddings are to be stored')
@click.argument('file_path')
def complex(folds,
            epochs,
            neg_ent,
            neg_rel,
            bern,
            workers,
            dims,
            weight,
            out,
            file_path):
    """Initializes the repository."""
    dataset = Dataset(filename=file_path)

    click.echo("Start training using the following parameters: ")
    click.echo("-----------------------------------------------")
    click.echo("Knowledge Base: {}".format(file_path))
    click.echo("Folds: {}".format(folds))
    click.echo("Epochs: {}".format(epochs))
    click.echo("Neg_Ent: {}".format(neg_ent))
    click.echo("Neg_Rel: {}".format(neg_rel))
    click.echo("bern: {}".format(bern))
    click.echo("Workers: {}".format(workers))
    click.echo("Dimensionality: {}".format(dims))
    click.echo("Weight: {}".format(weight))
    click.echo("Output directory: {}".format(out))
    click.echo("-----------------------------------------------")

    embedding = Embedding(
        dataset,
        ComplEx,
        folds=folds,
        epochs=epochs,
        neg_ent=neg_ent,
        neg_rel=neg_rel,
        bern=bern,
        workers=workers,
        dimension=dims,  # ComplEx-specific
        weight=weight,  # ComplEx-specific
    )

    checkpoint_path = Path('./checkpoints/ComplEx')
    out_path = Path(out)

    # Train the model. It is saved in the process.
    if not checkpoint_path.exists():
        click.echo('Creating checkpoint directory: {}'.format(checkpoint_path))
        checkpoint_path.mkdir(parents=True)

    embedding.train(prefix='{}/ComplEx'.format(checkpoint_path))

    if not out_path.exists():
        out_path.mkdir(parents=True)

    # Save the embedding to a JSON file
    embedding.save_to_json("{}/ComplEx.json".format(out_path))


@cli.command(help='Calculate HolE embeddings for knowledge base')
@click.option('-f', '--folds', type=int, default=20)
@click.option('-e', '--epochs', type=int, default=20, help='Number of training epochs')
@click.option('-ne', '--neg_ent', type=int, default=1)
@click.option('-nr', '--neg_rel', type=int, default=0)
@click.option('-b', '--bern', type=bool, default=False)
@click.option('-w', '--workers', type=int, default=4, help='Number of workers that will be used during training')
@click.option('-d', '--dims', type=int, default=50, help='Dimensionality of the generated embeddings')
@click.option('-m', '--margin', type=float, default=1.0)
@click.option('-o', '--out', type=str, default='./embeddings/HolE',
              help='Output directory in which the generated embeddings are to be stored')
@click.argument('file_path')
def hole(folds,
         epochs,
         neg_ent,
         neg_rel,
         bern,
         workers,
         dims,
         margin,
         out,
         file_path):
    """Initializes the repository."""
    dataset = Dataset(filename=file_path)

    click.echo("Start training using the following parameters: ")
    click.echo("-----------------------------------------------")
    click.echo("Knowledge Base: {}".format(file_path))
    click.echo("Folds: {}".format(folds))
    click.echo("Epochs: {}".format(epochs))
    click.echo("Neg_Ent: {}".format(neg_ent))
    click.echo("Neg_Rel: {}".format(neg_rel))
    click.echo("bern: {}".format(bern))
    click.echo("Workers: {}".format(workers))
    click.echo("Dimensionality: {}".format(dims))
    click.echo("Margin: {}".format(margin))
    click.echo("Output directory: {}".format(out))
    click.echo("-----------------------------------------------")

    embedding = Embedding(
        dataset,
        HolE,
        folds=folds,
        epochs=epochs,
        neg_ent=neg_ent,
        neg_rel=neg_rel,
        bern=bern,
        workers=workers,
        dimension=dims,  # TransE-specific
        margin=margin,  # TransE-specific
    )

    checkpoint_path = Path('./checkpoints/HolE')
    out_path = Path(out)

    # Train the model. It is saved in the process.
    if not checkpoint_path.exists():
        click.echo('Creating checkpoint directory: {}'.format(checkpoint_path))
        checkpoint_path.mkdir(parents=True)

    embedding.train(prefix='{}/HolE'.format(checkpoint_path))

    if not out_path.exists():
        out_path.mkdir(parents=True)

    # Save the embedding to a JSON file
    embedding.save_to_json("{}/HolE.json".format(out_path))
