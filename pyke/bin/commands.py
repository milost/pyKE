from pathlib import Path

import click

from pyke.dataset import Dataset
from pyke.embedding import Embedding
from pyke.models import TransE, TransD, TransH, TransR, HolE, ComplEx, DistMult, RESCAL


def get_model(model_type: str = 'TransE'):
    if model_type == 'TransE':
        return TransE
    elif model_type == 'TransD':
        return TransD
    elif model_type == 'TransH':
        return TransH
    elif model_type == 'TransR':
        return TransR
    elif model_type == 'HolE':
        return HolE
    elif model_type == 'ComplEx':
        return ComplEx
    elif model_type == 'DistMult':
        return DistMult
    elif model_type == 'RESCAL':
        return RESCAL
    else:
        raise ValueError(f'Unknown model_type {model_type}')


@click.group()
def cli():
    pass


@cli.command(help='Calculate TransE embeddings for knowledge base')
@click.option('-m', '--model', default='TransE', help='The model to be used for calculating the embeddings')
@click.option('-nb', '--n_batches', type=int, default=20, help='Number of batches to be used')
@click.option('-e', '--epochs', type=int, default=20, help='Number of training epochs')
@click.option('-ne', '--neg_ent', type=int, default=1)
@click.option('-nr', '--neg_rel', type=int, default=0)
@click.option('-b', '--bern', type=bool, default=False)
@click.option('-w', '--workers', type=int, default=4, help='Number of workers that will be used during training')
@click.option('-opt', '--optimizer', default='SGD', help='The optimizer to be used: SGD, Adagrad, Adadelta, Adam')
@click.option('-d', '--dims', type=int, default=50, help='Dimensionality of the generated embeddings')
@click.option('-m', '--margin', type=float, default=1.0)
@click.option('-o', '--out', type=str, default='./embeddings',
              help='Output directory in which the generated embeddings are to be stored')
@click.option('-j', '--json', default=False)
@click.argument('file_path')
def compute(model,
            n_batches,
            epochs,
            neg_ent,
            neg_rel,
            bern,
            workers,
            optimizer,
            dims,
            margin,
            out,
            json,
            file_path):
    """Initializes the repository."""
    # dataset = Dataset(filename=file_path)
    dataset = Dataset.from_npz('/Users/milost/Code/Python/pyKE/resources/test.npz')
    # dataset.to_npz(out_path='/Users/milost/Code/Python/pyKE/resources/test.npz')

    file_path = Path(file_path)

    click.echo("Start training using the following parameters: ")
    click.echo("-----------------------------------------------")
    click.echo(f"Knowledge Base: {file_path}")
    click.echo(f"Batch number: {n_batches} => {int(dataset.size / n_batches)} total batch size")
    click.echo(f"Epochs: {epochs}")
    click.echo(f"Neg_Ent: {neg_ent}")
    click.echo(f"Neg_Rel: {neg_rel}")
    click.echo(f"bern: {bern}")
    click.echo(f"Workers: {workers}")
    click.echo(f"Optimizer: {optimizer}")
    click.echo(f"Dimensionality: {dims}")
    click.echo(f"Margin: {margin}")
    click.echo(f"Output directory: {out}")
    click.echo("-----------------------------------------------")

    embedding = Embedding(
        dataset,
        get_model(model),
        folds=n_batches,
        epochs=epochs,
        neg_ent=neg_ent,
        neg_rel=neg_rel,
        bern=bern,
        workers=workers,
        optimizer=optimizer,
        dimension=dims,  # TransE-specific
        margin=margin,  # TransE-specific
        out_path=out
    )

    checkpoint_path = Path(f'./checkpoints/{model}')
    out_path = Path(f'{out}/{model}/{dataset.name}')

    if not out_path.exists():
        click.echo(f'Creating output path: {out_path}')
        out_path.mkdir(parents=True)

    # Train the model. It is saved in the process.
    if not checkpoint_path.exists():
        click.echo(f'Creating checkpoint directory: {checkpoint_path}')
        checkpoint_path.mkdir(parents=True)

    # embedding.train(prefix=str(checkpoint_path / dataset.name))
    results = embedding.evaluate_embeddings(rankings='/Users/milost/Code/Python/pyKE/resources/predictions.csv')
    print(results)

    # print(embedding.get_parameters())

    # Save the embedding to a JSON file
    if json:
        embedding.save_to_json(f"{out_path}/{file_path.name.rstrip(file_path.suffix)}_trans_e_embs.json")

    # Save the embedding as numpy (.npz) file
    archive_name = f'{out_path}/{file_path.name.rstrip(file_path.suffix)}_trans_e_embs.npz'
    embedding.save_to_npz(archive_name)


@cli.command(help='Build dataset from a file containing knowledge base triples')
@click.option('-g', '--generate_validation_test', type=bool, default=False, help='Generate validation and test sets')
@click.argument('file_in')
@click.argument('file_out')
def build_dataset(generate_validation_test,
                  file_in,
                  file_out):
    """Initializes the repository."""
    dataset = Dataset(filename=file_in, generate_valid_test=generate_validation_test)
    dataset.to_npz(out_path=file_out)


if __name__ == '__main__':
    cli()
