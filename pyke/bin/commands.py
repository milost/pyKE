from pathlib import Path

import click
import sys
import re
import pandas as pd

import plotly.plotly as py
import plotly.graph_objs as go

from pyke.utils import calc_metrics
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
@click.option('-eva', '--eval', type=bool, default=False)
@click.option('-o', '--out', type=str, default='./embeddings',
              help='Output directory in which the generated embeddings are to be stored')
@click.option('-j', '--json', default=False)
@click.option('-val', '--valid_file', default=None)
@click.option('-test', '--test_file', default=None)
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
            eval,
            out,
            json,
            valid_file,
            test_file,
            file_path):
    """Initializes the repository."""
    file_path = Path(file_path)

    if file_path.suffix == '.npz':
        dataset = Dataset.from_npz(file_path)
    elif valid_file is not None and test_file is not None:
        dataset = Dataset(train_file=str(file_path), valid_file=valid_file, test_file=test_file, generate_valid_test=True)
    elif valid_file is None and test_file is None and eval:
        dataset = Dataset(train_file=str(file_path), generate_valid_test=True)
    else:
        dataset = Dataset(train_file=str(file_path))

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

    # if dataset is not written out, do so
    # if not (out_path / f'{dataset.name}_dataset.npz').exists():
    #    dataset.to_npz(out_path / f'{dataset.name}_dataset.npz')

    embedding.train(prefix=str(checkpoint_path / dataset.name))

    # Save the embedding to a JSON file
    if json:
        embedding.save_to_json(f'{out_path}/{dataset.name}_{model.lower()}_{optimizer.lower()}_{dims}_embs.json')
    # Save the embedding as numpy (.npz) file
    archive_name = f'{out_path}/{dataset.name}_{model.lower()}_{optimizer.lower()}_{dims}_embs.npz'
    embedding.save_to_npz(archive_name)

    if eval:
        rank_predictions = embedding.get_predictions()
        # rank_predictions.to_csv(f'{out_path}/{dataset.name}_rank_predictions.csv')

        results = calc_metrics(rank_predictions=rank_predictions)
        if (out_path / f'{dataset.name}_metrics.csv').exists():
            df = pd.read_csv(str(out_path / f'{dataset.name}_metrics.csv'), index_col=0)
            prev_epochs = df.iloc[-1]['epochs']
            results['epochs'] = int(prev_epochs + epochs)
            df = df.append(results, ignore_index=True)
            df.to_csv(str(out_path / f'{dataset.name}_metrics.csv'))
            rank_predictions.to_csv(f'{out_path}/{dataset.name}_rank_predictions_{int(prev_epochs + epochs)}.csv')
            print(df)
        else:
            results['epochs'] = epochs
            results.to_csv(str(out_path / f'{dataset.name}_metrics.csv'))
            rank_predictions.to_csv(f'{out_path}/{dataset.name}_rank_predictions_{epochs}.csv')
            print(results)


@cli.command(help='Build dataset from a file containing knowledge base triples')
@click.option('-g', '--generate_validation_test', type=bool, default=False, help='Generate validation and test sets')
@click.option('-val', '--validation_file', default=None, help='A file containing validation triples')
@click.option('-test', '--test_file', default=None, help='A file containing test triples')
@click.option('-o', '--out', default=None, help='Where to write the generated dataset file')
@click.argument('file_in')
def build_dataset(generate_validation_test,
                  validation_file,
                  test_file,
                  out,
                  file_in):
    """Create npz dataset file"""
    file_in = Path(file_in)
    if not file_in.exists():
        click.echo(f'The file {file_in} does not exist')
        return sys.exit(1)

    if out is None:
        out = f'./{file_in.with_suffix(".npz")}'

    if validation_file is not None and test_file is not None:
        dataset = Dataset(train_file=str(file_in), valid_file=validation_file, test_file=test_file, generate_valid_test=True)
    elif validation_file is None and test_file is None and generate_validation_test:
        dataset = Dataset(train_file=str(file_in), generate_valid_test=True)
    else:
        dataset = Dataset(train_file=str(file_in))
    dataset.to_npz(out_path=out)


@cli.command(help='Plot evaluation metrics')
@click.option('-f', '--filename', default='dummy', help='filename used with plotly')
@click.option('-f', '--plot', default='hits_at_k', help='filename used with plotly')
@click.argument('folder_path')
def plot_eval_metrics(filename, plot, folder_path):
    folder_path = Path(folder_path)
    metrics = pd.read_csv(str(folder_path), index_col=0)
    print(metrics)

    data = []
    title = ''
    if plot == 'hits_at_k':
        title = 'Total Hits@K'
        hits_at_1 = go.Scatter(
            x=metrics['epochs'],
            y=metrics['hits_at_1'],
            name='Hits@1',
        )

        hits_at_3 = go.Scatter(
            x=metrics['epochs'],
            y=metrics['hits_at_3'],
            name='Hits@3'
        )

        hits_at_10 = go.Scatter(
            x=metrics['epochs'],
            y=metrics['hits_at_10'],
            name='Hits@10'
        )
        data = [hits_at_1, hits_at_3, hits_at_10]
    elif plot == 'mrr':
        title = 'Mean Reciprocal Rank'
        mrr = go.Scatter(
            x=metrics['epochs'],
            y=metrics['mrr'],
            name='MRR',
        )
        mrr_head = go.Scatter(
            x=metrics['epochs'],
            y=metrics['mean_reciprocal_head_rank'],
            name='Head MRR',
        )
        mrr_tail = go.Scatter(
            x=metrics['epochs'],
            y=metrics['mean_reciprocal_tail_rank'],
            name='Tail MRR',
        )

        data = [mrr, mrr_head, mrr_tail]
    elif plot == 'mean_rank':
        title = 'Plot of mean rank with increasing epochs'
        mean_rank = go.Scatter(
            x=metrics['epochs'],
            y=metrics['mean_rank'],
            name='Mean Rank'
        )
        mean_head_rank = go.Scatter(
            x=metrics['epochs'],
            y=metrics['mean_head_rank'],
            name='Mean Head Rank'
        )
        mean_tail_rank = go.Scatter(
            x=metrics['epochs'],
            y=metrics['mean_tail_rank'],
            name='Mean Tail Rank'
        )
        data = [mean_rank, mean_head_rank, mean_tail_rank]

    # Edit the layout
    layout = dict(title=title,
                  xaxis=dict(title='Epochs'),
                  yaxis=dict(title='Performance'),
                  )

    fig = dict(data=data, layout=layout)
    py.plot(fig, filename=filename)


if __name__ == '__main__':
    cli()
