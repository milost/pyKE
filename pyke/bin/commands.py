from pathlib import Path

import click
import os
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
            file_path):
    """Initializes the repository."""
    file_path = Path(file_path)
    if file_path.suffix == '.npz':
        dataset = Dataset.from_npz(file_path)
    else:
        dataset = Dataset(filename=str(file_path), generate_valid_test=True)

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
    if not (out_path / f'{dataset.name}_dataset.npz').exists():
        dataset.to_npz(out_path / f'{dataset.name}_dataset.npz')


    embedding.train(prefix=str(checkpoint_path / dataset.name))

    # Save the embedding to a JSON file
    if json:
        embedding.save_to_json(f'{out_path}/{dataset.name}_{model.lower()}_{optimizer.lower()}_{dims}_embs.json')
    # Save the embedding as numpy (.npz) file
    archive_name = f'{out_path}/{dataset.name}_{model.lower()}_{optimizer.lower()}_{dims}_embs.npz'
    embedding.save_to_npz(archive_name)

    if eval:
        rank_predictions = embedding.get_predictions()
        rank_predictions.to_csv(f'{out_path}/{dataset.name}_rank_predictions.csv')

        results = embedding.calc_metrics(rank_predictions=rank_predictions)
        print(results)


#

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


@cli.command(help='Compute evaluation metrics using all predictions in the specified folder')
@click.option('-k', type=int, default=10, help='The k value used for computing Hits@k')
@click.argument('folder_path')
def compute_eval_metrics(k, folder_path):
    folder_path = Path(folder_path)
    exclude = ['metrics.csv']
    regex = r"(\d+).csv"

    predictions = []
    for dir_name, subdirs, files in os.walk(folder_path):
        print('Found directory: %s' % dir_name)
        for file in files:
            if file.endswith('.csv') and file not in exclude:
                predictions.append(file)
    predictions = sorted(predictions)

    data = []
    column_headers = None
    for file in predictions:
        row = []
        matches = re.finditer(regex, file, re.MULTILINE)
        for match in matches:
            row.append(int(match.group(1)))

        df = pd.read_csv(str(folder_path / file))
        metrics = calc_metrics(rank_predictions=df, k=k)
        if column_headers is None:
            tmp = ['epochs']
            tmp.extend(list(metrics))
            column_headers = tmp

        for column in list(metrics):
            row.append(metrics[column].values[0])
        data.append(row)

    joined_df = pd.DataFrame(data, columns=column_headers)
    joined_df.to_csv(str(folder_path / 'metrics.csv'))


@cli.command(help='Plot evaluation metrics')
@click.option('-f', '--filename', default='dummy', help='filename used with plotly')
@click.option('-f', '--plot', default='hits_at_k', help='filename used with plotly')
@click.argument('folder_path')
def plot_eval_metrics(filename, plot, folder_path):
    folder_path = Path(folder_path)
    metrics = pd.read_csv(str(folder_path))
    print(metrics)

    data = []
    title = ''
    if plot == 'hits_at_k':
        title = 'Total Hits@K'
        head_hits_at_k = go.Scatter(
            x=metrics['epochs'],
            y=metrics['head_hits_at_10'],
            name='Head Hits@10',
        )

        tail_hits_at_k = go.Scatter(
            x=metrics['epochs'],
            y=metrics['tail_hits_at_10'],
            name='Tail Hits@10'
        )
        data = [head_hits_at_k, tail_hits_at_k]
    elif plot == 'mean_hits_at_k':
        title = 'Percentage Hits@K'
        head_hits_at_k = go.Scatter(
            x=metrics['epochs'],
            y=metrics['head_mean_hits_at_10'],
            name='Head Hits@10',
        )

        tail_hits_at_k = go.Scatter(
            x=metrics['epochs'],
            y=metrics['tail_mean_hits_at_10'],
            name='Tail Hits@10'
        )
        data = [head_hits_at_k, tail_hits_at_k]
    elif plot == 'mean_rank':
        title = 'Plot of mean rank with increasing epochs'
        mean_rank = go.Scatter(
            x=metrics['epochs'],
            y=metrics['mean_rank'],
            name='Mean Rank'
        )
        data = [mean_rank]

    # Edit the layout
    layout = dict(title=title,
                  xaxis=dict(title='Epochs'),
                  yaxis=dict(title='Performance'),
                  )

    fig = dict(data=data, layout=layout)
    py.plot(fig, filename=filename)


if __name__ == '__main__':
    cli()
