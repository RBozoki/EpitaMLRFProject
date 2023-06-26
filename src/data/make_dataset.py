# -*- coding: utf-8 -*-
import glob
import click
import logging
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv
import pickle
from tqdm import tqdm

import numpy as np
import pandas as pd

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # Load all data_batch_* files
    logger.info(f"Looking for data files in {input_filepath}")
    if not Path(input_filepath).exists():
        logger.error(f"The directory {input_filepath} does not exist.")
    else:
        files = glob.glob(str(Path(input_filepath) / 'data_batch_*'))
        if not files:
            logger.error(f"No files found in {input_filepath}")
        else:
            logger.info(f"Found files: {files}")
    data_batches = [unpickle(file) for file in files]

    # Save processed data
    for i, data_batch in enumerate(tqdm(data_batches, desc="Processing data batches")):
        data = data_batch[b'data'].tolist()  # Convert numpy array to list
        labels = data_batch[b'labels']
        filenames = [name.decode('utf-8') for name in data_batch[b'filenames']]  # Convert byte filenames to strings

        # Combine all columns into a DataFrame
        df = pd.DataFrame({
            'data': data,
            'labels': labels,
            'filenames': filenames
        })

        df.to_csv(Path(output_filepath) / f'data_batch_{i}.csv', index=False)







if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()
