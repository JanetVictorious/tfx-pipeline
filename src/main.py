from absl import logging
import argparse
import os

from tfx import v1 as tfx

from modules.generators.create_pipeline import create_schema_pipeline
from modules.trainers.create_pipeline import create_training_pipeline

from utils.schema_helpers import schemas_to_html, export_latest_shema

logging.set_verbosity(logging.INFO)  # Set default logging level

# TFX global variables
BASE_PATH = os.path.dirname(__file__)

# We will create two pipelines. One for schema generation and one for training.
PIPELINE_NAME = "titanic"

# Output directory to store artifacts generated from the pipeline.
PIPELINE_ROOT = os.path.join(BASE_PATH, 'pipelines', PIPELINE_NAME)

# Path to a SQLite DB file to use as an MLMD storage.
METADATA_PATH = os.path.join(
    BASE_PATH, 'metadata', PIPELINE_NAME, 'metadata.db')

# Output directory where created models from the pipeline will be exported.
SERVING_MODEL_DIR = os.path.join(BASE_PATH, 'serving_model', PIPELINE_NAME)

# Input data directory
DATA_ROOT = os.path.abspath(os.path.join(BASE_PATH, '..', 'data/01_raw'))

# Input data path
DATA_PATH = os.path.join(DATA_ROOT, 'data.csv')


def _parse_args():
    parser = argparse.ArgumentParser()

    required = parser.add_argument_group('required arguments')

    required.add_argument(
        '--config_dir',
        help='Config files directory.',
        dest='config_dir',
        default='./config'
    )

    required.add_argument(
        '--log_dir',
        help='Log files directory.',
        dest='log_dir',
        default='./logs'
    )

    required.add_argument(
        '--schema_dir',
        help='Latest version of schema.',
        dest='schema_dir',
        default='./src/schema'
    )

    required.add_argument(
        '--eda_dir',
        help='EDA output directory for html files.',
        dest='eda_dir',
        default='./src/eda'
    )

    required.add_argument(
        '--example_dir',
        help='ExampleValidator output directory for html files.',
        dest='example_dir',
        default='./src/example'
    )

    # Drop unknown arguments
    args, unknown = parser.parse_known_args()
    return args


if __name__ == '__main__':
    PARSER = _parse_args()

    #
    # Schemas generation
    #
    tfx.orchestration.LocalDagRunner().run(
        create_schema_pipeline(pipeline_name=PIPELINE_NAME,
                               pipeline_root=PIPELINE_ROOT,
                               data_root=DATA_ROOT,
                               metadata_path=METADATA_PATH,
                               log_dir=PARSER.log_dir))
    # schemas_to_html(schema_metadata_path=SCHEMA_METADATA_PATH,
    #                 schema_pipeline_name=SCHEMA_PIPELINE_NAME,
    #                 output_dir=PARSER.eda_dir)
    # export_latest_shema(schema_metadata_path=SCHEMA_METADATA_PATH,
    #                     schema_pipeline_name=SCHEMA_PIPELINE_NAME,
    #                     output_dir=PARSER.schema_dir)

    #
    # Feature engineering
    #

    #
    # Feature selection
    #

    # #
    # # Model training
    # #
    # tfx.orchestration.LocalDagRunner().run(
    #     create_training_pipeline(
    #         pipeline_name=PIPELINE_NAME,
    #         pipeline_root=PIPELINE_ROOT,
    #         data_root=DATA_ROOT,
    #         schema_path=PARSER.schema_dir,
    #         module_file=os.path.join(
    #             BASE_PATH, 'modules/trainers/pipeline_train.py'),
    #         serving_model_dir=SERVING_MODEL_DIR,
    #         metadata_path=METADATA_PATH))
    # schemas_to_html(schema_metadata_path=METADATA_PATH,
    #                 schema_pipeline_name=PIPELINE_NAME,
    #                 output_dir=PARSER.example_dir)

    #
    # Load data
    #

    #
    # Train model
    #

    #
    # Export model
    #
