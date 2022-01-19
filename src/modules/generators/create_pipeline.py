from tfx import v1 as tfx

from utils.logger_helpers import get_logger


def create_schema_pipeline(pipeline_name: str,
                           pipeline_root: str,
                           data_root: str,
                           metadata_path: str,
                           log_dir: str) -> tfx.dsl.Pipeline:
    """Creates a pipeline for schema generation."""
    # Logger name suffix
    log_suffix = 'schema_creation'
    logger = get_logger(log_dir, name_suffix=log_suffix)
    logger.info('+---------------------+')
    logger.info('|   SCHEMA CREATION   |')
    logger.info('+---------------------+')

    # Brings data into the pipeline.
    logger.info('Create ExampleGen...')
    example_gen = tfx.components.CsvExampleGen(input_base=data_root)

    # Computes statistics over data for visualization and schema
    # generation.
    logger.info('Create StatisticsGen...')
    statistics_gen = tfx.components.StatisticsGen(
        examples=example_gen.outputs['examples'])

    # Generates schema based on the generated statistics.
    logger.info('Create SchemaGen...')
    schema_gen = tfx.components.SchemaGen(
        statistics=statistics_gen.outputs['statistics'],
        infer_feature_shape=True)

    logger.info('Create pipeline components...')
    components = [example_gen,
                  statistics_gen,
                  schema_gen]

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        metadata_connection_config=tfx.orchestration.metadata
        .sqlite_metadata_connection_config(metadata_path),
        components=components)
