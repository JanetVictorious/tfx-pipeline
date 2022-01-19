import os
import shutil
from tfx import v1 as tfx
from tfx.orchestration.metadata import Metadata
from tfx.types import standard_component_specs

from utils.mlmd_helpers import get_latest_artifacts, visualize_artifacts


def schemas_to_html(schema_metadata_path: str,
                    schema_pipeline_name: str,
                    output_dir: str) -> None:
    metadata_connection_config = tfx.orchestration.metadata.\
        sqlite_metadata_connection_config(schema_metadata_path)

    with Metadata(metadata_connection_config) as metadata_handler:
        # Find output artifacts from MLMD.
        try:
            stat_gen_output = get_latest_artifacts(metadata_handler,
                                                   schema_pipeline_name,
                                                   'StatisticsGen')
            stats_artifacts = stat_gen_output[
                standard_component_specs.STATISTICS_KEY]

            # Visualize statistics
            visualize_artifacts(artifacts=stats_artifacts,
                                output_dir=output_dir)
        except AttributeError:
            print('StatisticsGen not available')

        try:
            schema_gen_output = get_latest_artifacts(metadata_handler,
                                                     schema_pipeline_name,
                                                     'SchemaGen')
            schema_artifacts = schema_gen_output[
                standard_component_specs.SCHEMA_KEY]

            # Visualize schema
            visualize_artifacts(artifacts=schema_artifacts,
                                output_dir=output_dir)
        except AttributeError:
            print('SchemaGen not available')

        try:
            ev_output = get_latest_artifacts(metadata_handler,
                                             schema_pipeline_name,
                                             'ExampleValidator')
            anomalies_artifacts = ev_output[
                standard_component_specs.ANOMALIES_KEY]

            # Visualize anomalies
            visualize_artifacts(artifacts=anomalies_artifacts,
                                output_dir=output_dir)
        except AttributeError:
            print('ExampleValidator not available')


def export_latest_shema(schema_metadata_path: str,
                        schema_pipeline_name: str,
                        output_dir: str) -> None:
    # Create output directory if not exists
    output_path = os.path.join(output_dir)
    os.makedirs(output_path, exist_ok=True)

    metadata_connection_config = tfx.orchestration.metadata.\
        sqlite_metadata_connection_config(schema_metadata_path)

    with Metadata(metadata_connection_config) as metadata_handler:
        # Find output artifacts from MLMD.
        schema_gen_output = get_latest_artifacts(metadata_handler,
                                                 schema_pipeline_name,
                                                 'SchemaGen')
        schema_artifacts = schema_gen_output[
            standard_component_specs.SCHEMA_KEY]

    for artifact in schema_artifacts:
        schema_path = os.path.join(artifact.uri, 'schema.pbtxt')

        vers = artifact.id
        schema_file = os.path.join(output_path, f'schema_{vers}.pbtxt')

        # Copy schema file
        shutil.copy(schema_path, schema_file)
