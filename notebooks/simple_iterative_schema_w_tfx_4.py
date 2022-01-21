import os
import pprint

import tensorflow_data_validation as tfdv

from tfx.components import CsvExampleGen
from tfx.components import StatisticsGen
from tfx.components import ExampleValidator
from tfx.components import SchemaGen
from tfx.dsl.components.common.importer import Importer
from tfx.types import standard_artifacts

from tfx.orchestration.experimental.interactive.interactive_context import \
    InteractiveContext
from tensorflow_metadata.proto.v0 import schema_pb2

import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2

pp = pprint.PrettyPrinter()

base_path = os.path.dirname(__file__)

# Location of the pipeline metadata store
_pipeline_root = './pipeline/'

# Directory of the raw data files
_data_root = os.path.abspath(
    os.path.join(base_path, '..', 'data/01_raw'))

# Path to the raw training data
_data_filepath = os.path.join(_data_root, 'data.csv')

# Initialize the InteractiveContext.
# If you leave `_pipeline_root` blank, then the db will be created in
# a temporary directory.
context = InteractiveContext(pipeline_root=_pipeline_root)

# Instantiate ExampleGen with the input CSV dataset
example_gen = CsvExampleGen(input_base=_data_root)

# Execute the component
context.run(example_gen)

# Instantiate StatisticsGen with the ExampleGen ingested dataset
statistics_gen = StatisticsGen(
    examples=example_gen.outputs['examples'])

# Execute the component
context.run(statistics_gen)

# Instantiate SchemaGen with the StatisticsGen ingested dataset
schema_gen = SchemaGen(
    statistics=statistics_gen.outputs['statistics'])

# Run the component
context.run(schema_gen)

# Visualize the schema
context.show(schema_gen.outputs['schema'])

# Get the schema uri
schema_uri = schema_gen.outputs['schema']._artifacts[0].uri

# Get the schema pbtxt file from the SchemaGen output
schema = tfdv.load_schema_text(os.path.join(schema_uri, 'schema.pbtxt'))

# Restrict the range of the `Age` feature
tfdv.set_domain(
    schema, 'Age', schema_pb2.FloatDomain(name='Age', min=0.0, max=80.0))

# Display the modified schema. Notice the `Domain` column of `age`.
tfdv.display_schema(schema)

# Create schema environments for training and serving
schema.default_environment.append('TRAINING')
schema.default_environment.append('SERVING')

# Omit label from the serving environment
tfdv.get_feature(schema, 'Survived').not_in_environment.append('SERVING')

# Declare the path to the updated schema directory
_updated_schema_dir = f'{_pipeline_root}/updated_schema'

# Create the said directory
os.makedirs(_updated_schema_dir, exist_ok=True)

# Declare the path to the schema file
schema_file = os.path.join(_updated_schema_dir, 'schema.pbtxt')

# Save the curated schema to the said file
tfdv.write_schema_text(schema, schema_file)

# Use an ImporterNode to put the curated schema to ML Metadata
user_schema_importer = Importer(
    source_uri=_updated_schema_dir,
    artifact_type=standard_artifacts.Schema
)

# Run the component
context.run(user_schema_importer, enable_cache=False)

# See the result
context.show(user_schema_importer.outputs['result'])

# Instantiate ExampleValidator with the StatisticsGen and
# SchemaGen ingested data
example_validator = ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=user_schema_importer.outputs['result'])

# Run the component
context.run(example_validator)

# Visualize the results
context.show(example_validator.outputs['anomalies'])

# Get the connection config to connect to the context's metadata store
connection_config = context.metadata_connection_config

# Instantiate a MetadataStore instance with the connection config
store = mlmd.MetadataStore(connection_config)

# Get artifact types
artifact_types = store.get_artifact_types()

# Print the results
[artifact_type.name for artifact_type in artifact_types]

# Get artifact types of Schema
schema_list = store.get_artifacts_by_type('Schema')

# Print artifacts
[(f'schema uri: {schema.uri}', f'schema id:{schema.id}')
    for schema in schema_list]

# Get 1st instance of ExampleAnomalies
example_anomalies = store.get_artifacts_by_type('ExampleAnomalies')[0]

# Print the artifact id
print(f'Artifact id: {example_anomalies.id}')

# Get first event related to the ID
anomalies_id_event = store.get_events_by_artifact_ids(
    [example_anomalies.id])[0]

# Print results
print(anomalies_id_event)

# Get execution ID
anomalies_execution_id = anomalies_id_event.execution_id

# Get events by the execution ID
events_execution = store.get_events_by_execution_ids([anomalies_execution_id])

# Print results
print(events_execution)

# Filter INPUT type events
inputs_to_exval = [event.artifact_id for event in events_execution
                   if event.type == metadata_store_pb2.Event.INPUT]

# Print results
print(inputs_to_exval)
