from tfx import v1 as tfx


def create_training_pipeline(pipeline_name: str, pipeline_root: str,
                             data_root: str, schema_path: str,
                             module_file: str, serving_model_dir: str,
                             metadata_path: str) -> tfx.dsl.Pipeline:
    """Creates a three component penguin pipeline with TFX."""
    # Brings data into the pipeline
    example_gen = tfx.components.CsvExampleGen(input_base=data_root)

    # Computes statistics over data for visualization and example validation
    statistics_gen = tfx.components.StatisticsGen(
        examples=example_gen.outputs['examples'])

    # Import the schema
    schema_importer = tfx.dsl.Importer(
        source_uri=schema_path,
        artifact_type=tfx.types.standard_artifacts.Schema).with_id(
            'schema_importer')

    # Performs anomaly detection based on statistics and data schema
    example_validator = tfx.components.ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_importer.outputs['result'])

    # Uses user-provided Python function that trains a model
    trainer = tfx.components.Trainer(
        module_file=module_file,
        examples=example_gen.outputs['examples'],
        schema=schema_importer.outputs['result'],  # Pass the imported schema
        train_args=tfx.proto.TrainArgs(num_steps=100),
        eval_args=tfx.proto.EvalArgs(num_steps=5))

    # Pushes the model to a filesystem destination.
    pusher = tfx.components.Pusher(
        model=trainer.outputs['model'],
        push_destination=tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(
                base_directory=serving_model_dir)))

    # Pipeline components
    components = [example_gen,
                  statistics_gen,
                  schema_importer,
                  example_validator,
                  trainer,
                  pusher]

    return tfx.dsl.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        metadata_connection_config=tfx.orchestration.metadata
        .sqlite_metadata_connection_config(metadata_path),
        components=components)
