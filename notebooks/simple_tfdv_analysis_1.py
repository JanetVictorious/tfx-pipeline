import os
import pandas as pd

import tensorflow_data_validation as tfdv
from tensorflow_metadata.proto.v0 import schema_pb2
from tensorflow_data_validation.utils import slicing_util
from tensorflow_metadata.proto.v0.statistics_pb2 import \
    DatasetFeatureStatisticsList

from sklearn.model_selection import train_test_split

base_path = os.path.dirname(__file__)

data_path = os.path.abspath(
    os.path.join(base_path, '..', 'data/01_raw/data.csv'))

# Load data
df = pd.read_csv(data_path)

# Inspect data
df.head()
df.shape
df.isna().mean()

# Split data
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
train_df.shape
eval_df.shape

# Generate training dataset statistics
train_stats = tfdv.generate_statistics_from_dataframe(train_df)
train_stats

# Visualize training dataset statistics
tfdv.visualize_statistics(train_stats)

# Infer schema from the computed statistics.
schema = tfdv.infer_schema(statistics=train_stats)

# Display the inferred schema
tfdv.display_schema(schema)

# Generate evaluation dataset statistics
eval_stats = tfdv.generate_statistics_from_dataframe(eval_df)

# Compare training with evaluation
tfdv.visualize_statistics(
    lhs_statistics=eval_stats,
    rhs_statistics=train_stats,
    lhs_name='EVAL_DATASET',
    rhs_name='TRAIN_DATASET'
)

# Check evaluation data for errors by validating the evaluation dataset
# statistics using the reference schema
anomalies = tfdv.validate_statistics(statistics=eval_stats, schema=schema)

# Visualize anomalies
tfdv.display_anomalies(anomalies)

# # Relax the minimum fraction of values that must come from the domain for
# the feature `Age`

# age_feature = tfdv.get_feature(schema, 'Age')
# age_feature.distribution_constraints.min_domain_mass = 0.8

# Relax the minimum fraction of values that must come from the domain for the
# feature `Embarked`
embarked_feature = tfdv.get_feature(schema, 'Embarked')
embarked_feature.distribution_constraints.min_domain_mass = 0.99

"""
# Add new value to the domain of the feature `race`
race_domain = tfdv.get_domain(schema, 'race')
race_domain.value.append('Asian')
"""

# Restrict the range of the `Age` feature
tfdv.set_domain(
    schema, 'Age', schema_pb2.FloatDomain(name='Age', min=0.0, max=81.0))

# Display the modified schema. Notice the `Domain` column of `age`.
tfdv.display_schema(schema)

# Validate eval stats after updating the schema
updated_anomalies = tfdv.validate_statistics(eval_stats, schema)
tfdv.display_anomalies(updated_anomalies)

# Investigate slices
slice_fn = slicing_util.get_feature_value_slicer(features={'Sex': None})

# Declare stats options
slice_stats_options = tfdv.StatsOptions(
    schema=schema,
    slice_functions=[slice_fn],
    infer_type_from_schema=True)

# Convert dataframe to CSV since `slice_functions` works only with
# `tfdv.generate_statistics_from_csv`
CSV_PATH = 'slice_sample.csv'
train_df.to_csv(CSV_PATH)

# Calculate statistics for the sliced dataset
sliced_stats = tfdv.generate_statistics_from_csv(
    CSV_PATH, stats_options=slice_stats_options)

print(f'Datasets generated: {[sliced.name for sliced in sliced_stats.datasets]}')
print(f'Type of sliced_stats elements: {type(sliced_stats.datasets[0])}')

# Convert `Male` statistics (index=1) to the correct type and get the
# dataset name
male_stats_list = DatasetFeatureStatisticsList()
male_stats_list.datasets.extend([sliced_stats.datasets[1]])
male_stats_name = sliced_stats.datasets[1].name

# Convert `Female` statistics (index=2) to the correct type and get the
# dataset name
female_stats_list = DatasetFeatureStatisticsList()
female_stats_list.datasets.extend([sliced_stats.datasets[2]])
female_stats_name = sliced_stats.datasets[2].name

# Visualize the two slices side by side
tfdv.visualize_statistics(
    lhs_statistics=male_stats_list,
    rhs_statistics=female_stats_list,
    lhs_name=male_stats_name,
    rhs_name=female_stats_name
)
