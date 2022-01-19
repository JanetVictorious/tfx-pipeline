# Features with string data types that will be converted to indices
CATEGORICAL_FEATURE_KEYS = ['Sex', 'Embarked']

# Numerical features that are marked as continuous
NUMERIC_FEATURE_KEYS = ['Pclass', 'SibSp', 'Parch', 'Fare']

# Feature that can be grouped into buckets
BUCKET_FEATURE_KEYS = ['Age']

# Number of buckets used by tf.transform for encoding each bucket feature.
FEATURE_BUCKET_COUNT = {'Age': 10}

# Feature that the model will predict
LABEL_KEY = 'Survived'


# Utility function for renaming the feature
def transformed_name(key):
    return key + '_xf'
