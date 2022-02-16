import tensorflow as tf
import tensorflow_transform as tft
# from tensorflow_transform.tf_metadata import dataset_metadata
# from tensorflow_transform.tf_metadata import schema_utils

from models import preprocessing

tf.compat.v1.disable_eager_execution()


class PreprocessingTest(tf.test.TestCase):

    # # Define sample data
    # raw_data = [
    #     {'x': 1.0, 'y': 1, 's': 'hello'},
    #     {'x': 2.0, 'y': 2, 's': 'world'},
    #     {'x': 3.0, 'y': 3, 's': 'hello'},
    #     {'x': tf.train.Feature(),
    #      'y': tf.train.Feature(),
    #      's': tf.train.Feature(bytes_list=tf.train.BytesList(value=[]))}]

    # # Define the schema as a DatasetMetadata object
    # raw_data_metadata = dataset_metadata.DatasetMetadata(
    #     # Use convenience function to build a Schema protobuf
    #     schema_utils.schema_from_feature_spec({
    #         # Define a dictionary mapping the keys to its feature spec type
    #         'y': tf.io.FixedLenFeature([], tf.int32),
    #         'x': tf.io.FixedLenFeature([], tf.float32),
    #         's': tf.io.FixedLenFeature([], tf.string)}))

    def test_preprocessing_fn(self):
        self.assertTrue(callable(preprocessing.preprocessing_fn))

    def test_fill_in_missing(self):
        self.assertTrue(callable(preprocessing._fill_in_missing))

    # def test_float_imputation(self):
    #     # Assign sparse tensor
    #     x = tf.sparse.SparseTensor(indices=[[0, 0], [2, 0]],
    #                                values=[1.0, 3.0],
    #                                dense_shape=[3, 1])

    #     # Impute missing value
    #     cmp_fill_val = tf.cast(tft.mean(x), x.dtype)
    #     x_imp = preprocessing._fill_in_missing(x, fill_value=cmp_fill_val)

    #     fill_val = 2.0
    #     x_res = tf.squeeze(tf.sparse.to_dense(
    #         tf.SparseTensor(x.indices,
    #                         x.values,
    #                         [x.dense_shape[0], 1]),
    #         fill_val), axis=1)

    #     # Assert equal
    #     self.assertEqual(x_imp, x_res)

    # def test_string_imputation(self):
    #     # Assign sparse tensor
    #     x = tf.sparse.SparseTensor(indices=[[0, 0], [2, 0]],
    #                                values=['hello', 'world'],
    #                                dense_shape=[3, 1])

    #     # Impute missing value
    #     x_imp = preprocessing._fill_in_missing(
    #         x, fill_value=b'missing')

    #     fill_val = b'missing'
    #     x_res = tf.squeeze(tf.sparse.to_dense(
    #         tf.SparseTensor(x.indices,
    #                         x.values,
    #                         [x.dense_shape[0], 1]),
    #         fill_val), axis=1)

    #     # Assert equal
    #     # self.assertEqual(x_res, x_imp)
    #     return True


if __name__ == '__main__':
    tf.test.main()
