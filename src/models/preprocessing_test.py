# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import tensorflow as tf

from models import preprocessing


class PreprocessingTest(tf.test.TestCase):
    def test_preprocessing_fn(self):
        self.assertTrue(callable(preprocessing.preprocessing_fn))

    def test_float_imputation(self):
        # TODO: Test if imputation is done correctly over float variables
        return True

    def test_string_imputation(self):
        # TODO: Test if imputation is done correctly over string variables
        return True


if __name__ == '__main__':
    tf.test.main()
