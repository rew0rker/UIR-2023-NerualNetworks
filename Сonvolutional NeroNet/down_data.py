import os
import tempfile
from azureml.opendatasets import MNIST

mnist_file = MNIST.get_file_dataset()
data_folder = tempfile.mkdtemp()
data_paths = mnist_file.download(data_folder, overwrite=True)
