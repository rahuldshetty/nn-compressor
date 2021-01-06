"""Main module."""

import numpy as np
import matplotlib.pyplot as plt
from nn_compressor.model.model import KerasModel


def compression_ratio(uncompressed_size, compressed_size):
    return (uncompressed_size/compressed_size)*100


class KerasImageCompressor:

    def __init__(self, enc, dec):
        """
        Initialize KerasImageCompressor
        """
        super().__init__()
        self.model = KerasModel(enc, dec)

    def compress_image(self, np_data, save_path):
        """
        Compresses a Single Image Data

        Args:
            np_data ([type]): [description]
            save_path ([type]): [description]
        """
        result = self.model.compress(np.array([np_data]))
        np.save(save_path, result[0])
        original_size = np_data.nbytes
        new_size = result.nbytes
        cmpr_ratio = compression_ratio(original_size, new_size)
        print("Compressed from {} bytes to {} bytes in ratio {}".format(original_size, new_size, cmpr_ratio))

    def uncompress_image(self, file_path):
        data = np.load(file_path)
        result = self.model.uncompress(np.array([data]))[0]
        return result
        

