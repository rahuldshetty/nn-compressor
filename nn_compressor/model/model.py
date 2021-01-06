import pickle
import numpy as np
import tensorflow as tf

class Model:

    encoder = None
    decoder = None

    def __init__(self, encoder, decoder):
        """
        Represents an Auto-Encoder Model.

        Args:
            encoder (keras.models.Model): Encoder Model
            decoder (keras.models.Model): Decoder Model
        """
        self.encoder = encoder
        self.decoder = decoder

    def save(self, path: str):
        """
        Pickle and store the Keras model.

        Args:
            path (str): Path to save the model.
        """
        with open(path, 'wb') as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path:str):
        """
        Unpickles the saved model.

        Args:
            path (str): Path from where model has to be load

        Returns:
            [Model]: A Generic Model File
        """
        fh = open(path, 'rb')
        model = pickle.load(fh)
        return model

    def save_decoder(self, path):
        with open(path, 'wb') as fh:
            pickle.dump(self.decoder, fh)


    @staticmethod
    def load_decoder(path):
        fh = open(path, 'rb')
        decoder = pickle.load(fh)
        return Model(None, decoder)



class KerasModel(Model):

    def compress(self, data):
        """
        Compresses given valid content using encoder
        """
        assert type(data) ==  type(np.array([]))

        result = self.encoder.predict(data)
        return result

    def uncompress(self, data):
        assert type(data) ==  type(np.array([]))

        result = self.decoder.predict(data)
        return result


        


    
