from nn_compressor import KerasImageCompressor
from keras.models import load_model
from skimage import io, transform
import matplotlib.pyplot as plt
import keras

print(keras.__version__)


enc = load_model('encoder.h5')
dec = load_model('decoder.h5')

def display(img):
    plt.imshow(img)
    plt.show()


def process(image_path):
    img = io.imread(image_path)
    img = transform.resize(img, (28,28))
    display(img)
    return img


image = "mnist.png"
save_path = 'compress.npy'

img = process(image)

KIP = KerasImageCompressor(enc, dec)


KIP.compress_image(img, save_path)

res = KIP.uncompress_image(save_path) 

display(res.reshape((28,28))*256)
