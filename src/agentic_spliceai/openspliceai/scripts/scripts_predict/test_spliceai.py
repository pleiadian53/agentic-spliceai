from keras.models import load_model
from pkg_resources import resource_filename
from spliceai.utils import one_hot_encode
import numpy as np

input_sequence = 'CGATCTGACGTGGGTGTCATCGCATTATCGATATTGCAT'
# Replace this with your custom sequence

context = 10000
paths = ('models/spliceai{}.h5'.format(x) for x in range(1, 6))
models = [load_model(resource_filename('spliceai', x)) for x in paths]
x = one_hot_encode('N'*(context//2) + input_sequence + 'N'*(context//2))[None, :]
y = np.mean([models[m].predict(x) for m in range(5)], axis=0)

acceptor_prob = y[0, :, 1]
donor_prob = y[0, :, 2]