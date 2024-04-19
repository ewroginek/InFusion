import time
from models import InFusionLayer, InFusionNet

ROOT = './sklearn_models/'
DATASET = 'mnist'
weighting_schemes = ['AC', 'WCDS']
BATCH_SIZE = 2048

# infusionlayer = InFusionLayer(ROOT = ROOT, DATASET = DATASET, weighting_schemes = weighting_schemes, BATCH_SIZE = BATCH_SIZE)
# infusionlayer.predict()

start_time = time.time()
infusionnet = InFusionNet(ROOT = ROOT, DATASET = DATASET, weighting_schemes = weighting_schemes, BATCH_SIZE = BATCH_SIZE)
infusionnet.predict()
print("Algorithm time:", time.time() - start_time)