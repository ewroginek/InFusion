from models import InFusionNet
from examples.mnist_example import train_models

T, G = train_models()
OUTPATH = f'./results/sklearn_models/mnist'
InFusionNet(T, G, OUTPATH, weighting_schemes=['AC', 'WCDS', 'WCP'], BATCH_SIZE=2048).predict()