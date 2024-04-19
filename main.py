from models import InFusionLayer, InFusionNet

infusionlayer = InFusionLayer(ROOT = './sklearn_models/', DATASET='mnist', BATCH_SIZE = 2048)
infusionlayer.predict()

infusionnet = InFusionNet(ROOT = './sklearn_models/', DATASET='mnist', BATCH_SIZE = 2048)
infusionnet.predict()