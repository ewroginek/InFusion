import argparse
import time
from models import InFusionLayer, InFusionNet

def main(args):
    ROOT = args.root
    DATASET = args.dataset
    BATCH_SIZE = args.batch_size
    MODEL_TYPE = args.model_type
    weighting_schemes = args.weighting_schemes.split(',')

    # Validate weighting schemes
    valid_schemes = {'AC', 'WCDS', 'WCP'}
    if not all(scheme in valid_schemes for scheme in weighting_schemes):
        raise ValueError(f"Invalid weighting scheme. Choose from: {', '.join(valid_schemes)}")

    # Record start time
    start_time = time.time()

    # Initialize the correct model based on user input
    if MODEL_TYPE == 'layer':
        model = InFusionLayer(ROOT=ROOT, DATASET=DATASET, weighting_schemes=weighting_schemes, BATCH_SIZE=BATCH_SIZE)
    elif MODEL_TYPE == 'net':
        model = InFusionNet(ROOT=ROOT, DATASET=DATASET, weighting_schemes=weighting_schemes, BATCH_SIZE=BATCH_SIZE)
    else:
        raise ValueError("Invalid model type. Choose 'layer' or 'net'.")

    # Use the selected model for prediction
    model.predict()

    # Print out the elapsed time
    print("Algorithm time:", time.time() - start_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the InFusion model with custom settings.")
    parser.add_argument('--root', type=str, default='./sklearn_models/', help='Root directory for saving models')
    parser.add_argument('--tensorset', type=str, default='mnist', help='Tensor outputs to be used')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--weighting_schemes', type=str, default='AC,WCDS,WCP', help='Comma-separated list of weighting schemes to use')
    parser.add_argument('--model_type', type=str, default='net', choices=['layer', 'net'], help='Type of model to use (layer or net)')

    args = parser.parse_args()
    main(args)
