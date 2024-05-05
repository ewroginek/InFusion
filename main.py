import argparse
import time
import os
from models import InFusionLayer, InFusionNet
from utils.utils import get_outputs

def main(args):
    ROOT = args.root
    DATASET = args.tensorset
    BATCH_SIZE = args.batch_size
    MODEL_TYPE = args.model_type
    weighting_schemes = args.weighting_schemes.split(',')
    model_selection = args.model_selection

    # Validate weighting schemes
    valid_schemes = {'AC', 'WC-CDS', 'WC-KDS', 'WCP'}
    if not all(scheme in valid_schemes for scheme in weighting_schemes):
        raise ValueError(f"Invalid weighting scheme. Choose from: {', '.join(valid_schemes)}")

    if not os.path.exists(f'./results/'): os.mkdir(f'./results/')
    if not os.path.exists(f'./results/{ROOT}'): os.mkdir(f'./results/{ROOT}')
    if not os.path.exists(f'./results/{ROOT}/{DATASET}'): os.mkdir(f'./results/{ROOT}/{DATASET}')
    OUTPATH = f'./results/{ROOT}/{DATASET}'

    score_data, ground_truth = get_outputs(f"{ROOT}/{DATASET}")

    # Record start time
    start_time = time.time()

    # Initialize the correct model based on user input
    if MODEL_TYPE == 'layer':
        model = InFusionLayer(score_data, ground_truth['0'], OUTPATH, weighting_schemes=weighting_schemes, BATCH_SIZE=BATCH_SIZE)
        model.predict()
    elif MODEL_TYPE == 'net':
        model = InFusionNet(score_data, ground_truth['0'], OUTPATH, weighting_schemes=weighting_schemes, BATCH_SIZE=BATCH_SIZE)
        model.predict(model_selection=model_selection)
    else:
        raise ValueError("Invalid model type. Choose 'layer' or 'net'.")

    # Print out the elapsed time
    print("Algorithm time:", time.time() - start_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the InFusion model with custom settings.")
    parser.add_argument('--root', type=str, default='./sklearn_models/', help='Root directory for saving models')
    parser.add_argument('--tensorset', type=str, default='mnist', help='Tensor outputs to be used')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--weighting_schemes', type=str, default='AC,WC-CDS,WC-KDS,WCP', help='Comma-separated list of weighting schemes to use')
    parser.add_argument('--model_type', type=str, default='net', choices=['layer', 'net'], help='Type of model to use (layer or net)')
    parser.add_argument('--model_selection', type=str, default='ER-Algorithm', choices=['ER-Algorithm', 'Majority Vote'], help='Model selection criteria to use (ER-Algorithm or Majority Vote: InFusionNet only)')

    args = parser.parse_args()
    main(args)