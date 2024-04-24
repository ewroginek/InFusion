## Usage

This script allows users to run the InFusion model with customizable settings via command-line arguments. You can specify the root directory, tensorset, batch size, weighting schemes, and choose between using the `InFusionLayer` or `InFusionNet` model. Both models are designed for boosting accuracy for classification tasks by using combinatorial fusion analysis on data item features for feature fusion. 

### Command-Line Arguments

- `--root`: Specifies the root directory where the models will be saved. Default is './sklearn_models/'.
- `--tensorset`: Specifies the tensor set to be used. Default is 'mnist'.
- `--batch_size`: Specifies the batch size for the model. Default is 2048.
- `--weighting_schemes`: Specifies a comma-separated list of weighting schemes to be used. Default is 'AC,WCDS,WCP'. Valid schemes are 'AC', 'WCDS', and 'WCP'.
- `--model_type`: Specifies the type of model to use. Choose 'layer' for `InFusionLayer` or 'net' for `InFusionNet`. Default is 'net'.

### Running the Script

To run the script, use the following command in your terminal. You can adjust the parameters as needed.

```bash
python main.py --root './tensor_predictions/' --tensorset 'mnist' --batch_size 1024 --weighting_schemes 'AC,WCP' --model_type 'layer'
