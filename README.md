## Usage

This script allows users to run the InFusion model with customizable settings via command-line arguments. You can specify the root directory, tensorset, batch size, weighting schemes, and choose between using the `InFusionLayer` or `InFusionNet` model. Both models are designed for boosting accuracy for classification tasks by using combinatorial fusion analysis on data item features for feature fusion. 

### Command-Line Arguments

- `--root`: Specifies the root directory where the models will be saved. Default is './sklearn_models/'.
- `--tensorset`: Specifies the tensor set to be used. Default is 'mnist'.
- `--batch_size`: Specifies the batch size for the model. Default is 2048.
- `--weighting_schemes`: Specifies a comma-separated list of weighting schemes to be used. Default is 'AC,WCDS,WCP'. Valid schemes are 'AC', 'WCDS', and 'WCP'.
- `--model_type`: Specifies the type of model to use. Choose 'layer' for `InFusionLayer` or 'net' for `InFusionNet`. Default is 'net'.

### Running the Script

Given a directory of prediction tensors as a .csv file, you can use the following command in your terminal. You can adjust the parameters as needed.

```bash
python main.py --root './tensor_predictions/' --tensorset 'mnist' --batch_size 1024 --weighting_schemes 'AC,WC-CDS,WC-KDS,WCP' --model_type 'layer'
```

If you'd like to improve classification accuracy immediately after model training, you can run the following code to see an example of how this is done using 5 sklearn models trained on the MNIST dataset.

```bash
python run_classification.py
```

`MajorityRankVoter` uses a majority voting scheme with combinatorial optimization to obtain the best votes (L. I. Kuncheva; Combining Pattern Classifiers: Methods and Algorithms, Wiley
Interscience, 2004)

If you have regression data, consider using the example in `run_regression.py`. `InFusionRegression` takes a dictionary of vectors and uses different weighting schemes and combination types to produce new models.

If you have no ground truth labels, your problem is an unsupervised learning problem. Consider using the example in `run_unsupervised.py`.