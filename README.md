# UCSD CSE 151B Final Project - Team Start Late Start Rarely
## Member(s): Alex Nguyen
## How to run inference
Under the `kaggle_data/` repository is `test_public.csv` which is the provided raw testing dataset.

`encoding.csv` is also included in this repository because the model's label encoder must be fitted to a cleaned version of the training dataset.

Both of these files are necessary to run the inference.

First, clone the repository using
```
git clone https://github.com/alextongdo/cse151b-start-late-start-rarely.git
```
Running the following Python script will perform the inference and produce a `submission.csv` file which contain the predictions.
```
python validation.py
```
The model weights are saved under `model_weights/`.

## Important
I was having a issue where my model would produce different results with the exact same code and exact same model weights,
so please ensure your environment is as close as possible to mine by running
```
pip install -r requirements.txt
```
I have also included `submission.csv` which is the output I get when running the inference command.
In case your results do not match this, please replicate the `kaggle_data/` and `model_weights/` file
structure on a Google Colab notebook, which I have verified produces the expected results. Thank you!
