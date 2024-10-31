## ME-HGNN


### Dependencies

Recent versions of the following packages for Python 3 are required:
* PyTorch 1.2.0
* DGL 0.3.1
* NetworkX 2.3
* scikit-learn 0.21.3
* NumPy 1.17.2
* SciPy 1.3.1

Dependencies for the preprocessing code are not listed here.

### Datasets

The preprocessed datasets are available at:
* VulKG - [Dropbox](https://github.com/happyResearcher/VulKG/tree/main/import)


### Usage

1. Create `checkpoint/` and `data/preprocessed` directories
2. Extract the zip file downloaded from the section above to `data/preprocessed`
    * E.g., extract the content of `IMDB_processed.zip` to `data/preprocessed/IMDB_processed`
2. Execute one of the following command from the project home directory:
    * `python run_VULKG.py`


For more information about the available options of the model, you may check by executing `python run_VULKG.py --help`


