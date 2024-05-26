# BERT Retrieval-Enhanced Models

This project demonstrates how to fine-tune a BERT model for retrieval-enhanced tasks using Azure Machine Learning.

## Project Structure

```
bert-retrieval-enhanced-models/
├── data/
│   ├── documents.csv
│   └── queries.csv
├── scripts/
│   ├── train.py
│   └── score.py
├── notebooks/
│   └── azure_ml_example.ipynb
├── environment.yml
├── config.json
├── README.md
└── .gitignore
```

## Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/davidcloudformation/bert-retrieval-enhanced-models.git
    cd bert-retrieval-enhanced-models
    ```

2. Create a conda environment:
    ```bash
    conda env create -f environment.yml
    conda activate bert-retrieval
    ```

3. Run the Jupyter notebook in `notebooks/azure_ml_example.ipynb` to train and deploy the model.

## Data

- `data/documents.csv`: Sample documents.
- `data/queries.csv`: Sample queries.

## Scripts

- `scripts/train.py`: Script to train the retrieval-enhanced model.
- `scripts/score.py`: Script to score queries using the trained model.

## Environment

The environment is defined in `environment.yml`.

## Azure ML

The Azure ML configuration is defined in `config.json`.

## License

This project is licensed under the MIT License.
