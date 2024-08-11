---

# Climate Sentiment Analysis with DistilBERT

This project demonstrates how to perform sentiment analysis on a climate-related dataset using a pre-trained DistilBERT model. The workflow includes loading a pre-trained model, preparing a dataset, evaluating the model before fine-tuning, fine-tuning the model, and evaluating the model after fine-tuning.

## Project Structure

- **`climate_dataset.csv`**: The dataset used for sentiment analysis. This file should contain at least two columns: `topic` (the text data) and `sentiment` (the sentiment labels).

- **`notebooks/`**: Contains the Jupyter Notebook or script for running the analysis.

- **`results/`**: Directory where evaluation results and model checkpoints are saved.

- **`logs/`**: Directory for logging information during training and evaluation.

## Requirements

- Python 3.x
- PyTorch
- Transformers
- Datasets
- pandas
- matplotlib

You can install the necessary libraries using pip:

```bash
pip install datasets transformers torch pandas matplotlib
```

## Instructions

### 1. Load the Pre-Trained Model

The project starts by loading the pre-trained DistilBERT model and tokenizer. The model used is `distilbert-base-uncased-finetuned-sst-2-english`, which is fine-tuned for sentiment analysis.

### 2. Data Preparation

- **Loading Data**: The dataset is loaded from a CSV file.
- **Visualization**: The distribution of sentiment labels is visualized using a bar chart.

### 3. Tokenization and Dataset Preparation

- **Tokenization**: The text data is tokenized using the DistilBERT tokenizer.
- **Dataset Preparation**: The data is split into training and testing sets. A custom `ClimateDataset` class is defined to handle tokenized inputs and labels.

### 4. Model Evaluation Before Fine-Tuning

The model is evaluated on the test set before any fine-tuning to establish a baseline performance.

### 5. Fine-Tuning the Model

The model is fine-tuned on the training dataset with specified training arguments. The `Trainer` class from the Hugging Face `transformers` library is used for training.

### 6. Evaluation After Fine-Tuning

The fine-tuned model is evaluated on the test set to assess improvements in performance.

### 7. Save the Fine-Tuned Model

The fine-tuned model is saved to the `./fine-tuned-model` directory for future use.

## Performance Comparison

The performance of the model is compared before and after fine-tuning. Evaluation results are printed to show improvements.

## Usage

To use this project:

1. Ensure you have the required libraries installed.
2. Place your dataset in the same directory or adjust the path in the code.
3. Run the provided Jupyter Notebook or Python script to perform the sentiment analysis.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)

---
