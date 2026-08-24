# README for GPT Few-Shot Classifier (Notebook Version)

## Code Overview
This code implements a GPT-based few-shot classifier for legal text classification. It processes text in two domains:
- Weapons-related cases
- Drug-related cases

It supports two processing methods:
1. Sequential Processing – Sends API requests one sentence at a time.
2. Batch Processing – Groups sentences by label and processes them in bulk.

## Setup Instructions

### 1. Install Dependencies
Ensure you have Python 3.8+ installed. Install the required libraries using:

```bash
pip install pandas tqdm scikit-learn openai
```

### 2. Set Up OpenAI API Key
Set your API key before running the notebook. You can do this by modifying the environment variable:

```python
os.environ["OPENAI_API_KEY"] = "api-key" 
```

### 3. Input Data Format

The classifier expects a CSV file with at least the following column:

text – the sentence to classify

Optional:

column for each label – the true label, for evaluation purposes


## How to Use the Notebook

Open the Jupyter Notebook (`GPT_classifier.ipynb`) and follow these steps:

### 1. Load Dependencies
Run the first few cells that import the required libraries.

### 2. Set Parameters
Modify the following variables in the relevant cell to suit your dataset:

```python
data_file = "/Data/tagged_data_manualy/weapon/test.csv"  # Input dataset file
output_directory = "/Data/results/weapon/"  # Directory for storing results
Drug_or_Wep = ""  # Choose between "Wep" or "Drug"
example_counts = 3  # Number of few-shot examples between 1-5
labels = ["Label1", "Label2", ...]  # Define labels
CSV_LOG_FILE = "log.csv" # Csv for storing log

```
### Example Parameters
```python

data_file = "/Data/tagged_data_manualy/weapon/test.csv"
output_directory = "/Data/results/weapon/" 
Drug_or_Wep = "Wep"  # Change here wich classifier to use; Drug or Wep
example_counts = [3] # Number of examples per label to use in the few-shot experiments
labels=["CONFESSION","CIR_TYPE_WEP","CIR_HELD_WAY_WEP","CIR_AMMU_AMOUNT_WEP","CIR_PURPOSE","GENERAL_CIRCUM","CIR_STATUS_WEP","REGRET","PUNISHMENT","CIR_PLANNING","RESPO","CIR_OBTAIN_WAY_WEP","CIR_USE"]
CSV_LOG_FILE = "/Data/results/weapon/batch_log.csv"
```

## 3. Choose Processing Method

### A. Sequential Processing (One sentence at a time)
- Runs GPT classification sentence-by-sentence.
- Logging and tracking enabled.

Run the notebook cells under `## Sequential Processing`.
Comment out the cells under `## Batch Processing` to avoid unnecessary execution.

### B. Batch Processing (Multiple requests at once)
- Creates a `.jsonl` file for each label and sends batch requests.
- Monitors API responses every 2 minutes.
- Can take 2-24 hours for completion.

Run the notebook cells under `## Batch Processing`.
Comment out the cells under `## Sequential Processing` to avoid redundancy.

## 4. Evaluation
After classification, you can evaluate the model’s performance. Run the cells under:

```
## Evaluations
```

These cells print:
- Classification results per label
- Precision, recall, F1-score, and accuracy metrics

## Notes
- Run only one processing method at a time to avoid redundant API requests.
- The OpenAI API has rate limits—adjust request frequency if needed.
- Batch processing is better for large datasets, but takes longer to complete.
