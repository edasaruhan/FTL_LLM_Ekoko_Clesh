# Install necessary libraries
!pip install pandas matplotlib seaborn transformers datasets

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
from torch.utils.data import Dataset as TorchDataset
from sklearn.model_selection import train_test_split

# **1. Create and Load the Dataset**

# Fictitious data
data = {
    'text': [
        'The creation of new jobs in the region is helping to boost the local economy.',
        'Unemployment is still a major issue in many areas, affecting economic growth.',
        'Initiatives for worker training have led to an increase in job opportunities.',
        'The economy is slowly recovering after the crisis, but many are still facing difficulties.',
        'Entrepreneurship incentives are having a positive impact on the job market.',
    ],
    'label': [
        'positive',
        'negative',
        'positive',
        'negative',
        'positive',
    ]
}

# Create the DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('economy_dataset.csv', index=False)

# Load the dataset from the CSV file
df = pd.read_csv('economy_dataset.csv')

# **2. Exploratory Data Analysis (EDA)**

# View the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe(include='all'))

# Class distribution
class_distribution = df['label'].value_counts()
print("\nClass Distribution:")
print(class_distribution)

# Visualize class distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='label', data=df, palette='viridis')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Analyze text length
df['text_length'] = df['text'].apply(len)

# Display descriptive statistics of text length
print("\nText Length Statistics:")
print(df['text_length'].describe())

# Visualize text length distribution
plt.figure(figsize=(8, 6))
sns.histplot(df['text_length'], bins=20, kde=True, color='teal')
plt.title('Text Length Distribution')
plt.xlabel('Text Length')
plt.ylabel('Count')
plt.show()

# Display sample texts with labels
print("\nSample Texts with Labels:")
print(df[['text', 'label']].sample(n=3, random_state=42))  # Sample 3 rows instead of 10

# **3. Load the Pre-Trained Model and Tokenizer**
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

# **4. Convert the DataFrame to a Hugging Face Dataset**

def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

dataset = Dataset.from_pandas(df)
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Define the custom dataset class for PyTorch
class EconomyDataset(TorchDataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.texts = df['text'].tolist()
        self.labels = torch.tensor(df['label'].map({'positive': 1, 'negative': 0}).values, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': label
        }

# Create the dataset for PyTorch
test_dataset = EconomyDataset(test_df, tokenizer)

# **5. Define Training Arguments for Evaluation**
training_args = TrainingArguments(
    output_dir='./results',
    per_device_eval_batch_size=8,
    logging_dir='./logs',
    logging_steps=10,
)

# Initialize the Trainer for evaluation
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset,
)

# Evaluate the model before fine-tuning
eval_results_before = trainer.evaluate(eval_dataset=test_dataset)
print(f"Evaluation results before fine-tuning: {eval_results_before}")
