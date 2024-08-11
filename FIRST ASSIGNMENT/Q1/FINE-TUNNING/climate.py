# Install necessary libraries
!pip install datasets transformers

# Importing libraries
from datasets import Dataset
from datasets import Dataset as HFDataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# Mount Google Drive (if needed)
from google.colab import drive
drive.mount('/content/drive')

# **PART 1: Loading the Pre-Trained Model**
model_name = 'distilbert-base-uncased-finetuned-sst-2-english'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

# **PART 2: Data Preparation**
# Loading the dataset from CSV
df = pd.read_csv('climate_dataset.csv')

# Visualizing sentiment distribution
df['stance'].value_counts().plot(kind='bar')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# **PART 3: Tokenization and Dataset Preparation**
# Tokenize the data
def tokenize_function(examples):
    return tokenizer(examples['topic'], truncation=True, padding='max_length', max_length=512)

# Convert the dataframe to a Hugging Face Dataset
dataset = HFDataset.from_pandas(df)

# Apply tokenization to the entire dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# Define the custom dataset class
class ClimateDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=512):
        self.texts = df['topic'].tolist()
        self.labels = torch.tensor(df['sentiment'].values, dtype=torch.long)
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

# Create training and testing datasets
train_dataset = ClimateDataset(train_df, tokenizer)
test_dataset = ClimateDataset(test_df, tokenizer)

# **PART 4: Model Evaluation Before Fine-Tuning**
training_args = TrainingArguments(
    output_dir='./results',          # Directory to save the results
    per_device_eval_batch_size=8,    # Evaluation batch size
    logging_dir='./logs',            # Directory for logs
    logging_steps=10,                # Log every 10 steps
)

# Initialize the Trainer for evaluation
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset,
)

# **Evaluate the model before fine-tuning**
eval_results_before = trainer.evaluate(eval_dataset=test_dataset)
print(f"Evaluation results before fine-tuning: {eval_results_before}")
# **PART 1: Fine-Tuning the Model**
training_args = TrainingArguments(
    output_dir='./results',        
    num_train_epochs=1,            
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=8,   
    evaluation_strategy='epoch',    
    logging_dir='./logs',           
    logging_steps=10,            
    learning_rate=5e-5,             
    load_best_model_at_end=True,   
    save_strategy='epoch',          
    fp16=True,                      
)

# Initialize the Trainer for fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# **PART 2: Fine-Tuning the Model**
trainer.train()

# **PART 3: Evaluation After Fine-Tuning**
eval_results_after = trainer.evaluate(eval_dataset=test_dataset)
print(f"Evaluation results after fine-tuning: {eval_results_after}")

# **PART 4: Save the Fine-Tuned Model**
trainer.save_model('./fine-tuned-model')

# Performance Comparison
print("\n--- Performance Comparison ---")
print("Before Fine-Tuning:")
print(eval_results_before)

print("\nAfter Fine-Tuning:")
print(eval_results_after)