import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorWithPadding
from torch.optim import AdamW
from datasets import load_dataset
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import transformer_lens.utils as utils
from transformer_lens import HookedTransformer
from torch.nn.utils.rnn import pad_sequence
import tqdm.auto as tqdm

# Load QNLI dataset
dataset = load_dataset('glue', 'qnli')

# Load tokenizer and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "gpt2"  # GPT-2 model, suitable for causal tasks
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 uses eos_token as pad_token
gpt2 = HookedTransformer.from_pretrained("gpt2-small", device=device)
tokenizer.padding_side = "left"

# Define classifier model
class GPT2Classifier(nn.Module):
    def __init__(self, gpt2_model, hidden_size, num_classes=2):
        super().__init__()
        self.gpt2 = gpt2_model
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, labels=None, **kwargs):
        logits, cache = self.gpt2.run_with_cache(input_ids)
        last_layer = self.gpt2.cfg.n_layers - 1
        key = f"blocks.{last_layer}.hook_resid_post"
        hidden_states = cache[key]
        last_token_states = hidden_states[:, -1, :]  # Representation of the last token
        logits = self.classifier(last_token_states)
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

# Instantiate the model
model = GPT2Classifier(gpt2, hidden_size=gpt2.cfg.d_model, num_classes=2)
model.to(device)

# Tokenization function
def tokenize_function(examples):
    inputs = tokenizer(
        examples['question'],
        examples['sentence'],
        padding='max_length',
        max_length=128,  # Adjust as needed
        truncation=True
    )
    inputs['labels'] = examples['label']
    return inputs

# Tokenize the datasets
train_dataset = dataset['train'].map(tokenize_function, batched=True)
val_dataset = dataset['validation'].map(tokenize_function, batched=True)

# Remove unnecessary columns
train_dataset = train_dataset.remove_columns(["question", "sentence", "idx"])
val_dataset = val_dataset.remove_columns(["question", "sentence", "idx"])

# Custom data collator
class CustomDataCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer, padding=True, return_tensors="pt")

    def __call__(self, features):
        batch = super().__call__(features)
        if "labels" in features[0]:
            labels = [f["labels"] for f in features]
            batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch

collator = CustomDataCollator(tokenizer=tokenizer)

# Evaluation metric function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    accuracy = accuracy_score(labels, predictions)
    return {"accuracy": accuracy}

# Training arguments
training_args = TrainingArguments(
    output_dir="./qnli_output",
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    learning_rate=2e-5,
    warmup_steps=500,
    weight_decay=0.01,
    fp16=True,  # Enable mixed-precision training if using GPU
    report_to="none"
)

# Instantiate Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collator,
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()

# Save the best model
trainer.save_model("./qnli_best_model")

# Evaluate the model
results = trainer.evaluate()
print(f"Validation Accuracy: {results['eval_accuracy']:.4f}")
