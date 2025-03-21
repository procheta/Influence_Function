import circuitsvis as cv
# Testing that the library works
cv.examples.hello("Neel")
try:
    from IPython import get_ipython

    ip = get_ipython()
    if ip is not None:
        # Load the autoreload extension, available only in the IPython environment
        if not ip.extension_manager.loaded:
            ip.extension_manager.load('autoreload')
            ip.run_line_magic('autoreload', '2')
except Exception as e:
    print("Not running in an IPython environment, skipping autoreload settings.", e)

import plotly.io as pio
pio.renderers.default = "notebook_connected"
print(f"Using renderer: {pio.renderers.default}")

import torch
import torch.nn as nn
import einops
from fancy_einsum import einsum
import tqdm.auto as tqdm
import plotly.express as px
from safetensors.torch import load_file

from jaxtyping import Float
from functools import partial
# import transformer_lens
import transformer_lens.utils as utils
from transformer_lens.hook_points import (
    HookPoint,
)  # Hooking utilities
from transformer_lens import HookedTransformer, FactoredMatrix

from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm
from datasets import load_dataset, Features, Sequence, Value
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback
from torch.utils.data import DataLoader
import numpy as np

def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(utils.to_numpy(tensor), color_continuous_midpoint=0.0, color_continuous_scale="RdBu", labels={"x": xaxis, "y": yaxis}, **kwargs).show(renderer)

def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x": xaxis, "y": yaxis}, **kwargs).show(renderer)

def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(y=y, x=x, labels={"x": xaxis, "y": yaxis, "color": caxis}, **kwargs).show(renderer)

device = utils.get_device()

gpt2 = HookedTransformer.from_pretrained("gpt2-small", device=device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

class GPT2Classifier(nn.Module):
    def __init__(self, gpt2_model, hidden_size, num_classes=2):
        super().__init__()
        self.gpt2 = gpt2_model  # TransformerLens' HookedTransformer instance
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, labels=None, **kwargs):
        # Call run_with_cache to obtain outputs and cache
        logits, cache = self.gpt2.run_with_cache(input_ids)
        last_layer = gpt2.cfg.n_layers - 1
        key = f"blocks.{last_layer}.hook_resid_post"
        hidden_states = cache[key]
        #print(hidden_states.shape)  # torch.Size([batch, sequence_length, 768])
        # Get the representation of the last token for each sample
        last_token_states = hidden_states[:, -1, :]  # [batch, hidden_size]
        logits = self.classifier(last_token_states)   # [batch, num_classes]
        # If labels are provided, compute the loss
        if labels is not None:
          loss_fn = nn.CrossEntropyLoss()
          loss = loss_fn(logits, labels)
          return {"loss": loss, "logits": logits}  # Return dictionary
        else:
          return {"logits": logits}

model = GPT2Classifier(gpt2, hidden_size=gpt2.cfg.d_model, num_classes=2)
model.to(device)

import pandas as pd
dataset = load_dataset('csv', data_files=r"D:\OneDrive - The University of Liverpool\LLMs\gpt-neo-classification\train.csv")
train_dataset = dataset["train"].select([i for i in range(len(dataset["train"])) if i % 10 != 0])  # Use 90% of the data for training
val_dataset = dataset["train"].select([i for i in range(len(dataset["train"])) if i % 10 == 0])

def tokenize_function(examples):
    inputs = tokenizer(examples['text'], padding='max_length', max_length=60, truncation=True)
    labels_ids = {'neg': 0, 'pos': 1}

    labels = [labels_ids.get(label, 0) for label in examples['label']]
    inputs['labels'] = labels
    return inputs

train_dataset = train_dataset.map(tokenize_function, batched=True)
train_dataset = train_dataset.remove_columns(["text", "label"])
val_dataset = val_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.remove_columns(["text", "label"])

new_features = Features({
    'input_ids': Sequence(feature=Value(dtype='int32')),
    'attention_mask': Sequence(feature=Value(dtype='int8')),
    'labels': Value(dtype='int64')
})

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
dataloader = DataLoader(train_dataset, batch_size=32, collate_fn=collator)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).astype(np.float32).mean().item()
    return {"accuracy": accuracy}

training_args = TrainingArguments(
    output_dir=r"D:\OneDrive - The University of Liverpool\LLMs\TransformerLnes\TransformerLens\model\movie_review\gpt2\output_model",
    fp16=True,
    num_train_epochs=10,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    logging_steps=50,
    learning_rate=4e-5,
    warmup_steps=70,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    report_to="none"
)

early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=2)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collator,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback]
)

#trainer.train()
#trainer.save_model(r"D:\OneDrive - The University of Liverpool\LLMs\TransformerLnes\TransformerLens\model\movie_review\gpt2\best_model")

def evaluate(model, dataloader, device):
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            labels_tensor = batch["labels"].to(device)
            outputs = model(input_ids)
            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=-1)
            true_labels.extend(labels_tensor.cpu().numpy().tolist())
            pred_labels.extend(preds.cpu().numpy().tolist())
    accuracy = (np.array(true_labels) == np.array(pred_labels)).mean()
    return accuracy

test_dataset = load_dataset('csv', data_files=r"D:\OneDrive - The University of Liverpool\LLMs\gpt-neo-classification\data\test_50.csv")['train']
test_dataset = test_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.remove_columns(["text", "label"])
new_features = Features({
    'input_ids': Sequence(feature=Value(dtype='int32')),
    'attention_mask': Sequence(feature=Value(dtype='int8')),
    'labels': Value(dtype='int64')
})
collator = CustomDataCollator(tokenizer=tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=collator)

model_bf = GPT2Classifier(gpt2, hidden_size=gpt2.cfg.d_model, num_classes=2).to(device)

# Create training DataLoader (assuming train_dataset is defined)
train_dataloader = DataLoader(train_dataset, batch_size=32, collate_fn=collator)

# # Calculate accuracy before fine-tuning
# train_accuracy_before = evaluate(model_bf, train_dataloader, device)
# print(f"Accuracy on training set before fine-tuning: {train_accuracy_before:.4f}")
#
# test_accuracy_before = evaluate(model_bf, test_dataloader, device)
# print(f"Accuracy on test set before fine-tuning: {test_accuracy_before:.4f}")

model = GPT2Classifier(gpt2, hidden_size=gpt2.cfg.d_model, num_classes=2)

state_dict = load_file(r"D:\OneDrive - The University of Liverpool\LLMs\TransformerLnes\MovieReview\model\gpt2\model.safetensors")
model.load_state_dict(state_dict)
model.to(device)
train_accuracy_after = evaluate(model, train_dataloader, device)
print(f"Accuracy on training set after fine-tuning: {train_accuracy_after:.4f}")
test_accuracy_after = evaluate(model, test_dataloader, device)
print(f"Accuracy on test set after fine-tuning: {test_accuracy_after:.4f}")
