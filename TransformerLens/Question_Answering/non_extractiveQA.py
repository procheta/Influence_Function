import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
import random
import math
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from fuzzywuzzy import fuzz
import numpy as np
from tqdm import tqdm
import re

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Detected device: {device}")

print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.__version__)

###############################################
# 1. Improved Data Preparation: Load SQuAD dataset and optimize sample filtering
###############################################
print("Loading SQuAD dataset...")
squad = load_dataset("squad")  # Load SQuAD v1.1

def is_single_word(answer_list):
    """Determine if the first answer in the list is a single word"""
    if answer_list and len(answer_list[0].strip().split()) == 1:
        return True
    return False

def prepare_examples(split, max_examples=None):
    """
    For each example, construct two versions and optimize sample filtering:
    - Ensure the answer appears in the context
    - Ensure the answer is a single word
    - Add clearer instruction prompts
    """
    examples = []
    for example in squad[split]:
        if is_single_word(example["answers"]["text"]):
            answer = example["answers"]["text"][0].strip()
            # Check if the answer appears in the context (case-insensitive)
            if answer.lower() in example["context"].lower():
                # Use a more explicit prompt format
                prompt_full = f"Answer the following question based on the given context.\nQuestion: {example['question']}\nContext: {example['context']}\nAnswer: {answer}"
                prompt_eval = f"Answer the following question based on the given context.\nQuestion: {example['question']}\nContext: {example['context']}\nAnswer:"
                examples.append({
                    "prompt_full": prompt_full,
                    "prompt_eval": prompt_eval,
                    "answer": answer,
                    "question": example['question'],
                    "context": example['context']
                })
        if max_examples and len(examples) >= max_examples:
            break
    return examples

# Use all samples that meet the criteria; adjust if memory is limited
train_examples = prepare_examples("train", max_examples=30267)
val_examples = prepare_examples("validation", max_examples=3429)
print(f"Number of training samples: {len(train_examples)}, Number of validation samples: {len(val_examples)}")

###############################################
# 2. Improved Tokenization Settings
###############################################
model_name = "gpt2-medium"  # Use GPT-2 model
tokenizer = AutoTokenizer.from_pretrained(model_name)
# GPT-2 does not have a default pad_token; set it to eos_token
tokenizer.pad_token = tokenizer.eos_token
max_length = 384  # Increase max sequence length to accommodate more context

def tokenize_prompt(prompt, padding="max_length", truncation=True):
    return tokenizer(
        prompt,
        truncation=truncation,
        padding=padding,
        max_length=max_length,
        return_tensors="pt"
    )

###############################################
# 3. Improved Dataset and DataLoader
###############################################
class QATrainingDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        prompt = self.examples[idx]["prompt_full"]
        tokenized = tokenize_prompt(prompt)
        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)
        
        # Get the position of the answer in the input (used for calculating focus loss)
        answer = self.examples[idx]["answer"]
        answer_tokens = tokenizer.encode(" " + answer, add_special_tokens=False)
        
        # Set the training target as input_ids (causal LM training target)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),
            "answer_tokens": torch.tensor(answer_tokens),
            "raw_answer": answer
        }

class QAEvaluationDataset(Dataset):
    def __init__(self, examples):
        # For evaluation, store the prompt without the answer and the ground truth answer
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

# Use a smaller batch size and gradient accumulation to reduce memory usage
batch_size = 8
train_dataset = QATrainingDataset(train_examples)

from torch.nn.utils.rnn import pad_sequence

def custom_collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    
    # Pad answer_tokens to the maximum length in the batch
    answer_tokens = [item["answer_tokens"] for item in batch]
    answer_tokens_padded = pad_sequence(answer_tokens, batch_first=True, padding_value=tokenizer.pad_token_id)
    
    # For raw_answer, collect into a list (no stacking needed)
    raw_answers = [item["raw_answer"] for item in batch]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "answer_tokens": answer_tokens_padded,
        "raw_answer": raw_answers
    }

# Update the DataLoader to use the custom collate function
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,
    num_workers=2,
    pin_memory=True,
    collate_fn=custom_collate_fn  # Use custom collate_fn
)

eval_dataset = QAEvaluationDataset(val_examples)
eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False)

###############################################
# 4. Load GPT-2 Model Built with TransformerLens and Set Up Progressive Fine-Tuning
###############################################
print("Loading TransformerLens GPT-2 model...")
model = HookedTransformer.from_pretrained(model_name, device=device)
# Set pad_token_id
model.cfg.pad_token_id = tokenizer.pad_token_id
model.to(device)
print(f"Model current device: {next(model.parameters()).device}")

# Initially, only train the last few layers
for param in model.parameters():
    param.requires_grad = False
    
# Only unfreeze the last 3 transformer blocks and the language modeling head
for i in range(model.cfg.n_layers - 3, model.cfg.n_layers):
    for param in model.blocks[i].parameters():
        param.requires_grad = True
for param in model.ln_final.parameters():
    param.requires_grad = True
for param in model.unembed.parameters():
    param.requires_grad = True

model.train()

###############################################
# 5. Improved Temperature Sampling Generation Function
###############################################
def temperature_sampling(model, input_ids, tokenizer, max_new_tokens=5, temperature=0.7, top_k=50, top_p=0.9):
    """
    Generate text using temperature sampling combined with top-k and top-p filtering
    """
    model.eval()
    input_len = input_ids.shape[1]
    attention_mask = torch.ones_like(input_ids)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids, return_type="logits")
            next_token_logits = outputs[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens whose cumulative probability exceeds the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep the first token (the one with the highest probability)
                sorted_indices_to_remove[..., 0] = 0
                
                # Rearrange to original order and filter
                indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool)
                indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('Inf')
            
            # Calculate probability distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Sample the next token
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append the generated token to the input sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
            
            # If EOS is generated, stop early
            if next_token.item() == tokenizer.eos_token_id:
                break
                
    model.train()
    return input_ids

###############################################
# 6. Improved Beam Search Implementation
###############################################
def improved_beam_search(model, input_ids, tokenizer, max_new_tokens=10, num_beams=5, temperature=0.7):
    """
    Optimized beam search implementation
    """
    model.eval()
    device = input_ids.device
    batch_size = input_ids.shape[0]
    input_ids_len = input_ids.shape[1]
    
    # Ensure that batch_size is 1
    if batch_size != 1:
        raise ValueError("This implementation only supports batch_size=1")
    
    # Initialize beams
    beam_scores = torch.zeros(num_beams, device=device)
    beam_sequences = input_ids.repeat(num_beams, 1)
    beam_finished = [False] * num_beams
    
    for _ in range(max_new_tokens):
        if all(beam_finished):
            break
            
        # Process only beams that are not finished
        active_idxs = [i for i, finished in enumerate(beam_finished) if not finished]
        if not active_idxs:
            break
        
        active_beam_sequences = beam_sequences[active_idxs].clone()  # using clone() to avoid unnessacesy share
        model.reset_hooks()  # clean cache of forward in model or the hook records
        
        with torch.no_grad():
            # Get logits for active beams
            outputs = model(active_beam_sequences, return_type="logits")
            next_token_logits = outputs[:, -1, :] / temperature
            
            # Compute log softmax probabilities
            next_token_scores = torch.log_softmax(next_token_logits, dim=-1)
            
            # Add cumulative score for each active beam
            for i, idx in enumerate(active_idxs):
                next_token_scores[i] += beam_scores[idx]
            
            # Obtain top-k candidates for each beam
            vocab_size = next_token_scores.shape[-1]
            # Each beam generates 2*num_beams candidate expansions, for a total of len(active_idxs)*2*num_beams candidates
            next_scores, next_tokens = torch.topk(
                next_token_scores.view(-1), 
                min(len(active_idxs) * 2 * num_beams, len(active_idxs) * vocab_size)
            )
            
            # Calculate corresponding beam indices
            beam_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size
            
            # Construct new candidate beams
            candidates = []
            for score_idx, (beam_idx, token_id, score) in enumerate(zip(beam_indices, next_tokens, next_scores)):
                # Get the original beam index
                orig_beam_idx = active_idxs[beam_idx]
                
                # Create new sequence
                new_sequence = torch.cat([
                    beam_sequences[orig_beam_idx],
                    token_id.unsqueeze(0)
                ], dim=0)
                
                # Check if EOS is generated
                is_eos = (token_id == tokenizer.eos_token_id)
                
                candidates.append({
                    "sequence": new_sequence,
                    "score": score.item(),
                    "is_finished": is_eos
                })
            
        # Select the top num_beams candidates as new beams
        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)[:num_beams]
        
        # Update beam status
        new_beam_sequences = []
        new_beam_scores = []
        new_beam_finished = []
        
        for i, candidate in enumerate(candidates):
            new_beam_sequences.append(candidate["sequence"])
            new_beam_scores.append(candidate["score"])
            new_beam_finished.append(candidate["is_finished"])
        
        # Update beam tracking variables
        beam_sequences = torch.stack(new_beam_sequences)
        beam_scores = torch.tensor(new_beam_scores, device=device)
        beam_finished = new_beam_finished
    
    # Return the highest scoring sequence
    best_sequence = beam_sequences[0].unsqueeze(0)
    model.train()
    return best_sequence

###############################################
# 7. Improved Evaluation Functions
###############################################
def extract_answer(generated_text, prompt_eval):
    """
    Extract the answer from the generated text, prioritizing the first word after 'Answer:'
    """
    # Remove the prompt portion
    generated_text = generated_text.strip()
    prompt_eval = prompt_eval.strip()
    answer_part = generated_text[len(prompt_eval):].strip()
    
    # Attempt to extract content after 'Answer:' using regex
    match = re.search(r'Answer:\s*(\w+)', generated_text, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()
    
    # If no match is found, take the first word
    words = answer_part.split()
    if words:
        return words[0].strip('.,;:!?"\'').lower()
    
    return ""

def evaluate_answer(predicted, ground_truth):
    """
    Combine multiple evaluation strategies for more flexible answer correctness judgment
    """
    # Exact match (case-insensitive)
    exact_match = predicted.lower() == ground_truth.lower()
    
    # Fuzzy match (using fuzzywuzzy library)
    fuzzy_match = fuzz.ratio(predicted.lower(), ground_truth.lower()) > 75
    
    # Containment match
    contains_match = (ground_truth.lower() in predicted.lower()) or (predicted.lower() in ground_truth.lower())
    
    # Stemming match (simplified version)
    stem_match = predicted.lower().rstrip('s') == ground_truth.lower().rstrip('s')
    
    return exact_match or fuzzy_match or contains_match or stem_match

###############################################
# 8. Define Optimizer and Loss Function
###############################################
num_epochs = 25  # Increase training epochs
warmup_epochs = 5  # Warmup for first 5 epochs
initial_lr = 5e-6  # Lower initial learning rate
max_lr = 2e-5  # Maximum learning rate

# Use AdamW optimizer with weight decay to reduce overfitting
optimizer = optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=initial_lr,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8
)

# Use cosine annealing learning rate scheduler
scheduler = CosineAnnealingWarmRestarts(
    optimizer,
    T_0=5,  # Initial period
    T_mult=2,  # Multiplier for period length increase after each restart
    eta_min=1e-7  # Minimum learning rate
)

# Use label-smoothed cross entropy loss
label_smoothing = 0.1
loss_fn = nn.CrossEntropyLoss(
    ignore_index=tokenizer.pad_token_id,
    label_smoothing=label_smoothing
)

# Mixed precision training
scaler = GradScaler()

# Gradient accumulation steps
accumulation_steps = 4

###############################################
# 9. Improved Training Loop
###############################################
print("Starting training...")
# Add early stopping
best_val_accuracy = 0
patience = 5  # Allow up to 5 epochs without improvement in validation accuracy
patience_counter = 0
unfreeze_epoch = 10  # Unfreeze all layers at epoch 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # At unfreeze_epoch, unfreeze all layers
    if epoch == unfreeze_epoch:
        print(f"Epoch {epoch + 1}: Unfreezing all layers for full fine-tuning")
        for param in model.parameters():
            param.requires_grad = True
        
        # Redefine optimizer (increase learning rate)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=initial_lr,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Reset learning rate scheduler
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=5,
            T_mult=2,
            eta_min=1e-7
        )
    
    # Use tqdm to display progress bar
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch["input_ids"].to(device)  # [B, L]
        attention_mask = batch["attention_mask"].to(device)  # [B, L]
        labels = batch["labels"].to(device)  # [B, L]
        
        # Perform gradient update every accumulation_steps steps
        if batch_idx % accumulation_steps == 0:
            optimizer.zero_grad()
        
        # Use mixed precision training
        with autocast():
            outputs = model(input_ids, return_type="logits")  # Output logits: [B, L, vocab_size]
            
            # Calculate causal LM loss (shifted by one for predicting t+1)
            shift_logits = outputs[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_labels.view(-1))
        
        # Scale loss and accumulate gradients
        scaler.scale(loss / accumulation_steps).backward()
        
        # Perform gradient update every accumulation_steps steps
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            # Apply gradient clipping to prevent exploding gradients
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update parameters
            scaler.step(optimizer)
            scaler.update()
            
            # Update learning rate
            scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        current_lr = scheduler.get_last_lr()[0]
        progress_bar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{total_loss / num_batches:.4f}",
            'lr': f"{current_lr:.7f}"
        })
    
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    
    ###############################################
    # 10. Improved Evaluation Process
    ###############################################
    if (epoch + 1) % 2 == 0 or epoch == num_epochs - 1:  # Evaluate every 2 epochs, and in the final epoch
        model.eval()
        print("Starting evaluation...")
        correct = 0
        total = 0
        
        eval_progress = tqdm(eval_dataset, desc="Evaluating")
        for example in eval_progress:
            prompt_eval = example["prompt_eval"]
            ground_truth = example["answer"].strip().lower()
            
            # Tokenize the evaluation prompt
            inputs = tokenize_prompt(prompt_eval, padding="max_length", truncation=True)
            input_ids = inputs["input_ids"].to(device)
            
            with torch.no_grad():
                # Use a combined strategy of temperature sampling and beam search
                if random.random() < 0.5:  # 50% chance to use temperature sampling
                    generated_ids = temperature_sampling(
                        model=model,
                        input_ids=input_ids,
                        tokenizer=tokenizer,
                        max_new_tokens=5,
                        temperature=0.7,
                        top_k=50,
                        top_p=0.9
                    )
                else:  # 50% chance to use beam search
                    generated_ids = improved_beam_search(
                        model=model,
                        input_ids=input_ids,
                        tokenizer=tokenizer,
                        max_new_tokens=5,
                        num_beams=5,
                        temperature=0.7
                    )
                
                # Decode the generated text
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                
                # Extract the answer
                predicted_answer = extract_answer(generated_text, prompt_eval)
                
                # Evaluate answer correctness using multiple strategies
                if evaluate_answer(predicted_answer, ground_truth):
                    correct += 1
                total += 1
                
                # Update progress bar
                eval_progress.set_postfix({
                    'accuracy': f"{correct / total * 100:.2f}%"
                })
        
        accuracy = correct / total if total > 0 else 0
        print(f"Epoch {epoch + 1} Evaluation Accuracy: {accuracy * 100:.2f}% (on {total} examples)")
        
        # Early stopping check
        if accuracy > best_val_accuracy:
            best_val_accuracy = accuracy
            patience_counter = 0
            
            # Save the best model
            print(f"Found a better model with accuracy: {best_val_accuracy * 100:.2f}%")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'accuracy': best_val_accuracy,
                'loss': avg_loss,
            }, f'best_qa_model_acc{best_val_accuracy:.4f}.pt')
        else:
            patience_counter += 1
            print(f"Model performance did not improve, patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"Early stopping triggered! Best accuracy: {best_val_accuracy * 100:.2f}%")
                break

###############################################
# 11. Load Best Model and Perform Final Evaluation
###############################################
print("Loading best model for final evaluation...")
best_model_path = f'best_qa_model_acc{best_val_accuracy:.4f}.pt'
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

correct = 0
total = 0
detailed_results = []

for example in tqdm(eval_dataset, desc="Final Evaluation"):
    prompt_eval = example["prompt_eval"]
    ground_truth = example["answer"].strip().lower()
    
    # Tokenize the evaluation prompt
    inputs = tokenize_prompt(prompt_eval)
    input_ids = inputs["input_ids"].to(device)
    
    with torch.no_grad():
        # Use beam search for final evaluation
        generated_ids = improved_beam_search(
            model=model,
            input_ids=input_ids,
            tokenizer=tokenizer,
            max_new_tokens=5,
            num_beams=5,
            temperature=0.7
        )
        
        # Decode the generated text
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Extract the answer
        predicted_answer = extract_answer(generated_text, prompt_eval)
        
        # Evaluate answer correctness
        is_correct = evaluate_answer(predicted_answer, ground_truth)
        if is_correct:
            correct += 1
        
        # Record detailed results
        detailed_results.append({
            "question": example["question"],
            "context": example["context"],
            "ground_truth": ground_truth,
            "predicted": predicted_answer,
            "is_correct": is_correct
        })
        
    total += 1

final_accuracy = correct / total if total > 0 else 0
print(f"Final evaluation accuracy: {final_accuracy * 100:.2f}% (on {total} examples)")

###############################################
# 12. Example Generation: Given a New QA Prompt, Generate an Answer
###############################################
test_prompts = [
    "Answer the following question based on the given context.\nQuestion: What is the capital of France?\nContext: France is a European country famous for its art, culture, and cuisine. Paris is known as the City of Light.\nAnswer:",
    "Answer the following question based on the given context.\nQuestion: Who invented the telephone?\nContext: Alexander Graham Bell is credited with inventing the first practical telephone in 1876.\nAnswer:",
    "Answer the following question based on the given context.\nQuestion: What is the largest planet in our solar system?\nContext: Jupiter is the largest planet in our solar system. It is a gas giant with a mass more than 300 times that of Earth.\nAnswer:"
]

print("\nExample Generation Results:")
for prompt in test_prompts:
    inputs = tokenize_prompt(prompt)
    input_ids = inputs["input_ids"].to(device)
    
    with torch.no_grad():
        # Generate answer using beam search
        generated_ids = improved_beam_search(
            model=model,
            input_ids=input_ids,
            tokenizer=tokenizer,
            max_new_tokens=5,
            num_beams=5,
            temperature=0.7
        )
        
        # Decode the generated text
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # Extract the answer
        predicted_answer = extract_answer(generated_text, prompt)
        
        # Extract the question text without backslashes in the f-string expression
        question_text = prompt.split("Question: ")[1].split("\n")[0]
        print(f"Question: {question_text}")
        print(f"Predicted Answer: {predicted_answer}")
        print("---")

# Print the most common error types
print("\nError Analysis:")
errors = [result for result in detailed_results if not result["is_correct"]]
if errors:
    print(f"Total errors: {len(errors)}")
    # Display example error cases (first 5)
    print("Example error cases:")
    for i, error in enumerate(errors[:5]):
        print(f"Case {i+1}:")
        print(f"Question: {error['question']}")
        print(f"Correct Answer: {error['ground_truth']}")
        print(f"Predicted Answer: {error['predicted']}")
        print("---")
else:
    print("No error cases!")

print("Training and evaluation completed!")
