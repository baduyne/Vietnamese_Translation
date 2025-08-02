from customzied_model import load_customized_model
from datasets import load_dataset
from transformers import Seq2SeqTrainingArguments , DataCollatorForSeq2Seq,Seq2SeqTrainer
from evaluate import load
import numpy as np
import torch
from torch.optim import AdamW

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from transformers import get_scheduler
from tqdm import tqdm

train_path = "data/train.csv"
valid_path = "data/test.csv"
model, tokenizer = load_customized_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"device:{model.device}")
model.train()

sacrebleu = load("sacrebleu")


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # decode predictions
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # sacrebleu expects a list of list of references
    decoded_labels = [[label] for label in decoded_labels]

    result = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}


def preprocess_data(tokenizer, train_path, valid_path):
    dataset = load_dataset("csv", data_files={
        "train": train_path,
        "valid": valid_path
    })

    def preprocess_function(examples):
        model_inputs = tokenizer(examples["English"], max_length=128, truncation=True, padding="max_length")

        labels = tokenizer(examples["Vietnamese"], max_length=128, truncation=True, padding="max_length")

        model_inputs["labels"] = [
            [(token if token != tokenizer.pad_token_id else -100) for token in seq]
            for seq in labels["input_ids"]
        ]

        return model_inputs

    tokenized = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    return tokenized

tokenized = preprocess_data(tokenizer= tokenizer, train_path = train_path, valid_path = valid_path)


# Criterion: seq2seq thường dùng CrossEntropyLoss, ignore padding index
criterion = CrossEntropyLoss(ignore_index=-100)

# Optimizer
optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

# Scheduler (linear warmup + decay)
num_epochs = 30
train_batch_size = 8
eval_batch_size = 8

train_dataset = tokenized["train"]
eval_dataset = tokenized["valid"]

data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )


train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, collate_fn=data_collator)
eval_loader = DataLoader(eval_dataset, batch_size=eval_batch_size, shuffle=False, collate_fn=data_collator)

num_training_steps = num_epochs * len(train_loader)
warmup_steps = int(0.05 * num_training_steps)

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=num_training_steps
)

print(device)
# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for batch in progress_bar:
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(**batch)
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)
        labels = batch["labels"]


        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=total_loss/(progress_bar.n+1))


    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            labels = batch["labels"]

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics((np.array(all_preds), np.array(all_labels)))
    print(f"Epoch {epoch+1} evaluation BLEU: {metrics['bleu']}")

    if metrics['bleu'] > best_bleu:
        best_bleu = metrics['bleu']
        print(f"New best BLEU: {best_bleu:.4f}, saving model...")
        model.save_pretrained("best_model")
        tokenizer.save_pretrained("best_model")
