import argparse
import math
import os
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Dict, List, Tuple
import tempfile

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM
from transformers import DataCollatorWithPadding, set_seed
from peft import LoraConfig, get_peft_model
import matplotlib.pyplot as plt
from tqdm import tqdm


# -------------------------------
# Utilities
# -------------------------------
def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_classifier_head_params(model: nn.Module) -> int:
    total = 0
    for name, p in model.named_parameters():
        if "classifier" in name and p.requires_grad:
            total += p.numel()
    return total


def plot_curves(train_acc: List[float], dev_acc: List[float], title: str, outfile: str):
    plt.figure(figsize=(5, 3.5), dpi=140)
    xs = np.arange(1, len(train_acc) + 1)
    plt.plot(xs, train_acc, marker="o", label="train acc")
    plt.plot(xs, dev_acc, marker="o", label="dev acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()


# -------------------------------
# Data
# -------------------------------
def load_text_classification(dataset_name: str, seed: int, tokenizer, max_len: int = 128):
    """
    Loads StrategyQA and creates train/dev/test splits.
    """
    if dataset_name == "strategyqa":
        # If a local dataset script named 'strategy-qa.py' is present, Datasets v3 will error.
        # Work around by loading from a clean temporary directory.
        needs_isolation = os.path.exists("strategy-qa.py")

        @contextmanager
        def _isolated_cwd():
            if not needs_isolation:
                yield
                return
            prev = os.getcwd()
            with tempfile.TemporaryDirectory() as tmpdir:
                os.chdir(tmpdir)
                try:
                    yield
                finally:
                    os.chdir(prev)

        with _isolated_cwd():
            # IMPORTANT for HF Datasets v3: allow remote dataset code execution
            ds = load_dataset("wics/strategy-qa", trust_remote_code=True)

        # StrategyQA repos vary; handle several layouts robustly.
        # Prefer existing splits; otherwise derive them from a single available split.
        if all(k in ds for k in ("train", "validation", "test")):
            base_train, dev_ds, test_ds = ds["train"], ds["validation"], ds["test"]
        elif "train" in ds and "validation" in ds:
            print("train and validation in ds")
            base_train, dev_ds = ds["train"], ds["validation"]
            # derive test from a slice of train to keep a held-out set
            tmp = base_train.train_test_split(test_size=0.2, seed=seed)
            base_train, test_ds = tmp["train"], tmp["test"]
        elif "train" in ds:
            print("train in ds")
            # only train provided → split into train/dev/test = 64/16/20
            tmp = ds["train"].train_test_split(test_size=0.2, seed=seed)   # 80% temp-train, 20% test
            temp_train, test_ds = tmp["train"], tmp["test"]
            split2 = temp_train.train_test_split(test_size=0.2, seed=seed) # 80→(64/16)
            base_train, dev_ds = split2["train"], split2["test"]
        elif "test" in ds:
            print("test in ds")
            # some mirrors only expose 'test' → reuse as pool, then split
            tmp = ds["test"].train_test_split(test_size=0.2, seed=seed)
            temp_train, test_ds = tmp["train"], tmp["test"]
            split2 = temp_train.train_test_split(test_size=0.2, seed=seed)
            base_train, dev_ds = split2["train"], split2["test"]
        else:
            raise RuntimeError("StrategyQA: unexpected split layout.")

        num_labels = 2
        text_col_candidates = ["question", "input", "text"]

        def _pick_text_col(example):
            for c in text_col_candidates:
                if c in example:
                    return c
            raise KeyError("No suitable text field found in StrategyQA example.")

        # Determine text column once from a sample
        sample = next(iter(base_train))
        text_col = _pick_text_col(sample)

        # Map labels → integer {0,1}
        def _label_map(ex):
            if "answer" in ex:
                ex["labels"] = 1 if bool(ex["answer"]) else 0
            elif "label" in ex:
                # some variants use 'label' already as 0/1
                ex["labels"] = int(ex["label"])
            else:
                raise KeyError("StrategyQA: no 'answer' or 'label' field present.")
            return ex

        train_ds = base_train.map(_label_map)
        dev_ds   = dev_ds.map(_label_map)
        test_ds  = test_ds.map(_label_map)
    else:
        raise ValueError("Unsupported dataset. Use --dataset strategyqa")

    def tokenize_fn(ex):
        enc = tokenizer(
            ex[text_col],
            truncation=True,
            padding=False,
            max_length=max_len,
        )
        # Preserve labels set by _label_map so the collator includes them
        if "labels" in ex:
            enc["labels"] = ex["labels"]
        return enc

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
    dev_ds = dev_ds.map(tokenize_fn, batched=True, remove_columns=dev_ds.column_names)
    test_ds = test_ds.map(tokenize_fn, batched=True, remove_columns=test_ds.column_names)

    collator = DataCollatorWithPadding(tokenizer)
    return (
        DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collator),
        DataLoader(dev_ds, batch_size=64, shuffle=False, collate_fn=collator),
        DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collator),
        num_labels,
    )


# -------------------------------
# (4.2) StrategyQA as prompted MLM (no classifier head)
# -------------------------------
def _pick_single_token_pair_ids(tokenizer: AutoTokenizer) -> Tuple[int, int]:
    """
    Prefer a verbalizer pair that is single-token for this tokenizer.
    Try ('yes','no'), then ('true','false'). Fallback uses the first token of each.
    """
    candidates = [("yes", "no"), ("true", "false")]
    for a, b in candidates:
        a_ids = tokenizer.encode(a, add_special_tokens=False)
        b_ids = tokenizer.encode(b, add_special_tokens=False)
        if len(a_ids) == 1 and len(b_ids) == 1:
            return a_ids[0], b_ids[0]
    # fallback: first subtoken
    a_ids = tokenizer.encode(candidates[0][0], add_special_tokens=False)
    b_ids = tokenizer.encode(candidates[0][1], add_special_tokens=False)
    return a_ids[0], b_ids[0]


def load_strategyqa_mlm_loaders(
    seed: int,
    tokenizer: AutoTokenizer,
    max_len: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    """
    Build train/dev/test DataLoaders for prompted MLM:
        Input:  "Question: {q} Answer: [MASK]"
        Labels: full-length tensor with -100 except mask position set to verbalizer id.
    Returns loaders and the (yes_id, no_id) verbalizer ids for evaluation.
    """
    # Reuse the same splitting logic as in load_text_classification
    # but do our own mapping/tokenization for MLM.
    ds = load_dataset("wics/strategy-qa", trust_remote_code=True)
    if all(k in ds for k in ("train", "validation", "test")):
        base_train, dev_ds, test_ds = ds["train"], ds["validation"], ds["test"]
    elif "train" in ds and "validation" in ds:
        tmp = ds["train"].train_test_split(test_size=0.2, seed=seed)
        base_train, test_ds = tmp["train"], tmp["test"]
        dev_ds = ds["validation"]
    elif "train" in ds:
        tmp = ds["train"].train_test_split(test_size=0.2, seed=seed)
        temp_train, test_ds = tmp["train"], tmp["test"]
        split2 = temp_train.train_test_split(test_size=0.2, seed=seed)
        base_train, dev_ds = split2["train"], split2["test"]
    elif "test" in ds:
        tmp = ds["test"].train_test_split(test_size=0.2, seed=seed)
        temp_train, test_ds = tmp["train"], tmp["test"]
        split2 = temp_train.train_test_split(test_size=0.2, seed=seed)
        base_train, dev_ds = split2["train"], split2["test"]
    else:
        raise RuntimeError("StrategyQA: unexpected split layout.")

    # find a question field
    text_col = None
    for c in ["question", "input", "text"]:
        if c in base_train.features:
            text_col = c
            break
    if text_col is None:
        raise KeyError("StrategyQA: no question/text field found.")

    def _label_map(ex):
        if "answer" in ex:
            ex["y"] = 1 if bool(ex["answer"]) else 0
        elif "label" in ex:
            ex["y"] = int(ex["label"])
        else:
            raise KeyError("StrategyQA: no 'answer' or 'label'.")
        return ex

    base_train = base_train.map(_label_map)
    dev_ds     = dev_ds.map(_label_map)
    test_ds    = test_ds.map(_label_map)

    yes_id, no_id = _pick_single_token_pair_ids(tokenizer)
    mask_tok = tokenizer.mask_token
    if mask_tok is None:
        raise ValueError("Tokenizer has no [MASK] token; ModernBERT should support masked LM.")
    mask_id = tokenizer.mask_token_id

    def _mlm_tokenize(ex):
        q = ex[text_col]
        prompt = f"Question: {q} Answer: {mask_tok}"
        enc = tokenizer(prompt, truncation=True, max_length=max_len)
        # build full labels: -100 everywhere except the single [MASK] pos
        labels = np.full_like(enc["input_ids"], fill_value=-100)
        # locate the single mask position
        try:
            mpos = enc["input_ids"].index(mask_id)
        except ValueError:
            # if truncated removed mask (rare), force-add at the end
            enc = tokenizer(prompt + f" {mask_tok}", truncation=True, max_length=max_len)
            mpos = enc["input_ids"].index(mask_id)
            labels = np.full_like(enc["input_ids"], fill_value=-100)
        labels[mpos] = yes_id if ex["y"] == 1 else no_id
        enc["labels"] = labels
        return enc

    train_mlm = base_train.map(_mlm_tokenize, remove_columns=base_train.column_names)
    dev_mlm   = dev_ds.map(_mlm_tokenize, remove_columns=dev_ds.column_names)
    test_mlm  = test_ds.map(_mlm_tokenize, remove_columns=test_ds.column_names)

    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(train_mlm, batch_size=32, shuffle=True, collate_fn=collator)
    dev_loader   = DataLoader(dev_mlm, batch_size=64, shuffle=False, collate_fn=collator)
    test_loader  = DataLoader(test_mlm, batch_size=64, shuffle=False, collate_fn=collator)
    return train_loader, dev_loader, test_loader, yes_id, no_id


# -------------------------------
# Training/Eval Loops
# -------------------------------
@dataclass
class TrainResult:
    train_curve: List[float]
    dev_curve: List[float]
    best_epoch: int
    best_dev_acc: float
    best_path: str
    trainable_params: int
    test_acc: float = float("nan")

# core training loop
def run_one_epoch(model, loader, device, optimizer=None, desc="Epoch"):
    is_train = optimizer is not None
    model.train(is_train)
    accs, losses = [], []
    loss_fn = nn.CrossEntropyLoss()
    
    # Add progress bar
    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = loss_fn(logits, labels)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        accs.append(accuracy_from_logits(logits.detach(), labels))
        
        # Update progress bar with current metrics
        current_loss = np.mean(losses)
        current_acc = np.mean(accs)
        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{current_acc:.4f}'
        })
    
    return float(np.mean(losses)), float(np.mean(accs))


# ----- MLM (prompted) training/eval for LoRA (no classifier head) -----
def run_one_epoch_mlm(
    model_mlm: nn.Module,
    loader: DataLoader,
    device: torch.device,
    yes_id: int,
    no_id: int,
    optimizer=None,
    desc="Epoch (MLM)",
):
    is_train = optimizer is not None
    model_mlm.train(is_train)
    accs, losses = [], []
    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)  # full-seq MLM labels

        outputs = model_mlm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # accuracy: choose between yes/no probs at the [MASK] position
        with torch.no_grad():
            logits = outputs.logits  # [B, T, V]
            # mask positions are where labels != -100
            mask_pos = (labels != -100).nonzero(as_tuple=False)
            # one mask per example assumed
            b_ix = mask_pos[:, 0]
            t_ix = mask_pos[:, 1]
            sel = logits[b_ix, t_ix, :]  # [B, V]
            pred_yes = sel[:, yes_id]
            pred_no = sel[:, no_id]
            preds = (pred_yes > pred_no).long()  # yes=1, no=0
            gold = (labels[mask_pos[:, 0], mask_pos[:, 1]] == yes_id).long()
            acc = (preds == gold).float().mean().item()
        losses.append(loss.item())
        accs.append(acc)
        pbar.set_postfix({'loss': f'{np.mean(losses):.4f}', 'acc': f'{np.mean(accs):.4f}'})
    return float(np.mean(losses)), float(np.mean(accs))


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    out_path: str,
) -> TrainResult:
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    train_curve, dev_curve = [], []
    best_dev, best_epoch = -1.0, -1
    best_path = out_path

    for ep in range(1, epochs + 1):
        _, train_acc = run_one_epoch(model, train_loader, device, optimizer=optimizer, desc=f"Epoch {ep}/{epochs} - Train")
        _, dev_acc = run_one_epoch(model, dev_loader, device, optimizer=None, desc=f"Epoch {ep}/{epochs} - Val")
        train_curve.append(train_acc)
        dev_curve.append(dev_acc)
        
        print(f"Epoch {ep}: Train Acc = {train_acc:.4f}, Dev Acc = {dev_acc:.4f}")
        
        if dev_acc > best_dev:
            best_dev = dev_acc
            best_epoch = ep
            torch.save(model.state_dict(), best_path)
            print(f"  -> New best dev accuracy! Saving checkpoint.")

    return TrainResult(
        train_curve=train_curve,
        dev_curve=dev_curve,
        best_epoch=best_epoch,
        best_dev_acc=best_dev,
        best_path=best_path,
        trainable_params=count_trainable_params(model),
    )


def train_model_mlm(
    model_mlm: nn.Module,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    device: torch.device,
    yes_id: int,
    no_id: int,
    epochs: int,
    lr: float,
    out_path: str,
) -> TrainResult:
    optimizer = torch.optim.AdamW([p for p in model_mlm.parameters() if p.requires_grad], lr=lr)
    train_curve, dev_curve = [], []
    best_dev, best_epoch = -1.0, -1
    best_path = out_path
    for ep in range(1, epochs + 1):
        _, train_acc = run_one_epoch_mlm(model_mlm, train_loader, device, yes_id, no_id, optimizer=optimizer, desc=f"Epoch {ep}/{epochs} - Train (MLM)")
        _, dev_acc   = run_one_epoch_mlm(model_mlm, dev_loader,   device, yes_id, no_id, optimizer=None,       desc=f"Epoch {ep}/{epochs} - Val (MLM)")
        train_curve.append(train_acc)
        dev_curve.append(dev_acc)
        print(f"Epoch {ep}: Train Acc = {train_acc:.4f}, Dev Acc = {dev_acc:.4f}")
        if dev_acc > best_dev:
            best_dev = dev_acc
            best_epoch = ep
            torch.save(model_mlm.state_dict(), best_path)
            print(f"  -> New best dev accuracy! Saving checkpoint.")
    return TrainResult(
        train_curve=train_curve,
        dev_curve=dev_curve,
        best_epoch=best_epoch,
        best_dev_acc=best_dev,
        best_path=best_path,
        trainable_params=count_trainable_params(model_mlm),
    )


def evaluate_best(model: nn.Module, ckpt_path: str, test_loader: DataLoader, device: torch.device) -> float:
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    _, test_acc = run_one_epoch(model, test_loader, device, optimizer=None, desc="Evaluating on test set")
    return test_acc


def evaluate_best_mlm(
    model_mlm: nn.Module,
    ckpt_path: str,
    test_loader: DataLoader,
    device: torch.device,
    yes_id: int,
    no_id: int,
) -> float:
    state = torch.load(ckpt_path, map_location=device)
    model_mlm.load_state_dict(state)
    _, test_acc = run_one_epoch_mlm(model_mlm, test_loader, device, yes_id, no_id, optimizer=None, desc="Evaluating on test set (MLM)")
    return test_acc


# -------------------------------
# (4.1) HEAD-ONLY setup
# -------------------------------
def build_head_only_model(model_name: str, num_labels: int, device: torch.device):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels).to(device)
    # Freeze everything
    for p in model.parameters():
        p.requires_grad = False
    # Unfreeze only the classification head
    for name, p in model.named_parameters():
        if "classifier" in name or name.endswith("score.weight") or name.endswith("score.bias"):
            p.requires_grad = True
            print(f"Unfreezing {name}")
    return model


# -------------------------------
# (4.2) LoRA setup with param budget matching
# -------------------------------
def list_linear_modules_for_lora(model: nn.Module) -> List[Tuple[str, nn.Linear]]:
    """Return all Linear modules except anything in the classification head."""
    linear_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "classifier" not in name:
            linear_names.append((name, module))
    return linear_names


def pick_lora_targets_close_to_budget(model: nn.Module, budget_params: int) -> Tuple[List[str], int]:
    """
    Choose a small subset of Linear modules and a rank r such that:
        trainable_params_lora = 2 * r * sum_i (in_i + out_i)  is close to 'budget_params'
    We keep it simple: try r=1, and pick the single Linear whose (2*(in+out)) is closest to budget.
    This works well on AG News with ModernBERT (head params ~= 3k).
    Returns (target_module_names, chosen_rank).
    """
    linear_list = list_linear_modules_for_lora(model)
    if not linear_list:
        return [], 1

    # Try r=1 with a single module closest to budget
    best_name, best_delta, best_cost = None, float("inf"), None
    for name, lin in linear_list:
        in_f, out_f = lin.in_features, lin.out_features
        cost = 2 * (in_f + out_f)  # params added when r=1
        delta = abs(cost - budget_params)
        if delta < best_delta:
            best_name, best_delta, best_cost = name, delta, cost

    chosen = [best_name] if best_name is not None else []
    r = 1
    print(f"Chosen targets: {chosen}, r: {r}")
    print(f"Best delta: {best_delta}, best cost: {best_cost}")
    return chosen, r


def build_lora_mlm_model_with_budget(model_name: str, device: torch.device, head_budget: int):
    """
    Build AutoModelForMaskedLM, freeze all weights including lm_head, and inject LoRA
    into a small subset of Linear modules to match the head-only parameter budget.
    """
    base = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    for _, p in base.named_parameters():
        p.requires_grad = False
    targets, r = pick_lora_targets_close_to_budget(base, budget_params=head_budget)
    lcfg = LoraConfig(
        r=r,
        lora_alpha=2 * r,
        lora_dropout=0.0,
        bias="none",
        target_modules=targets,
        task_type="SEQ_CLS",
    )
    lora_model = get_peft_model(base, lcfg)
    # Ensure lm_head stays frozen
    for name, p in lora_model.named_parameters():
        if "lm_head" in name:
            p.requires_grad = False
    return lora_model, targets, r


# -------------------------------
# Results Reporter
# -------------------------------
def print_results(
    head_params: int,
    head_classifier_cost: int,
    head_best_dev: float,
    head_test: float,
    lora_params: int,
    lora_best_dev: float,
    lora_test: float,
):
    """
    Prints concise classification results.
    """
    print("\n" + "="*60)
    print("CLASSIFICATION RESULTS")
    print("="*60)
    print(f"Head-only:")
    print(f"  Trainable Parameters: {head_params:,}")
    print(f"  Head tuning cost:    {head_classifier_cost:,}")
    print(f"  Best Validation Acc:  {head_best_dev:.4f}")
    print(f"  Test Accuracy:        {head_test:.4f}")
    print()
    print(f"LoRA:")
    print(f"  Trainable Parameters: {lora_params:,}")
    print(f"  Best Validation Acc:  {lora_best_dev:.4f}")
    print(f"  Test Accuracy:        {lora_test:.4f}")
    print("="*60)


# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser(description="ModernBERT head-only vs LoRA classification (StrategyQA)")
    parser.add_argument("--model", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--dataset", type=str, default="strategyqa", choices=["strategyqa"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr_head", type=float, default=5e-3)
    parser.add_argument("--lr_lora", type=float, default=2e-3)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    device = choose_device()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    train_loader, dev_loader, test_loader, num_labels = load_text_classification(
        args.dataset, args.seed, tokenizer, max_len=args.max_len
    )
    # Build parallel MLM dataloaders for LoRA (no classifier head)
    mlm_train_loader, mlm_dev_loader, mlm_test_loader, yes_id, no_id = load_strategyqa_mlm_loaders(
        seed=args.seed, tokenizer=tokenizer, max_len=args.max_len
    )

    # (4.1) Head-only training
    print("\n" + "="*60)
    print("TRAINING HEAD-ONLY MODEL")
    print("="*60)
    head_model = build_head_only_model(args.model, num_labels, device)
    head_res = train_model(
        head_model, train_loader, dev_loader, device, epochs=args.epochs, lr=args.lr_head, out_path="head_best.pt"
    )
    head_test_acc = evaluate_best(head_model, head_res.best_path, test_loader, device)
    head_res.test_acc = head_test_acc
    plot_curves(head_res.train_curve, head_res.dev_curve, "Head-only (StrategyQA): accuracy", "plots/head_acc.png")

    # (4.2) LoRA training with parameter budget ~ head-only, no classifier head (prompted MLM)
    print("\n" + "="*60)
    print("TRAINING LoRA MODEL")
    print("="*60)
    head_budget = head_res.trainable_params
    lora_model, targets, r = build_lora_mlm_model_with_budget(args.model, device, head_budget)
    lora_res = train_model_mlm(
        lora_model, mlm_train_loader, mlm_dev_loader, device, yes_id, no_id,
        epochs=args.epochs, lr=args.lr_lora, out_path="lora_best.pt"
    )
    lora_test_acc = evaluate_best_mlm(lora_model, lora_res.best_path, mlm_test_loader, device, yes_id, no_id)
    lora_res.test_acc = lora_test_acc
    plot_curves(lora_res.train_curve, lora_res.dev_curve, f"LoRA-MLM (StrategyQA) (targets={targets}, r={r}): accuracy", "plots/lora_acc.png")

    # Print results
    head_classifier_cost = count_classifier_head_params(head_model)
    print_results(
        head_params=head_res.trainable_params,
        head_classifier_cost=head_classifier_cost,
        head_best_dev=head_res.best_dev_acc,
        head_test=head_res.test_acc,
        lora_params=lora_res.trainable_params,
        lora_best_dev=lora_res.best_dev_acc,
        lora_test=lora_res.test_acc,
    )


if __name__ == "__main__":
    main()