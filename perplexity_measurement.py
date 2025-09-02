"""
Empirical: Measuring Perplexity and Sampling Strategies with DistilGPT2
"""
import argparse
import random
import textwrap
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


# ---------------------------
# Utilities
# ---------------------------
def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


def tokenize(text: str, tokenizer, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize text and return both input_ids and attention_mask."""
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False, padding=False)
    return encoded.input_ids.to(device), encoded.attention_mask.to(device)


def gpt2_like_nll(model, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> Tuple[float, int]:
    """
    Compute total negative log-likelihood (base-e) and the number of predicted tokens
    for a GPT-style LM using next-token prediction.
    """
    # Create labels for next-token prediction - HuggingFace models handle the shifting internally
    labels = input_ids.clone()
    # We do not compute loss for the first token (no context), mask with -100
    labels[:, 0] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
        # outputs.loss is mean NLL per predicted token (in nats)
        mean_nll = float(outputs.loss)
        num_pred_tokens = (labels != -100).sum().item()
        total_nll = mean_nll * num_pred_tokens
    return total_nll, num_pred_tokens


# calculate perplexity from negative log-likelihood
def perplexity_from_nll(total_nll: float, num_tokens: int) -> float:
    if num_tokens <= 0:
        return float("nan")
    return float(np.exp(total_nll / num_tokens))


def shuffle_tokens(input_ids: torch.Tensor, seed: int) -> torch.Tensor:
    """
    Randomly permute the token order (destroying local coherence),
    keeping the exact same multiset of tokens.
    """
    rng = random.Random(seed)
    ids = input_ids[0].tolist()
    rng.shuffle(ids)
    return torch.tensor([ids], device=input_ids.device, dtype=input_ids.dtype)


def shuffle_words(text: str, tokenizer, device: torch.device, seed: int) -> torch.Tensor:
    """
    Shuffle at word level for more dramatic coherence destruction.
    This creates a more extreme contrast than token-level shuffling.
    """
    rng = random.Random(seed)
    words = text.split()
    rng.shuffle(words)
    shuffled_text = " ".join(words)
    return tokenize(shuffled_text, tokenizer, device)


def decode(ids: torch.Tensor, tokenizer) -> str:
    return tokenizer.decode(ids[0], skip_special_tokens=True)


def generate(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 500,
    temperature: float = 1.0,
    greedy: bool = False,
) -> Tuple[str, torch.Tensor]:
    """
    Generate continuation and return the text and the generated *new* token ids (excluding the prompt).
    """
    model.eval()
    input_ids, attention_mask = tokenize(prompt, tokenizer, device)
    if greedy or temperature == 0.0:
        # Deterministic decoding
        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    else:
        # Stochastic with temperature
        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=0,           # no top-k truncation (full softmax)
            top_p=1.0,        # no nucleus truncation
            pad_token_id=tokenizer.pad_token_id,
        )
    # Return only the new tokens (after the prompt)
    new_tokens = gen_ids[:, input_ids.size(1):]
    return decode(gen_ids, tokenizer), new_tokens


def distinct_n(ids: List[int], n: int) -> float:
    """
    Distinct-n: number of unique n-grams divided by total n-grams.
    """
    if len(ids) < n:
        return 0.0
    total = len(ids) - n + 1
    ngrams = set(tuple(ids[i:i+n]) for i in range(total))
    return len(ngrams) / total if total > 0 else 0.0


def repetition_rate(ids: List[int], window: int = 50) -> float:
    """
    Simple repetition metric: fraction of positions where the current token
    equals any of the previous 'window' tokens.
    """
    if not ids:
        return 0.0
    rep = 0
    for i in range(1, len(ids)):
        lo = max(0, i - window)
        if ids[i] in ids[lo:i]:
            rep += 1
    return rep / max(1, len(ids) - 1)


def avg_token_logprob(model, context_ids: torch.Tensor, cont_ids: torch.Tensor, context_attention_mask: torch.Tensor = None) -> float:
    """
    Compute average log-prob (base-e) assigned *by the model* to the generated continuation,
    conditioned on the full growing prefix (teacher-forced).
    """
    # Concatenate context + continuation; predict continuation tokens auto-regressively
    input_ids = torch.cat([context_ids, cont_ids], dim=1)
    labels = input_ids.clone()
    labels[:, :context_ids.size(1)] = -100  # ignore context positions
    # Don't mask the final token - include it in the average
    
    # Create attention mask for the full sequence
    if context_attention_mask is not None:
        cont_attention_mask = torch.ones_like(cont_ids)
        attention_mask = torch.cat([context_attention_mask, cont_attention_mask], dim=1)
    else:
        attention_mask = None

    with torch.no_grad():
        outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
        mean_nll = float(outputs.loss)
    return -mean_nll  # negative NLL = average log-prob per predicted token


@dataclass
class SampleStats:
    temperature: float
    greedy: bool
    text: str
    distinct1: float
    distinct2: float
    repetition: float
    avg_logprob: float
    tokens: int


def analyze_sample(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    temperature: float,
    greedy: bool,
    max_new_tokens: int,
) -> SampleStats:
    full_text, new_ids = generate(
        model, tokenizer, prompt, device, max_new_tokens=max_new_tokens, temperature=temperature, greedy=greedy
    )
    ids_list = new_ids[0].tolist()
    d1 = distinct_n(ids_list, 1)
    d2 = distinct_n(ids_list, 2)
    rep = repetition_rate(ids_list)
    context_ids, context_attention_mask = tokenize(prompt, tokenizer, device)
    avg_lp = avg_token_logprob(model, context_ids, new_ids, context_attention_mask)
    return SampleStats(
        temperature=temperature,
        greedy=greedy,
        text=full_text,
        distinct1=d1,
        distinct2=d2,
        repetition=rep,
        avg_logprob=avg_lp,
        tokens=len(ids_list),
    )


def wrap_print(title: str, content: str, width: int = 100):
    print("=" * width)
    print(title)
    print("-" * width)
    print("\n".join(textwrap.wrap(content, width=width)))
    print("=" * width + "\n")


# ---------------------------
# Main Experiment
# ---------------------------
def run_perplexity_experiment(model, tokenizer, device, paragraph: str, seed: int) -> Dict[str, float]:
    model.eval()
    # Original
    orig_ids, orig_attention_mask = tokenize(paragraph, tokenizer, device)
    orig_nll, orig_tokens = gpt2_like_nll(model, orig_ids, orig_attention_mask)
    orig_ppl = perplexity_from_nll(orig_nll, orig_tokens)

    # Token-shuffled (token-order destroyed)
    shuf_ids = shuffle_tokens(orig_ids, seed=seed)
    shuf_nll, shuf_tokens = gpt2_like_nll(model, shuf_ids, orig_attention_mask)
    shuf_ppl = perplexity_from_nll(shuf_nll, shuf_tokens)

    # Word-shuffled (more dramatic coherence destruction)
    word_shuf_ids, word_shuf_attention_mask = shuffle_words(paragraph, tokenizer, device, seed=seed)
    word_shuf_nll, word_shuf_tokens = gpt2_like_nll(model, word_shuf_ids, word_shuf_attention_mask)
    word_shuf_ppl = perplexity_from_nll(word_shuf_nll, word_shuf_tokens)

    return {
        "orig_tokens": orig_tokens,
        "orig_nll": orig_nll,
        "orig_ppl": orig_ppl,
        "shuf_tokens": shuf_tokens,
        "shuf_nll": shuf_nll,
        "shuf_ppl": shuf_ppl,
        "word_shuf_tokens": word_shuf_tokens,
        "word_shuf_nll": word_shuf_nll,
        "word_shuf_ppl": word_shuf_ppl,
    }


def run_sampling_experiment(
    model,
    tokenizer,
    device,
    prompt: str,
    temps: List[float],
    max_new_tokens: int,
) -> List[SampleStats]:
    stats: List[SampleStats] = []

    # Greedy (temperature 0 equivalent)
    stats.append(analyze_sample(model, tokenizer, prompt, device, temperature=0.0, greedy=True, max_new_tokens=max_new_tokens))

    # Temperature sampling
    for t in temps:
        # Skip 0 because we already did greedy for that case
        if t == 0.0:
            continue
        stats.append(analyze_sample(model, tokenizer, prompt, device, temperature=t, greedy=False, max_new_tokens=max_new_tokens))
    return stats


def print_ppl_summary(ppl_results: Dict[str, float], paragraph: str, shuffled_preview: str, word_shuffled_preview: str):
    width = 100
    wrap_print("Paragraph (Original)", paragraph, width)
    print(f"Original:     tokens={ppl_results['orig_tokens']}, total NLL={ppl_results['orig_nll']:.2f}, perplexity={ppl_results['orig_ppl']:.3f}")
    print(f"Token-shuf:   tokens={ppl_results['shuf_tokens']}, total NLL={ppl_results['shuf_nll']:.2f}, perplexity={ppl_results['shuf_ppl']:.3f}")
    print(f"Word-shuf:    tokens={ppl_results['word_shuf_tokens']}, total NLL={ppl_results['word_shuf_nll']:.2f}, perplexity={ppl_results['word_shuf_ppl']:.3f}\n")
    print("Comment:")
    print("-" * width)
    print(
        "Shuffling destroys local syntactic/semantic dependencies the model relies on for next-token prediction. "
        "Word-level shuffling creates more dramatic coherence destruction than token-level shuffling, resulting in "
        "progressively higher NLL and perplexity (worse modeling), while the original maintains lower perplexity "
        "due to coherent context."
    )
    print("=" * width + "\n")


def print_sampling_summary(samples: List[SampleStats], show_text: bool, tokenizer, prompt: str):
    width = 100
    # Tabular summary
    print("=" * width)
    print("Sampling Summary (higher Distinct-1/2 = more diverse; higher repetition = more loops; higher avg_logprob = more model-preferred)")
    print("-" * width)
    header = f"{'Decoding':<12} {'Temp':<6} {'Tokens':<6} {'Distinct-1':<10} {'Distinct-2':<10} {'Repetition':<11} {'Avg LogProb':<12}"
    print(header)
    print("-" * width)
    for s in samples:
        name = "greedy" if s.greedy else "sample"
        print(f"{name:<12} {s.temperature:<6.1f} {s.tokens:<6d} {s.distinct1:<10.3f} {s.distinct2:<10.3f} {s.repetition:<11.3f} {s.avg_logprob:<12.3f}")
    print("=" * width + "\n")

    if show_text:
        for s in samples:
            title = f"Decoding={'greedy' if s.greedy else 'sample'}  |  Temperature={s.temperature}"
            wrap_print(title, s.text, width)


def build_default_paragraph() -> str:
    # 3–5 continuous sentences (public-domain style).
    return (
        "The field of artificial intelligence has rapidly advanced over the last decade, transforming industries from healthcare to education. "
        "Researchers now develop models capable of generating coherent text, analyzing medical images, and even assisting in scientific discovery. "
        "Despite these successes, concerns remain about fairness, transparency, and the ethical use of such technologies. "
        "Addressing these challenges requires collaboration across disciplines, including computer science, law, and philosophy."
    )

def main():
    parser = argparse.ArgumentParser(description="Perplexity & Sampling experiment with DistilGPT2")
    parser.add_argument("--model_name", type=str, default="distilgpt2", help="HuggingFace model id")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")
    parser.add_argument("--max_new_tokens", type=int, default=500, help="Continuation length for generation")
    parser.add_argument("--show_text", action="store_true", help="Print full generated texts")
    parser.add_argument("--paragraph", type=str, default=None, help="Custom paragraph for perplexity test")
    args = parser.parse_args()

    # Seed everything
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = choose_device()
    print(f"[Info] Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # Some GPT-2 variants have no pad token by default; set to eos for generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    
    model.eval()

    # (a) Perplexity Analysis
    paragraph = args.paragraph if args.paragraph is not None else build_default_paragraph()
    ppl_results = run_perplexity_experiment(model, tokenizer, device, paragraph, seed=args.seed)
    # For a quick "shuffled preview", decode the shuffled ids first 40 tokens to visualize the effect
    orig_ids, _ = tokenize(paragraph, tokenizer, device)
    shuf_ids = shuffle_tokens(orig_ids, seed=args.seed)
    shuf_preview_text = decode(shuf_ids[:, : min(40, shuf_ids.size(1))], tokenizer)
    
    # Generate word-shuffled preview
    word_shuf_ids, _ = shuffle_words(paragraph, tokenizer, device, seed=args.seed)
    word_shuf_preview_text = decode(word_shuf_ids[:, : min(40, word_shuf_ids.size(1))], tokenizer)
    
    print_ppl_summary(ppl_results, paragraph, shuf_preview_text, word_shuf_preview_text)

    # (b) Sampling Comparison
    prompt = "Once upon a time"
    temps = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]
    samples = run_sampling_experiment(model, tokenizer, device, prompt, temps, max_new_tokens=args.max_new_tokens)
    print_sampling_summary(samples, show_text=args.show_text, tokenizer=tokenizer, prompt=prompt)

    # Final note for report
    print("\n=== Notes ===")
    print("* Greedy (T=0) tends to be locally probable but can be dull and repetitive.")
    print("* Moderate temperatures (e.g., T≈0.6–0.9) often balance coherence and diversity.")
    print("* Very high temperatures (e.g., T≥1.2) increase diversity but risk incoherence and repetition.")


if __name__ == "__main__":
    main()