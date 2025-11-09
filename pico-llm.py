# starter code by matus & o1-pro
import argparse
import time
import random
import math
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# We do not import numpy or scikit-learn, so we implement a naive k-means in pure PyTorch.
# If you prefer scikit-learn, you can adapt the code.

from datasets import load_dataset
import tiktoken

################################################################################
# 1. Command-line arg parsing
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(description="Train multiple k-gram or sequence-based models on TinyStories and/or custom text files.")
    parser.add_argument("--input_files", nargs="*", default=None,
                        help="Optional list of text files to mix in as data sources. Each line is one example (up to block_size).")
    parser.add_argument("--tinystories_weight", type=float, default=0.5,
                        help="Probability of sampling from TinyStories if present. Default=0.5. (set to 0.0 to skip TinyStories).")
    parser.add_argument("--max_steps_per_epoch", type=int, default=None,
                        help="If set, each epoch ends after this many steps (for quick tests).")
    parser.add_argument("--num_inner_mlp_layers", type=int, default=1,
                        help="Number of (Linear->SiLU) blocks inside the k-gram MLP. Default=1.")
    parser.add_argument("--monosemantic_enabled", action="store_true",
                        help="(DISABLED BY DEFAULT) If set, run the monosemantic analysis.")
    parser.set_defaults(monosemantic_enabled=False)  # disable by default

    # Additional hyperparams to mitigate slow k-gram
    parser.add_argument("--kgram_k", type=int, default=3,
                        help="Sliding window size for k-gram MLP. Smaller can reduce memory usage. Default=3.")
    parser.add_argument("--kgram_chunk_size", type=int, default=1,
                        help="Process k-gram timesteps in micro-batches. Default=1.")

    parser.add_argument("--block_size", type=int, default=1024,
                        help="Maximum sequence length for each example. Default=1024.")
    parser.add_argument("--activation", type=str, default="silu",
                        help="Activation layer to use for MLP model. Default=silu, supported values : [relu,silu,sigmoid, gelu].")

    # New arguments:
    parser.add_argument("--embed_size", type=int, default=1024,
                        help="Dimension of the embedding layer for LSTM, MLP, etc. Default=1024.")
    parser.add_argument("--prompt", type=str, default="Once upon a",
                        help="Prompt used for generation. Default='Once upon a'.")

    # Newly added device argument:
    parser.add_argument("--device_id", type=str, default="cuda:0",
                        help="Torch device identifier (default='cuda:0'). If CUDA is unavailable, fallback to 'cpu'.")
    

    # Transformer hyperparameters
    parser.add_argument("--transformer_d_model", type=int, default=None,
                        help="Transformer model dimension (d_model). Defaults to --embed_size if not set.")
    parser.add_argument("--transformer_n_heads", type=int, default=8,
                        help="Number of attention heads in Transformer. Default=8.")
    parser.add_argument("--transformer_n_blocks", type=int, default=4,
                        help="Number of Transformer blocks (layers). Default=4.")
    parser.add_argument("--transformer_mlp_ratio", type=float, default=4.0,
                        help="Transformer MLP expansion ratio (inner = ratio * d_model). Default=4.0.")
    parser.add_argument("--transformer_max_seq_len", type=int, default=0,
                        help="Max sequence length for positional embeddings (0 => use --block_size).")

    # KV-cache flag (inference optimization for Transformer)
    parser.add_argument("--use_kv_cache", action="store_true",
                        help="If set, use incremental generation with key/value cache for Transformer (faster decoding).")

    # Buffered CSV logging and model saving options
    parser.add_argument("--log_csv", type=str, default="",
                        help="If set to a file path, write training losses in buffered batches.")
    parser.add_argument("--log_flush_steps", type=int, default=100,
                        help="Flush loss buffer to CSV every N steps (default=100).")
    parser.add_argument("--save_model_dir", type=str, default="",
                        help="Directory to save trained model weights + meta JSON (created if missing).")
    parser.add_argument("--save_model_name", type=str, default="transformer",
                        help="Base name for saved model files (default=transformer).")
    parser.add_argument("--depth_list", type=str, default="",
                        help="Comma-separated list of transformer block counts to train sequentially (e.g. 2,6,10). Overrides --transformer_n_blocks for multi-run.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")

    args = parser.parse_args()
    return args


##############################################f##################################
# 2. Data handling: entire sequences up to block_size => (seq_len, batch)
################################################################################

class MixedSequenceDataset(torch.utils.data.Dataset):
    """
    We store two lists of entire token sequences:
      - tinystories_seqs
      - other_seqs
    Each sequence is length <= block_size.

    During __getitem__, we randomly pick from one list or the other with probability p_tiny.
    Return that entire sequence as a 1D LongTensor.
    """
    def __init__(self, tinystories_seqs, other_seqs, p_tiny: float):
        super().__init__()
        self.tinystories_seqs = tinystories_seqs
        self.other_seqs = other_seqs
        self.p_tiny = p_tiny

        self.has_tinystories = (len(self.tinystories_seqs) > 0)
        self.has_other = (len(self.other_seqs) > 0)

        self.total_length = len(self.tinystories_seqs) + len(self.other_seqs)
        if self.total_length == 0:
            raise ValueError("No data found! Both TinyStories and other sets are empty.")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        r = random.random()
        if self.has_tinystories and self.has_other:
            if r < self.p_tiny:
                i = random.randint(0, len(self.tinystories_seqs) - 1)
                seq = self.tinystories_seqs[i]
            else:
                i = random.randint(0, len(self.other_seqs) - 1)
                seq = self.other_seqs[i]
        elif self.has_tinystories:
            i = random.randint(0, len(self.tinystories_seqs) - 1)
            seq = self.tinystories_seqs[i]
        else:
            i = random.randint(0, len(self.other_seqs) - 1)
            seq = self.other_seqs[i]

        return torch.tensor(seq, dtype=torch.long)


def seq_collate_fn(batch):
    """
    batch: list of 1D LongTensors of various lengths [<= block_size].
    1) find max length
    2) pad with zeros
    3) shape => (max_len, batch_size)
    """
    max_len = max(len(seq) for seq in batch)
    batch_size = len(batch)

    padded = torch.zeros(max_len, batch_size, dtype=torch.long)
    for i, seq in enumerate(batch):
        seq_len = seq.size(0)
        padded[:seq_len, i] = seq

    return padded


################################################################################
# 3. K-gram MLP in a sequence-to-sequence approach
################################################################################

def compute_next_token_loss(logits, tokens):
    """
    logits: (seq_len, batch, vocab_size)
    tokens: (seq_len, batch)
    Next-token prediction => we shift target by 1.
    """
    seq_len, batch_size, vocab_size = logits.shape
    if seq_len < 2:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    preds = logits[:-1, :, :]  # (seq_len-1, batch, vocab_size)
    gold = tokens[1:, :]       # (seq_len-1, batch)

    preds = preds.reshape(-1, vocab_size)
    gold = gold.reshape(-1)
    return F.cross_entropy(preds, gold)


class KGramMLPSeqModel(nn.Module):
    """
    For each position t in [0..seq_len-1], gather the last k tokens => one-hot => MLP => logits.
    Return (seq_len, batch, vocab_size).

    Potentially very large memory usage for big vocab or seq_len. chunk_size helps mitigate overhead.
    """

    def __init__(self, vocab_size, k=3, embed_size=1024, num_inner_layers=1, chunk_size=1, activation=nn.SiLU):
        super().__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_inner_layers = num_inner_layers
        self.chunk_size = chunk_size
        # fill in
        self.activation = activation
        self.net = nn.Sequential(
            nn.Linear(self.k * self.vocab_size, self.embed_size),
            self.activation(),
            *[
                layer for _ in range(self.num_inner_layers) for layer in (
                    nn.Linear(self.embed_size, self.embed_size),
                    self.activation()
                )],
                nn.Linear(self.embed_size, self.vocab_size))

    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch)
        return: (seq_len, batch, vocab_size)
        We'll do a loop over time steps. chunk_size can reduce overhead.
        """
        seq_len, batch_size = tokens_seq.shape
        outputs = []

        start = 0
        while start < seq_len:
            end = min(start + self.chunk_size, seq_len)
            block_outputs = []
            for t in range(start, end):
                batch_logits = []
                for b in range(batch_size):
                    if t < self.k:
                        needed = self.k - t
                        context_ids = [0]*needed + tokens_seq[:t, b].tolist()
                    else:
                        context_ids = tokens_seq[t-self.k:t, b].tolist()

                    context_oh = F.one_hot(
                        torch.tensor(context_ids, dtype=torch.long, device=tokens_seq.device),
                        num_classes=self.vocab_size
                    )
                    context_flat = context_oh.flatten().float().unsqueeze(0)
                    logits_b = self.net(context_flat)  # (1, vocab_size)
                    batch_logits.append(logits_b)
                block_outputs.append(torch.cat(batch_logits, dim=0).unsqueeze(0))  # (1, batch, vocab_size)

            block_outputs = torch.cat(block_outputs, dim=0)  # (chunk_size, batch, vocab_size)
            outputs.append(block_outputs)
            start = end

        outputs = torch.cat(outputs, dim=0)  # (seq_len, batch, vocab_size)
        return outputs


################################################################################
# 4. LSTM-based seq2seq
################################################################################

class LSTMSeqModel(nn.Module):
    def __init__(self, vocab_size, embed_size=1024, hidden_size=1024):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=False)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, tokens_seq):
        """
        tokens_seq: (seq_len, batch)
        => (seq_len, batch, vocab_size)
        """
        emb = self.embedding(tokens_seq)   # (seq_len, batch, embed)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(emb)           # (seq_len, batch, hidden)
        logits = self.linear(out)         # (seq_len, batch, vocab_size)
        return logits


################################################################################
# 5. Our "stub" Transformer with KV-cache 
#    Very slow Python loop for training. Multi-head sums head outputs.
################################################################################
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    y = weight * (x / sqrt(mean(x^2) + eps)) + bias  (bias optional)
    We keep bias for flexibility although many implementations omit it.
    """
    def __init__(self, dim: int, eps: float = 1e-6, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.use_bias = bias
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        y = (x / rms) * self.weight
        if self.use_bias:
            y = y + self.bias
        return y


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional KV cache.
    Input x: (T, B, C) -> Output: (T, B, C)
    If past_kv is provided, it should be a tuple (k_cache, v_cache) of shape (B,H,T_prev,D).
    Returns (out, new_kv) when caching is used; otherwise returns out.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, t: torch.Tensor, T: int, B: int) -> torch.Tensor:
        return t.view(T, B, self.n_heads, self.head_dim).permute(1, 2, 0, 3)  # (B,H,T,D)

    def forward(self, x: torch.Tensor, past_kv=None):
        T, B, C = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self._split_heads(q, T, B)
        k = self._split_heads(k, T, B)
        v = self._split_heads(v, T, B)

        if past_kv is not None:
            k_cache, v_cache = past_kv
            if k_cache is not None:
                k = torch.cat([k_cache, k], dim=2)
                v = torch.cat([v_cache, v], dim=2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B,H,T,T_total)
        # Apply causal mask only when not using cache and sequence length > 1
        if past_kv is None and T > 1:
            # Here T_total == T
            causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            attn_scores = attn_scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        ctx = torch.matmul(attn_probs, v)  # (B,H,T,D)
        ctx = ctx.permute(2, 0, 1, 3).contiguous().view(T, B, C)
        out = self.o_proj(ctx)

        if past_kv is not None:
            return out, (k, v)
        else:
            return out


class TransformerBlock(nn.Module):
    """Single decoder block: RMSNorm -> CausalAttn (residual) -> RMSNorm -> MLP (residual)."""
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.mlp_norm = RMSNorm(d_model)
        inner = int(mlp_ratio * d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, inner, bias=False),
            nn.SiLU(),
            nn.Linear(inner, d_model, bias=False),
        )
    def forward(self, x: torch.Tensor, past_kv=None):
        if past_kv is None:
            x = x + self.attn(self.attn_norm(x))
        else:
            attn_out, new_kv = self.attn(self.attn_norm(x), past_kv=past_kv)
            x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        if past_kv is not None:
            return x, new_kv
        return x


class TransformerModel(nn.Module):
    """Decoder-only causal Transformer producing logits for next-token prediction.

    Forward input: tokens_seq (T, B) LongTensor
    Output: logits (T, B, vocab_size)
    """
    def __init__(self,
                 vocab_size: int = 50257,
                 d_model: int = 512,
                 n_heads: int = 8,
                 n_blocks: int = 6,
                 max_seq_len: int = 2048,
                 mlp_ratio: float = 4.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_ratio) for _ in range(n_blocks)
        ])
        self.final_norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying (optional; improves parameter efficiency)
        self.lm_head.weight = self.token_embed.weight

    def forward(self, tokens_seq: torch.Tensor, kv_cache=None):
        """If kv_cache is provided (list of (k,v)), we perform incremental forward for the new tokens.
        tokens_seq may be (T,B) full sequence (training) or (1,B) single step (inference).
        kv_cache: list of tuples per block or None.
        Returns logits and updated kv_cache when caching.
        """
        T, B = tokens_seq.shape
        if T > self.max_seq_len:
            tokens_seq = tokens_seq[-self.max_seq_len:]
            T = tokens_seq.shape[0]
        # Compute absolute positions; when using cache, continue from cached length
        if kv_cache is not None and len(kv_cache) > 0 and kv_cache[0][0] is not None:
            cached_len = kv_cache[0][0].size(2)
        else:
            cached_len = 0
        pos = torch.arange(cached_len, cached_len + T, device=tokens_seq.device)
        x = self.token_embed(tokens_seq) + self.pos_embed(pos).unsqueeze(1)
        new_cache = [] if kv_cache is not None else None
        if kv_cache is None:
            for blk in self.blocks:
                x = blk(x)
        else:
            # Incremental: assume tokens_seq is last token(s); pass cache per block.
            for blk, past in zip(self.blocks, kv_cache):
                x, updated = blk(x, past_kv=past)
                new_cache.append(updated)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        if kv_cache is not None:
            return logits, new_cache
        return logits


################################################################################
# 6. K-Means Monosemantic (DISABLED by default)
################################################################################


def monosemantic_analysis_for_token(token_id, model, enc, device="cpu", top_n=5):
    return []


################################################################################
# 7. Single code path for text generation
################################################################################

def nucleus_sampling(logits, p=0.95):
    # Convert logits to probabilities
    prob_dist = torch.softmax(logits, dim=-1)

    # Sort probabilities in descending order and get indices
    sorted_probs, sorted_indices = torch.sort(prob_dist, descending=True)

    # Compute cumulative sum
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find the cutoff index where cumulative probability exceeds p
    # We want to include all tokens up to where cumsum first exceeds p
    cutoff_mask = cumsum_probs <= p

    # Always include at least the first token (highest probability)
    # This handles edge case where first token alone has prob > p
    cutoff_mask[0] = True

    # Zero out probabilities beyond the nucleus
    filtered_probs = sorted_probs.clone()
    filtered_probs[~cutoff_mask] = 0.0

    # Renormalize the remaining probabilities
    filtered_probs = filtered_probs / filtered_probs.sum()

    # Sample from the filtered distribution
    sampled_index = torch.multinomial(filtered_probs, num_samples=1).item()

    # Map back to original token index
    chosen_token = sorted_indices[sampled_index].item()

    return chosen_token


def generate_text(model, enc, init_text, max_new_tokens=20, device="cpu",
                  top_p=None,
                  monosemantic_info=None,
                  do_monosemantic=False,
                  use_kv_cache=False):
    """
    A single code path for all models:
      - We keep a growing list 'context_tokens'.
      - At each step, we feed the entire context as (seq_len,1) to model(...).
      - We get model(...)->(seq_len,1,vocab_size). We take the final step's logits => logits[-1,0,:].
      - We pick next token (greedy or top-p), append to context_tokens.
      - Optionally do monosemantic analysis on that newly generated token.
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        context_tokens = enc.encode(init_text)
        annotation_list = []

        kv_cache = None

        # Prime KV cache step-by-step over the initial prompt for strict causality
        if use_kv_cache and hasattr(model, 'blocks') and len(context_tokens) > 0:
            kv_cache = [(None, None) for _ in range(len(model.blocks))]
            for tid in context_tokens:
                tok = torch.tensor([tid], dtype=torch.long, device=device).unsqueeze(1)  # (1,1)
                _, kv_cache = model(tok, kv_cache=kv_cache)

        for step_i in range(max_new_tokens):
            if use_kv_cache and hasattr(model, 'blocks'):
                # Use only the last token and advance cache
                if len(context_tokens) == 0:
                    # Fallback: no context, create a space token as a starter
                    last_id = enc.encode(" ")[-1]
                else:
                    last_id = context_tokens[-1]
                last_token = torch.tensor([last_id], dtype=torch.long, device=device).unsqueeze(1)  # (1,1)
                logits_seq, kv_cache = model(last_token, kv_cache=kv_cache if kv_cache is not None else [(None, None) for _ in range(len(model.blocks))])
                next_logits = logits_seq[-1, 0, :]
            else:
                # Fallback: full context each step (works for all models)
                seq_tensor = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(1)
                logits_seq = model(seq_tensor)              # (seq_len,1,vocab_size)
                next_logits = logits_seq[-1, 0, :]         # shape (vocab_size,)

            if top_p is None:
                # greedy
                chosen_token = torch.argmax(next_logits).item()
            else:
                chosen_token = nucleus_sampling(next_logits, p=top_p)

            context_tokens.append(chosen_token)

            if do_monosemantic and monosemantic_info is not None:
                neighbors = monosemantic_analysis_for_token(
                    chosen_token, model, monosemantic_info, enc, device=device, top_n=5
                )
                annotation_list.append((chosen_token, neighbors))
            else:
                annotation_list.append((chosen_token, []))

    model.train(was_training)

    final_text = enc.decode(context_tokens)
    prefix_text = enc.decode(context_tokens[:-max_new_tokens])
    annotated_strs = [prefix_text]
    for (tid, neighs) in annotation_list:
        token_str = enc.decode([tid])
        if neighs:
            neighbor_strs = [f"{enc.decode([x[1]])}" for x in neighs]
            annotated = f"{token_str}[NN={neighbor_strs}]"
        else:
            annotated = token_str
        annotated_strs.append(annotated)

    annotated_text = "".join(annotated_strs)
    return final_text, annotated_text


################################################################################
# 8. Training
################################################################################

def train_one_model(model,
                    loader,
                    epochs,
                    model_name,
                    device,
                    lr=1e-3,
                    log_steps=100,
                    sample_interval=30,
                    max_steps_per_epoch=None,
                    enc=None,
                    monosemantic_info=None,
                    prompt="Once upon a",
                    log_csv_path: str = "",
                    log_flush_steps: int = 100):
    """
    We add `prompt` as an explicit argument so we can pass it down from main().
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Buffered logging setup
    loss_buffer = []
    csv_file = None
    if log_csv_path:
        log_dir = os.path.dirname(log_csv_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_exists = os.path.exists(log_csv_path)
        csv_file = open(log_csv_path, 'a', newline='')
        # Write header if empty
        if not file_exists or os.path.getsize(log_csv_path) == 0:
            csv_file.write('timestamp,model,epoch,step_in_epoch,global_step,loss\n')

    start_time = time.time()
    next_sample_time = start_time
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        partial_loss = 0.0
        partial_count = 0

        step_in_epoch = 0
        for batch_idx, batch_tokens in enumerate(loader, start=1):
            step_in_epoch += 1
            global_step += 1

            batch_tokens = batch_tokens.to(device)  # (seq_len, batch)

            logits = model(batch_tokens)  # (seq_len, batch, vocab_size)
            loss = compute_next_token_loss(logits, batch_tokens)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            partial_loss += loss.item()
            partial_count += 1

            # Buffer this step's loss
            if csv_file is not None:
                loss_buffer.append(f"{time.time()},{model_name},{epoch},{step_in_epoch},{global_step},{loss.item()}\n")
                if len(loss_buffer) >= log_flush_steps:
                    csv_file.writelines(loss_buffer)
                    csv_file.flush()
                    loss_buffer.clear()

            if batch_idx % log_steps == 0:
                avg_part_loss = partial_loss / partial_count
                print(f"[{model_name}] Epoch {epoch}/{epochs}, "
                      f"Step {batch_idx}/{len(loader)} (global step: {global_step}) "
                      f"Partial Avg Loss: {avg_part_loss:.4f}")
                partial_loss = 0.0
                partial_count = 0

            current_time = time.time()
            if current_time >= next_sample_time and enc is not None:
                with torch.no_grad():
                    print(f"\n[{model_name}] Generating sample text (greedy) at epoch={epoch}, step={batch_idx}...")
                    text_greedy, ann_greedy = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=None,
                        use_kv_cache=False,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Greedy Sample: {text_greedy}")
                    print(f" Annotated: {ann_greedy}\n")

                    print(f"[{model_name}] Generating sample text (top-p=0.95) at epoch={epoch}, step={batch_idx}...")
                    text_topp, ann_topp = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=0.95,
                        use_kv_cache=False,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Top-p (p=0.95) Sample: {text_topp}")
                    print(f" Annotated: {ann_topp}\n")

                    # third generation => top-p=1.0 => full distribution random sampling
                    print(f"[{model_name}] Generating sample text (top-p=1.0) at epoch={epoch}, step={batch_idx}...")
                    text_topp1, ann_topp1 = generate_text(
                        model, enc, prompt, max_new_tokens=20, device=device,
                        top_p=1.0,
                        use_kv_cache=False,
                        monosemantic_info=monosemantic_info,
                        do_monosemantic=(monosemantic_info is not None)
                    )
                    print(f" Top-p (p=1.0) Sample: {text_topp1}")
                    print(f" Annotated: {ann_topp1}\n")

                next_sample_time = current_time + sample_interval

            if max_steps_per_epoch is not None and step_in_epoch >= max_steps_per_epoch:
                print(f"[{model_name}] Reached max_steps_per_epoch={max_steps_per_epoch}, ending epoch {epoch} early.")
                break

        avg_loss = total_loss / step_in_epoch
        print(f"[{model_name}] *** End of Epoch {epoch} *** Avg Loss: {avg_loss:.4f}")

    # Final flush
    if csv_file is not None:
        if loss_buffer:
            csv_file.writelines(loss_buffer)
            csv_file.flush()
        csv_file.close()


################################################################################
# 9. Main
################################################################################

def return_activation(activation_str):
    activation_str = activation_str.lower()
    if activation_str == "relu":
        return nn.ReLU
    elif activation_str == "silu":
        return nn.SiLU
    elif activation_str == "sigmoid":
        return nn.Sigmoid
    elif activation_str == "gelu":
        return nn.GELU
    else:
        raise ValueError(f"Unsupported activation: {activation_str}. Supported: relu, silu, sigmoid, gelu.")
def main():
    args = parse_args()

    # Additional local variables from arguments
    k = args.kgram_k
    chunk_size = args.kgram_chunk_size

    embed_size = args.embed_size
    batch_size = 16
    num_epochs = args.num_epochs
    learning_rate = 1e-3

    block_size = args.block_size
    train_subset_size = 20000
    log_interval_steps = 100
    sample_interval_seconds = 30

    max_steps_per_epoch = args.max_steps_per_epoch
    num_inner_layers = args.num_inner_mlp_layers

    #newly added local variables
    activation = args.activation.lower()
    act = return_activation(activation)
    



    # NEW: pick device from args.device_id, fallback to cpu if needed
    requested_device_id = args.device_id
    print(f"Requested device ID: {requested_device_id}")
    if requested_device_id.startswith("cuda") and not torch.cuda.is_available():
        print(f"Requested device '{requested_device_id}' but CUDA not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(requested_device_id)

    print(f"Using device: {device}, block_size={block_size}, kgram_k={k}, chunk_size={chunk_size}, embed_size={embed_size}")

    ############################################################################
    # Data
    ############################################################################
    tinystories_seqs = []
    other_seqs = []

    if args.tinystories_weight > 0.0:
        print(f"Loading TinyStories from huggingface with weight={args.tinystories_weight}...")
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        dataset = dataset.select(range(train_subset_size))
    else:
        print("TinyStories weight=0 => skipping TinyStories.")
        dataset = None

    enc = tiktoken.get_encoding("gpt2")
    vocab_size = enc.n_vocab
    print(f"Vocab size: {vocab_size}")

    if dataset is not None:
        for sample in dataset:
            text = sample['text']
            tokens = enc.encode(text)
            tokens = tokens[:block_size]
            if len(tokens) > 0:
                tinystories_seqs.append(tokens)
        print(f"TinyStories sequences: {len(tinystories_seqs)}")

    if args.input_files:
        for filepath in args.input_files:
            print(f"Reading custom text file: {filepath}")
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                tokens = enc.encode(line)
                tokens = tokens[:block_size]
                if len(tokens) > 0:
                    other_seqs.append(tokens)
        print(f"Custom input files: {len(other_seqs)} sequences loaded.")
    else:
        print("No custom input files provided.")

    p_tiny = args.tinystories_weight
    if len(tinystories_seqs) == 0 and p_tiny>0:
        print("Warning: TinyStories is empty but tinystories_weight>0. That's okay, no data from it.")
    combined_dataset = MixedSequenceDataset(
        tinystories_seqs=tinystories_seqs,
        other_seqs=other_seqs,
        p_tiny=p_tiny
    )

    train_loader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=seq_collate_fn
    )

    ############################################################################
    # Models
    ############################################################################
    kgram_model = KGramMLPSeqModel(
        vocab_size=vocab_size,
        k=k,
        embed_size=embed_size,
        num_inner_layers=num_inner_layers,
        chunk_size=chunk_size,
        activation=act
    ).to(device)

    lstm_model = LSTMSeqModel(
        vocab_size=vocab_size,
        embed_size=embed_size,
        hidden_size=embed_size
    ).to(device)

    # Resolve Transformer hyperparameters (defaults tied to embed_size/block_size when unspecified)
    t_d_model = args.transformer_d_model if args.transformer_d_model is not None else embed_size
    t_max_seq_len = args.transformer_max_seq_len if args.transformer_max_seq_len and args.transformer_max_seq_len > 0 else block_size

    transformer = TransformerModel(
        vocab_size=vocab_size,
        d_model=t_d_model,
        n_heads=args.transformer_n_heads,
        n_blocks=args.transformer_n_blocks,
        max_seq_len=t_max_seq_len,
        mlp_ratio=args.transformer_mlp_ratio,
    ).to(device)

    models = {
      # "kgram_mlp_seq": kgram_model,
        #"lstm_seq": lstm_model,
        "transformer": transformer,
      "kgram_mlp_seq": kgram_model,
      #  "lstm_seq": lstm_model,
      # "kvcache_transformer": kv_transformer,
    }


    ############################################################################
    # Train each model
    ############################################################################
    def run_and_optionally_save(model_name, model, n_blocks_for_name=None):
        print(f"\n=== Training model: {model_name} ===")
        print(f"Model architecture:\n{model}\n")
        csv_name = args.log_csv.split('.csv')[0]
        new_csv_path = f"{csv_name}_{model_name}_{n_blocks_for_name}.csv" if n_blocks_for_name is not None else args.log_csvs
        train_one_model(
            model=model,
            loader=train_loader,
            epochs=num_epochs,
            model_name=model_name,
            device=device,
            lr=learning_rate,
            log_steps=log_interval_steps,
            sample_interval=sample_interval_seconds,
            max_steps_per_epoch=max_steps_per_epoch,
            enc=enc,
            prompt=args.prompt,  # <--- Pass the user-specified prompt here
            log_csv_path=new_csv_path,
            log_flush_steps=args.log_flush_steps,
        )

    # Final generation from the user-provided prompt (args.prompt).
        with torch.no_grad():
            # 1) Greedy
            text_greedy, ann_greedy = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=None,
                use_kv_cache=args.use_kv_cache,
            )
            # 2) top-p=0.95
            text_topp, ann_topp = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=0.95,
                use_kv_cache=args.use_kv_cache,
            )
            # 3) top-p=1.0 => full distribution random sampling
            text_topp1, ann_topp1 = generate_text(
                model, enc, args.prompt, max_new_tokens=20, device=device,
                top_p=1.0,
                use_kv_cache=args.use_kv_cache,
            )

        print(f"[{model_name}] Final sample (greedy) from prompt: '{args.prompt}'")
        print(text_greedy)
        print(f"Annotated:\n{ann_greedy}\n")

        print(f"[{model_name}] Final sample (top-p=0.95) from prompt: '{args.prompt}'")
        print(text_topp)
        print(f"Annotated:\n{ann_topp}\n")

        print(f"[{model_name}] Final sample (top-p=1.0) from prompt: '{args.prompt}'")
        print(text_topp1)
        print(f"Annotated:\n{ann_topp1}")
        print("--------------------------------------------------")

        # Save trained model if requested
        if args.save_model_dir:
            os.makedirs(args.save_model_dir, exist_ok=True)
            base = args.save_model_name
            if n_blocks_for_name is not None:
                base = f"{base}_n{n_blocks_for_name}"
            save_path = os.path.join(args.save_model_dir, f"{base}_state.pt")
            meta_path = os.path.join(args.save_model_dir, f"{base}_meta.json")
            try:
                torch.save(model.state_dict(), save_path)
                meta = {
                    "model_name": model_name,
                    "vocab_size": int(vocab_size),
                    "args": {
                        "embed_size": int(embed_size),
                        "block_size": int(block_size),
                        "transformer_d_model": int(t_d_model),
                        "transformer_n_heads": int(args.transformer_n_heads),
                        "transformer_n_blocks": int(n_blocks_for_name if n_blocks_for_name is not None else args.transformer_n_blocks),
                        "transformer_mlp_ratio": float(args.transformer_mlp_ratio),
                        "transformer_max_seq_len": int(t_max_seq_len),
                    },
                }
                with open(meta_path, 'w') as f:
                    json.dump(meta, f, indent=2)
                print(f"Saved model to {save_path} and metadata to {meta_path}")
            except Exception as e:
                print(f"Failed to save model: {e}")

    depth_list = []
    if args.depth_list:
        try:
            depth_list = [int(x.strip()) for x in args.depth_list.split(',') if x.strip()]
        except Exception as e:
            print(f"Could not parse --depth_list: {e}")
            depth_list = []

    if depth_list:
        # Train specified transformer depths sequentially
        for n_blocks_val in depth_list:
            print(f"\n>>> Running transformer with n_blocks={n_blocks_val}")
            transformer_cfg = TransformerModel(
                vocab_size=vocab_size,
                d_model=t_d_model,
                n_heads=args.transformer_n_heads,
                n_blocks=n_blocks_val,
                max_seq_len=t_max_seq_len,
                mlp_ratio=args.transformer_mlp_ratio,
            ).to(device)
            run_and_optionally_save("transformer", transformer_cfg, n_blocks_for_name=n_blocks_val)
    else:
        # Fallback: run the prebuilt models dict
        for model_name, model in models.items():
            run_and_optionally_save(model_name, model)

    # Finally, a friendly sign-off
    print("\n*** Training complete. ***")


if __name__ == "__main__":
    main()
