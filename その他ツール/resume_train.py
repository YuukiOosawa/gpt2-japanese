import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
import random
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================
# 0. 初期設定と引数パース
# ==========================================
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

parser = argparse.ArgumentParser(description="GPT-2 途中再開ファインチューニング")
parser.add_argument("--config", type=str, default="resume_config.yml", help="再開用設定ファイルのパス")
args, unknown = parser.parse_known_args()

config_path = args.config
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

system_config = config.get('system', {})
seed = system_config.get('seed', None)
if seed is not None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# モデル・データ・学習設定
modelname = config['model']['name']
do_lower_case = config['model'].get('do_lower_case', True)

data_config = config.get('data', {})
num_epochs = config['training']['epochs']
start_epoch = config['training'].get('start_epoch', 10)  # ★ 10エポック目から開始
shortcut = data_config.get('shortcut', False)
valid_subset_size = data_config.get('valid_subset_size', False)
valid_subset_seed = data_config.get('valid_subset_seed', 2025)
ADD_BOS_TOKEN = data_config.get('add_bos_token', True)
src_col = data_config.get('src_column', 'j_lists')
tgt_col = data_config.get('tgt_column', 'sentence')

training_config = config.get('training', {})
batch_size = training_config['batch_size']  # OOM防止のため小さくしたバッチサイズ（例: 8）
eval_batch_size = int(training_config.get('eval_batch_size', 4))
accumulation_steps = training_config.get('accumulation_steps', 1)
learning_rate = float(training_config['learning_rate'])
weight_decay = float(training_config.get('weight_decay', 0.01))
generation_mode = training_config.get('generation_mode', 'gumbel')
eval_generation_mode = training_config.get('eval_generation_mode', 'argmax')

resume_checkpoint_dir = training_config.get('resume_checkpoint_dir', "models_2_1009/finetuned_model_epoch_10")

output_config = config.get('output', {})
save_base_dir = output_config.get('save_dir', 'models_2_1009')
save_prefix = output_config.get('save_prefix', 'finetuned_model_epoch')
save_intermediate_checkpoints = output_config.get('save_intermediate_checkpoints', True)
save_epoch_interval = output_config.get('save_epoch_interval', 1)

mask_schedule = training_config.get('mask_schedule')

# ==========================================
# 1. データセットの読み込み
# ==========================================
df_train = pd.read_csv(data_config['train_path'])
df_test = pd.read_csv(data_config['test_path'])
df_valid = pd.read_csv(data_config['valid_path'])

if shortcut:
    df_train = df_train[:data_config['shortcut_sizes']['train']]
    df_test = df_test[:data_config['shortcut_sizes']['test']]
    df_valid = df_valid[:data_config['shortcut_sizes']['valid']]

# 追加
train_subset_size = data_config.get('train_subset_size', False)
train_subset_seed = data_config.get('train_subset_seed', 2025)
if train_subset_size and isinstance(train_subset_size, int) and len(df_train) > train_subset_size:
    df_train = df_train.sample(n=train_subset_size, random_state=train_subset_seed).reset_index(drop=True)

if valid_subset_size and isinstance(valid_subset_size, int) and len(df_valid) > valid_subset_size:
    df_valid = df_valid.sample(n=valid_subset_size, random_state=valid_subset_seed).reset_index(drop=True)

# ==========================================
# 2. トークナイザーとチェックポイントからのモデル復元
# ==========================================
print(f"Resuming model & tokenizer from checkpoint: {resume_checkpoint_dir}")
tokenizer = AutoTokenizer.from_pretrained(resume_checkpoint_dir, use_fast=False)
tokenizer.do_lower_case = do_lower_case
#model = AutoModelForCausalLM.from_pretrained(resume_checkpoint_dir)
model = AutoModelForCausalLM.from_pretrained(
    resume_checkpoint_dir,
    attn_implementation="sdpa"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if model.dtype == torch.float16:
    model = model.float()

model.to(device)
use_amp = device.type == 'cuda'
if use_amp:
    #model.gradient_checkpointing_enable()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

# Optimizer定義と状態（optimizer.pt）の復元
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
optimizer_path = os.path.join(resume_checkpoint_dir, "optimizer.pt")
if os.path.exists(optimizer_path):
    optimizer.load_state_dict(torch.load(optimizer_path, map_location=device))
    print(f"Loaded optimizer state from {optimizer_path}")
else:
    print("Warning: optimizer.pt not found. Starting with fresh optimizer state.")

# ==========================================
# 3. Dataset & DataLoader
# ==========================================
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, intexts, outtexts):
        self.data = []
        self.labels = []
        for i in tqdm(range(len(intexts)), desc="Building Dataset"):
            inputs = tokenizer(intexts[i], add_special_tokens=False)
            if ADD_BOS_TOKEN:
                inputs['input_ids'] = [tokenizer.bos_token_id] + inputs['input_ids']
                inputs['attention_mask'] = [1] + inputs['attention_mask']
            inputs['input_ids'] = inputs['input_ids'] + [tokenizer.sep_token_id]
            inputs['attention_mask'] = inputs['attention_mask'] + [1]
            self.data.append(inputs.copy())
            inputs['input_ids'] = inputs['input_ids'][1:]
            inputs['attention_mask'] = inputs['attention_mask'][1:]
            self.labels.append(inputs.copy())
            inputs_out = tokenizer(outtexts[i], add_special_tokens=True)
            self.data[-1]['input_ids'] += inputs_out['input_ids'][:-1]
            self.data[-1]['attention_mask'] += inputs_out['attention_mask'][:-1]
            self.labels[-1]['input_ids'] += inputs_out['input_ids']
            self.labels[-1]['attention_mask'] += inputs_out['attention_mask']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

train_dataset = TextDataset(df_train[src_col].apply(lambda x: ''.join(x[2:-2].split("', '"))).tolist(), df_train[tgt_col].tolist())
valid_dataset = TextDataset(df_valid[src_col].apply(lambda x: ''.join(x[2:-2].split("', '"))).tolist(), df_valid[tgt_col].tolist())

def collate_fnNTF(batch):
    data_list = {'input_ids': [], 'attention_mask': []}
    label_list = {'input_ids': [], 'attention_mask': []}
    for d, l in batch:
        src_sep_idx = d['input_ids'].index(tokenizer.sep_token_id) + 1
        data_list['input_ids'].append(torch.tensor(d['input_ids'][:src_sep_idx], dtype=torch.long))
        data_list['attention_mask'].append(torch.tensor(d['attention_mask'][:src_sep_idx], dtype=torch.long))
        tgt_sep_idx = l['input_ids'].index(tokenizer.sep_token_id) + 1
        label_list['input_ids'].append(torch.tensor(l['input_ids'][tgt_sep_idx:], dtype=torch.long))
        label_list['attention_mask'].append(torch.tensor(l['attention_mask'][tgt_sep_idx:], dtype=torch.long))
    src_batch = {
        'input_ids': torch.nn.utils.rnn.pad_sequence(data_list['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id, padding_side='left'),
        'attention_mask': torch.nn.utils.rnn.pad_sequence(data_list['attention_mask'], batch_first=True, padding_value=0, padding_side='left')
    }
    tgt_batch = {
        'input_ids': torch.nn.utils.rnn.pad_sequence(label_list['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id, padding_side='right'),
        'attention_mask': torch.nn.utils.rnn.pad_sequence(label_list['attention_mask'], batch_first=True, padding_value=0, padding_side='right')
    }
    return src_batch, tgt_batch

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fnNTF, shuffle=True)
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=eval_batch_size, collate_fn=collate_fnNTF, shuffle=False)

# ==========================================
# 4. 生成・マスク・DEBLEU損失関数
# ==========================================
def parse_mask_pattern(pattern_str: str) -> tuple:
    if not pattern_str or ":" not in str(pattern_str):
        return (1, 0)
    parts = str(pattern_str).strip().split(":")
    return int(parts[0]), int(parts[1])

def get_mask_pattern(epoch_idx: int):
    if mask_schedule is None:
        return "1:0"
    if isinstance(mask_schedule, dict):
        matched = [int(k) for k in mask_schedule.keys() if int(k) <= epoch_idx]
        if matched:
            best_k = max(matched)
            return str(mask_schedule.get(best_k, mask_schedule.get(str(best_k), "1:0")))
        return "1:0"
    return str(mask_schedule)

bleu_config = config.get('bleu', {})
bleu_enabled = bleu_config.get('enabled', True)
bleu_max_ngram = bleu_config.get('max_ngram', 4)
bleu_weights = tuple(bleu_config.get('weights', [0.1, 0.3, 0.3, 0.3]))
bleu_alpha_schedule = bleu_config.get('alpha_schedule', {})

def get_alpha(epoch_idx: int) -> float:
    if not bleu_enabled:
        return 0.0
    if isinstance(bleu_alpha_schedule, dict):
        matched = [int(k) for k in bleu_alpha_schedule.keys() if int(k) <= epoch_idx]
        if matched:
            best_k = max(matched)
            return float(bleu_alpha_schedule.get(best_k, bleu_alpha_schedule.get(str(best_k), 0.0)))
        return 0.0
    return 0.0

def autonomous_generate_stepwise(
    model, src_input_ids: torch.Tensor, src_attention_mask: torch.Tensor,
    max_steps: int, generation_mode: str = "argmax", temperature: float = 1.0, gumbel_hard: bool = True
) -> torch.Tensor:
    batch_size = src_input_ids.size(0)
    device = src_input_ids.device
    curr_attention_mask = src_attention_mask
    logits_list = []

    if generation_mode == "gumbel":
        embed_layer = model.get_input_embeddings()
        curr_inputs_embeds = embed_layer(src_input_ids)
        for t in range(max_steps):
            outputs = model(inputs_embeds=curr_inputs_embeds, attention_mask=curr_attention_mask)
            step_logits = outputs.logits[:, -1, :]
            logits_list.append(step_logits)
            if t < max_steps - 1:
                soft_dist = torch.nn.functional.gumbel_softmax(step_logits, tau=max(temperature, 1e-5), hard=gumbel_hard)
                next_embed = torch.matmul(soft_dist.unsqueeze(1), embed_layer.weight)
                curr_inputs_embeds = torch.cat([curr_inputs_embeds, next_embed], dim=1)
                curr_attention_mask = torch.cat([curr_attention_mask, torch.ones((batch_size, 1), device=device, dtype=curr_attention_mask.dtype)], dim=1)
    else:
        curr_input_ids = src_input_ids
        for t in range(max_steps):
            outputs = model(input_ids=curr_input_ids, attention_mask=curr_attention_mask)
            step_logits = outputs.logits[:, -1, :]
            logits_list.append(step_logits)
            if t < max_steps - 1:
                if generation_mode == "sample":
                    probs = torch.softmax(step_logits / max(temperature, 1e-5), dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = step_logits.argmax(dim=-1, keepdim=True)
                curr_input_ids = torch.cat([curr_input_ids, next_token], dim=1)
                curr_attention_mask = torch.cat([curr_attention_mask, torch.ones((batch_size, 1), device=device, dtype=curr_attention_mask.dtype)], dim=1)
    return torch.stack(logits_list, dim=1)

def decode_with_teacher_mask(
    model, src_input_ids: torch.Tensor, src_attention_mask: torch.Tensor, labels: torch.Tensor,
    max_steps: int, mask_pattern: str = "1:0", generation_mode: str = "gumbel",
    temperature: float = 1.0, gumbel_hard: bool = True, random_shift: bool = True, pad_token_id: int = tokenizer.pad_token_id
) -> tuple:
    batch_size = src_input_ids.size(0)
    device = src_input_ids.device
    embed_layer = model.get_input_embeddings()
    vocab_size = embed_layer.weight.size(0)
    u, m = parse_mask_pattern(mask_pattern)

    if u == 0:
        prompt_len = src_input_ids.size(1)
        if max_steps > 1:
            tf_inputs = torch.cat([src_input_ids, labels[:, :max_steps - 1]], dim=1)
            tf_mask = torch.cat([src_attention_mask, (labels[:, :max_steps - 1] != pad_token_id).to(src_attention_mask.dtype)], dim=1)
        else:
            tf_inputs = src_input_ids
            tf_mask = src_attention_mask
        outputs = model(input_ids=tf_inputs, attention_mask=tf_mask)
        generated_logits = outputs.logits[:, prompt_len - 1 : prompt_len - 1 + max_steps, :]
        debleu_probs = torch.nn.functional.one_hot(labels[:, :max_steps], num_classes=vocab_size).to(generated_logits.dtype)
        return generated_logits, debleu_probs

    C = u + m
    offset = random.randint(0, C - 1) if (random_shift and C > 0) else 0
    curr_inputs_embeds = embed_layer(src_input_ids)
    curr_attention_mask = src_attention_mask
    logits_list = []
    probs_list = []

    for t in range(max_steps):
        is_masked = (m > 0 and ((t + offset) % C) >= u)
        outputs = model(inputs_embeds=curr_inputs_embeds, attention_mask=curr_attention_mask)
        step_logits = outputs.logits[:, -1, :]
        logits_list.append(step_logits)

        if is_masked:
            gt_token = labels[:, t]
            step_prob = torch.nn.functional.one_hot(gt_token, num_classes=vocab_size).to(step_logits.dtype)
        else:
            if generation_mode == "gumbel":
                step_prob = torch.nn.functional.gumbel_softmax(step_logits, tau=max(temperature, 1e-5), hard=gumbel_hard)
            else:
                step_prob = torch.softmax(step_logits, dim=-1)
        probs_list.append(step_prob)

        if t < max_steps - 1:
            if is_masked:
                gt_token = labels[:, t].unsqueeze(1)
                next_embed = embed_layer(gt_token)
            else:
                if generation_mode == "gumbel":
                    next_embed = torch.matmul(step_prob.unsqueeze(1), embed_layer.weight)
                elif generation_mode == "sample":
                    probs = torch.softmax(step_logits / max(temperature, 1e-5), dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    next_embed = embed_layer(next_token)
                else:
                    next_token = step_logits.argmax(dim=-1, keepdim=True)
                    next_embed = embed_layer(next_token)
            curr_inputs_embeds = torch.cat([curr_inputs_embeds, next_embed], dim=1)
            curr_attention_mask = torch.cat([curr_attention_mask, torch.ones((batch_size, 1), device=device, dtype=curr_attention_mask.dtype)], dim=1)

    return torch.stack(logits_list, dim=1), torch.stack(probs_list, dim=1)

criterionCE = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

def criterionBLEU(
    logits: torch.Tensor, labels: torch.Tensor, pad_token_id: int = tokenizer.pad_token_id,
    sep_token_id: int = tokenizer.sep_token_id, target_only: bool = True, N: int = 4,
    weights: tuple = (0.1, 0.3, 0.3, 0.3), return_loss: bool = True, return_score: bool = False, eps: float = 1e-8
):
    if logits.ndim != 3:
        raise ValueError(f"Expected 3D logits tensor, got shape {logits.shape}")
    logits_f32 = logits.float()
    if logits_f32.shape[1] == labels.shape[1]:
        probs = logits_f32
    elif logits_f32.shape[2] == labels.shape[1]:
        probs = logits_f32.transpose(1, 2)
    else:
        probs = logits_f32

    is_prob = (probs >= 0.0).all() and torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)), atol=1e-2)
    if not is_prob:
        probs = torch.softmax(probs, dim=-1)

    batch_size = labels.shape[0]
    device = logits.device
    dtype = torch.float32
    if weights is None or len(weights) < N:
        weights = [0.1, 0.3, 0.3, 0.3] if N == 4 else [1.0 / N] * N

    batch_losses = []
    batch_scores = []
    for b in range(batch_size):
        sample_labels = labels[b]
        sample_probs = probs[b]
        valid_mask = sample_labels.ne(pad_token_id)
        if target_only and sep_token_id is not None:
            sep_positions = (sample_labels == sep_token_id).nonzero(as_tuple=False)
            if len(sep_positions) > 0:
                sep_idx = sep_positions[0].item()
                target_mask = torch.zeros_like(valid_mask)
                target_mask[sep_idx + 1:] = True
                valid_mask = valid_mask & target_mask

        ref = sample_labels[valid_mask]
        p = sample_probs[valid_mask]
        T = p.shape[0]
        T_star = ref.shape[0]
        if T == 0 or T_star == 0:
            continue

        log_bp = torch.tensor(1.0 - T_star / T, device=device, dtype=dtype) if T < T_star else torch.tensor(0.0, device=device, dtype=dtype)
        bp = float(torch.exp(log_bp)) if T < T_star else 1.0

        M = p.index_select(dim=1, index=ref)
        R = (ref.unsqueeze(0) == ref.unsqueeze(1)).to(dtype)
        precisions = []
        log_precisions = []
        M_n = M
        R_n = R

        for n in range(1, N + 1):
            cur_T = T - n + 1
            cur_T_star = T_star - n + 1
            if cur_T <= 0 or cur_T_star <= 0:
                break
            if n > 1:
                M_n = M_n_prev[:-1, :-1] * M[n - 1:, n - 1:]
                R_n = R_n_prev[:-1, :-1] * R[n - 1:, n - 1:]
            C_n = R_n.sum(dim=0).clamp(min=1.0)
            S = M_n.sum(dim=0, keepdim=True)
            denom = 1.0 + (S - M_n)
            C_inv = 1.0 / C_n.unsqueeze(0)
            denom_inv = 1.0 / denom
            term = M_n * torch.minimum(C_inv, denom_inv)
            o_tilde = term.sum(dim=1)
            prec_n = o_tilde.mean()
            precisions.append(prec_n)
            log_precisions.append(torch.log(torch.clamp(prec_n, min=eps)))
            M_n_prev = M_n
            R_n_prev = R_n

        actual_N = len(precisions)
        if actual_N == 0:
            continue
        cur_weights = weights[:actual_N]
        w_sum = sum(cur_weights)
        cur_weights = [w / w_sum for w in cur_weights]

        sample_loss = -log_bp - sum(w * lp for w, lp in zip(cur_weights, log_precisions))
        sample_score = bp * torch.exp(sum(w * lp for w, lp in zip(cur_weights, log_precisions)))
        batch_losses.append(sample_loss)
        batch_scores.append(sample_score)

    if not batch_losses:
        zero_val = logits.new_zeros(())
        return (zero_val, zero_val) if (return_loss and return_score) else zero_val

    total_loss = torch.stack(batch_losses).mean().to(logits.dtype)
    total_score = torch.stack(batch_scores).mean().to(logits.dtype)
    if return_loss and return_score:
        return total_loss, total_score
    return total_loss if return_loss else total_score

# ==========================================
# 5. 途中再開用のトレーニングループ (start_epoch 〜 num_epochs)
# ==========================================
print(f"Starting resume training loop from Epoch {start_epoch} to {num_epochs}...")

for epoch in range(start_epoch, num_epochs):
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_bleu_loss = 0
    total_bleu_score = 0
    bleu_count = 0
    optimizer.zero_grad(set_to_none=True)

    alpha = get_alpha(epoch)
    cur_mask_pattern = get_mask_pattern(epoch)
    print(f"\n[Epoch {epoch}] alpha = {alpha}, mask_pattern = {cur_mask_pattern}")

    for step, batch in enumerate(tqdm(train_dataloader, desc=f"Train Epoch {epoch}")):
        src_inputs = batch[0]['input_ids'].to(device)
        src_attention_mask = batch[0]['attention_mask'].to(device)
        labels = batch[1]['input_ids'].to(device)
        target_len = labels.shape[1]

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            generated_logits, debleu_probs = decode_with_teacher_mask(
                model=model, src_input_ids=src_inputs, src_attention_mask=src_attention_mask,
                labels=labels, max_steps=target_len, mask_pattern=cur_mask_pattern, generation_mode=generation_mode
            )
            ce_loss = criterionCE(torch.transpose(generated_logits, 1, 2), labels)

            if alpha > 0.0:
                bleu_loss, bleu_score = criterionBLEU(
                    debleu_probs, labels, pad_token_id=tokenizer.pad_token_id, sep_token_id=tokenizer.sep_token_id,
                    target_only=False, N=bleu_max_ngram, weights=bleu_weights, return_loss=True, return_score=True
                )
                loss = ce_loss + alpha * bleu_loss
                total_bleu_loss += bleu_loss.item()
                total_bleu_score += bleu_score.item()
                bleu_count += 1
            else:
                loss = ce_loss

        scaler.scale(loss / accumulation_steps).backward()
        if (step + 1) % accumulation_steps == 0 or step + 1 == len(train_dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        # 追加
        del generated_logits, debleu_probs, ce_loss, loss
        if step % 50 == 0:  # 50ステップごとに実行（毎ステップだと重くなるため）
            torch.cuda.empty_cache()

    avg_train_loss = total_loss / len(train_dataloader)
    avg_ce_loss = total_ce_loss / len(train_dataloader)
    avg_bleu_loss = (total_bleu_loss / bleu_count) if bleu_count > 0 else 0.0
    avg_bleu_score = (total_bleu_score / bleu_count) if bleu_count > 0 else 0.0

    optimizer.zero_grad(set_to_none=True)
    # 追加
    import gc
    gc.collect()
    # ここまで
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 検証ループ
    model.eval()
    eval_loss = 0
    eval_ce_loss = 0
    eval_bleu_loss = 0
    eval_bleu_score = 0
    eval_bleu_count = 0

    with torch.inference_mode():
        for batch in tqdm(valid_dataloader, desc=f"Valid Epoch {epoch}"):
            src_inputs = batch[0]['input_ids'].to(device)
            src_attention_mask = batch[0]['attention_mask'].to(device)
            labels = batch[1]['input_ids'].to(device)
            target_len = labels.shape[1]

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                generated_logits = autonomous_generate_stepwise(
                    model=model, src_input_ids=src_inputs, src_attention_mask=src_attention_mask,
                    max_steps=target_len, generation_mode=eval_generation_mode
                )
                ce_loss = criterionCE(torch.transpose(generated_logits, 1, 2), labels)
                if alpha > 0.0:
                    bleu_loss, bleu_score = criterionBLEU(
                        generated_logits, labels, pad_token_id=tokenizer.pad_token_id, sep_token_id=tokenizer.sep_token_id,
                        target_only=False, N=bleu_max_ngram, weights=bleu_weights, return_loss=True, return_score=True
                    )
                    loss = ce_loss + alpha * bleu_loss
                    eval_bleu_loss += bleu_loss.item()
                    eval_bleu_score += bleu_score.item()
                    eval_bleu_count += 1
                else:
                    loss = ce_loss

            eval_loss += loss.item()
            eval_ce_loss += ce_loss.item()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    avg_eval_loss = eval_loss / len(valid_dataloader)
    avg_eval_ce_loss = eval_ce_loss / len(valid_dataloader)
    avg_eval_bleu_score = (eval_bleu_score / eval_bleu_count) if eval_bleu_count > 0 else 0.0

    if alpha > 0.0:
        print(f"Epoch {epoch}: Avg Train Loss = {avg_train_loss:.4f} (CE: {avg_ce_loss:.4f}, DEBLEU Loss: {avg_bleu_loss:.4f}, Score: {avg_bleu_score:.4f}) / Eval Loss = {avg_eval_loss:.4f} (Eval Score: {avg_eval_bleu_score:.4f})")
    else:
        print(f"Epoch {epoch}: Avg Train Loss = {avg_train_loss:.4f} (CE: {avg_ce_loss:.4f}) / Eval Loss = {avg_eval_loss:.4f}")

    # チェックポイント保存
    current_epoch = epoch + 1
    is_final_epoch = (current_epoch == num_epochs)
    is_interval_epoch = save_intermediate_checkpoints and (current_epoch % save_epoch_interval == 0)

    if is_interval_epoch or is_final_epoch:
        save_d = f"{save_base_dir}/{save_prefix}_{current_epoch}"
        os.makedirs(save_d, exist_ok=True)
        model.save_pretrained(save_d)
        tokenizer.save_pretrained(save_d)
        torch.save(optimizer.state_dict(), f"{save_d}/optimizer.pt")
        with open(f"{save_d}/meta.txt", "w") as f:
            f.write(f"epoch={current_epoch}\n")
            f.write(f"avg_train_loss={avg_train_loss:.4f}\n")
            f.write(f"avg_eval_loss={avg_eval_loss:.4f}\n")
            f.write(f"alpha={alpha}\n")

print("Resumed training finished successfully!")