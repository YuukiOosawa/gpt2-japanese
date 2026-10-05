# %% [markdown]
# 必要なパッケージの読み込み、使用モデル等の指定

# %%
from tqdm import tqdm
import time
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import argparse

# コマンドライン引数の設定
parser = argparse.ArgumentParser(description="GPT-2 日本語モデルのファインチューニング")
parser.add_argument("--config", type=str, default="config.yml", help="設定ファイルのパス")
parser.add_argument("--save_intermediate_checkpoints", type=str, default=None, choices=["true", "false", "True", "False"],
                    help="指定エポック毎にモデルを保存するか (True/False)")
parser.add_argument("--save_epoch_interval", type=int, default=None, help="モデルを保存するエポック間隔")
parser.add_argument("--epochs", type=int, default=None, help="学習エポック数")
parser.add_argument("--save_dir", type=str, default=None, help="保存先親ディレクトリ")
parser.add_argument("--alpha", type=float, default=None, help="DEBLEU損失の重みalpha (固定値指定用)")
parser.add_argument("--learning_rate", type=float, default=None, help="学習率")
parser.add_argument("--batch_size", type=int, default=None, help="バッチサイズ")
args, unknown = parser.parse_known_args()

# 設定ファイルの読み込み
config_path = args.config
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# コマンドライン引数による設定の上書き
if args.save_intermediate_checkpoints is not None:
    config.setdefault('output', {})['save_intermediate_checkpoints'] = (args.save_intermediate_checkpoints.lower() == "true")
if args.save_epoch_interval is not None:
    config.setdefault('output', {})['save_epoch_interval'] = args.save_epoch_interval
if args.epochs is not None:
    config.setdefault('training', {})['epochs'] = args.epochs
if args.save_dir is not None:
    config.setdefault('output', {})['save_dir'] = args.save_dir
if args.learning_rate is not None:
    config.setdefault('training', {})['learning_rate'] = args.learning_rate
if args.batch_size is not None:
    config.setdefault('training', {})['batch_size'] = args.batch_size
if args.alpha is not None:
    config.setdefault('bleu', {})['alpha_schedule'] = {0: args.alpha}

# システム設定
system_config = config.get('system', {})
seed = system_config.get('seed', None)
if seed is not None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# モデル設定
modelname = config['model']['name']
do_lower_case = config['model'].get('do_lower_case', True)

# データ設定
data_config = config.get('data', {})
num_epochs = config['training']['epochs']
shortcut = data_config.get('shortcut', False)
ADD_BOS_TOKEN = data_config.get('add_bos_token', True)
src_col = data_config.get('src_column', 'j_lists')
src_test_col = data_config.get('src_test_column', 'j_lists_test')
tgt_col = data_config.get('tgt_column', 'sentence')

# 学習設定
training_config = config.get('training', {})
batch_size = training_config['batch_size']
accumulation_steps = training_config.get('accumulation_steps', 1)
learning_rate = float(training_config['learning_rate'])
weight_decay = float(training_config.get('weight_decay', 0.01))
teacher_forcing_ratio = float(training_config.get('teacher_forcing_ratio', 0.5))

# DEBLEU設定
bleu_config = config.get('bleu', {})
bleu_enabled = bleu_config.get('enabled', True)
bleu_max_ngram = bleu_config.get('max_ngram', 4)
bleu_weights = tuple(bleu_config.get('weights', [0.1, 0.3, 0.3, 0.3]))
bleu_target_only = bleu_config.get('target_only', True)
bleu_alpha_schedule = bleu_config.get('alpha_schedule', {0: 0.0, 4: 0.001, 5: 0.002})

def get_alpha(epoch_idx: int) -> float:
    """configのalpha_scheduleから指定エポックのalphaを取得"""
    if not bleu_enabled:
        return 0.0
    if isinstance(bleu_alpha_schedule, dict):
        matched = [int(k) for k in bleu_alpha_schedule.keys() if int(k) <= epoch_idx]
        if matched:
            best_k = max(matched)
            val = bleu_alpha_schedule.get(best_k, bleu_alpha_schedule.get(str(best_k), 0.0))
            return float(val)
        return 0.0
    elif isinstance(bleu_alpha_schedule, list):
        if epoch_idx < len(bleu_alpha_schedule):
            return float(bleu_alpha_schedule[epoch_idx])
        return float(bleu_alpha_schedule[-1])
    elif 'alpha' in bleu_config:
        return float(bleu_config['alpha'])
    return 0.0

# 出力・保存設定
output_config = config.get('output', {})
save_base_dir = output_config.get('save_dir', 'models')
save_prefix = output_config.get('save_prefix', 'finetuned_model_epoch')
save_intermediate_checkpoints = output_config.get('save_intermediate_checkpoints', True)
save_epoch_interval = output_config.get('save_epoch_interval', 5)

# 推論プレビュー設定
inference_config = config.get('inference', {})
max_new_tokens = inference_config.get('max_new_tokens', 40)

# %% [markdown]
# データセットの読み込み

# %%
import pandas as pd
df_train = pd.read_csv(config['data']['train_path'])
df_test = pd.read_csv(config['data']['test_path'])
df_valid = pd.read_csv(config['data']['valid_path'])
print(len(df_train),len(df_test),len(df_valid))
if shortcut:
    df_train = df_train[:config['data']['shortcut_sizes']['train']]
    df_test = df_test[:config['data']['shortcut_sizes']['test']]
    df_valid = df_valid[:config['data']['shortcut_sizes']['valid']]
    print(len(df_train),len(df_test),len(df_valid))
df_train

# %%
from transformers import AutoTokenizer, AutoModelForCausalLM

#df_train = pd.read_csv('amazon_review/amazon_review_train_valid.csv')
#df_test = pd.read_csv('amazon_review/amazon_review_test.csv')

# モデルとトークナイザーの読み込み
tokenizer = AutoTokenizer.from_pretrained(modelname, use_fast=False)
tokenizer.do_lower_case = do_lower_case
model = AutoModelForCausalLM.from_pretrained(modelname)

device_setting = system_config.get('device', 'auto')
if device_setting == 'auto':
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            device = torch.device("cuda:0")
            print('device = cuda:0')
        else:
            device = torch.device("cuda")
            print('device = cuda')
    else:
        device = torch.device("cpu")
        print('device = cpu')
else:
    device = torch.device(device_setting)
    print(f'device = {device_setting}')

model.to(device)
use_amp = device.type == 'cuda'
if use_amp:
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

# Optimizerの定義
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# %%
ADD_BOS_TOKEN = config['data']['add_bos_token']  # 文頭に<s>トークンを追加するかどうか

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, intexts, outtexts, transform=None, target_transform=None):
        self.data = []
        self.labels = []
        if len(intexts) != len(outtexts):
            raise ValueError("Input texts and output texts must have the same length.")
        for i in tqdm(range(len(intexts))):
            # 自立語系列
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
            # 完成された文
            inputs = tokenizer(outtexts[i], add_special_tokens=True)
            self.data[-1]['input_ids'] += inputs['input_ids'][:-1]
            self.data[-1]['attention_mask'] += inputs['attention_mask'][:-1]
            self.labels[-1]['input_ids'] += inputs['input_ids']
            self.labels[-1]['attention_mask'] += inputs['attention_mask']
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        d = self.data[idx]
        if self.transform:
            d = self.transform(d)
        l = self.labels[idx]
        if self.target_transform:
            l = self.target_transform(l)
        return d, l

train_dataset = TextDataset(df_train[src_col].apply(lambda x: ''.join(x[2:-2].split("', '"))).tolist(), df_train[tgt_col].tolist())
print('train_dataset:', len(train_dataset))
valid_dataset = TextDataset(df_valid[src_col].apply(lambda x: ''.join(x[2:-2].split("', '"))).tolist(), df_valid[tgt_col].tolist())
print('valid_dataset:', len(valid_dataset))
test_dataset = TextDataset(df_test[src_test_col].apply(lambda x: ''.join(x[2:-2].split("', '"))).tolist(), df_test[tgt_col].tolist())
print('test_dataset:', len(test_dataset))

# %%
print(train_dataset[0][0])
print(train_dataset[0][1])
for i in range(len(train_dataset[0][0]['input_ids'])):
    print(f"{i:03d}: {train_dataset[0][0]['input_ids'][i]:5d}  {tokenizer.convert_ids_to_tokens(train_dataset[0][0]['input_ids'][i])}    |   {train_dataset[0][1]['input_ids'][i]:5d}  {tokenizer.convert_ids_to_tokens(train_dataset[0][1]['input_ids'][i])}")

# %%
batch_size = config['training']['batch_size']
accumulation_steps = config['training']['accumulation_steps']  # 有効バッチサイズは維持

print('For Teacher Forcing Training')
def collate_fnTF(batch):
    data_list = {'input_ids': [], 'attention_mask': []}
    label_list = {'input_ids': [], 'attention_mask': []}
    # ミニバッチの中のデータ(d)とラベル(l)を一つずつ取り出してリストに追加
    for d, l in batch:
        data_list['input_ids'].append(torch.tensor(d['input_ids'], dtype=torch.int))
        data_list['attention_mask'].append(torch.tensor(d['attention_mask'], dtype=torch.int))
        label_list['input_ids'].append(torch.tensor(l['input_ids'], dtype=torch.long))
        label_list['attention_mask'].append(torch.tensor(l['attention_mask'], dtype=torch.long))
    # [pad]を詰めて長さを揃える
    src_batch = {'input_ids': torch.nn.utils.rnn.pad_sequence(data_list['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id, padding_side='right'), 
                 'attention_mask': torch.nn.utils.rnn.pad_sequence(data_list['attention_mask'], batch_first=True, padding_value=0)}
    tgt_batch = {'input_ids': torch.nn.utils.rnn.pad_sequence(label_list['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id, padding_side='right'), 
                 'attention_mask': torch.nn.utils.rnn.pad_sequence(label_list['attention_mask'], batch_first=True, padding_value=0)}
    return src_batch, tgt_batch

train_dataloaderTF = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fnTF, shuffle=True)
print('Train:', len(train_dataset), 'samples /', len(train_dataloaderTF), 'batches')
valid_dataloaderTF = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fnTF, shuffle=False)
print('Valid:', len(valid_dataset), 'samples /', len(valid_dataloaderTF), 'batches')
test_dataloaderTF = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fnTF, shuffle=False)
print('Test: ', len(test_dataset), 'samples /', len(test_dataloaderTF), 'batches')

# %%
print('For Non-teacher Forcing Training')
def collate_fnNTF(batch):
    data_list = {'input_ids': [], 'attention_mask': []}
    label_list = {'input_ids': [], 'attention_mask': []}
    # ミニバッチの中のデータ(d)とラベル(l)を一つずつ取り出してリストに追加
    for d, l in batch:
        print(d['input_ids'], tokenizer.convert_ids_to_tokens(d['input_ids']))
        i = d['input_ids'].index(tokenizer.sep_token_id) + 1
        d['input_ids'] = d['input_ids'][:i]
        d['attention_mask'] = d['attention_mask'][:i]
        data_list['input_ids'].append(torch.tensor(d['input_ids'], dtype=torch.int))
        data_list['attention_mask'].append(torch.tensor(d['attention_mask'], dtype=torch.int))
        print(l['input_ids'], tokenizer.convert_ids_to_tokens(l['input_ids']))
        i = l['input_ids'].index(tokenizer.sep_token_id) + 1
        l['input_ids'] = l['input_ids'][i:]
        l['attention_mask'] = l['attention_mask'][i:]
        label_list['input_ids'].append(torch.tensor(l['input_ids'], dtype=torch.long))
        label_list['attention_mask'].append(torch.tensor(l['attention_mask'], dtype=torch.long))
    # [pad]を詰めて長さを揃える
    src_batch = {'input_ids': torch.nn.utils.rnn.pad_sequence(data_list['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id, padding_side='left'), 
                 'attention_mask': torch.nn.utils.rnn.pad_sequence(data_list['attention_mask'], batch_first=True, padding_value=0)}
    tgt_batch = {'input_ids': torch.nn.utils.rnn.pad_sequence(label_list['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id, padding_side='right'), 
                 'attention_mask': torch.nn.utils.rnn.pad_sequence(label_list['attention_mask'], batch_first=True, padding_value=0)}
    return src_batch, tgt_batch

train_dataloaderNTF = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fnNTF, shuffle=True)
print('Train:', len(train_dataset), 'samples /', len(train_dataloaderTF), 'batches')
valid_dataloaderNTF = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate_fnNTF, shuffle=False)
print('Valid:', len(valid_dataset), 'samples /', len(valid_dataloaderTF), 'batches')
test_dataloaderNTF = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fnNTF, shuffle=False)
print('Test: ', len(test_dataset), 'samples /', len(test_dataloaderTF), 'batches')

# %%
# Inference mode (非teacher forcing)で使う model.generate() の使用例
# Inference mode は未実装

for batch in train_dataloaderNTF:
    print(batch[0]['input_ids'].shape[1], batch[1]['input_ids'].shape[1])
    outputs = model.generate(batch[0]['input_ids'].to(device), attention_mask=batch[0]['attention_mask'].to(device), 
                             pad_token_id=tokenizer.pad_token_id, 
                             max_new_tokens=max_new_tokens, 
                             return_dict_in_generate=True, 
                             output_scores=True)
    break

for i in range(len(outputs.sequences)):
    for j in range(len(outputs.scores)):
        print(tokenizer.decode(outputs.scores[j][i].argmax().item(), skip_special_tokens=False), end=' ')
    print('')

# %%
teacher_forcing_ratio = config['training']['teacher_forcing_ratio']


# CE損失
criterionCE = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# 微分可能なBLEU（DEBLEU: Differentiable Expected BLEU）損失
# 論文: "DIFFERENTIABLE EXPECTED BLEU FOR TEXT GENERATION" (ICLR 2019)
def criterionBLEU(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pad_token_id: int = tokenizer.pad_token_id,
    sep_token_id: int = tokenizer.sep_token_id,
    target_only: bool = True,
    N: int = 4,
    weights: tuple = (0.1, 0.3, 0.3, 0.3),
    return_loss: bool = True,
    return_score: bool = False,
    eps: float = 1e-8,
):
    """
    DEBLEU (Differentiable Expected BLEU) 損失・スコア計算関数

    Args:
        logits: モデル出力ロジット [batch, seq_len, vocab_size] または [batch, vocab_size, seq_len]
        labels: 正解ラベル [batch, seq_len]
        pad_token_id: パディングトークンID
        sep_token_id: プロンプトと生成文の区切りトークンID ([SEP])
        target_only: Trueの場合、[SEP]以降の生成文トークンのみを対象にBLEUを計算 (評価時の一致)
        N: 最大n-gram次数 (論文では N=4)
        weights: n-gram重み (論文第4節の推奨値: w1=0.1, w2=w3=w4=0.3)
        return_loss: Trueの場合、DEBLEU損失 (式17: -log BP - sum w_n log prec_n) を返す
        return_score: Trueの場合、期待BLEUスコア (式16) も同時に返す
        eps: log計算の数値安定化用微小値
    """
    if logits.ndim != 3:
        raise ValueError(f"Expected 3D logits tensor, got shape {logits.shape}")

    # AMP (float16) のアンダーフローやNaNを防ぐため、内部計算はfloat32にキャスト
    logits_f32 = logits.float()

    # テンソル形状の正規化 -> [B, T, V]
    if logits_f32.shape[1] == labels.shape[1]:
        probs = logits_f32
    elif logits_f32.shape[2] == labels.shape[1]:
        probs = logits_f32.transpose(1, 2)
    else:
        probs = logits_f32

    # 未正規化ロジットの場合はsoftmaxを適用
    is_prob = (probs >= 0.0).all() and torch.allclose(
        probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)), atol=1e-2
    )
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

        # 有効トークンマスク（パディング除外）
        valid_mask = sample_labels.ne(pad_token_id)

        # target_onlyがTrueかつ[SEP]が存在する場合、[SEP]より後の完成文のみを対象とする
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

        # Brevity Penalty (BP) の計算 (式1)
        if T < T_star:
            log_bp = torch.tensor(1.0 - T_star / T, device=device, dtype=dtype)
            bp = float(torch.exp(log_bp))
        else:
            log_bp = torch.tensor(0.0, device=device, dtype=dtype)
            bp = 1.0

        # M^1: M^1_{i, j} = p(y_i = y^*_j) = P_{i, y^*_j} (Appendix A, 式18)
        M = p[:, ref]  # [T, T_star]

        # 参照文同士の一致行列 R^1: R^1_{j1, j2} = 1[y^*_{j1} == y^*_{j2}]
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
                # 再帰的計算: M^n = M^{n-1}[:-1, :-1] * M[n-1:, n-1:] (Appendix A, 式20)
                M_n = M_n_prev[:-1, :-1] * M[n - 1:, n - 1:]
                R_n = R_n_prev[:-1, :-1] * R[n - 1:, n - 1:]

            # C_n[j]: 参照文におけるn-gram y^*_{j:j+n}の出現回数 (式3, 9)
            C_n = R_n.sum(dim=0).clamp(min=1.0)  # [cur_T_star]

            # 列和 S_j = sum_{i'=0}^{cur_T-1} M^n_{i', j}
            S = M_n.sum(dim=0, keepdim=True)  # [1, cur_T_star]

            # 分母: 1 + sum_{i' != i} p(y_{i':i'+n} = y^*_{j:j+n}) (式13, 14)
            denom = 1.0 + (S - M_n)  # [cur_T, cur_T_star]

            # o_tilde_{n, i} の計算 (式14):
            # (M^n / C_n) * min(1, C_n / denom) = M^n * min(1 / C_n, 1 / denom)
            C_inv = 1.0 / C_n.unsqueeze(0)
            denom_inv = 1.0 / denom
            term = M_n * torch.minimum(C_inv, denom_inv)
            o_tilde = term.sum(dim=1)  # [cur_T]

            # n-gram精度 prec_n (式15):
            prec_n = o_tilde.mean()
            precisions.append(prec_n)
            log_precisions.append(torch.log(torch.clamp(prec_n, min=eps)))

            M_n_prev = M_n
            R_n_prev = R_n

        actual_N = len(precisions)
        if actual_N == 0:
            continue

        # 実効次数に応じた重みの正規化
        cur_weights = weights[:actual_N]
        w_sum = sum(cur_weights)
        cur_weights = [w / w_sum for w in cur_weights]

        # DEBLEU損失 (式17): -log BP - sum_{n=1}^N w_n log(prec_n)
        sample_loss = -log_bp - sum(w * lp for w, lp in zip(cur_weights, log_precisions))
        # 期待BLEUスコア (式16): BP * prod_{n=1}^N (prec_n)^{w_n}
        sample_score = bp * torch.exp(sum(w * lp for w, lp in zip(cur_weights, log_precisions)))

        batch_losses.append(sample_loss)
        batch_scores.append(sample_score)

    if not batch_losses:
        zero_val = logits.new_zeros(())
        if return_loss and return_score:
            return zero_val, zero_val
        return zero_val

    total_loss = torch.stack(batch_losses).mean().to(logits.dtype)
    total_score = torch.stack(batch_scores).mean().to(logits.dtype)

    if return_loss and return_score:
        return total_loss, total_score
    elif return_loss:
        return total_loss
    else:
        return total_score

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_bleu_loss = 0
    total_bleu_score = 0
    bleu_count = 0
    optimizer.zero_grad(set_to_none=True)

    # configのalpha_scheduleからalphaを取得
    alpha = get_alpha(epoch)
    print(f"[Epoch {epoch}] alpha = {alpha}")

    for step, batch in enumerate(tqdm(train_dataloaderTF, desc="Train")):
        use_teacher_forcing = True if np.random.rand() < teacher_forcing_ratio else False
        inputs = batch[0]['input_ids'].to(device)
        attention_mask = batch[0]['attention_mask'].to(device)
        labels = batch[1]['input_ids'].to(device)
        
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(input_ids=inputs, attention_mask=attention_mask)
            ce_loss = criterionCE(torch.transpose(outputs.logits, 1, 2), labels)
            
            if alpha > 0.0:
                bleu_loss, bleu_score = criterionBLEU(
                    outputs.logits,
                    labels,
                    pad_token_id=tokenizer.pad_token_id,
                    sep_token_id=tokenizer.sep_token_id,
                    target_only=bleu_target_only,
                    N=bleu_max_ngram,
                    weights=bleu_weights,
                    return_loss=True,
                    return_score=True,
                )
                loss = ce_loss + alpha * bleu_loss
                total_bleu_loss += bleu_loss.item()
                total_bleu_score += bleu_score.item()
                bleu_count += 1
            else:
                loss = ce_loss

        scaler.scale(loss / accumulation_steps).backward()
        if (step + 1) % accumulation_steps == 0 or step + 1 == len(train_dataloaderTF):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()

    avg_train_loss = total_loss / len(train_dataloaderTF)
    avg_ce_loss = total_ce_loss / len(train_dataloaderTF)
    avg_bleu_loss = (total_bleu_loss / bleu_count) if bleu_count > 0 else 0.0
    avg_bleu_score = (total_bleu_score / bleu_count) if bleu_count > 0 else 0.0

    # 検証ループ
    model.eval()
    eval_loss = 0
    eval_ce_loss = 0
    eval_bleu_loss = 0
    eval_bleu_score = 0
    eval_bleu_count = 0
    with torch.no_grad():
        for batch in tqdm(valid_dataloaderTF, desc="Valid"):
            inputs = batch[0]['input_ids'].to(device)
            attention_mask = batch[0]['attention_mask'].to(device)
            labels = batch[1]['input_ids'].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(input_ids=inputs, attention_mask=attention_mask)
                ce_loss = criterionCE(torch.transpose(outputs.logits, 1, 2), labels)
                if alpha > 0.0:
                    bleu_loss, bleu_score = criterionBLEU(
                        outputs.logits,
                        labels,
                        pad_token_id=tokenizer.pad_token_id,
                        sep_token_id=tokenizer.sep_token_id,
                        target_only=bleu_target_only,
                        N=bleu_max_ngram,
                        weights=bleu_weights,
                        return_loss=True,
                        return_score=True,
                    )
                    loss = ce_loss + alpha * bleu_loss
                    eval_bleu_loss += bleu_loss.item()
                    eval_bleu_score += bleu_score.item()
                    eval_bleu_count += 1
                else:
                    loss = ce_loss
            eval_loss += loss.item()
            eval_ce_loss += ce_loss.item()
    avg_eval_loss = eval_loss / len(valid_dataloaderTF)
    avg_eval_ce_loss = eval_ce_loss / len(valid_dataloaderTF)
    avg_eval_bleu_score = (eval_bleu_score / eval_bleu_count) if eval_bleu_count > 0 else 0.0

    if alpha > 0.0:
        print(f"Epoch {epoch}: Avg Train Loss = {avg_train_loss:.4f} (CE: {avg_ce_loss:.4f}, DEBLEU Loss: {avg_bleu_loss:.4f}, Score: {avg_bleu_score:.4f}) / Avg Eval Loss: {avg_eval_loss:.4f} (Eval Score: {avg_eval_bleu_score:.4f})")
    else:
        print(f"Epoch {epoch}: Avg Train Loss = {avg_train_loss:.4f} (CE: {avg_ce_loss:.4f}) / Avg Eval Loss: {avg_eval_loss:.4f}")

    # モデル保存判定（指定エポック毎 または 最終エポック）
    current_epoch = epoch + 1
    is_final_epoch = (current_epoch == num_epochs)
    is_interval_epoch = save_intermediate_checkpoints and (current_epoch % save_epoch_interval == 0)

    if is_interval_epoch or is_final_epoch:
        save_d = f"{save_base_dir}/{save_prefix}_{current_epoch}"
        os.makedirs(save_d, exist_ok=True)
        reason = "最終エポック" if is_final_epoch and not is_interval_epoch else f"指定間隔 (interval={save_epoch_interval})"
        print(f"Epoch {current_epoch}/{num_epochs} のモデル保存を実行 ({reason}): {save_d}")
        model.save_pretrained(save_d)
        tokenizer.save_pretrained(save_d)
        torch.save(optimizer.state_dict(), f"{save_d}/optimizer.pt")
        with open(f"{save_d}/meta.txt", "w") as f:
            f.write(f"epoch={current_epoch}\n")
            f.write(f"avg_train_loss={avg_train_loss:.4f}\n")
            f.write(f"avg_eval_loss={avg_eval_loss:.4f}\n")
            f.write(f"alpha={alpha}\n")




