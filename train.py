# %% [markdown]
# 必要なパッケージの読み込み、使用モデル等の指定

# %%
import os
# CUDAアロケータの断片化防止
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from tqdm import tqdm
import time
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import random
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
parser.add_argument("--batch_size", type=int, default=None, help="バッチサイズ (学習時)")
parser.add_argument("--eval_batch_size", type=int, default=None, help="検証・テスト時のバッチサイズ (例: 2, 4, 8)")
parser.add_argument("--generation_mode", type=str, default=None, choices=["argmax", "sample", "gumbel"],
                    help="自律生成時のトークン選択手法 (argmax/sample/gumbel)")
parser.add_argument("--eval_generation_mode", type=str, default=None, choices=["argmax", "sample", "gumbel"],
                    help="検証時のトークン選択手法 (argmax/sample/gumbel)")
parser.add_argument("--teacher_forcing_ratio", type=float, default=None,
                    help="Teacher Forcingの適用割合 (0.0で完全自律生成)")
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
if args.eval_batch_size is not None:
    config.setdefault('training', {})['eval_batch_size'] = args.eval_batch_size
if args.alpha is not None:
    config.setdefault('bleu', {})['alpha_schedule'] = {0: args.alpha}
if args.generation_mode is not None:
    config.setdefault('training', {})['generation_mode'] = args.generation_mode
if args.eval_generation_mode is not None:
    config.setdefault('training', {})['eval_generation_mode'] = args.eval_generation_mode
if args.teacher_forcing_ratio is not None:
    config.setdefault('training', {})['teacher_forcing_ratio'] = args.teacher_forcing_ratio

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
valid_subset_size = data_config.get('valid_subset_size', False)
valid_subset_seed = data_config.get('valid_subset_seed', 2025)
ADD_BOS_TOKEN = data_config.get('add_bos_token', True)
src_col = data_config.get('src_column', 'j_lists')
src_test_col = data_config.get('src_test_column', 'j_lists_test')
tgt_col = data_config.get('tgt_column', 'sentence')

# 学習設定
training_config = config.get('training', {})
batch_size = training_config['batch_size']
eval_batch_size = int(training_config.get('eval_batch_size', training_config.get('valid_batch_size', 4)))
accumulation_steps = training_config.get('accumulation_steps', 1)
learning_rate = float(training_config['learning_rate'])
weight_decay = float(training_config.get('weight_decay', 0.01))
teacher_forcing_ratio = float(training_config.get('teacher_forcing_ratio', 0.0))
generation_mode = training_config.get('generation_mode', 'argmax')
eval_generation_mode = training_config.get('eval_generation_mode', 'argmax')
mask_schedule = training_config.get('mask_schedule') or bleu_config.get('mask_schedule')

def parse_mask_pattern(pattern_str: str) -> tuple:
    """
    Teacher mask パターン文字列 (例: '2:2', '0:1', '1:0') をパース
    戻り値: (unmasked_steps, masked_steps)
    """
    if not pattern_str or ":" not in str(pattern_str):
        return (1, 0)
    parts = str(pattern_str).strip().split(":")
    return int(parts[0]), int(parts[1])

def get_mask_pattern(epoch_idx: int):
    """configのmask_scheduleから指定エポックのteacher maskパターンを取得"""
    if mask_schedule is None:
        return None
    if isinstance(mask_schedule, dict):
        matched = [int(k) for k in mask_schedule.keys() if int(k) <= epoch_idx]
        if matched:
            best_k = max(matched)
            val = mask_schedule.get(best_k, mask_schedule.get(str(best_k), "1:0"))
            return str(val)
        return "1:0"
    elif isinstance(mask_schedule, list):
        if epoch_idx < len(mask_schedule):
            return str(mask_schedule[epoch_idx])
        return str(mask_schedule[-1])
    return str(mask_schedule)

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

if valid_subset_size and isinstance(valid_subset_size, int) and len(df_valid) > valid_subset_size:
    df_valid = df_valid.sample(n=valid_subset_size, random_state=valid_subset_seed).reset_index(drop=True)
    print(f"Validation dataset downsampled to {len(df_valid)} samples (seed={valid_subset_seed}).")

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

# AMP (GradScaler) を使用するため、モデルのマスターパラメータを float32 で保持
# (rinna/japanese-gpt-1b 等、デフォルトで float16 のモデルの GradScaler エラーを防止)
if model.dtype == torch.float16:
    model = model.float()

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
eval_batch_size = int(config['training'].get('eval_batch_size', config['training'].get('valid_batch_size', 4)))
accumulation_steps = config['training']['accumulation_steps']  # 有効バッチサイズは維持

print('For Autonomous Generation (Non-teacher Forcing) & Teacher Forcing')

def collate_fnNTF(batch):
    """
    自律生成 (Non-teacher Forcing) 用の collate 関数:
    - src_batch: 入力部（[BOS] + 自立語系列 + [SEP]）を left-padding
    - tgt_batch: 完成文（正解トークン列）を right-padding
    """
    data_list = {'input_ids': [], 'attention_mask': []}
    label_list = {'input_ids': [], 'attention_mask': []}
    for d, l in batch:
        # 入力部: [SEP] トークンまでを取得 (元データを破壊しないよう安全にスライス)
        src_sep_idx = d['input_ids'].index(tokenizer.sep_token_id) + 1
        src_ids = d['input_ids'][:src_sep_idx]
        src_mask = d['attention_mask'][:src_sep_idx]
        data_list['input_ids'].append(torch.tensor(src_ids, dtype=torch.long))
        data_list['attention_mask'].append(torch.tensor(src_mask, dtype=torch.long))

        # 完成文 (正解ターゲット): [SEP] の直後から末尾までを取得
        tgt_sep_idx = l['input_ids'].index(tokenizer.sep_token_id) + 1
        tgt_ids = l['input_ids'][tgt_sep_idx:]
        tgt_mask = l['attention_mask'][tgt_sep_idx:]
        label_list['input_ids'].append(torch.tensor(tgt_ids, dtype=torch.long))
        label_list['attention_mask'].append(torch.tensor(tgt_mask, dtype=torch.long))

    # CausalLMでの生成用に、入力部は left-padding、ターゲットは right-padding
    src_batch = {
        'input_ids': torch.nn.utils.rnn.pad_sequence(data_list['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id, padding_side='left'), 
        'attention_mask': torch.nn.utils.rnn.pad_sequence(data_list['attention_mask'], batch_first=True, padding_value=0, padding_side='left')
    }
    tgt_batch = {
        'input_ids': torch.nn.utils.rnn.pad_sequence(label_list['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id, padding_side='right'), 
        'attention_mask': torch.nn.utils.rnn.pad_sequence(label_list['attention_mask'], batch_first=True, padding_value=0, padding_side='right')
    }
    return src_batch, tgt_batch

def collate_fnTF(batch):
    """従来の Teacher Forcing 用 collate 関数 (互換性用)"""
    data_list = {'input_ids': [], 'attention_mask': []}
    label_list = {'input_ids': [], 'attention_mask': []}
    for d, l in batch:
        data_list['input_ids'].append(torch.tensor(d['input_ids'], dtype=torch.long))
        data_list['attention_mask'].append(torch.tensor(d['attention_mask'], dtype=torch.long))
        label_list['input_ids'].append(torch.tensor(l['input_ids'], dtype=torch.long))
        label_list['attention_mask'].append(torch.tensor(l['attention_mask'], dtype=torch.long))
    src_batch = {
        'input_ids': torch.nn.utils.rnn.pad_sequence(data_list['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id, padding_side='right'), 
        'attention_mask': torch.nn.utils.rnn.pad_sequence(data_list['attention_mask'], batch_first=True, padding_value=0)
    }
    tgt_batch = {
        'input_ids': torch.nn.utils.rnn.pad_sequence(label_list['input_ids'], batch_first=True, padding_value=tokenizer.pad_token_id, padding_side='right'), 
        'attention_mask': torch.nn.utils.rnn.pad_sequence(label_list['attention_mask'], batch_first=True, padding_value=0)
    }
    return src_batch, tgt_batch

train_dataloaderNTF = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fnNTF, shuffle=True)
valid_dataloaderNTF = torch.utils.data.DataLoader(valid_dataset, batch_size=eval_batch_size, collate_fn=collate_fnNTF, shuffle=False)
test_dataloaderNTF = torch.utils.data.DataLoader(test_dataset, batch_size=eval_batch_size, collate_fn=collate_fnNTF, shuffle=False)

train_dataloaderTF = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fnTF, shuffle=True)
valid_dataloaderTF = torch.utils.data.DataLoader(valid_dataset, batch_size=eval_batch_size, collate_fn=collate_fnTF, shuffle=False)
test_dataloaderTF = torch.utils.data.DataLoader(test_dataset, batch_size=eval_batch_size, collate_fn=collate_fnTF, shuffle=False)

# 自律生成用のメインDataLoader
train_dataloader = train_dataloaderNTF
valid_dataloader = valid_dataloaderNTF
test_dataloader = test_dataloaderNTF

print('Train (NTF):', len(train_dataset), 'samples /', len(train_dataloader), 'batches')
print('Valid (NTF):', len(valid_dataset), 'samples /', len(valid_dataloader), 'batches')
print('Test  (NTF):', len(test_dataset), 'samples /', len(test_dataloader), 'batches')

def autonomous_generate_stepwise(
    model,
    src_input_ids: torch.Tensor,
    src_attention_mask: torch.Tensor,
    max_steps: int,
    generation_mode: str = "argmax",
    temperature: float = 1.0,
    gumbel_hard: bool = True,
) -> torch.Tensor:
    """
    自己回帰（自律）逐次文生成関数:
    - 最初の1ステップ目は入力部のみを参考に出力トークンを生成
    - 以降のステップでは入力部 ＋ 直前までのモデル自身の生成結果を参考にしてトークン出力を行う
    - generation_mode='gumbel' の場合:
      Gumbel-Softmax による soft token 分布と Embedding 行列の積により inputs_embeds を生成・連結し、
      自己回帰系列全体にわたる勾配逆伝播 (BPTT) を可能にする (論文 DEBLEU Section 3.3, Fig. 2)。
    - generation_mode='argmax' / 'sample' の場合:
      離散トークンID列 (input_ids) を用いて省メモリかつ高速に逐次生成を行う。

    Args:
        model: CausalLMモデル
        src_input_ids: 入力部（プロンプト）のトークンID列 [batch_size, prompt_len] (left padding)
        src_attention_mask: 入力部のアテンションマスク [batch_size, prompt_len] (left padding)
        max_steps: 生成するステップ数（正解文長 T*）
        generation_mode: トークン選択手法 ('argmax', 'sample', 'gumbel')
        temperature: サンプリング・Gumbel-Softmax時の温度パラメータ
        gumbel_hard: generation_mode='gumbel' のとき Straight-Through (hard=True) を適用するか否か

    Returns:
        generated_logits: 各ステップで出力されたロジットテンソル [batch_size, max_steps, vocab_size]
    """
    batch_size = src_input_ids.size(0)
    device = src_input_ids.device
    curr_attention_mask = src_attention_mask
    logits_list = []

    if generation_mode == "gumbel":
        embed_layer = model.get_input_embeddings()
        curr_inputs_embeds = embed_layer(src_input_ids)

        for t in range(max_steps):
            outputs = model(inputs_embeds=curr_inputs_embeds, attention_mask=curr_attention_mask)
            step_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            logits_list.append(step_logits)

            if t < max_steps - 1:
                # Gumbel-Softmax による soft token 分布を取得 [batch_size, vocab_size]
                soft_dist = torch.nn.functional.gumbel_softmax(
                    step_logits, tau=max(temperature, 1e-5), hard=gumbel_hard
                )
                # soft token @ W_emb -> [batch_size, 1, hidden_size]
                next_embed = torch.matmul(soft_dist.unsqueeze(1), embed_layer.weight)
                curr_inputs_embeds = torch.cat([curr_inputs_embeds, next_embed], dim=1)
                curr_attention_mask = torch.cat(
                    [curr_attention_mask, torch.ones((batch_size, 1), device=device, dtype=curr_attention_mask.dtype)],
                    dim=1
                )
    else:
        curr_input_ids = src_input_ids

        for t in range(max_steps):
            outputs = model(input_ids=curr_input_ids, attention_mask=curr_attention_mask)
            step_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            logits_list.append(step_logits)

            if t < max_steps - 1:
                if generation_mode == "sample":
                    probs = torch.softmax(step_logits / max(temperature, 1e-5), dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:  # argmax
                    next_token = step_logits.argmax(dim=-1, keepdim=True)

                curr_input_ids = torch.cat([curr_input_ids, next_token], dim=1)
                curr_attention_mask = torch.cat(
                    [curr_attention_mask, torch.ones((batch_size, 1), device=device, dtype=curr_attention_mask.dtype)],
                    dim=1
                )

    return torch.stack(logits_list, dim=1)


def decode_with_teacher_mask(
    model,
    src_input_ids: torch.Tensor,
    src_attention_mask: torch.Tensor,
    labels: torch.Tensor,
    max_steps: int,
    mask_pattern: str = "1:0",
    generation_mode: str = "gumbel",
    temperature: float = 1.0,
    gumbel_hard: bool = True,
    random_shift: bool = True,
    pad_token_id: int = tokenizer.pad_token_id,
) -> tuple:
    """
    Teacher Mask付きデコーディング関数 (論文 DEBLEU Section 3.3, Fig. 2):
    - mask_pattern ("#unmasked:#masked") に従って、maskedステップとunmaskedステップを交互に適用
    - maskedステップ:
      次ステップ入力として正解トークン (ground truth) を与え、
      DEBLEU評価用の確率分布としても正解トークンのone-hot表現を使用する。
    - unmaskedステップ:
      次ステップ入力としてモデル自身の出力 (Gumbel-Softmaxソフトトークン等) を与え、
      DEBLEU評価用の確率分布としてもモデル予測分布を使用する。
    - ランダムシフト: 各イテレーションでマスク開始位置をランダムにシフトして正則化・汎化性を向上。
    - 特殊ケース "0:1" (全ステップmasked):
      すべてのステップが正解トークン参照 (Teacher Forcing) となるため、並列forwardにより高速に計算。

    Args:
        model: CausalLMモデル
        src_input_ids: 入力部（プロンプト）のトークンID列 [batch_size, prompt_len] (left padding)
        src_attention_mask: 入力部のアテンションマスク [batch_size, prompt_len] (left padding)
        labels: 正解文のトークンID列 [batch_size, max_steps]
        max_steps: 生成ステップ数（正解文長 T*）
        mask_pattern: "unmasked:masked" 形式の文字列 (例: "0:1", "2:2", "4:2", "1:0")
        generation_mode: トークン選択手法 ('argmax', 'sample', 'gumbel')
        temperature: Gumbel-Softmax / サンプリング時の温度
        gumbel_hard: Gumbel-Softmax時にStraight-Through (hard=True) を適用するか否か
        random_shift: イテレーションごとのマスク位置ランダムシフトを行うか否か
        pad_token_id: パディングトークンID

    Returns:
        generated_logits: モデル出力の生ロジット [batch_size, max_steps, vocab_size] (CE損失計算用)
        debleu_probs: DEBLEU計算用確率テンソル [batch_size, max_steps, vocab_size] (maskedステップはone-hot)
    """
    batch_size = src_input_ids.size(0)
    device = src_input_ids.device
    embed_layer = model.get_input_embeddings()
    vocab_size = embed_layer.weight.size(0)

    u, m = parse_mask_pattern(mask_pattern)

    # 1. 完全なTeacher Forcing ("0:1" 等、unmasked_steps == 0) の場合は一括並列forwardで高速化
    if u == 0:
        prompt_len = src_input_ids.size(1)
        if max_steps > 1:
            tf_inputs = torch.cat([src_input_ids, labels[:, :max_steps - 1]], dim=1)
            tf_mask = torch.cat([
                src_attention_mask,
                (labels[:, :max_steps - 1] != pad_token_id).to(src_attention_mask.dtype)
            ], dim=1)
        else:
            tf_inputs = src_input_ids
            tf_mask = src_attention_mask

        outputs = model(input_ids=tf_inputs, attention_mask=tf_mask)
        generated_logits = outputs.logits[:, prompt_len - 1 : prompt_len - 1 + max_steps, :]
        debleu_probs = torch.nn.functional.one_hot(labels[:, :max_steps], num_classes=vocab_size).to(generated_logits.dtype)
        return generated_logits, debleu_probs

    # 2. Teacher Mask ("2:2", "4:2", "1:0" 等) の逐次デコーディング
    C = u + m
    offset = random.randint(0, C - 1) if (random_shift and C > 0) else 0

    curr_inputs_embeds = embed_layer(src_input_ids)
    curr_attention_mask = src_attention_mask
    logits_list = []
    probs_list = []

    for t in range(max_steps):
        is_masked = (m > 0 and ((t + offset) % C) >= u)

        outputs = model(inputs_embeds=curr_inputs_embeds, attention_mask=curr_attention_mask)
        step_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
        logits_list.append(step_logits)

        # DEBLEU評価用の確率分布 (maskedステップは正解one-hot、unmaskedステップは予測確率)
        if is_masked:
            gt_token = labels[:, t]
            step_prob = torch.nn.functional.one_hot(gt_token, num_classes=vocab_size).to(step_logits.dtype)
        else:
            if generation_mode == "gumbel":
                step_prob = torch.nn.functional.gumbel_softmax(step_logits, tau=max(temperature, 1e-5), hard=gumbel_hard)
            else:
                step_prob = torch.softmax(step_logits, dim=-1)
        probs_list.append(step_prob)

        # 最終ステップでなければ次ステップの入力埋め込みを作成して連結
        if t < max_steps - 1:
            if is_masked:
                # maskedステップ: 正解トークンの埋め込みを入力
                gt_token = labels[:, t].unsqueeze(1)
                next_embed = embed_layer(gt_token)
            else:
                # unmaskedステップ: モデル出力の埋め込みを入力
                if generation_mode == "gumbel":
                    next_embed = torch.matmul(step_prob.unsqueeze(1), embed_layer.weight)
                elif generation_mode == "sample":
                    probs = torch.softmax(step_logits / max(temperature, 1e-5), dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    next_embed = embed_layer(next_token)
                else:  # argmax
                    next_token = step_logits.argmax(dim=-1, keepdim=True)
                    next_embed = embed_layer(next_token)

            curr_inputs_embeds = torch.cat([curr_inputs_embeds, next_embed], dim=1)
            curr_attention_mask = torch.cat(
                [curr_attention_mask, torch.ones((batch_size, 1), device=device, dtype=curr_attention_mask.dtype)],
                dim=1
            )

    generated_logits = torch.stack(logits_list, dim=1)
    debleu_probs = torch.stack(probs_list, dim=1)
    return generated_logits, debleu_probs


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
        M = p.index_select(dim=1, index=ref)  # [T, T_star]

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
    # configのmask_scheduleからteacher maskパターンを取得
    cur_mask_pattern = get_mask_pattern(epoch)
    print(f"[Epoch {epoch}] alpha = {alpha}, mask_pattern = {cur_mask_pattern if cur_mask_pattern is not None else '(teacher_forcing_ratio)'}")

    for step, batch in enumerate(tqdm(train_dataloader, desc="Train")):
        src_inputs = batch[0]['input_ids'].to(device)
        src_attention_mask = batch[0]['attention_mask'].to(device)
        labels = batch[1]['input_ids'].to(device)
        target_len = labels.shape[1]
        
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            if cur_mask_pattern is not None:
                # 論文 DEBLEU Section 3.3: Teacher Mask によるデコーディング
                generated_logits, debleu_probs = decode_with_teacher_mask(
                    model=model,
                    src_input_ids=src_inputs,
                    src_attention_mask=src_attention_mask,
                    labels=labels,
                    max_steps=target_len,
                    mask_pattern=cur_mask_pattern,
                    generation_mode=generation_mode,
                )
            else:
                # 従来の teacher_forcing_ratio によるフォールバック
                use_teacher_forcing = (np.random.rand() < teacher_forcing_ratio) if teacher_forcing_ratio > 0.0 else False
                if use_teacher_forcing:
                    # Teacher Forcing: 入力部 + 直前の正解参照文トークン
                    prompt_len = src_inputs.shape[1]
                    if target_len > 1:
                        tf_inputs = torch.cat([src_inputs, labels[:, :-1]], dim=1)
                        tf_attention_mask = torch.cat([src_attention_mask, (labels[:, :-1] != tokenizer.pad_token_id).to(src_attention_mask.dtype)], dim=1)
                    else:
                        tf_inputs = src_inputs
                        tf_attention_mask = src_attention_mask
                    outputs = model(input_ids=tf_inputs, attention_mask=tf_attention_mask)
                    generated_logits = outputs.logits[:, prompt_len - 1 : prompt_len - 1 + target_len, :]
                else:
                    # 自律逐次生成
                    generated_logits = autonomous_generate_stepwise(
                        model=model,
                        src_input_ids=src_inputs,
                        src_attention_mask=src_attention_mask,
                        max_steps=target_len,
                        generation_mode=generation_mode,
                    )
                debleu_probs = generated_logits

            ce_loss = criterionCE(torch.transpose(generated_logits, 1, 2), labels)
            
            if alpha > 0.0:
                bleu_loss, bleu_score = criterionBLEU(
                    debleu_probs,
                    labels,
                    pad_token_id=tokenizer.pad_token_id,
                    sep_token_id=tokenizer.sep_token_id,
                    target_only=False,  # labelsは完成文のみ
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
        if (step + 1) % accumulation_steps == 0 or step + 1 == len(train_dataloader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()

    avg_train_loss = total_loss / len(train_dataloader)
    avg_ce_loss = total_ce_loss / len(train_dataloader)
    avg_bleu_loss = (total_bleu_loss / bleu_count) if bleu_count > 0 else 0.0
    avg_bleu_score = (total_bleu_score / bleu_count) if bleu_count > 0 else 0.0

    # 学習ループ終了後、オプティマイザの勾配をクリアしGPUキャッシュを解放して検証時のOOMを防止
    optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 検証ループ
    model.eval()
    eval_loss = 0
    eval_ce_loss = 0
    eval_bleu_loss = 0
    eval_bleu_score = 0
    eval_bleu_count = 0
    # 勾配計算の完全無効化・バージョン追跡無効化による省メモリ・高速化
    with torch.inference_mode():
        for batch in tqdm(valid_dataloader, desc="Valid"):
            src_inputs = batch[0]['input_ids'].to(device)
            src_attention_mask = batch[0]['attention_mask'].to(device)
            labels = batch[1]['input_ids'].to(device)
            target_len = labels.shape[1]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                # 検証時も自律生成で次トークンを出力・評価
                generated_logits = autonomous_generate_stepwise(
                    model=model,
                    src_input_ids=src_inputs,
                    src_attention_mask=src_attention_mask,
                    max_steps=target_len,
                    generation_mode=eval_generation_mode,
                )
                ce_loss = criterionCE(torch.transpose(generated_logits, 1, 2), labels)
                if alpha > 0.0:
                    bleu_loss, bleu_score = criterionBLEU(
                        generated_logits,
                        labels,
                        pad_token_id=tokenizer.pad_token_id,
                        sep_token_id=tokenizer.sep_token_id,
                        target_only=False,
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

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    avg_eval_loss = eval_loss / len(valid_dataloader)
    avg_eval_ce_loss = eval_ce_loss / len(valid_dataloader)
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




