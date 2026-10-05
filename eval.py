import os
import json
import yaml
import argparse
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from bert_score import score as bertscore_score

# ==========================================
# 1. 設定とモデル・データセットの読み込み
# ==========================================
parser = argparse.ArgumentParser(description="GPT-2 日本語モデル評価スクリプト (BLEU / BERTScore)")
parser.add_argument("--config", type=str, default="config.yml", help="設定ファイルのパス (デフォルト: config.yml)")
parser.add_argument("--model_dir", type=str, default=None, help="評価対象モデルディレクトリのパス (デフォルト: ./finetuned_model_1)")
parser.add_argument("--test_csv_path", type=str, default=None, help="テストデータCSVのパス (未指定時はconfig.ymlから取得)")
parser.add_argument("--output_dir", type=str, default=".", help="結果を保存するディレクトリパス")
parser.add_argument("--output_filename", type=str, default=None, help="結果CSVファイル名 (指定しない場合は自動命名)")
parser.add_argument("--shortcut", type=str, default=None, choices=["true", "false", "True", "False"],
                    help="テストデータの件数を絞るか (true/false, 未指定時はconfig.ymlの設定に従う)")
parser.add_argument("--add_bos_token", type=str, default=None, choices=["true", "false", "True", "False"],
                    help="BOSトークンを追加するか (true/false, 未指定時はconfig.ymlの設定に従う)")

args = parser.parse_args()

# config.yml の読み込み（存在する場合）
config = {}
if os.path.exists(args.config):
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}

data_config = config.get('data', {})

# パラメータの決定 (CLI引数 > config.yml > デフォルト値)
model_dir = args.model_dir if args.model_dir is not None else "./finetuned_model_1"
test_csv_path = args.test_csv_path if args.test_csv_path is not None else data_config.get('test_path', 'amazon_review/amazon_review_test.csv')
output_dir = args.output_dir

if args.shortcut is not None:
    shortcut = args.shortcut.lower() == "true"
else:
    shortcut = data_config.get('shortcut', False)

if args.add_bos_token is not None:
    ADD_BOS_TOKEN = args.add_bos_token.lower() == "true"
else:
    ADD_BOS_TOKEN = data_config.get('add_bos_token', True)

shortcut_test_size = data_config.get('shortcut_sizes', {}).get('test', 560)
src_test_col = data_config.get('src_test_column', 'j_lists_test')
tgt_col = data_config.get('tgt_column', 'sentence')

os.makedirs(output_dir, exist_ok=True)
model_name = os.path.basename(os.path.normpath(model_dir))

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# トークナイザーと学習済みモデルのロード
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_dir)
model.to(device)
model.eval()

# テストデータの読み込み
df_test = pd.read_csv(test_csv_path)
if shortcut:
    df_test = df_test[:shortcut_test_size]
print(f"Test samples: {len(df_test)} (shortcut={shortcut})")

# ==========================================
# 2. テキスト生成（Inference）
# ==========================================
generated_list = [""] * len(df_test)
pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

print("Generating sentences...")
for i in tqdm(range(len(df_test))):
    input_text = ''.join(df_test.loc[i, src_test_col][2:-2].split("', '"))
    inputs = tokenizer(input_text)
    
    if ADD_BOS_TOKEN:
        inputs['input_ids'] = [tokenizer.bos_token_id] + inputs['input_ids']
        inputs['attention_mask'] = [1] + inputs['attention_mask']
        
    inputs['input_ids'] = torch.tensor(inputs['input_ids'] + [tokenizer.sep_token_id], dtype=torch.long)[None, :]
    inputs['attention_mask'] = torch.tensor(inputs['attention_mask'] + [1], dtype=torch.long)[None, :]

    min_length = inputs["input_ids"].shape[1] + 5
    max_length = inputs["input_ids"].shape[1] * 2

    with torch.no_grad():
        output = model.generate(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            min_length=min_length,
            max_length=max_length,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.2,
            num_return_sequences=1,
            pad_token_id=pad_token_id,
        )

    input_length = inputs["input_ids"].shape[1]
    generated_tokens = output[0][input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    generated_list[i] = generated_text

    if i % 100 == 0:
        print(f"\n[{i}件目]\n入力文: {input_text}\n生成文: {generated_text}\n")

# 生成結果をCSVに保存
df_test['generated_sentence'] = generated_list
if args.output_filename:
    output_filename = args.output_filename
else:
    output_filename = "evaluation_results.csv" if output_dir == "." else f"evaluation_results_{model_name}.csv"
output_csv_path = os.path.join(output_dir, output_filename)
df_test.to_csv(output_csv_path, index=False)
print(f"生成結果を '{output_csv_path}' に保存しました。")

# ==========================================
# 3. 評価指標の計算 (BLEU / BERTScore)
# ==========================================
print("\nCalculating metrics...")
references = df_test[tgt_col].astype(str).tolist()
hypotheses = [str(text) for text in generated_list]

metrics = {
    "model_name": model_name,
    "model_dir": model_dir,
    "num_samples": len(df_test),
}

# BLEU-1 から BLEU-4 の計算
reference_tokens = [[tokenizer.tokenize(text)] for text in references]
hypothesis_tokens = [tokenizer.tokenize(text) for text in hypotheses]
smoothing_function = SmoothingFunction().method3
for n in range(1, 5):
    weights = tuple([1 / n] * n)
    bleu = corpus_bleu(
        reference_tokens,
        hypothesis_tokens,
        weights=weights,
        smoothing_function=smoothing_function,
    )
    print(f'BLEU-{n}: {bleu:.6f}')
    metrics[f'BLEU-{n}'] = float(bleu)

# BERTScore の計算
precision, recall, f1 = bertscore_score(
    hypotheses,
    references,
    lang='ja',
    device=str(device),
    verbose=True,
)

p_val = precision.mean().item()
r_val = recall.mean().item()
f1_val = f1.mean().item()

print(f'BERTScore Precision: {p_val:.6f}')
print(f'BERTScore Recall:    {r_val:.6f}')
print(f'BERTScore F1:        {f1_val:.6f}')

metrics['BERTScore_Precision'] = float(p_val)
metrics['BERTScore_Recall'] = float(r_val)
metrics['BERTScore_F1'] = float(f1_val)

metrics_filename = "evaluation_metrics.json" if output_dir == "." else f"metrics_{model_name}.json"
metrics_json_path = os.path.join(output_dir, metrics_filename)
with open(metrics_json_path, 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print(f"評価メトリクスを '{metrics_json_path}' に保存しました。")