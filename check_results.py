import os
import json
import glob
import pandas as pd

# 出力ディレクトリのパス（eval.py / 評価ラッパースクリプトで指定した output_dir）
output_dir = "./eval_results_1"

# 方法A: すでに生成されている summary_metrics.csv を読み込む場合
summary_csv_path = os.path.join(output_dir, "summary_metrics.csv")

if os.path.exists(summary_csv_path):
    print(f"'{summary_csv_path}' から結果を読み込みます。\n")
    df_summary = pd.read_csv(summary_csv_path)
    
else:
    print("summary_metrics.csv が見つからないため、metrics_*.json から直接読み込みます。\n")
    json_files = glob.glob(os.path.join(output_dir, "metrics_*.json"))
    records = []
    for jf in json_files:
        with open(jf, 'r', encoding='utf-8') as f:
            records.append(json.load(f))
    df_summary = pd.DataFrame(records)

if not df_summary.empty:
    # 自然順（エポック番号順など）にソートしたい場合の処理（必要に応じて）
    # df_summary = df_summary.sort_values(by="model_name")

    # 表示する列の選択（存在する列のみ抽出）
    target_columns = [
        'model_name', 
        'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 
        'BERTScore_Precision', 'BERTScore_Recall', 'BERTScore_F1'
    ]
    cols_to_show = [c for c in target_columns if c in df_summary.columns]

    print("=" * 80)
    print("【各エポックの評価スコア一覧】")
    print("=" * 80)
    print(df_summary[cols_to_show].to_string(index=False))
    print("=" * 80)
    
    # 必要であれば、エポックごとの結果を新たなCSVとして再保存することも可能
    # df_summary[cols_to_show].to_csv(os.path.join(output_dir, "epoch_scores_extracted.csv"), index=False)
else:
    print("評価結果のデータが見つかりませんでした。")
    