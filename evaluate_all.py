import os
import sys
import re
import json
import yaml
import argparse
import subprocess
import pandas as pd


def natural_sort_key(s):
    """文字列内の数字を数値として扱い、自然順ソートを行うキー関数"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def is_model_directory(path):
    """指定パスがモデルディレクトリ（重みファイルや設定ファイルを含む）かチェック"""
    if not os.path.isdir(path):
        return False
    files = os.listdir(path)
    model_indicators = [
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "generation_config.json",
        "tokenizer_config.json",
    ]
    return any(f in files for f in model_indicators)


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser(
        description="指定フォルダ内の全モデルに対して eval.py を順次実行し、結果を指定フォルダに格納するスクリプト"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yml",
        help="設定ファイルのパス (デフォルト: config.yml)",
    )
    parser.add_argument(
        "--models_dir",
        type=str,
        default="./models",
        help="評価対象モデル群が格納されているフォルダパス (デフォルト: ./models)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="評価結果 (CSV / JSON / サマリー) を保存するフォルダパス (デフォルト: ./eval_results)",
    )
    parser.add_argument(
        "--test_csv_path",
        type=str,
        default=None,
        help="テストデータのCSVパス (未指定時はconfig.ymlから取得)",
    )
    parser.add_argument(
        "--shortcut",
        type=str,
        default=None,
        choices=["true", "false", "True", "False"],
        help="テストデータを絞るか (true/false, 未指定時はconfig.ymlの設定に従う)",
    )
    parser.add_argument(
        "--add_bos_token",
        type=str,
        default=None,
        choices=["true", "false", "True", "False"],
        help="BOSトークンを追加するか (true/false, 未指定時はconfig.ymlの設定に従う)",
    )
    parser.add_argument(
        "--eval_script",
        type=str,
        default="eval.py",
        help="実行する評価スクリプトのパス (デフォルト: eval.py)",
    )

    args = parser.parse_args()

    # config.yml の読み込み（存在する場合）
    config = {}
    if os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}

    data_config = config.get('data', {})

    models_dir = os.path.abspath(args.models_dir)
    output_dir = os.path.abspath(args.output_dir)
    eval_script = os.path.abspath(args.eval_script)

    test_csv_path = args.test_csv_path if args.test_csv_path is not None else data_config.get('test_path', 'amazon_review/amazon_review_test.csv')

    if args.shortcut is not None:
        shortcut_str = args.shortcut.lower()
    else:
        shortcut_str = str(data_config.get('shortcut', False)).lower()

    if args.add_bos_token is not None:
        add_bos_str = args.add_bos_token.lower()
    else:
        add_bos_str = str(data_config.get('add_bos_token', True)).lower()

    if not os.path.exists(models_dir):
        print(f"エラー: 指定されたモデルフォルダが見つかりません: {models_dir}")
        sys.exit(1)

    if not os.path.exists(eval_script):
        print(f"エラー: 評価スクリプトが見つかりません: {eval_script}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # モデルフォルダの抽出
    subdirs = [
        d for d in os.listdir(models_dir)
        if os.path.isdir(os.path.join(models_dir, d))
    ]

    model_dirs = [
        os.path.join(models_dir, d) for d in subdirs
        if is_model_directory(os.path.join(models_dir, d))
    ]

    if not model_dirs:
        model_dirs = [os.path.join(models_dir, d) for d in subdirs]

    # 自然順（エポック番号順）にソート
    model_dirs.sort(key=lambda p: natural_sort_key(os.path.basename(p)))

    if not model_dirs:
        print(f"エラー: '{models_dir}' 内に対象となるモデルフォルダが見つかりませんでした。")
        sys.exit(1)

    print("=" * 60)
    print("全モデル一括評価を開始します")
    print(f"設定ファイル  : {args.config} (存在: {os.path.exists(args.config)})")
    print(f"モデルフォルダ: {models_dir}")
    print(f"出力フォルダ  : {output_dir}")
    print(f"テストデータ  : {test_csv_path} (shortcut={shortcut_str})")
    print(f"対象モデル数  : {len(model_dirs)} 件")
    for idx, m in enumerate(model_dirs, 1):
        print(f"  [{idx:2d}] {os.path.basename(m)}")
    print("=" * 60)

    summary_records = []

    for i, model_path in enumerate(model_dirs, 1):
        model_name = os.path.basename(model_path)
        print(f"\n>>>>> [{i}/{len(model_dirs)}] モデル評価中: {model_name} <<<<<")

        cmd = [
            sys.executable,
            eval_script,
            "--config", args.config,
            "--model_dir", model_path,
            "--test_csv_path", test_csv_path,
            "--output_dir", output_dir,
            "--shortcut", shortcut_str,
            "--add_bos_token", add_bos_str,
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"警告: {model_name} の評価でエラーが発生しました (終了コード {e.returncode})")
            continue

        # 生成されたメトリクスJSONの読み込み
        metrics_json_path = os.path.join(output_dir, f"metrics_{model_name}.json")
        if os.path.exists(metrics_json_path):
            with open(metrics_json_path, 'r', encoding='utf-8') as f:
                record = json.load(f)
                summary_records.append(record)

    # 全モデルのサマリー保存
    if summary_records:
        df_summary = pd.DataFrame(summary_records)
        summary_csv_path = os.path.join(output_dir, "summary_metrics.csv")
        df_summary.to_csv(summary_csv_path, index=False)

        summary_md_path = os.path.join(output_dir, "summary_metrics.md")
        with open(summary_md_path, 'w', encoding='utf-8') as f:
            f.write("# 全モデル評価サマリー\n\n")
            try:
                f.write(df_summary.to_markdown(index=False))
            except Exception:
                f.write(df_summary.to_string())
            f.write("\n")

        print("\n" + "=" * 60)
        print("全モデルの評価が完了しました！")
        print(f"サマリーCSV : {summary_csv_path}")
        print(f"サマリーMD  : {summary_md_path}")
        print("=" * 60)
        print("\n【評価結果サマリー】")
        cols_to_show = [
            c for c in ['model_name', 'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4', 'BERTScore_F1']
            if c in df_summary.columns
        ]
        print(df_summary[cols_to_show].to_string(index=False))
    else:
        print("\n評価結果のメトリクスが取得できませんでした。")


if __name__ == "__main__":
    main()
