import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

# --- 設定 ---
base_dir = "models"
prefix = "finetuned_model_epoch_"
use_alpha = False

# --- meta.txt の読み込み処理 ---
data = []

# 対象フォルダを検索 (例: models/finetuned_model_epoch_1)
folder_paths = glob.glob(os.path.join(base_dir, f"{prefix}*"))

for folder in folder_paths:
    meta_path = os.path.join(folder, "meta.txt")
    if os.path.exists(meta_path):
        meta_dict = {}
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if "=" in line:
                    key, val = line.strip().split("=", 1)
                    meta_dict[key] = float(val) if "." in val or "e" in val.lower() else int(val)
        data.append(meta_dict)

# --- DataFrame化とソート ---
if not data:
    print(f"エラー: {base_dir} 内に meta.txt が見つかりませんでした。")
else:
    df_history = pd.DataFrame(data)
    # epoch順に昇順ソート
    df_history = df_history.sort_values("epoch").reset_index(drop=True)
    
    # 集計結果をCSVとして保存
    csv_path = os.path.join(base_dir, "learning_curve_from_meta.csv")
    df_history.to_csv(csv_path, index=False)
    print(f"集計データを保存しました: {csv_path}")
    print(df_history)

    # --- グラフの描画 ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Lossのプロット (左軸)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="tab:blue")
    line1 = ax1.plot(df_history["epoch"], df_history["avg_train_loss"], label="Train Loss", color="tab:blue", marker="o")
    line2 = ax1.plot(df_history["epoch"], df_history["avg_eval_loss"], label="Eval Loss", color="tab:orange", marker="s")
    ax1.tick_params(axis="y")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # alpha のプロット (右軸: スケジュールの変化を確認用)
    lines = line1 + line2
    if "alpha" in df_history.columns and df_history["alpha"].max() > 0 and use_alpha == True:
        ax2 = ax1.twinx()
        ax2.set_ylabel("Alpha (DEBLEU Weight)", color="tab:green")
        line3 = ax2.plot(df_history["epoch"], df_history["alpha"], label="Alpha", color="tab:green", linestyle=":", marker="^")
        ax2.tick_params(axis="y")
        lines += line3

    # 凡例の集約表示
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")

    plt.title("Training and Validation Loss Curve")
    fig.tight_layout()

    # グラフ画像の保存
    plot_path = os.path.join(base_dir, "loss_curve.png")
    plt.savefig(plot_path, dpi=300)
    plt.show()
    print(f"学習曲線の画像を保存しました: {plot_path}")