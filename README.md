# BLEU損失関数（DEBLEU）を使用したGPT-2の学習

## 実行方法

モデル学習スクリプトは `train.py` です。基本設定は `config.yml` から読み込まれますが、ターミナル上のコマンドライン引数でパラメータを上書きして実行できます。

SSH接続を切断（ログアウト）した後も学習を継続させるため、**`nohup`** と **`&`** を使ったバックグラウンド実行を推奨します。  
（※ `-u` オプションを付けることでログがバッファリングされず、`output.log` にリアルタイムで書き込まれます。）

---

### モデル保存の状態別 実行コマンド（nohup によるバックグラウンド実行）

#### 0. 仮想環境の有効化
必要なパッケージは次の通り。
train時： `tqdm`, `numpy`, `pandas`, `matplotlib`, `pyyaml`, `torch`
評価時： `tqdm`, `numpy`, `pandas`, `torch`, `nltk`, `bert-score`

```bash
conda activate myenv
```

---

#### 1. 指定したエポック毎にモデルを保存する場合（中間チェックポイント保存）
`--save_intermediate_checkpoints true` を指定します。`--save_epoch_interval` で保存間隔（エポック数）を指定できます。  
※ 指定間隔のエポックに加えて、**最終エポックも自動的に保存**されます。

- **毎エポック保存する場合 (例: config.yml の設定通り、または明示指定)**
  ```bash
  nohup python3 -u train.py --save_intermediate_checkpoints true --save_epoch_interval 1 > output.log 2>&1 &
  ```

- **5エポック毎に保存する場合**
  ```bash
  nohup python -u train.py --save_intermediate_checkpoints true --save_epoch_interval 5 > output.log 2>&1 &
  ```

- **エポック数もターミナルから指定する場合 (例: 全20エポック、毎エポック保存)**
  ```bash
  nohup python -u train.py --epochs 20 --save_intermediate_checkpoints true --save_epoch_interval 1 > output.log 2>&1 &
  ```

保存先フォルダ: `models/finetuned_model_epoch_{epoch}/`

---

#### 2. 最終エポックのみモデルを保存する場合
途中エポックの保存を行わず、全エポック完了時の最終モデルのみを保存します。

```bash
nohup python -u train.py --save_intermediate_checkpoints false > output.log 2>&1 &
```

- **エポック数を指定して最終モデルのみ保存する場合**
  ```bash
  nohup python -u train.py --epochs 20 --save_intermediate_checkpoints false > output.log 2>&1 &
  ```

保存先フォルダ: `models/finetuned_model_epoch_{最終エポック}/`

---

#### 3. 設定ファイル（config.yml）の内容でそのままバックグラウンド実行する場合
引数を省略すると `config.yml` の値がすべて適用されます。

```bash
nohup python -u train.py > output.log 2>&1 &
```

別ファイルを指定して実行する場合:
```bash
nohup python -u train.py --config config_custom.yml > output.log 2>&1 &
```

#### 4. 途中から実行する場合
途中再開用のコードとconfigを使用して実行

```bash
nohup python resume_train.py > train_resume.log 2>&1 &
```

---

### 学習の進捗確認とプロセスの管理

#### ログのリアルタイム確認
SSH再接続後など、学習の進捗ログを確認するには以下のコマンドを実行します。
```bash
tail -f output.log
```
※ `Ctrl + C` を押すとログ表示を終了できます（学習プロセス自体はバックグラウンドで継続して動作します）。

#### 実行中プロセスの確認
```bash
ps aux | grep train.py
```
（または `nvidia-smi` でGPU使用状況を確認）

#### 学習プロセスの停止（強制終了）
```bash
kill <PID>
# 停止しない場合は強制終了: kill -9 <PID>
```

---

## モデルの評価（Evaluation）

学習済みモデルに対してテストデータを入力し、文生成および評価指標（BLEU-1〜4、BERTScore）の算出を行います。  
※ 評価の実行には `nltk` および `bert_score` が必要です（インストールされている環境、例: `base` 環境等で実行してください）。

### 1. フォルダ内の全モデルを一括評価 (`evaluate_all.py`)

指定した親フォルダ（例: `models/`）内にある全モデルディレクトリ（`finetuned_model_epoch_1`, `epoch_2`, ...）を自動探索し、エポック番号順にソートして順次 `eval.py` を実行します。  
生成文CSVや評価スコアJSON、および全モデルの比較サマリー表を指定フォルダに自動格納します。

#### 基本実行コマンド（フォアグラウンド）
```bash
python evaluate_all.py --models_dir models --output_dir eval_results
```

#### バックグラウンド実行（nohup による実行推奨）
全モデルの評価は時間を要するため、SSH接続切断後も継続実行できるようバックグラウンド実行を推奨します。

```bash
nohup python -u evaluate_all.py --models_dir models --output_dir eval_results > eval.log 2>&1 &
```

- **ログの確認:**
  ```bash
  tail -f eval.log
  ```
- **実行中プロセスの確認:**
  ```bash
  ps aux | grep evaluate_all.py
  ```

---

### 2. 単一モデルの評価 (`eval.py`)

特定のモデル単体を評価する場合は、`eval.py` に引数を渡して直接実行できます。

```bash
python eval.py --model_dir models/finetuned_model_epoch_20 --output_dir eval_results
```

---

### 3. Bash ループによる一括実行（シェルコマンドの場合）

スクリプトを使わず、Bashのコマンドループで直接順次実行する場合は以下を実行します：

```bash
mkdir -p eval_results
for model_dir in $(ls -d models/*/ | sort -V); do
    echo "Evaluating ${model_dir}..."
    python eval.py --model_dir "${model_dir}" --output_dir eval_results
done
```

## その他pyファイルの説明

`check_result.py` : eval_results内のBLEU, BERTScoreを確認するコード。ファイル内でdirを指定。
`draw_learning_curve.py` : 学習曲線を描くコード。ファイル内でdirを指定。