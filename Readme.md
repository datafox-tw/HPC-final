# HPC期末專案

## 前言
傳統金融預測波動度時經常使用 GARCH 類等線性模型，其核心假設未來波動度僅由過去殘差、過去變異數線性與少量變數(Covariates)相關。然而，在複雜且高維的金融市場中，波動度僅由線性關係驅動的假設過於強硬，且變數的稀疏性也難以將公司的重大資訊公布、成交量、報酬率和實證上已經發現的大量潛在變數一起納入考慮。

本研究為了解決實證上波動度可能發生的非線性關係與大量潛在變數，我們採用 Chen et al.(2023) 提出的 TSMixer 模型對波動度進行建模，相對於TFT, DeepAR等模型，TSMixer 是一款全利用 MLP 架構進行時間序列預測的深度學習模型，它在 2023 年是當代 state-of-the-art 模型，且在S&P500指數波動度預測中也達到SOTA地位。

TSMixer 模型的好處在於 RNN, LSTM 以及 GRU 等方式並沒有辦法同時對多隻股票建模，導致模型的訓練、儲存、維護都變得相當複雜。TSMixer 也支援對多條時間序列建模共享參數組，通過共享參數組，模型可以捕捉到金融市場上普遍發生的現象，相比於對每一條時間序列都建立一個模型，這完美解決了金融市場容易過擬合的痛點。

由於在台灣股票上，並沒有針對tsmixer進行波動度預測的contribution，同時在tsmixer在time-series預測上也暫時沒有相關文獻做 High performance computing 加速，儘管與基於 Transformer 的模型相比，TSMixer 相對輕量級，但其運算模式以矩陣乘法為主，使其成為 GPU 加速和高效能最佳化的理想目標。

本研究對學術與實務的貢獻如下：(1) TSMixer 在 MAE/RMSE 等指標皆贏過 GARCH 模型 (2) 共享參數組對多條時間序列建模在波動度預測上是有效的 (3) 在此基礎上，進行profiling and optimization. 

## 參考訓練方法
0. source .venv/bin/activate
1. 下載dataset (資料前處理已經搞定了但不是我搞的，但我覺得應該不用特地描述)
2. pip install -r requirements.txt
3. 資料前處理
python src/preprocess.py  --input data/ml_dataset_alpha101_volatility.csv     --output data/clean.pkl  --disabled_features close log_return u_hat_90 gjrgarch_var_90 tgarch_var_90    --use_log_target     --target_col var_true_90     --garch_col garch_var_90 
（我忘記為什麼不能塞return了 我先使用他應該可以當作獨立的new alphas）(但也不是我弄的)
這個版本採用close 大概100秒

python src/preprocess.py  --input data/ml_dataset_alpha101_volatility.csv     --output data/clean.pkl  --disabled_features close  gjrgarch_var_90 tgarch_var_90    --use_log_target     --target_col var_true_90     --garch_col garch_var_90 

4. Build dataset 大概40秒
python src/dataset_builder.py --input data/clean.pkl  --output data/ts_data.pkl --val_frac 0.2 --test_frac 0.1 --input_chunk_length 90 --static_mode ticker --target_col var_true_90 --garch_col garch_var_90 

5. python src/alpha_eda_multi.py

<!-- # How to reenact the result and further comparison
### 0. source .venv/bin/activate
0-1. new requirements is built by pipreqs to further allocate pip dependency.
pip install pipreqs
pipreqs ./src
資料集我不提供到github上因為太大了

### 1. download newest dataset from **Releases**
1-1. I have renamed it to Dataset_reenact_yuchi to make it different and reconogizable

### 2. run following code to complete data preprocessing(14:38-14:41)
```
python src/preprocess.py     --input Dataset_reenact_yuchi/data/ml_dataset_alpha101_volatility.csv     --output Dataset_reenact_yuchi/clean.pkl     --disabled_features close log_return u_hat_90 gjrgarch_var_90 tgarch_var_90    --use_log_target     --target_col var_true_90     --garch_col garch_var_90 
```
2-1. final dataset is in clean.pkl

### 3. build dataset (30 secs)
```
python src/dataset_builder.py --input Dataset_reenact_yuchi/clean.pkl  --output Dataset_reenact_yuchi/ts_data.pkl --val_frac 0.2 --test_frac 0.1 --input_chunk_length 90 --static_mode ticker --target_col var_true_90 --garch_col garch_var_90 
```

### 4. 訓練模型，自訂部分超參數 (1447-1451 一個iter 100秒)
- `--covariate_mode`: 預設為`none`，不使用alpha資料；設定為`lagged`會使用alpha資料訓練

#### TSMixer
```
python src/model_train.py     --data Dataset_reenact_yuchi/ts_data.pkl --lambda 0 --epochs 6 --lr 3e-4 --lr_scheduler exponential --lr_gamma 0.99  --grad_clip 0.5 --hidden_size 32 --ff_size 64 --num_blocks 4 --dropout 0.1 --model_path models/tsmixer_lambda0.pth
```
#### LSTM
```
python src/model_train_lstm.py     --data Dataset_reenact_yuchi/ts_data.pkl --lambda 0 --epochs 6 --lr 2e-4 --lr_scheduler exponential --lr_gamma 0.99  --grad_clip 0.5 --hidden_size 32 --dropout 0.1 
```
#### bash批次訓練
```
bash train_all_lambdas.sh --parallel 4 --nohup
```
- 腳本訓練多組lambda (lambda=0: All GARCH; lambda=1: No GARCH)
- 超參數需至bash檔設定
- 原始的方法如果只用cpi要跑8小時1個epoch我叫gpt幫我生成mac晶片加速跟gpu加速： 結果光是用電腦內建的mac晶片就可以壓到10分鐘內一個epoch

### 5. Predict，輸出結果
- `--covariate_mode`: 預設為`none`，不使用alpha資料；設定為`lagged`會使用alpha資料
#### TSMixer
```
python src/model_predict_eval.py --data Dataset_reenact_yuchi/ts_data.pkl --model models/tsmixer_lambda0_reviselr.pth --output outputs/lambda_0_1/
```
#### LSTM
```
python src/predict_lstm.py --data Dataset_reenact_yuchi/ts_data.pkl --model models/lstm.pth --split test --output outputs/lambda_0_1 --save_plots
```
#### bash批次預測
```
bash eval_all_lambdas.sh --parallel 4
```

### 6. 資料視覺化和資訊整理 
5. 輸出結果 python src/model_predict_eval.py --data Dataset_reenact_yuchi/ts_data.pkl --model models/tsmixer_lambda0_reviselr.pth --output outputs/lambda_0_1/


or lstm prediction
python src/predict_lstm.py \
  --data Dataset_reenact_yuchi/ts_data.pkl \
  --model models/lstm.pth \
  --split test \
  --output outputs/lambda_0_1
  --save_plots

### 如果用bash:使用 ./eval_all_lambdas.sh --parallel 4
6. 資料視覺化和資訊整理 
python src/data_visualization.py \
  --input outputs/lambda_0_1/metrics.csv \
  --input outputs/lambda_0_1/metrics.png \
python src/data_visualization.py \
  --input outputs/lamb0-iter2-new/metrics_tsmixer_lambda0.csv \
  --output outputs/lamb0-iter2-new/metrics_tsmixer_lambda0.png


如何使用自動測試最佳化的腳本？ ＊我有使用平行化加速、背景處理、nohup  
./train_all_lambdas.sh --parallel 4 --nohup  
./train_all_lambdas.sh --parallel 4 --log-dir my_logs
監控訓練進度
# 查看日誌檔案
tail -f logs/tsmixer_lambda0.log

# 查看所有模型檔案
ls -lh models/

# 查看執行中的 Python 進程
ps aux | grep python

7. 用compare_all_model.py來看到底誰更好
python src/compare_all_model.py --input outputs/all_metrics_combined.csv 
以下是每次比較的資料結果
Lambda = 0, garch+tsmixer, epoch=2, lr=2e-4:
--- Average Metrics ---
|            |   Average Value |
|:-----------|----------------:|
| MAE_model  |        0.818576 |
| RMSE_model |        0.981375 |
| MAE_garch  |        0.827502 |
| RMSE_garch |        1.00899  |

--- Win Counts (Lower error is a win) ---
| Metric   |   Model Wins |   GARCH Wins |   Ties |
|:---------|-------------:|-------------:|-------:|
| MAE      |           27 |           19 |      0 |
| RMSE     |           39 |            7 |      0 |

(.venv) blackwingedkite@keyouqideMacBook-Air ADL-Final % python src/data_visualization.py --input outputs/lamb0-iter10/metrics_tsmixer_lambda0.csv


Lambda = 0, garch+tsmixer, epoch=10, lr=3e-4
--- Average Metrics ---
|            |   Average Value |
|:-----------|----------------:|
| MAE_model  |        0.808775 |
| RMSE_model |        0.97595  |
| MAE_garch  |        0.827502 |
| RMSE_garch |        1.00899  |

--- Win Counts (Lower error is a win) ---
| Metric   |   Model Wins |   GARCH Wins |   Ties |
|:---------|-------------:|-------------:|-------:|
| MAE      |           34 |           12 |      0 |
| RMSE     |           40 |            6 |      0 |

Lambda = 0, lstm+garch, epoch=10, lr=3e-4
--- Average Metrics ---
| Metric   | Method   |   Average Value |
|:---------|:---------|----------------:|
| MAE      | LSTM     |        0.806965 |
| MAE      | GARCH    |        0.827502 |
| RMSE     | LSTM     |        0.980534 |
| RMSE     | GARCH    |        1.00899  |

--- Win Counts (Lower error is a win) ---
| Metric   |   Model Wins |   GARCH Wins |   Ties |
|:---------|-------------:|-------------:|-------:|
| MAE      |           45 |            1 |      0 |
| RMSE     |           45 |            1 |      0 |

Lambda = 0, tsmixer+garch, epoch=2, lr=3e-4
--- Average Metrics ---
| Metric   | Method   |   Average Value |
|:---------|:---------|----------------:|
| MAE      | model    |        0.802012 |
| MAE      | garch    |        0.827502 |
| RMSE     | model    |        0.967023 |
| RMSE     | garch    |        1.00899  |

--- Win Counts (Lower error is a win) ---
| Metric   |   Model Wins |   GARCH Wins |   Ties |
|:---------|-------------:|-------------:|-------:|
| MAE      |           36 |           10 |      0 |
| RMSE     |           41 |            5 |      0 |

Lambda = 0, lstm+tsmixer, epoch=2, lr=3e-4

FAQ:
model在哪裡？
->可以自己訓練或者跟我要我上傳雲端 因為有點多 -->
