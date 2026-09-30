# LSTM and Transformer forecasting study

Status: prospective revised protocol. No new result is claimed until its run manifest exists.

## Research question
Under a documented parameter-matched training budget, how do vanilla LSTM and Transformer forecasting errors change with input length, training history, forecast horizon, and dataset? Do either improve over persistence, seasonal naive, and a channel-shared linear Ridge forecaster? Stationarity and seasonality are descriptive associations, not identified causes.

## Data and leakage controls
Use original ETTh1 and ETTm1 from ETDataset and Exchange and Traffic from the LSTNet data repository. Record downloaded byte hashes, source URLs, dimensions, missingness, and selected channels. Never substitute synthetic Traffic.

Chronological train/validation/test fractions are 70/15/15. All 7 ETT channels are retained. For Exchange and Traffic, select the seven largest-variance channels using training data only, recording their identities. Traffic conclusions concern this subset, not all 862 sensors. Missing or nonfinite input causes an error rather than silent imputation. ETTm1 is 15-minute, ETTh1/Traffic hourly, and Exchange daily (dataset observation steps); comparisons across datasets are not equal wall-clock horizons.

Each training-history ratio uses a suffix of the original training interval. Its input scaler is fit on that suffix only. Evaluation reports channel-wise raw errors plus MAE and RMSE normalized by ONE fixed full-training standard deviation per dataset/split, independent of ratio; this reference is used for evaluation, never passed to a restricted-history model. This metric is called reference-standardized error and is not pooled raw-unit MAE.

Validation and test origins are fixed using a 336-step maximum input and 192-step maximum forecast, regardless of the condition being evaluated. Training windows stay entirely inside the available training interval. This sacrifices some boundary observations to keep evaluation origins identical. Sample stride is one. Target is all selected channels at all forecast offsets, with equal element weights.

## Models and training
LSTM: hidden=256, layers=2, dropout=.1. Transformer: width=256, heads=8, encoder layers=2, feedforward=256, dropout=.1, sinusoidal positions. Both use the final hidden position and a direct multihorizon linear head. Counts are reported for each configuration.

Adam, batch 32, MSE training objective, max 100 epochs, patience 10, gradient clipping at 1.0 for both models. FP32, explicit deterministic settings. Validation chooses the best checkpoint. A prospective equal-budget tuning phase compares learning rates 1e-4 and 1e-3 at input=96/horizon=24/full training on the first three seeds separately for each dataset/model. Select mean validation reference-standardized MSE; test data is never evaluated in tuning. Freeze the chosen learning rate for the condition grid. Ridge alpha is chosen from .01,.1,1,10,100 on validation only.

Seeds: 42,123,456,789,2024,7,88,314,1004,9999. Smoke results use a separate directory and configuration hash and are never accepted as full results. Each run is written atomically to its own file; multi-GPU workers receive disjoint jobs. No shared CSV writer. Include environment, code/config/data hash, best epoch, wall time, peak GPU memory, and per-epoch history.

## Experiments
1. ETTh1 input lengths 24,48,96,168,336; horizon=24.
2. ETTh1, ETTm1, Exchange, Traffic at input=96/horizon=24.
3. ETTh1 training ratios .3,.5,.7,1 at input=96/horizon=24.
4. ETTh1 horizons 12,24,48,96,192 at input=96.
Overlapping default conditions are run once and reused transparently. The 15 unique dataset/condition jobs produce 300 neural-model fits for 10 seeds, plus 48 tuning fits. Baselines are deterministic and run once per condition. Recompute actual manifest size rather than relying on this prose count.

5. Attribution: compare the same absolute input-gradient method on both models over dispersed, identical test origins, across all available seeds for the ETTh1 default condition. Gradients are of a specified output scalar (mean predictions); they are sensitivity measures, not Integrated Gradients. Report recent 19/96 mass against a uniform reference, and distribution by seed. Do not claim explanation faithfulness or rank explanation quality from this alone.

## Statistics and scope
Report every condition and both MAE/RMSE. Pair neural model errors by seed, report mean/SD, paired difference 95% t interval, paired t and exact Wilcoxon, Cohen's d_z. Apply Holm across the unique confirmatory condition comparisons and report raw and adjusted p-values. Seed variability is optimization variability on one chronological split, not sampling uncertainty over future regimes. Do not claim generalization to unseen periods or causal influence of unit roots. Unit-root diagnostics use training-only per-channel ADF/KPSS with both constant and trend specifications; report lag settings, statistics, p bounds, and conflicting outcomes. KPSS statistic is not seasonality strength.

## Manuscript rule
All numerical tables and graphs must derive from completed manifests. Missing conditions remain explicitly pending. Results favoring naive models or contradicting hypotheses must be retained. Hardware changes are disclosed and time comparisons are made on the same GPU. This protocol corrects the earlier manuscript; it is not a preregistration of the original experiments.
