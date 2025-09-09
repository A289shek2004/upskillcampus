# turbofan_rul_final.py
import os
import json
import joblib
import random
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LSTM, GRU, Conv1D, MaxPooling1D,
    GlobalAveragePooling1D, Bidirectional, Attention, LayerNormalization, MultiHeadAttention)
from tensorflow.keras.callbacks import EarlyStopping

# --------------------------
# Config (edit these)
# --------------------------
DATA_DIR = r"E:\DS UCT internship\project 6\Turbofan engine"

FD = "FD001"
SEQ_LENS_TO_TRY = [30, 40, 50]   # try these for LSTM/GRU grid
RUL_CAPS = [125]                 # you can try [100,125,None]
LSTM_GRU_UNITS = [128, 256]
DROPOUTS = [0.3, 0.5]
LRS = [5e-4]
EPOCHS = 50
BATCH_SIZE = 128
OUT_DIR = r"E:\DS UCT internship\project 6\Turbofan engine\outputs"


# --------------------------
# Utility functions
# --------------------------
def load_cmapss(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    n_cols = df.shape[1]
    cols = ["unit", "cycle"] + [f"op{i}" for i in range(1,4)]
    n_sensors = n_cols - 5
    sensor_cols = [f"s{i}" for i in range(1, n_sensors+1)]
    df.columns = cols + sensor_cols
    return df

def add_rul(df: pd.DataFrame, cap: int | None = None) -> pd.DataFrame:
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    out = df.copy()
    out["RUL"] = max_cycle - out["cycle"]
    if cap is not None:
        out["RUL"] = out["RUL"].clip(upper=cap)
    return out

def drop_constant_sensors(train_df: pd.DataFrame) -> List[str]:
    sensors = [c for c in train_df.columns if c.startswith("s")]
    keep = [c for c in sensors if train_df[c].std() > 0]
    return keep

def create_sequences(df: pd.DataFrame, seq_len: int, sensor_cols: List[str], label_col="RUL"):
    X, y, groups = [], [], []
    for unit_id, unit_df in df.groupby("unit"):
        unit_df = unit_df.sort_values("cycle")
        data = unit_df[sensor_cols].values
        labels = unit_df[label_col].values
        for end in range(seq_len, len(unit_df)+1):
            start = end - seq_len
            X.append(data[start:end])
            y.append(labels[end-1])
            groups.append(unit_id)
    return np.array(X), np.array(y), np.array(groups)

def scale_sequences(X: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    n_samples, seq_len, n_feats = X.shape
    X_flat = X.reshape(-1, n_feats)
    X_scaled = scaler.transform(X_flat)
    return X_scaled.reshape(n_samples, seq_len, n_feats)

def prepare_test_windows(test_df: pd.DataFrame, seq_len: int, sensor_cols: List[str]):
    X_test, unit_ids = [], sorted(test_df["unit"].unique())
    for u in unit_ids:
        ud = test_df[test_df["unit"] == u].sort_values("cycle")
        data = ud[sensor_cols].values
        if len(data) < seq_len:
            pad = np.repeat(data[0:1, :], seq_len - len(data), axis=0)
            data = np.vstack([pad, data])
        else:
            data = data[-seq_len:]
        assert data.shape[0] == seq_len
        X_test.append(data)
    return np.array(X_test), unit_ids

def metrics_basic(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred))
    }

def nasa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = y_pred - y_true
    score = np.where(d < 0, np.exp(-d/13.0) - 1, np.exp(d/10.0) - 1)
    return float(np.sum(score))

# --------------------------
# Model builders (with requested defaults)
# --------------------------
def build_lstm(seq_len: int, n_feats: int, units=128, dropout=0.3, lr=5e-4):
    model = Sequential([
        LSTM(units, input_shape=(seq_len, n_feats)),
        Dropout(dropout),
        Dense(units//2, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    return model

def build_gru(seq_len: int, n_feats: int, units=128, dropout=0.3, lr=5e-4):
    model = Sequential([
        GRU(units, input_shape=(seq_len, n_feats)),
        Dropout(dropout),
        Dense(units//2, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    return model

def build_bilstm(seq_len: int, n_feats: int, units=64, dropout=0.3, lr=5e-4):
    model = Sequential([
        Bidirectional(LSTM(units), input_shape=(seq_len, n_feats)),
        Dropout(dropout),
        Dense(units//2, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    return model

def build_cnn_lstm(seq_len: int, n_feats: int, kernel_size=5, lstm_units=128, dropout=0.3, lr=5e-4):
    model = Sequential([
        Conv1D(64, kernel_size=kernel_size, activation="relu", padding="same", input_shape=(seq_len, n_feats)),
        MaxPooling1D(2),
        LSTM(lstm_units),
        Dropout(dropout),
        Dense(64, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    return model

def build_attn_lstm(seq_len: int, n_feats: int, lstm_units=64, dropout=0.3, lr=5e-4):
    inp = Input(shape=(seq_len, n_feats))
    x = LSTM(lstm_units, return_sequences=True)(inp)
    attn = Attention()([x, x])
    x = GlobalAveragePooling1D()(attn)
    x = Dropout(dropout)(x)
    out = Dense(1)(x)
    m = Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    return m

def build_transformer(seq_len: int, n_feats: int, d_model=64, heads=4, ff_dim=128, dropout=0.1, lr=5e-4):
    inp = Input(shape=(seq_len, n_feats))
    x = Dense(d_model)(inp)  # project to d_model
    attn = MultiHeadAttention(num_heads=heads, key_dim=d_model)(x, x)
    x = LayerNormalization()(x + attn)
    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dense(d_model)(ff)
    x = LayerNormalization()(x + ff)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    out = Dense(1)(x)
    m = Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    return m

# --------------------------
# Training helper (train & return val preds)
# --------------------------
def train_model(X_tr, y_tr, X_val, y_val, builder_fn, builder_kwargs, epochs=EPOCHS, batch_size=BATCH_SIZE):
    model = builder_fn(**builder_kwargs)
    es = EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])
    val_pred = model.predict(X_val, verbose=0).flatten()
    return model, val_pred

# --------------------------
# Main pipeline
# --------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load
    train = load_cmapss(os.path.join(DATA_DIR, f"train_{FD}.txt"))
    test = load_cmapss(os.path.join(DATA_DIR, f"test_{FD}.txt"))
    rul_vector = pd.read_csv(os.path.join(DATA_DIR, f"RUL_{FD}.txt"), sep=r"\s+", header=None).values.flatten()

    results = {}

    # We'll iterate over RUL caps (easy to add others)
    for rul_cap in RUL_CAPS:
        train_cap = add_rul(train, cap=rul_cap)

        # choose sensors once (train-based)
        sensor_cols = drop_constant_sensors(train_cap)
        n_sensors = len(sensor_cols)
        print(f"RUL cap {rul_cap}: using {n_sensors} sensors -> {sensor_cols}")

        # Prepare baseline sequences with default SEQ_LEN for classical models and some deep models
        # But for LSTM/GRU hyper-search we'll rebuild windows as needed
        default_seq_len = SEQ_LENS_TO_TRY[0]

        # Create windows with default_seq_len to fit scaler
        X_all, y_all, groups_all = create_sequences(train_cap, default_seq_len, sensor_cols)
        scaler = MinMaxScaler().fit(X_all.reshape(-1, n_sensors))

        # We'll create a GroupKFold on default windows to get train/val split for classical baseline and deep models that use default seq_len
        Xs = scale_sequences(X_all, scaler)
        gkf = GroupKFold(n_splits=5)
        tr_idx, val_idx = next(gkf.split(Xs, y_all, groups_all))
        X_tr_def, X_val_def = Xs[tr_idx], Xs[val_idx]
        y_tr_def, y_val_def = y_all[tr_idx], y_all[val_idx]

        # Classical baselines (flattened)
        X_tr_flat = X_tr_def.reshape(X_tr_def.shape[0], -1)
        X_val_flat = X_val_def.reshape(X_val_def.shape[0], -1)
        imputer = SimpleImputer(strategy="mean").fit(X_tr_flat)
        X_tr_flat = imputer.transform(X_tr_flat)
        X_val_flat = imputer.transform(X_val_flat)

        lr = LinearRegression().fit(X_tr_flat, y_tr_def)
        rf = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1).fit(X_tr_flat, y_tr_def)

        results[f"LinearRegression_RUL{rul_cap}"] = metrics_basic(y_val_def, lr.predict(X_val_flat))
        results[f"RandomForest_RUL{rul_cap}"] = metrics_basic(y_val_def, rf.predict(X_val_flat))

        # --------------------------
        # Hyperparameter search for LSTM & GRU (simple grid, low budget)
        # --------------------------
        best_deep = {"nasa": np.inf}
        deep_records = []

        for seq_len in SEQ_LENS_TO_TRY:
            X, y, groups = create_sequences(train_cap, seq_len, sensor_cols)
            if len(X) == 0:
                continue
            scaler_seq = MinMaxScaler().fit(X.reshape(-1, n_sensors))
            Xs = scale_sequences(X, scaler_seq)

            gkf = GroupKFold(n_splits=5)
            tr_idx, val_idx = next(gkf.split(Xs, y, groups))
            X_tr, X_val = Xs[tr_idx], Xs[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]

            for units in LSTM_GRU_UNITS:
                for dropout in DROPOUTS:
                    for lr_val in LRS:
                        # LSTM config
                        builder_kwargs = {"seq_len": seq_len, "n_feats": n_sensors, "units": units, "dropout": dropout, "lr": lr_val}
                        # Train LSTM
                        model_lstm, val_pred_lstm = train_model(X_tr, y_tr, X_val, y_val, build_lstm, {"seq_len": seq_len, "n_feats": n_sensors, "units": units, "dropout": dropout, "lr": lr_val})
                        metrics_l = metrics_basic(y_val, val_pred_lstm)
                        nasa_l = nasa_score(y_val, val_pred_lstm)
                        deep_records.append({"model":"LSTM", "seq_len":seq_len, "units":units, "dropout":dropout, "lr":lr_val, **metrics_l, "nasa": nasa_l})

                        if nasa_l < best_deep["nasa"]:
                            best_deep = {"model":"LSTM", "seq_len":seq_len, "units":units, "dropout":dropout, "lr":lr_val, "scaler":scaler_seq, "nasa":nasa_l}
                            # keep the model weights path optional (we can save later)

                        # GRU config
                        model_gru, val_pred_gru = train_model(X_tr, y_tr, X_val, y_val, build_gru, {"seq_len": seq_len, "n_feats": n_sensors, "units": units, "dropout": dropout, "lr": lr_val})
                        metrics_g = metrics_basic(y_val, val_pred_gru)
                        nasa_g = nasa_score(y_val, val_pred_gru)
                        deep_records.append({"model":"GRU", "seq_len":seq_len, "units":units, "dropout":dropout, "lr":lr_val, **metrics_g, "nasa": nasa_g})

                        if nasa_g < best_deep["nasa"]:
                            best_deep = {"model":"GRU", "seq_len":seq_len, "units":units, "dropout":dropout, "lr":lr_val, "scaler":scaler_seq, "nasa":nasa_g}

        # Save the grid search summary
        results[f"deep_grid_RUL{rul_cap}"] = deep_records
        print("Best deep config (by val NASA):", best_deep)

        # --------------------------
        # Train other deep models with reasonable defaults and record val metrics + nasa
        # --------------------------
        # use default seq_len for these models (first in SEQ_LENS_TO_TRY)
        seq_len_default = SEQ_LENS_TO_TRY[0]
        X_def, y_def, groups_def = create_sequences(train_cap, seq_len_default, sensor_cols)
        scaler_def = MinMaxScaler().fit(X_def.reshape(-1, n_sensors))
        Xs_def = scale_sequences(X_def, scaler_def)
        tr_idx, val_idx = next(GroupKFold(n_splits=5).split(Xs_def, y_def, groups_def))
        X_tr0, X_val0 = Xs_def[tr_idx], Xs_def[val_idx]
        y_tr0, y_val0 = y_def[tr_idx], y_def[val_idx]

        # BiLSTM
        m_bilstm, val_pred_bilstm = train_model(X_tr0, y_tr0, X_val0, y_val0, build_bilstm, {"seq_len": seq_len_default, "n_feats": n_sensors, "units":64, "dropout":0.3, "lr":5e-4})
        results[f"BiLSTM_RUL{rul_cap}"] = {**metrics_basic(y_val0, val_pred_bilstm), "nasa": nasa_score(y_val0, val_pred_bilstm)}

        # CNN-LSTM (fixed kernel 5)
        m_cnnlstm, val_pred_cnnlstm = train_model(X_tr0, y_tr0, X_val0, y_val0, build_cnn_lstm, {"seq_len": seq_len_default, "n_feats": n_sensors, "kernel_size":5, "lstm_units":128, "dropout":0.3, "lr":5e-4})
        results[f"CNN_LSTM_RUL{rul_cap}"] = {**metrics_basic(y_val0, val_pred_cnnlstm), "nasa": nasa_score(y_val0, val_pred_cnnlstm)}

        # AttnLSTM
        m_attn, val_pred_attn = train_model(X_tr0, y_tr0, X_val0, y_val0, build_attn_lstm, {"seq_len": seq_len_default, "n_feats": n_sensors, "lstm_units":64, "dropout":0.3, "lr":5e-4})
        results[f"AttnLSTM_RUL{rul_cap}"] = {**metrics_basic(y_val0, val_pred_attn), "nasa": nasa_score(y_val0, val_pred_attn)}

        # Transformer
        m_trf, val_pred_trf = train_model(X_tr0, y_tr0, X_val0, y_val0, build_transformer, {"seq_len": seq_len_default, "n_feats": n_sensors, "d_model":64, "heads":4, "ff_dim":128, "dropout":0.1, "lr":5e-4})
        results[f"Transformer_RUL{rul_cap}"] = {**metrics_basic(y_val0, val_pred_trf), "nasa": nasa_score(y_val0, val_pred_trf)}

        # --------------------------
        # Pick best deep config found in grid (LSTM/GRU) and retrain on FULL train windows (all windows at that seq_len)
        # --------------------------
        if best_deep["model"] in ["LSTM", "GRU"]:
            chosen = best_deep
            seq_len_best = chosen["seq_len"]
            scaler_best = chosen["scaler"]
            X_full, y_full, groups_full = create_sequences(train_cap, seq_len_best, sensor_cols)
            X_full_scaled = scale_sequences(X_full, scaler_best)

            # build model with chosen params and retrain on full training windows (no val split)
            if chosen["model"] == "LSTM":
                final_model = build_lstm(seq_len_best, n_sensors, units=chosen["units"], dropout=chosen["dropout"], lr=chosen["lr"])
            else:
                final_model = build_gru(seq_len_best, n_sensors, units=chosen["units"], dropout=chosen["dropout"], lr=chosen["lr"])

            # early stopping on loss to avoid overfitting
            es_final = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
            final_model.fit(X_full_scaled, y_full, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es_final], verbose=0)
            
            # Save final model & scaler for deployment
            os.makedirs("models", exist_ok=True)
            final_model.save(f"models/best_{chosen['model'].lower()}.h5")
            joblib.dump(scaler_best, f"models/scaler_{chosen['model'].lower()}.pkl")


            # Prepare test windows at seq_len_best (same scaler)
            X_test, test_units = prepare_test_windows(test, seq_len_best, sensor_cols)
            X_test_scaled = scale_sequences(X_test, scaler_best)
            y_test_pred = final_model.predict(X_test_scaled, verbose=0).flatten()

            results[f"BEST_DEEP_{chosen['model']}_RUL{rul_cap}"] = {**metrics_basic(rul_vector, y_test_pred), "nasa": nasa_score(rul_vector, y_test_pred)}
            print("BEST_DEEP test metrics:", results[f"BEST_DEEP_{chosen['model']}_RUL{rul_cap}"])

            # Simple ensemble: average predictions of final_model and another variant (same model with other units)
            alt_units = 128 if chosen["units"] == 256 else 256
            if chosen["model"] == "LSTM":
                alt_model = build_lstm(seq_len_best, n_sensors, units=alt_units, dropout=chosen["dropout"], lr=chosen["lr"])
            else:
                alt_model = build_gru(seq_len_best, n_sensors, units=alt_units, dropout=chosen["dropout"], lr=chosen["lr"])
            alt_model.fit(X_full_scaled, y_full, epochs=max(10, EPOCHS//2), batch_size=BATCH_SIZE, callbacks=[es_final], verbose=0)
            alt_pred = alt_model.predict(X_test_scaled, verbose=0).flatten()
            ens_pred = 0.5 * (y_test_pred + alt_pred)
            results[f"ENSEMBLE_{chosen['model']}_RUL{rul_cap}"] = {**metrics_basic(rul_vector, ens_pred), "nasa": nasa_score(rul_vector, ens_pred)}

        # Save intermediate results for this RUL cap
        with open(os.path.join(OUT_DIR, f"results_RUL{rul_cap}.json"), "w") as f:
            json.dump(results, f, indent=2)

    print("All done. Results saved to", OUT_DIR)
    return results

if __name__ == "__main__":
    res = main()
    print(json.dumps(res, indent=2))
