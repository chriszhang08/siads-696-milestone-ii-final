import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras import layers, models
import polars as pl
import pickle
import sys
from pathlib import Path

from model_factory import build_lstm


def configure_gpu():
    """Configure GPU settings for optimal performance."""
    print("=== GPU CONFIGURATION ===")

    # Check if GPUs are available
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Number of GPUs Available: {len(gpus)}")

    if gpus:
        try:
            # Enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ Memory growth enabled for: {gpu}")

            # Set the first GPU as visible (if multiple GPUs)
            tf.config.set_visible_devices(gpus[0], 'GPU')
            print(f"✅ Using GPU: {gpus[0]}")

            # Test GPU computation
            print("Testing GPU computation...")
            with tf.device('/GPU:0'):
                a = tf.random.normal([1000, 1000])
                b = tf.matmul(a, a)
                result = tf.reduce_sum(b)
            print(f"✅ GPU test successful: {result.numpy():.2f}")

            return True

        except RuntimeError as e:
            print(f"❌ GPU configuration error: {e}")
            return False
    else:
        print("❌ No GPU devices found. Running on CPU.")
        return False


def evaluate_model(y_true, y_pred, model_name):
    """Evaluate model performance and print metrics."""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    print(f"\n{model_name} Performance:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")

    return {'mse': mse, 'mae': mae, 'rmse': rmse}


def load_split_data(train_path, test_path, feature_cols, target_col):
    """Load pre-split train and test data from separate parquet files."""
    print(f"Loading train data from {train_path}...")
    train_df = pl.read_parquet(train_path).to_pandas()
    print(f"Loaded {len(train_df)} training rows")

    print(f"Loading test data from {test_path}...")
    test_df = pl.read_parquet(test_path).to_pandas()
    print(f"Loaded {len(test_df)} test rows")

    # Separate features and targets for train set
    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values

    # Separate features and targets for test set
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    print(f"Train features shape: {X_train.shape}")
    print(f"Train targets shape: {y_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    print(f"Test targets shape: {y_test.shape}")

    return X_train, y_train, X_test, y_test, train_df, test_df


def train_lstm_model_presplit(train_path, test_path, feature_cols, target_col, output_dir,
                              units=64, depth=2, dropout=0.3, epochs=50, batch_size=32,
                              validation_split=0.2):
    """
    Train LSTM model on pre-split parquet data files with GPU support.

    Args:
        train_path: Path to training parquet file
        test_path: Path to test parquet file
        feature_cols: List of feature column names
        target_col: Target column name
        output_dir: Directory to save model and results
        units: Number of LSTM units
        depth: Number of LSTM layers
        dropout: Dropout rate
        epochs: Training epochs
        batch_size: Batch size
        validation_split: Fraction for validation (from training set)
    """

    # Configure GPU first
    gpu_available = configure_gpu()

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load pre-split data
    X_train, y_train, X_test, y_test, train_df, test_df = load_split_data(
        train_path, test_path, feature_cols, target_col
    )

    # Scale features - fit on training data only
    print("Scaling features...")
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)  # Transform test data using train scaler

    # Scale targets - fit on training data only
    print("Scaling targets...")
    target_scaler = MinMaxScaler()
    if len(y_train.shape) == 1:
        y_train = y_train.reshape(-1, 1)
    if len(y_test.shape) == 1:
        y_test = y_test.reshape(-1, 1)

    y_train_scaled = target_scaler.fit_transform(y_train)
    y_test_scaled = target_scaler.transform(y_test)  # Transform test data using train scaler

    if len(X_train_scaled.shape) == 2:
        # Assume each row is a single timestep
        X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
        X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

    input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
    H = y_train_scaled.shape[1]

    print(f"Input shape: {input_shape}")
    print(f"Output dimension (H): {H}")
    print(f"Train data shape: {X_train_scaled.shape}")
    print(f"Test data shape: {X_test_scaled.shape}")

    # Build model with GPU device specification
    print("Building LSTM model...")
    if gpu_available:
        with tf.device('/GPU:0'):
            model = build_lstm(input_shape, H, units=units, depth=depth, dropout=dropout)
    else:
        model = build_lstm(input_shape, H, units=units, depth=depth, dropout=dropout)

    model.summary()

    # Setup callbacks with GPU memory monitoring
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=10,
            restore_best_weights=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=5,
            monitor='val_loss'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=output_path / 'best_model.h5',
            save_best_only=True,
            monitor='val_loss'
        )
    ]

    # Train model using validation split from training data
    print("Training model...")
    print(f"Using device: {'/GPU:0' if gpu_available else '/CPU:0'}")

    history = model.fit(
        X_train_scaled, y_train_scaled,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Make predictions on both train and test sets
    print("Making predictions on training set...")
    y_train_pred_scaled = model.predict(X_train_scaled)
    y_train_pred_orig = target_scaler.inverse_transform(y_train_pred_scaled)
    y_train_true_orig = target_scaler.inverse_transform(y_train_scaled)

    print("Making predictions on test set...")
    y_test_pred_scaled = model.predict(X_test_scaled)
    y_test_pred_orig = target_scaler.inverse_transform(y_test_pred_scaled)
    y_test_true_orig = target_scaler.inverse_transform(y_test_scaled)

    # Evaluate model on both sets
    train_metrics = evaluate_model(y_train_true_orig, y_train_pred_orig, "LSTM Model (Training)")
    test_metrics = evaluate_model(y_test_true_orig, y_test_pred_orig, "LSTM Model (Test)")

    # Save model and scalers
    print("Saving model and scalers...")
    model.save(output_path / 'lstm_model.h5')

    with open(output_path / 'feature_scaler.pkl', 'wb') as f:
        pickle.dump(feature_scaler, f)

    with open(output_path / 'target_scaler.pkl', 'wb') as f:
        pickle.dump(target_scaler, f)

    # Save training history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(output_path / 'training_history.csv', index=False)

    # Save metrics for both train and test
    all_metrics = {
        'train_mse': train_metrics['mse'],
        'train_mae': train_metrics['mae'],
        'train_rmse': train_metrics['rmse'],
        'test_mse': test_metrics['mse'],
        'test_mae': test_metrics['mae'],
        'test_rmse': test_metrics['rmse'],
        'gpu_used': gpu_available
    }
    metrics_df = pd.DataFrame([all_metrics])
    metrics_df.to_csv(output_path / 'metrics.csv', index=False)

    # Save predictions for analysis
    train_results = pd.DataFrame({
        'true': y_train_true_orig.flatten(),
        'predicted': y_train_pred_orig.flatten(),
        'dataset': 'train'
    })
    test_results = pd.DataFrame({
        'true': y_test_true_orig.flatten(),
        'predicted': y_test_pred_orig.flatten(),
        'dataset': 'test'
    })

    all_predictions = pd.concat([train_results, test_results], ignore_index=True)
    all_predictions.to_csv(output_path / 'predictions.csv', index=False)

    print(f"\nTraining complete! Results saved to {output_path}")
    print(f"Training MSE: {train_metrics['mse']:.6f}")
    print(f"Test MSE: {test_metrics['mse']:.6f}")
    print(f"GPU Used: {gpu_available}")

    return model, history, train_metrics, test_metrics


if __name__ == "__main__":
    # Print TensorFlow and system info
    print("=== SYSTEM INFO ===")
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

    # Configuration - Update paths to your pre-split files
    TRAIN_PATH = "train_dataset.parquet"
    TEST_PATH = "test_dataset.parquet"
    OUTPUT_DIR = "/scratch/siads696f25s012_class_root/siads696f25s012_class/chrzhang/lstm_results"

    FEATURE_COLS = [
        'volume', 'open_interest', 'impl_volatility', 'delta', 'vega', 'theta',
        'prc', 'vol', 'price_diff_1d', 'price_diff_2d', 'price_diff_3d',
        'price_diff_5d', 'price_diff_8d', 'price_diff_13d', 'price_diff_21d',
        'price_diff_34d', 'price_diff_55d', 'price_diff_89d', 'price_diff_144d',
        'price_diff_233d', 'vol_delta_product', 'moneyness', 'volume_ma5',
        'iv_rolling_std', 'atm_iv', 'skew', 'curvature'
    ]

    TARGET_COL = 'iv_30d'

    # Model parameters
    UNITS = 64
    DEPTH = 2
    DROPOUT = 0.3
    EPOCHS = 100
    BATCH_SIZE = 64
    VALIDATION_SPLIT = 0.2  # Use 20% of training data for validation

    # Train model on pre-split data
    model, history, train_metrics, test_metrics = train_lstm_model_presplit(
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        feature_cols=FEATURE_COLS,
        target_col=TARGET_COL,
        output_dir=OUTPUT_DIR,
        units=UNITS,
        depth=DEPTH,
        dropout=DROPOUT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT
    )
