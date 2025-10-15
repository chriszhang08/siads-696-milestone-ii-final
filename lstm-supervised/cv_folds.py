# cv_folds.py
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from model_factory import build_lstm
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from itertools import product
import os

def custom_expanding_multi_dataset_cv_split(
        market_df_in,
        date_column='date',
        anchor_years=[(2005, 3), (2015, 2)],
        initial_train_years=2,
        test_duration_years=1,
        gap_days=0
):
    """
    Create expanding-window time series CV splits across multiple anchored periods.
    Each anchor represents a distinct market regime, and folds within each anchor
    expand the training window forward.

    Args:
        market_df_in: DataFrame containing market data
        date_column: name of the datetime column
        anchor_years: list of tuples (start_year, n_splits_from_anchor)
        initial_train_years: size of initial training window (years)
        test_duration_years: test set window (years)
        gap_days: gap between training and testing

    Returns:
        cv_splits: list of dictionaries containing fold info
        combined_df: sorted DataFrame
    """
    combined_df = market_df_in.copy()
    combined_df[date_column] = pd.to_datetime(combined_df[date_column])
    combined_df = combined_df.sort_values(by=date_column).reset_index(drop=True)

    print(f"Combined dataset shape: {combined_df.shape}")
    print(f"Date range: {combined_df[date_column].min()} to {combined_df[date_column].max()}")

    cv_splits = []
    fold_id = 0

    for anchor_year, n_splits in anchor_years:
        print(f"\n{'=' * 60}")
        print(f"Creating {n_splits} expanding folds from anchor year {anchor_year}")
        print(f"{'=' * 60}")

        for split_idx in range(n_splits):
            fold_id += 1

            train_start_year = anchor_year
            train_end_year = anchor_year + initial_train_years + (split_idx * test_duration_years)
            test_start_year = train_end_year
            test_end_year = test_start_year + test_duration_years

            train_start_date = pd.Timestamp(f"{train_start_year}-01-01")
            train_end_date = pd.Timestamp(f"{train_end_year}-01-01")
            test_start_date = pd.Timestamp(f"{test_start_year}-01-01") + pd.Timedelta(days=gap_days)
            test_end_date = pd.Timestamp(f"{test_end_year}-01-01")

            train_mask = (
                    (combined_df[date_column] >= train_start_date) &
                    (combined_df[date_column] < train_end_date)
            )
            test_mask = (
                    (combined_df[date_column] >= test_start_date) &
                    (combined_df[date_column] < test_end_date)
            )

            train_idx = combined_df[train_mask].index.values
            test_idx = combined_df[test_mask].index.values

            if len(train_idx) == 0 or len(test_idx) == 0:
                print(f"⚠️  Fold {fold_id} skipped — insufficient data for {train_start_year}-{test_end_year}")
                continue

            fold_info = {
                'fold_id': fold_id,
                'anchor_year': anchor_year,
                'split_within_anchor': split_idx + 1,
                'train_start_date': train_start_date,
                'train_end_date': train_end_date,
                'test_start_date': test_start_date,
                'test_end_date': test_end_date,
                'train_idx': train_idx,
                'test_idx': test_idx,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'train_period': f"{train_start_year}-{train_end_year}",
                'test_period': f"{test_start_year}-{test_end_year}"
            }

            cv_splits.append(fold_info)

            print(f"\nFold {fold_id} (Anchor {anchor_year}, Expanding {split_idx + 1}/{n_splits}):")
            print(f"  Train: {train_start_year}-{train_end_year} ({len(train_idx):,} samples)")
            print(f"  Test:  {test_start_year}-{test_end_year} ({len(test_idx):,} samples)")

    print(f"\n{'=' * 60}")
    print(f"Total expanding CV folds created: {len(cv_splits)}")
    print(f"{'=' * 60}\n")

    return cv_splits, combined_df


def nested_cv_evaluate_models(
        combined_df,
        cv_splits,
        feature_cols,
        target_col,
        model_configs,
        inner_cv_folds=3,
        verbose=True
):
    """
    Unified nested cross-validation with hyperparameter tuning for multiple models.

    Implements:
    - OUTER LOOP: Custom time series CV splits for model evaluation
    - INNER LOOP: GridSearchCV for hyperparameter tuning

    Args:
        combined_df: DataFrame with features and target
        cv_splits: Custom CV splits from custom_expanding_multi_dataset_cv_split
        feature_cols: List of feature column names
        target_col: Target column name
        model_configs: Dictionary of model configurations with 'model' and 'params'
        inner_cv_folds: Number of folds for inner hyperparameter tuning CV
        verbose: Print detailed progress

    Returns:
        results_summary: Dictionary with results for each model
        detailed_results: DataFrame with per-fold results
    """

    all_results = {}
    detailed_fold_results = []

    print("\n" + "=" * 80)
    print("STARTING NESTED CROSS-VALIDATION WITH HYPERPARAMETER TUNING")
    print("=" * 80)
    print(f"Number of models to evaluate: {len(model_configs)}")
    print(f"Number of outer CV folds: {len(cv_splits)}")
    print(f"Inner CV folds for hyperparameter tuning: {inner_cv_folds}")
    print("=" * 80 + "\n")

    # Evaluate each model
    for model_name, config in model_configs.items():
        print(f"\n{'#' * 80}")
        print(f"# EVALUATING MODEL: {model_name}")
        print(f"{'#' * 80}")

        model_template = config['model']
        param_grid = config['params']

        print(f"Hyperparameter grid size: {np.prod([len(v) for v in param_grid.values()])} combinations")

        outer_fold_scores = []
        outer_fold_details = []

        # OUTER LOOP: Evaluate model across custom CV splits
        for fold in cv_splits:
            fold_id = fold['fold_id']

            if verbose:
                print(f"\n{'-' * 60}")
                print(f"Outer Fold {fold_id}: {fold['train_period']} → {fold['test_period']}")
                print(f"{'-' * 60}")

            # Extract data for this fold
            X_train = combined_df.loc[fold['train_idx'], feature_cols].values
            y_train = combined_df.loc[fold['train_idx'], target_col].values
            X_test = combined_df.loc[fold['test_idx'], feature_cols].values
            y_test = combined_df.loc[fold['test_idx'], target_col].values

            # Handle NaN values, should be redundant
            train_valid_mask = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
            test_valid_mask = ~np.isnan(X_test).any(axis=1) & ~np.isnan(y_test)

            X_train_clean = X_train[train_valid_mask]
            y_train_clean = y_train[train_valid_mask]
            X_test_clean = X_test[test_valid_mask]
            y_test_clean = y_test[test_valid_mask]

            if len(X_train_clean) == 0 or len(X_test_clean) == 0:
                print(f"  ⚠️ Skipping fold {fold_id} - insufficient clean data")
                continue

            if verbose:
                print(f"  Training samples: {len(X_train_clean):,} (after cleaning)")
                print(f"  Test samples: {len(X_test_clean):,} (after cleaning)")

            # INNER LOOP: Hyperparameter tuning with GridSearchCV
            try:
                grid_search = GridSearchCV(
                    estimator=config['model'],
                    param_grid=config['params'],
                    cv=inner_cv_folds,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,  # ✅ Use all cores for CV
                    verbose=0,
                    pre_dispatch='2*n_jobs'  # ✅ Memory-efficient parallelization
                )

                if verbose:
                    print(f"  Running inner CV hyperparameter search...")

                grid_search.fit(X_train_clean, y_train_clean)

                # Get best model from inner CV
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                best_inner_score = -grid_search.best_score_  # Convert back to MSE

                if verbose:
                    print(f"  Best inner CV MSE: {best_inner_score:.6f}")
                    print(f"  Best params: {best_params}")

                # Evaluate on outer fold test set
                y_pred = best_model.predict(X_test_clean)

                # Calculate metrics
                mse = mean_squared_error(y_test_clean, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test_clean, y_pred)
                r2 = r2_score(y_test_clean, y_pred)

                outer_fold_scores.append(rmse)

                fold_result = {
                    'model_name': model_name,
                    'fold_id': fold_id,
                    'anchor_year': fold['anchor_year'],
                    'train_period': fold['train_period'],
                    'test_period': fold['test_period'],
                    'train_size': len(X_train_clean),
                    'test_size': len(X_test_clean),
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'best_params': str(best_params),
                    'inner_cv_mse': best_inner_score
                }

                outer_fold_details.append(fold_result)
                detailed_fold_results.append(fold_result)

                if verbose:
                    print(f"  Outer fold test RMSE: {rmse:.6f}")
                    print(f"  Outer fold test R²: {r2:.6f}")

            except Exception as e:
                print(f"  ❌ Error in fold {fold_id}: {str(e)}")
                continue

        # Aggregate results for this model
        if len(outer_fold_scores) > 0:
            avg_rmse = np.mean(outer_fold_scores)
            std_rmse = np.std(outer_fold_scores)

            # Group by anchor year
            anchor_groups = {}
            for detail in outer_fold_details:
                anchor = detail['anchor_year']
                if anchor not in anchor_groups:
                    anchor_groups[anchor] = []
                anchor_groups[anchor].append(detail['rmse'])

            all_results[model_name] = {
                'avg_rmse': avg_rmse,
                'std_rmse': std_rmse,
                'fold_scores': outer_fold_scores,
                'fold_details': outer_fold_details,
                'n_folds': len(outer_fold_scores),
                'anchor_performance': {
                    anchor: {
                        'avg_rmse': np.mean(scores),
                        'std_rmse': np.std(scores),
                        'n_folds': len(scores)
                    }
                    for anchor, scores in anchor_groups.items()
                }
            }

            print(f"\n{'=' * 60}")
            print(f"{model_name} - OVERALL PERFORMANCE")
            print(f"{'=' * 60}")
            print(f"Average RMSE: {avg_rmse:.6f} ± {std_rmse:.6f}")
            print(f"Number of folds: {len(outer_fold_scores)}")

            for anchor, perf in all_results[model_name]['anchor_performance'].items():
                print(f"\nAnchor {anchor}:")
                print(f"  RMSE: {perf['avg_rmse']:.6f} ± {perf['std_rmse']:.6f}")
                print(f"  Folds: {perf['n_folds']}")
        else:
            print(f"\n⚠️ No valid results for {model_name}")
            all_results[model_name] = None

    # Create detailed results DataFrame
    detailed_df = pd.DataFrame(detailed_fold_results)

    # Print final comparison
    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON")
    print("=" * 80)

    valid_models = {k: v for k, v in all_results.items() if v is not None}

    if valid_models:
        # Sort by RMSE
        sorted_models = sorted(valid_models.items(), key=lambda x: x[1]['avg_rmse'])

        print(f"\n{'Model':<30} {'RMSE':<20} {'Std':<15} {'Folds':<10}")
        print("-" * 80)

        for model_name, results in sorted_models:
            print(f"{model_name:<30} {results['avg_rmse']:<20.6f} "
                  f"{results['std_rmse']:<15.6f} {results['n_folds']:<10}")

        best_model_name = sorted_models[0][0]
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"   RMSE: {sorted_models[0][1]['avg_rmse']:.6f} ± {sorted_models[0][1]['std_rmse']:.6f}")

    print("=" * 80 + "\n")

    return all_results, detailed_df


def visualize_cv_splits(cv_splits):
    """Print a visual representation of the CV split structure."""
    print("\n📊 Cross-Validation Structure:")
    print("=" * 80)

    for fold in cv_splits:
        fold_id = fold['fold_id']
        anchor = fold['anchor_year']
        train_period = fold['train_period']
        test_period = fold['test_period']

        print(f"Fold {fold_id} [Anchor {anchor}]:")
        print(f"  ├─ Train: {train_period} ({fold['train_size']:,} samples)")
        print(f"  └─ Test:  {test_period} ({fold['test_size']:,} samples)")

    print("=" * 80)


def build_sequences(df, feature_cols, target_col, W=60, H=3, date_col='date'):
    """
    Build rolling input/output sequences for multi-step forecasting.

    Args:
        df: DataFrame with features and target
        feature_cols: list of feature column names
        target_col: str, column to forecast
        W: int, input window length (lookback)
        H: int, forecast horizon (number of steps ahead)
        date_col: str, date column name

    Returns:
        X: np.array, shape (N, W, F) - sequences
        y: np.array, shape (N, H) - targets
        meta: pd.DataFrame, metadata with date information
    """
    Xs, ys, meta = [], [], []

    # Sort by date
    df = df.sort_values(date_col).reset_index(drop=True)

    X_mat = df[feature_cols].values
    y_vec = df[target_col].values
    dates = df[date_col].values

    for i in range(W - 1, len(df) - H):
        x_win = X_mat[i - W + 1:i + 1]  # shape (W, F)
        y_target = y_vec[i + 1:i + H + 1] if H > 1 else y_vec[i + 1]  # shape (H,) or scalar

        # Skip if any NaN
        if np.isnan(y_target).any() if isinstance(y_target, np.ndarray) else np.isnan(y_target):
            continue
        if np.isnan(x_win).any():
            continue

        Xs.append(x_win)
        ys.append(y_target if H > 1 else [y_target])
        meta.append({'end_date': dates[i], 'window_end_idx': i})

    X = np.array(Xs)
    y = np.array(ys)
    meta = pd.DataFrame(meta)

    return X, y, meta


def time_series_split_sequences(X, y, meta, train_size=0.75, val_size=0.15):
    """
    Split sequences into train/val/test by time (no shuffling).

    Args:
        X, y, meta: outputs from build_sequences
        train_size: fraction of data for training
        val_size: fraction of data for validation (rest goes to test)

    Returns:
        (X_train, y_train, meta_train),
        (X_val, y_val, meta_val),
        (X_test, y_test, meta_test)
    """
    N = len(X)
    train_end = int(N * train_size)
    val_end = train_end + int(N * val_size)

    X_train, y_train, meta_train = X[:train_end], y[:train_end], meta.iloc[:train_end]
    X_val, y_val, meta_val = X[train_end:val_end], y[train_end:val_end], meta.iloc[train_end:val_end]
    X_test, y_test, meta_test = X[val_end:], y[val_end:], meta.iloc[val_end:]

    return (X_train, y_train, meta_train), (X_val, y_val, meta_val), (X_test, y_test, meta_test)


def evaluate_lstm_fold(
        combined_df,
        fold,
        feature_cols,
        target_col,
        hyperparams,
        model_builder=build_lstm,
        W=60,
        H=30,
        epochs=50,
        batch_size=32,
        verbose=0
):
    """
    Evaluate LSTM on a single CV fold with given hyperparameters.

    Args:
        combined_df: DataFrame with features
        fold: Single fold dictionary from cv_splits
        feature_cols: List of feature column names
        target_col: Target column name
        model_builder: Function that builds LSTM model
        hyperparams: Dictionary of LSTM hyperparameters
        W: Lookback window
        H: Forecast horizon
        epochs: Training epochs
        batch_size: Batch size
        verbose: Verbosity level

    Returns:
        Dictionary with fold results
    """
    # Get data for this fold
    train_data = combined_df.loc[fold['train_idx']]
    test_data = combined_df.loc[fold['test_idx']]

    # Build sequences
    X_train, y_train, _ = build_sequences(train_data, feature_cols, target_col, W=W, H=H)
    X_test, y_test, _ = build_sequences(test_data, feature_cols, target_col, W=W, H=H)

    if len(X_train) == 0 or len(X_test) == 0:
        return None

    # Split train into train/val for early stopping
    val_split_idx = int(len(X_train) * 0.85)
    X_train_fit = X_train[:val_split_idx]
    y_train_fit = y_train[:val_split_idx]
    X_val = X_train[val_split_idx:]
    y_val = y_train[val_split_idx:]

    # Build model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (W, num_features)

    model = model_builder(
        input_shape=input_shape,
        H=H,
        units=hyperparams.get('units', 64),
        depth=hyperparams.get('depth', 1),
        dropout=hyperparams.get('dropout', 0.2)
    )

    # Recompile with custom learning rate
    model.compile(
        optimizer=Adam(learning_rate=hyperparams.get('learning_rate', 0.001)),
        loss='mse',
        metrics=['mae']
    )

    # Callbacks
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=0
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=0
    )

    # Train
    history = model.fit(
        X_train_fit, y_train_fit,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=verbose
    )

    # Evaluate on fold test set
    y_pred = model.predict(X_test, verbose=0)

    # Flatten if needed for metrics
    y_test_flat = y_test.flatten()
    y_pred_flat = y_pred.flatten()

    mse = mean_squared_error(y_test_flat, y_pred_flat)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_flat, y_pred_flat)
    r2 = r2_score(y_test_flat, y_pred_flat)

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mse': mse,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'final_epoch': len(history.history['loss']),
        'best_val_loss': min(history.history['val_loss'])
    }


def nested_cv_evaluate_lstm(
        combined_df,
        cv_splits,
        feature_cols,
        target_col,
        lstm_hyperparam_grid,
        H=30,
        max_combinations=10,
        epochs=50,
        batch_size=32,
        verbose=True
):
    """
    Nested CV for LSTM with hyperparameter tuning using grid search.

    Args:
        combined_df: DataFrame with features
        cv_splits: Custom CV splits
        feature_cols: List of feature columns
        target_col: Target column name
        lstm_hyperparam_grid: Dictionary with hyperparameter options
            Example: {
                'units': [32, 64, 128],
                'depth': [1, 2],
                'dropout': [0.2, 0.3],
                'learning_rate': [0.001, 0.0001],
                'W': [30, 60, 90]  # <-- NOW INCLUDED!
            }
        H: Forecast horizon (fixed)
        max_combinations: Maximum hyperparameter combinations to try
        epochs: Training epochs per trial
        batch_size: Batch size
        verbose: Verbosity

    Returns:
        results_summary: Dictionary with LSTM results
        detailed_results: DataFrame with per-fold results
    """

    print(f"\n{'#' * 80}")
    print(f"# EVALUATING LSTM MODEL")
    print(f"{'#' * 80}")
    print(f"Forecast horizon (H): {H}")

    # Check if W is in hyperparameter grid
    if 'W' in lstm_hyperparam_grid:
        print(f"Lookback windows (W) to test: {lstm_hyperparam_grid['W']}")
    else:
        print("WARNING: 'W' not found in hyperparameter grid. Using default W=60")
        lstm_hyperparam_grid['W'] = [60]

    print(f"Total hyperparameter grid size: {np.prod([len(v) for v in lstm_hyperparam_grid.values()])}")

    # Generate hyperparameter combinations
    keys = lstm_hyperparam_grid.keys()
    values = lstm_hyperparam_grid.values()
    all_combinations = [dict(zip(keys, v)) for v in product(*values)]

    # Limit combinations if too many
    if len(all_combinations) > max_combinations:
        print(f"Limiting to {max_combinations} random combinations (out of {len(all_combinations)})")
        np.random.seed(42)
        selected_indices = np.random.choice(len(all_combinations), max_combinations, replace=False)
        combinations_to_try = [all_combinations[i] for i in selected_indices]
    else:
        combinations_to_try = all_combinations

    print(f"Testing {len(combinations_to_try)} hyperparameter combinations\n")

    # Store results for each combination
    combination_results = []
    detailed_fold_results = []

    for combo_idx, hyperparams in enumerate(combinations_to_try):
        # Extract W from hyperparameters
        W = hyperparams.pop('W', 60)  # Remove W from dict, default to 60 if not present

        if verbose:
            print(f"\n{'-' * 60}")
            print(f"Combination {combo_idx + 1}/{len(combinations_to_try)}")
            print(f"  W (lookback): {W}, H (forecast): {H}")
            print(f"  Model params: {hyperparams}")
            print(f"{'-' * 60}")

        fold_scores = []

        # Evaluate this hyperparameter combination across all folds
        for fold in cv_splits:
            fold_id = fold['fold_id']

            if verbose:
                print(f"  Fold {fold_id}: {fold['train_period']} → {fold['test_period']}", end=" ")

            try:
                fold_result = evaluate_lstm_fold(
                    combined_df=combined_df,
                    fold=fold,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    model_builder=build_lstm,
                    hyperparams=hyperparams,
                    W=W,  # Pass W to fold evaluation
                    H=H,
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0
                )

                if fold_result is None:
                    if verbose:
                        print("SKIPPED (insufficient data)")
                    continue

                fold_scores.append(fold_result['rmse'])

                # Store detailed results (include W in results)
                detailed_fold_results.append({
                    'model_name': 'LSTM',
                    'fold_id': fold_id,
                    'anchor_year': fold['anchor_year'],
                    'train_period': fold['train_period'],
                    'test_period': fold['test_period'],
                    'W': W,  # Store W value
                    'H': H,
                    'units': hyperparams.get('units'),
                    'depth': hyperparams.get('depth'),
                    'dropout': hyperparams.get('dropout'),
                    'learning_rate': hyperparams.get('learning_rate'),
                    'hyperparams': str({**hyperparams, 'W': W}),  # Full hyperparam string
                    'rmse': fold_result['rmse'],
                    'mae': fold_result['mae'],
                    'r2': fold_result['r2'],
                    'train_size': fold_result['train_size'],
                    'test_size': fold_result['test_size']
                })

                if verbose:
                    print(f"RMSE: {fold_result['rmse']:.6f}, R²: {fold_result['r2']:.6f}")

            except Exception as e:
                if verbose:
                    print(f"ERROR: {str(e)}")
                    import traceback
                    traceback.print_exc()
                continue

        # Aggregate results for this combination (restore W to dict for storage)
        if len(fold_scores) > 0:
            avg_rmse = np.mean(fold_scores)
            std_rmse = np.std(fold_scores)

            full_hyperparams = {**hyperparams, 'W': W}  # Add W back to hyperparams

            combination_results.append({
                'hyperparams': full_hyperparams,
                'avg_rmse': avg_rmse,
                'std_rmse': std_rmse,
                'n_folds': len(fold_scores),
                'fold_scores': fold_scores
            })

            if verbose:
                print(f"  Combination avg RMSE: {avg_rmse:.6f} ± {std_rmse:.6f}")

    # Find best hyperparameter combination
    if len(combination_results) > 0:
        best_combo = min(combination_results, key=lambda x: x['avg_rmse'])

        print(f"\n{'=' * 60}")
        print(f"LSTM - BEST HYPERPARAMETERS")
        print(f"{'=' * 60}")
        print(f"Lookback window (W): {best_combo['hyperparams']['W']}")
        print(f"Forecast horizon (H): {H}")
        print(f"Model architecture:")
        for param, value in best_combo['hyperparams'].items():
            if param != 'W':
                print(f"  {param}: {value}")
        print(f"Average RMSE: {best_combo['avg_rmse']:.6f} ± {best_combo['std_rmse']:.6f}")
        print(f"Number of folds: {best_combo['n_folds']}")
        print(f"{'=' * 60}\n")

        # Prepare results summary
        results_summary = {
            'LSTM': {
                'avg_rmse': best_combo['avg_rmse'],
                'std_rmse': best_combo['std_rmse'],
                'n_folds': best_combo['n_folds'],
                'fold_scores': best_combo['fold_scores'],
                'best_hyperparams': best_combo['hyperparams'],
                'best_W': best_combo['hyperparams']['W'],
                'H': H,
                'all_combinations_tried': combination_results
            }
        }

        detailed_df = pd.DataFrame(detailed_fold_results)

        return results_summary, detailed_df
    else:
        print("\n⚠️ No valid LSTM results obtained")
        return None, pd.DataFrame()

def compare_all_models(
        traditional_results,
        lstm_results,
        save_path=None
):
    """
    Compare traditional ML models with LSTM model.

    Args:
        traditional_results: Results from nested_cv_evaluate_models
        lstm_results: Results from nested_cv_evaluate_lstm
        save_path: Optional path to save comparison table

    Returns:
        comparison_df: DataFrame with all model comparisons
    """
    print("\n" + "=" * 80)
    print("COMPLETE MODEL COMPARISON: TRADITIONAL ML vs DEEP LEARNING")
    print("=" * 80 + "\n")

    comparison_data = []

    # Add traditional models
    for model_name, results in traditional_results.items():
        if results is not None:
            comparison_data.append({
                'Model': model_name,
                'Type': 'Traditional ML',
                'Avg RMSE': results['avg_rmse'],
                'Std RMSE': results['std_rmse'],
                'N Folds': results['n_folds']
            })

    # Add LSTM
    if lstm_results is not None and 'LSTM' in lstm_results:
        lstm_res = lstm_results['LSTM']
        comparison_data.append({
            'Model': 'LSTM',
            'Type': 'Deep Learning',
            'Avg RMSE': lstm_res['avg_rmse'],
            'Std RMSE': lstm_res['std_rmse'],
            'N Folds': lstm_res['n_folds']
        })

    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Avg RMSE').reset_index(drop=True)
    comparison_df['Rank'] = range(1, len(comparison_df) + 1)

    # Display
    print(comparison_df.to_string(index=False))

    # Highlight best model
    best_model = comparison_df.iloc[0]
    print(f"\n🏆 BEST OVERALL MODEL: {best_model['Model']}")
    print(f"   Type: {best_model['Type']}")
    print(f"   RMSE: {best_model['Avg RMSE']:.6f} ± {best_model['Std RMSE']:.6f}")
    print("=" * 80 + "\n")

    # Save if requested
    if save_path:
        comparison_df.to_csv(save_path, index=False)
        print(f"Comparison saved to: {save_path}")

    return comparison_df


def load_cv_fold(fold_info, data_type='both'):
    """
    Load a specific CV fold from parquet files.

    Args:
        fold_info: Dictionary containing fold information (from cv_splits)
        data_type: 'train', 'test', or 'both'

    Returns:
        train_df, test_df (if data_type='both')
        train_df (if data_type='train')
        test_df (if data_type='test')
    """

    if data_type in ['train', 'both']:
        if not os.path.exists(fold_info['train_file']):
            raise FileNotFoundError(f"Train file not found: {fold_info['train_file']}")
        train_df = pl.read_parquet(fold_info['train_file'])

    if data_type in ['test', 'both']:
        if not os.path.exists(fold_info['test_file']):
            raise FileNotFoundError(f"Test file not found: {fold_info['test_file']}")
        test_df = pl.read_parquet(fold_info['test_file'])

    if data_type == 'both':
        return train_df, test_df
    elif data_type == 'train':
        return train_df
    elif data_type == 'test':
        return test_df


def load_cv_metadata(output_dir):
    """
    Load CV fold metadata from the output directory.

    Args:
        output_dir: Directory containing the CV fold files

    Returns:
        metadata_df: DataFrame with fold information
        cv_splits: List of dictionaries (same format as original function)
    """
    metadata_file = os.path.join(output_dir, 'cv_folds_metadata.csv')
    if not os.path.exists(metadata_file):
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    metadata_df = pd.read_csv(metadata_file)
    cv_splits = metadata_df.to_dict('records')

    return metadata_df, cv_splits
