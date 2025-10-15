import tensorflow as tf
from tensorflow.keras import layers, models

# Scikit-learn models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================================
# DEEP LEARNING MODELS (TensorFlow/Keras)
# ============================================================================

def build_lstm(input_shape, H, units=64, depth=1, dropout=0.2):
    """
    Build LSTM model for sequence prediction.

    Args:
        input_shape: tuple (timesteps, features)
        H: output horizon (number of timesteps to predict)
        units: LSTM units per layer
        depth: number of LSTM layers
        dropout: dropout rate

    Returns:
        Compiled Keras model
    """
    model = models.Sequential()
    model.add(layers.Input(shape=input_shape))  # Explicit Input layer
    for _ in range(depth):
        model.add(layers.LSTM(units, return_sequences=True if _ < depth - 1 else False))
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(H))
    model.compile(optimizer='adam', loss='mse')
    return model


def build_gru(input_shape, H, units=64, depth=1, dropout=0.2):
    """
    Build GRU model for sequence prediction.

    Args:
        input_shape: tuple (timesteps, features)
        H: output horizon (number of timesteps to predict)
        units: GRU units per layer
        depth: number of GRU layers
        dropout: dropout rate

    Returns:
        Compiled Keras model
    """
    model = models.Sequential()
    for i in range(depth):
        return_seq = i < depth - 1
        if i == 0:
            model.add(layers.GRU(units, return_sequences=return_seq, input_shape=input_shape))
        else:
            model.add(layers.GRU(units, return_sequences=return_seq))
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(H))
    model.compile(optimizer='adam', loss='mse')
    return model


def build_seq2seq(input_shape, H, units=64, dropout=0.2, cell='LSTM'):
    """
    Build Sequence-to-Sequence model for multi-step prediction.

    Args:
        input_shape: tuple (timesteps, features)
        H: output horizon (number of timesteps to predict)
        units: encoder/decoder units
        dropout: dropout rate
        cell: 'LSTM' or 'GRU'

    Returns:
        Compiled Keras model
    """
    encoder_inputs = tf.keras.Input(shape=input_shape, name="encoder_inputs")
    if cell == 'LSTM':
        context = tf.keras.layers.LSTM(units, dropout=dropout, name="encoder_lstm")(encoder_inputs)
        decoder_seq = tf.keras.layers.LSTM(units, return_sequences=True, dropout=dropout, name="decoder_lstm")(
            tf.keras.layers.RepeatVector(H, name="repeat_vector")(context))
    else:
        context = tf.keras.layers.GRU(units, dropout=dropout, name="encoder_gru")(encoder_inputs)
        decoder_seq = tf.keras.layers.GRU(units, return_sequences=True, dropout=dropout, name="decoder_gru")(
            tf.keras.layers.RepeatVector(H, name="repeat_vector")(context))
    outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1), name="decoder_dense")(decoder_seq)
    model = tf.keras.Model(encoder_inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model


# ============================================================================
# TRADITIONAL ML MODELS (Scikit-learn)
# ============================================================================

def build_linear_regression(alpha=None, fit_intercept=True, normalize=False, with_scaling=True):
    """
    Build Linear Regression model (with optional regularization).

    Args:
        alpha: Regularization strength (None for OLS, float for Ridge/Lasso)
        fit_intercept: Whether to calculate intercept
        normalize: Whether to normalize features (deprecated, use with_scaling)
        with_scaling: Whether to include StandardScaler in pipeline

    Returns:
        LinearRegression model or Pipeline with scaler
    """
    if alpha is None:
        # Ordinary Least Squares
        model = LinearRegression(fit_intercept=fit_intercept, n_jobs=-1)
    elif alpha > 0:
        # Ridge Regression (L2 regularization)
        model = Ridge(alpha=alpha, fit_intercept=fit_intercept)

    if with_scaling:
        # Return pipeline with standardization
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])
        return pipeline

    return model


def build_ridge_regression(alpha=1.0, fit_intercept=True, solver='auto', with_scaling=True):
    """
    Build Ridge Regression model (L2 regularization).

    Args:
        alpha: Regularization strength (larger values = more regularization)
        fit_intercept: Whether to calculate intercept
        solver: Solver to use ('auto', 'svd', 'cholesky', 'lsqr', 'saga')
        with_scaling: Whether to include StandardScaler in pipeline

    Returns:
        Ridge model or Pipeline
    """
    model = Ridge(alpha=alpha, fit_intercept=fit_intercept, solver=solver)

    if with_scaling:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])
        return pipeline

    return model


def build_lasso_regression(alpha=1.0, fit_intercept=True, max_iter=1000, with_scaling=True):
    """
    Build Lasso Regression model (L1 regularization for feature selection).

    Args:
        alpha: Regularization strength
        fit_intercept: Whether to calculate intercept
        max_iter: Maximum number of iterations
        with_scaling: Whether to include StandardScaler in pipeline

    Returns:
        Lasso model or Pipeline
    """
    model = Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=max_iter)

    if with_scaling:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])
        return pipeline

    return model


# ============================================================================
# TREE-BASED MODELS
# ============================================================================

def build_random_forest(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        **kwargs
):
    """
    Build Random Forest Regressor.

    Args:
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees (None = unlimited)
        min_samples_split: Minimum samples required to split a node
        min_samples_leaf: Minimum samples required at leaf node
        max_features: Number of features to consider for best split
        random_state: Random seed for reproducibility
        n_jobs: Number of parallel jobs (-1 = use all processors)
        **kwargs: Additional RandomForestRegressor parameters

    Returns:
        RandomForestRegressor model
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
        **kwargs
    )
    return model


def build_gradient_boosting(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=1.0,
        random_state=42,
        **kwargs
):
    """
    Build Gradient Boosting Regressor.

    Args:
        n_estimators: Number of boosting stages
        learning_rate: Step size shrinkage
        max_depth: Maximum depth of individual trees
        min_samples_split: Minimum samples required to split
        min_samples_leaf: Minimum samples required at leaf
        subsample: Fraction of samples used for fitting trees
        random_state: Random seed
        **kwargs: Additional GradientBoostingRegressor parameters

    Returns:
        GradientBoostingRegressor model
    """
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        random_state=random_state,
        **kwargs
    )
    return model

