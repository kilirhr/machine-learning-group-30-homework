# Shared code of the three project notebooks (Group 30).
# Notebook 1 explains and checks every step, notebooks 2 and 3 import the steps from here.
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "repeat_purchase_90d"
RANDOM_STATE = 42

PLOT_BLUE = "#1779A8"
PLOT_PURPLE = "#6A3D9A"

# candidate variables by group (the groups are also used in the open-ended section)
CURRENT_INVOICE = ["Total Liquido", "Total Imp_", "Total Portes", "Total Desc_ Global",
                   "Total Desc_ Linha", "Custo", "N_ Detalhes", "invoice_count_current_purchase"]
HISTORY_TIMING = ["days_since_previous_purchase", "customer_tenure_days", "prior_purchase_count",
                  "prior_mean_gap_days", "prior_std_gap_days"]
HISTORY_VALUE = ["prior_total_spend", "prior_mean_purchase_value", "prior_std_purchase_value",
                 "prior_min_purchase_value", "prior_max_purchase_value"]
ROLLING_COUNT = ["purchase_count_30d", "purchase_count_90d", "purchase_count_180d",
                 "purchase_count_365d"]
ROLLING_SPEND = ["spend_30d", "spend_90d", "spend_180d", "spend_365d"]
SUPPLIED_NUMERIC = CURRENT_INVOICE + HISTORY_TIMING + HISTORY_VALUE + ROLLING_COUNT + ROLLING_SPEND

ROW_FEATURES = ["discount_rate", "purchases_per_year"]
CUSTOMER_HISTORY = ["prior_repeat_share", "recent_repeat_share", "same_season_last_year",
                    "same_season_two_years_ago"]
STRUCTURAL_FLAGS = ["is_first_purchase", "has_gap_history"]
ROW_FLAGS = ["zero_cost"]
PURCHASE_CATEGORIES = ["payment_type", "salesperson", "month"]

# shares between 0 and 1 and the noise column are scaled without log1p
PLAIN_COLUMNS = ["discount_rate", "prior_repeat_share", "recent_repeat_share", "random_noise"]

# number of earlier purchases a history value needs before it can exist
PURCHASES_NEEDED = {"days_since_previous_purchase": 1, "prior_mean_purchase_value": 1,
                    "prior_min_purchase_value": 1, "prior_max_purchase_value": 1,
                    "prior_std_purchase_value": 2, "prior_mean_gap_days": 2,
                    "prior_std_gap_days": 3}


# drop the training rows that have no feature value at all
def drop_empty_rows(train):
    feature_columns = [col for col in train.columns if col not in ["ID", "customer_id", TARGET]]
    is_empty = train[feature_columns].isna().all(axis=1)
    return train[~is_empty].reset_index(drop=True)


# replace impossible values by a recomputed value or by a missing value
def repair_values(purchases):
    purchases = purchases.copy()
    prior_count = purchases["prior_purchase_count"]
    prior_spend = purchases["prior_total_spend"]

    # -4 days is a code and not a number of days
    for col in ["days_since_previous_purchase", "prior_mean_gap_days"]:
        purchases.loc[purchases[col] < 0, col] = np.nan

    # the mean purchase value follows from two other columns
    purchases["prior_mean_purchase_value"] = np.nan
    has_history = prior_count > 0
    purchases.loc[has_history, "prior_mean_purchase_value"] = prior_spend / prior_count

    # one purchase cannot be bigger than the total spend
    too_big = purchases["prior_max_purchase_value"] > prior_spend + 0.01
    purchases.loc[too_big, "prior_max_purchase_value"] = np.nan

    # spend_365d: placeholder to missing, wrong sign removed, values above the total to missing
    purchases.loc[purchases["spend_365d"] >= 1e8, "spend_365d"] = np.nan
    purchases["spend_365d"] = purchases["spend_365d"].abs()
    purchases.loc[purchases["spend_365d"] > prior_spend + 0.01, "spend_365d"] = np.nan
    return purchases


# fill history values that cannot exist yet and mark these rows with two flags
def fill_structural_gaps(purchases, fill_value=0):
    purchases = purchases.copy()
    prior_count = purchases["prior_purchase_count"]
    for col, needed in PURCHASES_NEEDED.items():
        purchases.loc[prior_count < needed, col] = fill_value
    purchases.loc[prior_count == 0, "spend_365d"] = 0
    purchases["is_first_purchase"] = (prior_count == 0).astype(int)
    purchases["has_gap_history"] = (prior_count >= 2).astype(int)
    return purchases


# new features that only use values of the same row
def add_row_features(purchases):
    purchases = purchases.copy()
    # month as text, so the encoder treats it as a category
    purchases["month"] = pd.to_datetime(purchases["purchase_date"]).dt.strftime("%b")

    discounts = purchases["Total Desc_ Linha"] + purchases["Total Desc_ Global"]
    purchases["discount_rate"] = (discounts / purchases["Total Bruto"].replace(0, np.nan)).fillna(0)

    tenure_years = purchases["customer_tenure_days"] / 365
    purchases["purchases_per_year"] = 0.0
    is_known = tenure_years > 0
    purchases.loc[is_known, "purchases_per_year"] = purchases["prior_purchase_count"] / tenure_years

    purchases["zero_cost"] = (purchases["Custo"] == 0).astype(int)
    return purchases


# purchases of one customer that lie between far_days and near_days before each purchase
def count_purchases_between(purchase_days, far_days, near_days):
    until_near = np.searchsorted(purchase_days, purchase_days - near_days, side="right")
    until_far = np.searchsorted(purchase_days, purchase_days - far_days, side="right")
    return until_near - until_far


# history features of a row, built only from earlier purchases of the same customer
def add_customer_history(purchases):
    purchases = purchases.copy()
    purchases["purchase_day"] = pd.to_datetime(purchases["purchase_date"])
    ordered = purchases.sort_values(["customer_id", "purchase_day"], kind="stable")
    customer = ordered["customer_id"]

    # a gap of at most 90 days means that the purchase before was followed by a repeat purchase
    gap = ordered["days_since_previous_purchase"]
    gap = gap.where(gap >= 0)
    was_repeat = (gap <= 90).astype(float).where(gap.notna())
    gaps_seen = gap.notna().groupby(customer).cumsum()
    repeats_seen = was_repeat.fillna(0).groupby(customer).cumsum()
    ordered["prior_repeat_share"] = (repeats_seen / gaps_seen.replace(0, np.nan)).fillna(0)
    recent_share = was_repeat.groupby(customer).rolling(5, min_periods=1).mean()
    ordered["recent_repeat_share"] = recent_share.reset_index(level=0, drop=True).fillna(0)

    # purchases in the 90 days after the same calendar date, one and two years ago
    ordered["same_season_last_year"] = 0
    ordered["same_season_two_years_ago"] = 0
    day_number = ordered["purchase_day"].to_numpy().astype("datetime64[D]").astype(int)
    last_year = ordered.columns.get_loc("same_season_last_year")
    two_years = ordered.columns.get_loc("same_season_two_years_ago")
    for rows in ordered.groupby("customer_id").indices.values():
        ordered.iloc[rows, last_year] = count_purchases_between(day_number[rows], 365, 275)
        ordered.iloc[rows, two_years] = count_purchases_between(day_number[rows], 730, 640)
    return ordered.loc[purchases.index].drop(columns="purchase_day")


# all row rules in their order: cleaning, customer history, new features
# new purchases go through the same rules, with the training purchases as their history
def prepare_purchases(train, new_purchases=None):
    purchases = drop_empty_rows(train).assign(part="train")
    if new_purchases is not None:
        purchases = pd.concat([purchases, new_purchases.assign(part="new")], ignore_index=True)
    purchases = repair_values(purchases)
    purchases = add_customer_history(purchases)
    purchases = fill_structural_gaps(purchases)
    purchases = add_row_features(purchases)

    is_train = purchases["part"] == "train"
    prepared_train = purchases[is_train].drop(columns="part")
    prepared_train = prepared_train.sort_values(["purchase_date", "ID"], kind="stable")
    prepared_train = prepared_train.reset_index(drop=True)
    prepared_train[TARGET] = prepared_train[TARGET].astype(int)
    if new_purchases is None:
        return prepared_train
    prepared_new = purchases[~is_train].drop(columns=["part", TARGET]).reset_index(drop=True)
    return prepared_train, prepared_new


# train.csv and clientes.csv, test.csv is only read in the deployment (Notebook 3)
def load_train(folder="../data/raw"):
    train = pd.read_csv(f"{folder}/train.csv")
    clientes = pd.read_csv(f"{folder}/clientes.csv")
    return train, clientes


# prepared training purchases saved by Notebook 1
def load_interim(folder="../data/interim"):
    return pd.read_csv(f"{folder}/purchases_train.csv", float_precision="round_trip")


# log1p for amounts, negative values are set to 0 first
def log_amount(values):
    return np.log1p(np.clip(values, 0, None))


# clips the named columns of both parts to the quantiles of the training part
def winsorize(train, test, columns, lower=0.01, upper=0.99):
    low, high = train[columns].quantile(lower), train[columns].quantile(upper)
    train, test = train.copy(), test.copy()
    train[columns] = train[columns].clip(low, high, axis=1)
    test[columns] = test[columns].clip(low, high, axis=1)
    return train, test


# imputation, log1p and scaling for numbers, "missing" and one-hot for categories
# the defaults are the choices of the project, the arguments exist to test other values
def build_preprocessor(numeric, flags, categorical, use_log=True, scaler=StandardScaler,
                       impute="median", min_frequency=30):
    log_columns = [col for col in numeric if col not in PLAIN_COLUMNS] if use_log else []
    plain_columns = [col for col in numeric if col not in log_columns]
    log_steps = Pipeline([
        ("imputer", SimpleImputer(strategy=impute)),
        ("log1p", FunctionTransformer(log_amount, feature_names_out="one-to-one")),
        ("scaler", scaler()),
    ])
    plain_steps = Pipeline([
        ("imputer", SimpleImputer(strategy=impute)),
        ("scaler", scaler()),
    ])
    category_steps = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(min_frequency=min_frequency, handle_unknown="infrequent_if_exist",
                                 sparse_output=False)),
    ])
    return ColumnTransformer([
        ("log", log_steps, log_columns),
        ("plain", plain_steps, plain_columns),
        ("flag", "passthrough", flags),
        ("cat", category_steps, categorical),
    ], remainder="drop", verbose_feature_names_out=False)


# pre-processing and model in one pipeline, so every step is fitted on training rows only
def build_pipeline(model, numeric, flags, categorical, **prep_options):
    return Pipeline([
        ("prep", build_preprocessor(numeric, flags, categorical, **prep_options)),
        ("model", model),
    ])


# TimeSeriesSplit with a gap in days: train on the past, validate on the next period
# a training row only stays if its 90-day target window ends before the validation period starts,
# otherwise its target would already know purchases from the validation period
def time_folds(dates, n_splits, gap_days=90):
    dates = pd.to_datetime(dates).reset_index(drop=True)
    folds = []
    for train_idx, val_idx in TimeSeriesSplit(n_splits=n_splits).split(dates):
        window_closed = dates.iloc[train_idx] + pd.Timedelta(days=gap_days) < dates.iloc[val_idx[0]]
        folds.append((train_idx[window_closed.to_numpy()], val_idx))
    return folds


# newest outer fold = holdout, the rest (minus the gap) = development part with inner folds
def split_development_holdout(train, n_outer=5, n_inner=4):
    dev_idx, holdout_idx = time_folds(train["purchase_date"], n_outer)[-1]
    development = train.iloc[dev_idx].reset_index(drop=True)
    holdout = train.iloc[holdout_idx].reset_index(drop=True)
    inner_folds = time_folds(development["purchase_date"], n_inner)
    return development, holdout, inner_folds


# original variable of an encoded column (a one-hot column belongs to its category)
def source_variable(column_name, categorical):
    for variable in categorical:
        if column_name.startswith(variable + "_"):
            return variable
    return column_name
