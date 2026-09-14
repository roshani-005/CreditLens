import numpy as np
from config import MODEL_FEATURES, TARGET
from data_prep import add_indian_fintech_layer, generate_demo_lending_club


def test_synthetic_layer_is_structurally_valid():
    raw = generate_demo_lending_club(n=300, seed=7)
    df = add_indian_fintech_layer(raw, seed=7)
    assert set(MODEL_FEATURES).issubset(df.columns)
    assert (df["upi_txn_frequency_30d"] <= df["upi_txn_frequency_60d"]).all()
    assert (df["upi_txn_frequency_60d"] <= df["upi_txn_frequency_90d"]).all()
    assert df["bill_payment_punctuality_score"].between(0, 100).all()
    assert df["proxy_low_income_area"].isin([0, 1]).all()
    assert TARGET not in df.columns  # feature synthesis never creates/reads target


def test_synthetic_features_do_not_depend_on_target_column():
    raw = generate_demo_lending_club(n=120, seed=11)
    a = add_indian_fintech_layer(raw, seed=11)
    raw_with_fake_target = raw.copy(); raw_with_fake_target[TARGET] = np.arange(len(raw)) % 2
    b = add_indian_fintech_layer(raw_with_fake_target, seed=11)
    for col in MODEL_FEATURES:
        assert a[col].equals(b[col])
