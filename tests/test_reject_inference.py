from config import MODEL_FEATURES, TARGET
from data_prep import add_indian_fintech_layer, generate_demo_lending_club
from reject_inference import apply_legacy_policy, fuzzy_augmentation_reject_inference, legacy_cutoff


def test_fuzzy_augmentation_trains_and_preserves_probabilities():
    raw = generate_demo_lending_club(n=700, seed=13)
    df = add_indian_fintech_layer(raw, seed=13)
    df[TARGET] = raw["loan_status"].eq("Charged Off").astype(int).to_numpy()
    cutoff = legacy_cutoff(df, approval_rate=0.7)
    approved = apply_legacy_policy(df, cutoff)
    result = fuzzy_augmentation_reject_inference(df, df[TARGET], approved, MODEL_FEATURES, seed=13)
    p = result.inferred_model.predict_proba(df.head(25))
    assert ((p >= 0) & (p <= 1)).all()
    assert result.rejected_rows > 0
    assert result.augmented_rows == approved.sum() + 2 * (~approved).sum()
