# tests/test_models.py
import pytest
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

SAMPLE_CSV    = "df_weather_sample_20k.csv"
XGB_MODEL     = "cuda_xgb_severity_model.joblib"
LOGREG_MODEL  = "logreg_balanced_01.joblib"


@pytest.fixture(scope="module")
def df_sample():
    """Load the 20k‐row sample once."""
    return pd.read_csv(SAMPLE_CSV)


def test_xgb_severity_model_runs(df_sample):
    # try importing xgboost here
    try:
        import xgboost
    except Exception as e:
        pytest.skip(f"Skipping XGB test; cannot import xgboost: {e}")

    # 1) exclude weather-cancellations
    df = df_sample[df_sample["delayDueWeather"] != 400].copy()

    # 2) engineered features
    wt_sev = {0:3.0,1:2.0,2:1.0,3:3.0,4:3.0,5:3.0,-1:1.0}
    for side in ("Origin", "Dest"):
        sev = f"Severity{side}Score"
        if sev not in df:
            df[sev] = df[f"weatherType{side}"].map(wt_sev).astype("float32")
        imp = f"WeatherImpact{side}"
        if imp not in df:
            df[imp] = (df[f"precipitation{side}"].fillna(0.0)*df[sev]).astype("float32")

    # 3) calendar columns
    if {"year","month","day"}.isdisjoint(df.columns) and "flDate" in df:
        dt = pd.to_datetime(df["flDate"], errors="coerce")
        df["year"], df["month"], df["day"] = dt.dt.year, dt.dt.month, dt.dt.day
    cal = [c for c in ("year","month","day") if c in df]

    # 4) assemble features
    feat = [
        "crsDepTime","crsArrTime","crsElapsedTime","distance",
        "startTimeOrigin_min","endTimeOrigin_min",
        "startTimeDest_min","endTimeDest_min",
        "precipitationOrigin","precipitationDest",
        "SeverityOriginScore","SeverityDestScore",
        "WeatherImpactOrigin","WeatherImpactDest",
        "weatherTypeOrigin","weatherTypeDest",
        "airlineCode","originAirport","destAirport",
    ] + cal

    missing = set(feat) - set(df.columns)
    if missing:
        pytest.skip(f"Missing XGB cols: {missing}")

    # 5) load & predict
    try:
        xgb_pipe = joblib.load(XGB_MODEL)
    except Exception as e:
        pytest.skip(f"Skipping XGB test; cannot load model: {e}")

    y_true = df["delayDueWeather"].astype("float32")
    y_pred = xgb_pipe.predict(df[feat])

    # 6) basic asserts
    assert len(y_pred) == len(y_true)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    assert np.isfinite(rmse)
    assert 0.2 <= r2 <= 1.0


def test_logreg_cancellation_model_runs(df_sample):
    # 1) build binary target
    df = df_sample.copy()
    df["is_canceled"] = (df["delayDueWeather"] == 400).astype(int)

    # 2) calendar columns
    if {"year","month","day"}.isdisjoint(df.columns) and "flDate" in df:
        dt = pd.to_datetime(df["flDate"], errors="coerce")
        df["year"], df["month"], df["day"] = dt.dt.year, dt.dt.month, dt.dt.day
    cal = [c for c in ("year","month","day") if c in df]

    # 3) assemble features
    feats = [
        "airlineCode","originAirport","destAirport",
        "crsDepTime","crsArrTime","crsElapsedTime","distance",
        "weatherTypeOrigin","severityOrigin","precipitationOrigin",
        "weatherTypeDest","severityDest","precipitationDest",
        "startTimeOrigin_min","endTimeOrigin_min",
        "startTimeDest_min","endTimeDest_min",
    ] + cal

    missing = set(feats) - set(df.columns)
    if missing:
        pytest.skip(f"Missing logreg cols: {missing}")

    X = df[feats].copy()
    y_true = df["is_canceled"]

    # 4) cast categoricals to str
    cat_cols = ["airlineCode","originAirport","destAirport",
                "weatherTypeOrigin","weatherTypeDest"]
    X[cat_cols] = X[cat_cols].fillna("Missing").astype(str)

    # 5) load & predict
    try:
        logreg = joblib.load(LOGREG_MODEL)
    except Exception as e:
        pytest.skip(f"Skipping logreg test; cannot load model: {e}")

    y_pred = logreg.predict(X)

    # 6) basic asserts
    assert len(y_pred) == len(y_true)
    acc = accuracy_score(y_true, y_pred)
    assert 0.4 <= acc <= 1.0
    assert set(y_pred) <= {0, 1}
