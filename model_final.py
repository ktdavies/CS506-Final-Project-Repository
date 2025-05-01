#!/usr/bin/env python
# coding: utf-8

# In[1]:

def main():
    import pandas as pd
    import numpy as np

    df = pd.read_csv('sampledForModelCleanV2.csv')
    # # 1) Columns to load (no airTime or delayDueLateAircraft)
    # cols = [
    #     'flDate',
    #     'airlineCode',
    #     'originAirport',
    #     'destAirport',
    #     'crsDepTime',
    #     'crsArrTime',
    #     'cancelled',
    #     'cancellationCode',
    #     'crsElapsedTime',
    #     'distance',
    #     'delayDueWeather',
    #     'weatherTypeOrigin',
    #     'severityOrigin',
    #     'startTimeOrigin',
    #     'endTimeOrigin',
    #     'precipitationOrigin',
    #     'weatherTypeDest',
    #     'severityDest',
    #     'startTimeDest',
    #     'endTimeDest',
    #     'precipitationDest'
    # ]

    # # 2) Load only those columns, parsing the date‐time fields
    # df = pd.read_csv(
    #     'cleaned.csv',
    #     usecols=cols,
    #     parse_dates=[
    #         'flDate',
    #         'startTimeOrigin',
    #         'endTimeOrigin',
    #         'startTimeDest',
    #         'endTimeDest'
    #     ]
    # )

    # 2b) Drop the COVID period (March 2020 through December 2021)
    covid_start = pd.Timestamp('2020-03-01')
    covid_end   = pd.Timestamp('2021-12-31')

    df['flDate'] = pd.to_datetime(df['flDate'])
    df = df[~df['flDate'].between(covid_start, covid_end)]

    # 3) Fill NaNs in weather delay, then convert cancellations ‘B’ → 400 min
    df['delayDueWeather'] = df['delayDueWeather'].fillna(0)
    mask = (df['cancelled'] == 1) & (df['cancellationCode'] == 'B')
    df.loc[mask, 'delayDueWeather'] = 400

    print(f"Loaded DataFrame (excl. COVID) with shape: {df.shape}")
    df.head()


    # In[2]:


    # 1) Filter to only flights with a weather‐based delay
    df_weather = df[df['delayDueWeather'] > 0].copy()

    # 2) Quick sanity check
    print(f"Rows with non-zero weather delay: {df_weather.shape[0]}")

    # 3) Peek at the first few
    df_weather.head(10)


    # In[3]:


    df_weather = df_weather.drop(columns=['cancelled', 'cancellationCode'])


    # In[4]:


    print(df_weather.columns)
    print(len(df_weather))
    df_weather.head()


    # In[5]:


    import pandas as pd

    # Use the original datetime columns (before you convert → _min)
    origin_cols = [
        'weatherTypeOrigin',
        'severityOrigin',
        'startTimeOrigin',
        'endTimeOrigin',
        'precipitationOrigin'
    ]
    dest_cols = [
        'weatherTypeDest',
        'severityDest',
        'startTimeDest',
        'endTimeDest',
        'precipitationDest'
    ]

    # Masks for fully‐missing vs partially‐missing on the ORIGINAL columns
    origin_all_missing = df_weather[origin_cols].isnull().all(axis=1)
    origin_any_missing = df_weather[origin_cols].isnull().any(axis=1)
    dest_all_missing   = df_weather[dest_cols].isnull().all(axis=1)
    dest_any_missing   = df_weather[dest_cols].isnull().any(axis=1)

    # Print counts
    print("=== Origin Weather Data Missingness ===")
    print("Completely missing:", origin_all_missing.sum())
    print("Partially missing :", ((origin_any_missing) & (~origin_all_missing)).sum())
    print("Fully present     :", (~origin_any_missing).sum(), "\n")

    print("=== Destination Weather Data Missingness ===")
    print("Completely missing:", dest_all_missing.sum())
    print("Partially missing :", ((dest_any_missing) & (~dest_all_missing)).sum())
    print("Fully present     :", (~dest_any_missing).sum(), "\n")

    # Crosstab of intersection
    print("=== Rows by (origin_all_missing, dest_all_missing) ===")
    print(pd.crosstab(
        origin_all_missing,
        dest_all_missing,
        rownames=['origin_all_missing'],
        colnames=['dest_all_missing']
    ))


    # In[6]:


    # 1) Record the original size
    before_count = len(df_weather)

    # 2) If you still have raw “UNK” strings, drop them:
    for col in ['weatherTypeOrigin','weatherTypeDest','severityOrigin','severityDest']:
        if df_weather[col].dtype == object:
            df_weather = df_weather[df_weather[col] != 'UNK']

    # 3) If you’ve already mapped “UNK” → -1, drop those too:
    for col in ['weatherTypeOrigin','weatherTypeDest','severityOrigin','severityDest']:
        if pd.api.types.is_integer_dtype(df_weather[col]):
            df_weather = df_weather[df_weather[col] != -1]

    # 4) Record the new size
    after_count = len(df_weather)

    # 5) Print before & after
    print(f"Rows before dropping UNK/-1 entries: {before_count}")
    print(f"Rows after  dropping UNK/-1 entries: {after_count}")
    print(f"Total dropped: {before_count - after_count}")


    # In[7]:


    # Recompute the “all-missing” masks on df_weather
    origin_cols = [
        'weatherTypeOrigin','severityOrigin',
        'startTimeOrigin','endTimeOrigin',
        'precipitationOrigin'
    ]
    dest_cols = [
        'weatherTypeDest','severityDest',
        'startTimeDest','endTimeDest',
        'precipitationDest'
    ]

    origin_all_missing = df_weather[origin_cols].isna().all(axis=1)
    dest_all_missing   = df_weather[dest_cols].isna().all(axis=1)

    print("Before drop, rows:", df_weather.shape[0])
    # Drop rows where both blocks are missing
    df_weather = df_weather.loc[~(origin_all_missing & dest_all_missing)].copy()
    print("After  drop, rows:", df_weather.shape[0])


    # In[9]:


    # Boolean masks (re-compute or reuse from above)
    origin_all_missing = df_weather[['weatherTypeOrigin','severityOrigin',
                                    'startTimeOrigin','endTimeOrigin',
                                    'precipitationOrigin']].isna().all(axis=1)
    dest_all_missing   = df_weather[['weatherTypeDest','severityDest',
                                    'startTimeDest','endTimeDest',
                                    'precipitationDest']].isna().all(axis=1)

    # Fill origin-missing rows
    df_weather.loc[origin_all_missing, [
        'startTimeOrigin', 'endTimeOrigin',
        'weatherTypeOrigin','severityOrigin',
        'precipitationOrigin'
    ]] = [-1, -1, -1, -1, 0]

    # Fill dest-missing rows
    df_weather.loc[dest_all_missing, [
        'startTimeDest', 'endTimeDest',
        'weatherTypeDest','severityDest',
        'precipitationDest'
    ]] = [-1, -1, -1, -1, 0]

    # (Optional) Verify
    print("After block-fill, any origin NaNs left?", 
        df_weather[['weatherTypeOrigin','severityOrigin',
                    'startTimeOrigin','endTimeOrigin',
                    'precipitationOrigin']].isna().any().any())
    print("After block-fill, any dest NaNs left?", 
        df_weather[['weatherTypeDest','severityDest',
                    'startTimeDest','endTimeDest',
                    'precipitationDest']].isna().any().any())


    # In[10]:


    df_weather.head(10)


    # In[11]:


    # … earlier in the same cell …

    # 1) Ensure those four columns are true datetimes
    for col in ['startTimeOrigin','endTimeOrigin','startTimeDest','endTimeDest']:
        df_weather[col] = pd.to_datetime(df_weather[col], errors='coerce')

    # 2) Now convert datetime → minutes‐of‐day, missing→‐1
    for col in ['startTimeOrigin','endTimeOrigin','startTimeDest','endTimeDest']:
        df_weather[col + '_min'] = (
            df_weather[col]
            .dt.hour.mul(60)
            .add(df_weather[col].dt.minute)
            .fillna(-1)
            .astype(int)
        )

    # 3) Drop the originals
    df_weather.drop(columns=[
        'startTimeOrigin','endTimeOrigin','startTimeDest','endTimeDest'
    ], inplace=True)

    # … rest of your mapping steps …


    # In[12]:


    import pandas as pd

    print("Columns before:", df_weather.columns.tolist())

    # 1) Fill precipitation NaNs with 0.0 (float)
    df_weather['precipitationOrigin'] = df_weather['precipitationOrigin'].fillna(0.0)
    df_weather['precipitationDest']   = df_weather['precipitationDest'].fillna(0.0)

    # 2) Datetime → minutes-from-midnight; sentinel = –1.0 (float)
    for col in ['startTimeOrigin','endTimeOrigin','startTimeDest','endTimeDest']:
        if col in df_weather.columns:
            df_weather[col + '_min'] = (
                df_weather[col].dt.hour.mul(60)
                .add(df_weather[col].dt.minute)
                .fillna(-1.0)
                .astype('float32')
            )
        else:
            print(f"Warning: {col} not in df_weather, skipping conversion")

    # 3) Drop original datetime columns
    df_weather.drop(columns=[c for c in
            ['startTimeOrigin','endTimeOrigin','startTimeDest','endTimeDest']
            if c in df_weather.columns], inplace=True)

    # 4) Severity mapping ➜ floats  (sentinel –1.0)
    severity_map = {'Light': 0.0, 'Moderate': 1.0, 'Heavy': 2.0}
    for side in ['Origin','Dest']:
        col = f'severity{side}'
        df_weather[col] = (
            df_weather[col]
            .map(severity_map)
            .fillna(-1.0)
            .astype('float32')
        )

    print("\nSeverity mapping (string ➜ code):")
    for k,v in severity_map.items():
        print(f"  {k:<8} → {v:.1f}")
    print("  (missing) → -1.0")

    # 5) Weather-type mapping ➜ floats  (sentinel –1.0)
    all_types = (
        pd.concat([df_weather['weatherTypeOrigin'],
                df_weather['weatherTypeDest']])
        .dropna()
        .loc[lambda s: s != -1]            # keep real strings only
        .unique()
    )

    weather_map = {wt: float(i) for i, wt in enumerate(all_types)}

    for side in ['Origin','Dest']:
        col = f'weatherType{side}'
        df_weather[col] = (
            df_weather[col]
            .map(weather_map)
            .fillna(-1.0)
            .astype('float32')
        )

    print("\nWeather-type mapping (code ➜ string):")
    for wt, code in weather_map.items():
        print(f"  {int(code):2d} → {wt}")
    print("  (missing) → -1.0")

    df_weather.head()


    # In[27]:


    # ===============================================================
    #  XGBoost “delay-severity” model — WITH calendar features
    #  (assumes df_weather already in memory, with string col “flDate”)
    # ===============================================================
    import numpy as np
    import pandas as pd
    import joblib
    from xgboost import XGBRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    # ───────────────────────────────────────────────────────────────
    # 0)  Keep *non-cancellations* & build severity / impact columns
    # ───────────────────────────────────────────────────────────────
    df_nc = df_weather[df_weather["delayDueWeather"] != 400].copy()

    # integer weatherType → severity weight (floats)
    # 0-Snow • 1-Rain • 2-Fog • 3-Storm • 4-Cold • 5-Hail • –1-Missing
    wt_sev = {0:3.0, 1:2.0, 2:1.0, 3:3.0, 4:3.0, 5:3.0, -1:1.0}
    df_nc["SeverityOriginScore"] = df_nc["weatherTypeOrigin"].map(wt_sev).astype("float32")
    df_nc["SeverityDestScore"]   = df_nc["weatherTypeDest"  ].map(wt_sev).astype("float32")

    df_nc["WeatherImpactOrigin"] = df_nc["precipitationOrigin"] * df_nc["SeverityOriginScore"]
    df_nc["WeatherImpactDest"]   = df_nc["precipitationDest"]   * df_nc["SeverityDestScore"]

    # ───────────────────────────────────────────────────────────────
    # 0a)  Calendar parts  (year / month / day)
    # ───────────────────────────────────────────────────────────────
    DATE_COL = "flDate"          # <-- change if your column name differs
    if DATE_COL in df_nc.columns:
        dt              = pd.to_datetime(df_nc[DATE_COL], errors="coerce")
        df_nc["year"]   = dt.dt.year.astype("int16")
        df_nc["month"]  = dt.dt.month.astype("int8")
        df_nc["day"]    = dt.dt.day.astype("int8")
        cal_cols        = ["year", "month", "day"]
    else:
        print(f"⚠️  column “{DATE_COL}” not found – calendar features skipped")
        cal_cols        = []

    # ───────────────────────────────────────────────────────────────
    # 1)  Feature matrix & target
    # ───────────────────────────────────────────────────────────────
    feat_cols = [
        # schedule / distance
        "crsDepTime","crsArrTime","crsElapsedTime","distance",
        "startTimeOrigin_min","endTimeOrigin_min",
        "startTimeDest_min","endTimeDest_min",
        # precipitation & engineered scores
        "precipitationOrigin","precipitationDest",
        "SeverityOriginScore","SeverityDestScore",
        "WeatherImpactOrigin","WeatherImpactDest",
        # encoded weather & IDs
        "weatherTypeOrigin","weatherTypeDest",
        "airlineCode","originAirport","destAirport",
    ] + cal_cols

    X = df_nc[feat_cols]
    y = df_nc["delayDueWeather"].astype("float32")

    # ───────────────────────────────────────────────────────────────
    # 2)  Train / test split
    # ───────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # ───────────────────────────────────────────────────────────────
    # 3)  Column lists
    # ───────────────────────────────────────────────────────────────
    num_cols = [
        "crsDepTime","crsArrTime","crsElapsedTime","distance",
        "startTimeOrigin_min","endTimeOrigin_min",
        "startTimeDest_min","endTimeDest_min",
        "precipitationOrigin","precipitationDest",
        "SeverityOriginScore","SeverityDestScore",
        "WeatherImpactOrigin","WeatherImpactDest"
    ] + cal_cols

    cat_cols = [
        "weatherTypeOrigin","weatherTypeDest",
        "airlineCode","originAirport","destAirport"
    ]

    # ───────────────────────────────────────────────────────────────
    # 4)  Preprocessor + XGB (CUDA) pipeline
    # ───────────────────────────────────────────────────────────────
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols)
    ])

    xgb_pipe = Pipeline([
        ("pre" , pre),
        ("xgb" , XGBRegressor(
            tree_method="hist",     # XGBoost ≥2.0
            device="cuda",
            random_state=42,
            learning_rate=0.10,
            max_depth=6,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=1,
            verbosity=0
        ))
    ])

    # ───────────────────────────────────────────────────────────────
    # 5)  Fit & evaluate
    # ───────────────────────────────────────────────────────────────
    xgb_pipe.fit(X_train, y_train)
    preds = xgb_pipe.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    print(f"XGBoost (CUDA) → RMSE: {rmse:.2f}")
    print(f"                    R²:  {r2:.3f}")

    # ───────────────────────────────────────────────────────────────
    # 6)  Save model
    # ───────────────────────────────────────────────────────────────
    joblib.dump(xgb_pipe, "cuda_xgb_severity_model.joblib")
    print("✅  Model saved to 'cuda_xgb_severity_model.joblib'")


    # In[28]:


    # ===============================================================
    #  Logistic-regression (balanced) — WITH calendar features
    #  (assumes df_weather already present, with “flDate” string col)
    # ===============================================================
    import pandas as pd, joblib
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report

    # ───────────────────────────────────────────────────────────────
    # 0)  Binary target
    # ───────────────────────────────────────────────────────────────
    df_weather["is_canceled"] = (df_weather["delayDueWeather"] == 400).astype(int)

    # ───────────────────────────────────────────────────────────────
    # 0a)  Calendar parts from “flDate”
    # ───────────────────────────────────────────────────────────────
    DATE_COL = "flDate"               # rename if your column differs
    if DATE_COL in df_weather.columns:
        dt                     = pd.to_datetime(df_weather[DATE_COL], errors="coerce")
        df_weather["year"]     = dt.dt.year.astype("int16")
        df_weather["month"]    = dt.dt.month.astype("int8")
        df_weather["day"]      = dt.dt.day.astype("int8")
        cal_cols               = ["year", "month", "day"]
    else:
        print(f"⚠️  column “{DATE_COL}” not found – calendar features skipped")
        cal_cols               = []

    # ───────────────────────────────────────────────────────────────
    # 1)  Feature list  (weather-type & severity already numeric)
    # ───────────────────────────────────────────────────────────────
    features = [
        "airlineCode", "originAirport", "destAirport",
        "crsDepTime", "crsArrTime", "crsElapsedTime", "distance",
        "weatherTypeOrigin", "severityOrigin", "precipitationOrigin",
        "weatherTypeDest",   "severityDest",   "precipitationDest",
        "startTimeOrigin_min", "endTimeOrigin_min",
        "startTimeDest_min",   "endTimeDest_min"
    ] + cal_cols

    X = df_weather[features]
    y = df_weather["is_canceled"]

    # ───────────────────────────────────────────────────────────────
    # 2)  Train / test split (stratified)
    # ───────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ───────────────────────────────────────────────────────────────
    # 3)  Column groups
    #     – airline / airports & weather codes are treated as categories
    #     – everything else (incl. year/month/day) is numeric
    # ───────────────────────────────────────────────────────────────
    cat_cols = ["airlineCode", "originAirport", "destAirport",
                "weatherTypeOrigin", "weatherTypeDest"]
    num_cols = [c for c in features if c not in cat_cols]

    # OneHotEncoder needs strings
    for df_part in (X_train, X_test):
        df_part[cat_cols] = df_part[cat_cols].astype("string")

    # ───────────────────────────────────────────────────────────────
    # 4)  Pre-processor
    # ───────────────────────────────────────────────────────────────
    preproc = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(),                       num_cols)
    ])

    # ───────────────────────────────────────────────────────────────
    # 5)  Logistic-regression pipeline (balanced, C=0.01)
    # ───────────────────────────────────────────────────────────────
    logreg_bal_01 = Pipeline([
        ("pre", preproc),
        ("clf", LogisticRegression(
            C=0.01,
            penalty="l2",
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
            n_jobs=1
        ))
    ])

    # ───────────────────────────────────────────────────────────────
    # 6)  Fit, evaluate, save
    # ───────────────────────────────────────────────────────────────
    logreg_bal_01.fit(X_train, y_train)

    y_pred = logreg_bal_01.predict(X_test)
    print("\nClassification report (balanced, C=0.01):\n")
    print(classification_report(y_test, y_pred))

    joblib.dump(logreg_bal_01, "logreg_balanced_01.joblib")
    print("✅  Model saved to 'logreg_balanced_01.joblib'")


    # In[ ]:


    # ===============================================================
    #   FUTURE-SCENARIO DELAY FORECAST  (XGB-severity model, v2)
    #   – calendar “year / month / day” features added
    #   – severity & window bumps use float columns
    # ===============================================================
    import pandas as pd, joblib
    from sklearn.model_selection import train_test_split

    # ════════════════════════════════════════════════════════════════
    # 0)  Merge cluster  ▸  filter non-cancellations  ▸  add calendar
    # ════════════════════════════════════════════════════════════════
    clusters = (
        pd.read_csv("airport_cluster_assignments_iata.csv")
        .rename(columns={"AirportCode": "originAirport", "Cluster": "cluster"})
    )

    df = (
        df_weather.drop(columns=["cluster"], errors="ignore")
                .merge(clusters[["originAirport", "cluster"]],
                        on="originAirport", how="left")
    )

    df_nc = df[df["delayDueWeather"] != 400].copy()

    # ── calendar parts from string “flDate” (YYYY-MM-DD)
    if "flDate" in df_nc.columns and {"year","month","day"}.isdisjoint(df_nc.columns):
        dt               = pd.to_datetime(df_nc["flDate"], errors="coerce")
        df_nc["year"]    = dt.dt.year.astype("int16")
        df_nc["month"]   = dt.dt.month.astype("int8")
        df_nc["day"]     = dt.dt.day.astype("int8")

    # ════════════════════════════════════════════════════════════════
    # 1)  Feature matrix expected by the XGB model
    #     (cast severity & window cols → float32 so bumps stay valid)
    # ════════════════════════════════════════════════════════════════
    float_cols = [
        "severityOrigin", "severityDest",
        "startTimeOrigin_min", "endTimeOrigin_min",
        "startTimeDest_min",   "endTimeDest_min"
    ]
    df_nc[float_cols] = df_nc[float_cols].astype("float32")

    # engineered impacts
    df_nc["SeverityOriginScore"] = df_nc["severityOrigin"]
    df_nc["SeverityDestScore"]   = df_nc["severityDest"]
    df_nc["WeatherImpactOrigin"] = df_nc["precipitationOrigin"] * df_nc["SeverityOriginScore"]
    df_nc["WeatherImpactDest"]   = df_nc["precipitationDest"]   * df_nc["SeverityDestScore"]

    base_feats = [
        # schedule / distance
        "crsDepTime","crsArrTime","crsElapsedTime","distance",
        "startTimeOrigin_min","endTimeOrigin_min",
        "startTimeDest_min","endTimeDest_min",
        # precip & encoded weather / severity
        "precipitationOrigin","precipitationDest",
        "severityOrigin","severityDest",
        "weatherTypeOrigin","weatherTypeDest",
        # engineered impacts
        "SeverityOriginScore","SeverityDestScore",
        "WeatherImpactOrigin","WeatherImpactDest",
        # IDs + cluster
        "airlineCode","originAirport","destAirport","cluster",
        # calendar parts (added above)
        "year","month","day"
    ]

    # keep only columns that actually exist (e.g. if year/month/day missing)
    base_feats = [c for c in base_feats if c in df_nc.columns]

    X = df_nc[base_feats]
    y = df_nc["delayDueWeather"]

    _, X_test, _, _ = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    X_test[float_cols] = X_test[float_cols].astype("float32")  # make sure view is float

    # ════════════════════════════════════════════════════════════════
    # 2)  Baseline delay predictions
    # ════════════════════════════════════════════════════════════════
    xgb_sev = joblib.load("cuda_xgb_severity_model.joblib")
    delay_base = xgb_sev.predict(X_test.drop(columns=["cluster"]))

    # ════════════════════════════════════════════════════════════════
    # 3)  Ten-year future scenario
    #     Snow=0 • Rain=1 • Fog=2 • Storm=3   (integer codes)
    # ════════════════════════════════════════════════════════════════
    years = 10
    SNOW, RAIN, FOG, STORM = 0, 1, 2, 3

    cluster_pct = {
        0: {"precip":0.0303, FOG:0.0110, SNOW:0.0050},
        1: {              FOG:0.0170, SNOW:0.0050},
        2: {                          SNOW:0.0050},
        3: {STORM:0.0100, RAIN:-0.0100, SNOW:0.0050}
    }

    X_future = X_test.copy()
    X_future[float_cols] = X_future[float_cols].astype("float32")

    for cl, changes in cluster_pct.items():
        m_cl = X_future["cluster"] == cl
        if not m_cl.any():
            continue

        # ── precipitation scaling
        if "precip" in changes:
            fac = 1 + changes["precip"] * years
            X_future.loc[m_cl, ["precipitationOrigin","precipitationDest"]] = (
                X_future.loc[m_cl, ["precipitationOrigin","precipitationDest"]].fillna(0) * fac
            )

        # ── severity bumps & duration stretch
        for wcode, pct_per_year in changes.items():
            if wcode == "precip":
                continue
            bump  = pct_per_year * years
            scale = 1 + pct_per_year * years

            mo = m_cl & (X_future["weatherTypeOrigin"] == wcode)
            md = m_cl & (X_future["weatherTypeDest"]   == wcode)

            # severity bump
            X_future.loc[mo, "severityOrigin"] += bump
            X_future.loc[md, "severityDest"]   += bump

            # symmetric stretch / compression
            win_o = X_future.loc[mo, "endTimeOrigin_min"] - X_future.loc[mo, "startTimeOrigin_min"]
            X_future.loc[mo, "startTimeOrigin_min"] -= (scale-1) * win_o / 2
            X_future.loc[mo, "endTimeOrigin_min"]   += (scale-1) * win_o / 2

            win_d = X_future.loc[md, "endTimeDest_min"] - X_future.loc[md, "startTimeDest_min"]
            X_future.loc[md, "startTimeDest_min"]  -= (scale-1) * win_d / 2
            X_future.loc[md, "endTimeDest_min"]    += (scale-1) * win_d / 2

    for col in ("year","month","day"):
        if col in X_future.columns and col == "year":
            X_future[col] += years             

    # recompute impacts
    X_future["SeverityOriginScore"] = X_future["severityOrigin"]
    X_future["SeverityDestScore"]   = X_future["severityDest"]
    X_future["WeatherImpactOrigin"] = X_future["precipitationOrigin"] * X_future["SeverityOriginScore"]
    X_future["WeatherImpactDest"]   = X_future["precipitationDest"]   * X_future["SeverityDestScore"]

    # ════════════════════════════════════════════════════════════════
    # 4)  Future prediction & report
    # ════════════════════════════════════════════════════════════════
    delay_future = xgb_sev.predict(X_future.drop(columns=["cluster"]))

    print(f"Avg current delay : {delay_base.mean():.1f} min")
    print(f"Avg future  delay : {delay_future.mean():.1f} min")
    print(f"Change over {years} yr: {delay_future.mean() - delay_base.mean():+4.1f} min")


    # In[46]:


    import numpy as np, pandas as pd, joblib
    from sklearn.model_selection import train_test_split

    # ══════════════════════════════════════════════════════════════
    # 0)  CONSTANTS  ▸  MODEL  ▸  FEATURE LIST (+ calendar parts)
    # ══════════════════════════════════════════════════════════════
    YEARS = 10
    SNOW, RAIN, FOG, STORM = 0, 1, 2, 3        # integer weather codes
    cluster_pct = {
        0: {"precip":0.303, FOG:0.011, SNOW:0.005},
        1: {              FOG:0.017, SNOW:0.005},
        2: {                            SNOW:0.005},
        3: { STORM:0.010, RAIN:-0.010, SNOW:0.005},
    }

    cat_cols = ["airlineCode","originAirport","destAirport",
                "weatherTypeOrigin","weatherTypeDest"]         # treated as categories

    num_cols = [
        "crsDepTime","crsArrTime","crsElapsedTime","distance",
        "severityOrigin","precipitationOrigin",
        "severityDest","precipitationDest",
        "startTimeOrigin_min","endTimeOrigin_min",
        "startTimeDest_min","endTimeDest_min",
        "year","month","day"                                    # ← new calendar parts
    ]
    feat_cols = [*cat_cols, *num_cols]

    logreg = joblib.load("logreg_balanced_01.joblib")

    # ══════════════════════════════════════════════════════════════
    # 1)  BASELINE SAMPLE  (current weather + calendar features)
    # ══════════════════════════════════════════════════════════════
    # add calendar columns once if they don’t already exist
    if {"year","month","day"}.isdisjoint(df_weather.columns) and "flDate" in df_weather.columns:
        dt                  = pd.to_datetime(df_weather["flDate"], errors="coerce")
        df_weather["year"]  = dt.dt.year.astype("int16")
        df_weather["month"] = dt.dt.month.astype("int8")
        df_weather["day"]   = dt.dt.day.astype("int8")

    # stratified sample identical to training split
    _, X_test = train_test_split(
        df_weather[feat_cols],
        test_size=0.20,
        random_state=42,
        stratify=df_weather["is_canceled"]
    )

    # encoder expects strings for categoricals
    X_test[cat_cols] = X_test[cat_cols].fillna("Missing").astype(str)

    curr_rate = logreg.predict(X_test).mean() * 100
    print(f"Current cancellation rate           : {curr_rate:5.2f}%")

    # ══════════════════════════════════════════════════════════════
    # 2)  BUILD TEN-YEAR FUTURE SCENARIO
    # ══════════════════════════════════════════════════════════════
    clusters = (
        pd.read_csv("airport_cluster_assignments_iata.csv")
        .rename(columns={"AirportCode":"originAirport","Cluster":"cluster"})
    )

    X_future = (
        X_test.copy()
            .merge(clusters[["originAirport","cluster"]], on="originAirport", how="left")
    )

    # cast columns that will get fractional bumps
    float_cols = ["severityOrigin","severityDest",
                "startTimeOrigin_min","endTimeOrigin_min",
                "startTimeDest_min","endTimeDest_min"]
    X_future[float_cols] = X_future[float_cols].astype("float32")
    X_future[["precipitationOrigin","precipitationDest"]] = \
        X_future[["precipitationOrigin","precipitationDest"]].fillna(0.0)

    for cl, changes in cluster_pct.items():
        m_cl = X_future["cluster"] == cl
        if not m_cl.any():
            continue

        # ── precipitation scaling
        if "precip" in changes:
            X_future.loc[m_cl, ["precipitationOrigin","precipitationDest"]] *= \
                1 + changes["precip"] * YEARS

        # ── severity / duration adjustments per weather code
        for wcode, pct in changes.items():
            if wcode == "precip":
                continue
            bump  = pct * YEARS
            scale = 1 + pct * YEARS

            mo = m_cl & (X_future["weatherTypeOrigin"] == str(wcode))
            md = m_cl & (X_future["weatherTypeDest"]   == str(wcode))

            X_future.loc[mo, "severityOrigin"] += bump
            X_future.loc[md, "severityDest"]   += bump

            win_o = X_future.loc[mo, "endTimeOrigin_min"] - X_future.loc[mo, "startTimeOrigin_min"]
            X_future.loc[mo, "startTimeOrigin_min"] -= (scale-1) * win_o / 2
            X_future.loc[mo, "endTimeOrigin_min"]   += (scale-1) * win_o / 2

            win_d = X_future.loc[md, "endTimeDest_min"] - X_future.loc[md, "startTimeDest_min"]
            X_future.loc[md, "startTimeDest_min"] -= (scale-1) * win_d / 2
            X_future.loc[md, "endTimeDest_min"]   += (scale-1) * win_d / 2


    # ensure categoricals are strings after manipulations
    X_future = X_future.drop(columns="cluster")
    X_future[cat_cols] = X_future[cat_cols].astype(str)

    # ══════════════════════════════════════════════════════════════
    # 3)  FUTURE PREDICTION & DELTA
    # ══════════════════════════════════════════════════════════════
    future_rate = logreg.predict(X_future[feat_cols]).mean() * 100
    print(f"Predicted cancellation rate in {YEARS} yrs : {future_rate:5.2f}%")
    print(f"Change over {YEARS} years                  : {future_rate-curr_rate:+5.2f}%")

if __name__ == "__main__":
    main()

    # In[ ]:




