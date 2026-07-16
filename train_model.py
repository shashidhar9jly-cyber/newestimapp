"""
Phase 2 - Model Training
Trains one Gradient Boosting Regressor per output component (33 targets),
evaluates each, and saves a single bundle (models + metadata) for the
Streamlit prediction app to load.

Run once: python3 train_model.py
"""

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

DATA_PATH = "data/boiler_synthetic_dataset.csv"
MODEL_PATH = "models/boiler_weight_models.joblib"

INPUT_CATEGORICAL = ["Boiler_Type"]
INPUT_NUMERIC = [
    "Capacity_MW",
    "Steam_Flow_TPH",
    "Rated_Pressure_kgcm2",
    "Rated_Steam_Temperature_C",
]

CATEGORY_GROUPS = {
    "Pressure Parts": [
        "Superheater_Weight_kg", "Reheater_Weight_kg",
        "Water_Circulation_System_Weight_kg", "Economiser_Weight_kg",
        "Steam_Drum_Weight_kg",
    ],
    "Structural": [
        "Pipe_Supports_kg", "Buckstay_System_kg", "Boiler_Rough_Mountings_kg",
        "Foundation_Bolts_kg", "Grouting_kg", "Insert_Materials_kg",
    ],
    "Thermal": [
        "Insulation_Weight_kg", "Seal_Boxes_kg", "Sealing_Arrangements_kg",
    ],
    "Mechanical Equipment": [
        "Hoppers_kg", "Chutes_kg", "Deaerator_kg", "Soot_Blowers_kg",
        "Boiler_Mountings_Fittings_kg", "Maintenance_Tools_kg",
    ],
    "Instrumentation": [
        "Field_Instruments_kg", "Instrumentation_Erection_Hardware_kg",
    ],
    "Electrical": [
        "Electrical_Panels_kg", "Electrical_Erection_Materials_kg",
        "Illumination_System_kg",
    ],
    "Chemical Systems": [
        "HP_Dosing_System_kg", "LP_Dosing_System_kg",
        "Common_Blowdown_Equipment_kg", "Sample_Coolers_kg",
    ],
    "Miscellaneous": [
        "Fasteners_kg", "Gaskets_kg", "Spares_kg", "Miscellaneous_Items_kg",
    ],
}
ALL_TARGETS = [t for group in CATEGORY_GROUPS.values() for t in group]


def build_features(df):
    """One-hot encode Boiler_Type, keep numerics as-is. Returns X, feature_cols, boiler_types."""
    boiler_types = sorted(df["Boiler_Type"].unique().tolist())
    dummies = pd.get_dummies(df["Boiler_Type"], prefix="Type")
    # Ensure consistent column order regardless of what's present in a given df
    for bt in boiler_types:
        col = f"Type_{bt}"
        if col not in dummies.columns:
            dummies[col] = 0
    dummies = dummies[[f"Type_{bt}" for bt in boiler_types]]
    X = pd.concat([df[INPUT_NUMERIC].reset_index(drop=True),
                    dummies.reset_index(drop=True)], axis=1)
    feature_cols = INPUT_NUMERIC + [f"Type_{bt}" for bt in boiler_types]
    return X[feature_cols], feature_cols, boiler_types


def train_bundle(data_path=DATA_PATH, verbose=True):
    """Trains all 33 component models and returns the bundle dict (does not
    write to disk). Used both by the CLI entry point below and by app.py as
    a live-retrain fallback if a pre-saved .joblib file can't be loaded
    (e.g. scikit-learn version mismatch between environments)."""
    df = pd.read_csv(data_path)
    X, feature_cols, boiler_types = build_features(df)
    y_all = df[ALL_TARGETS]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_all, test_size=0.2, random_state=42
    )

    models = {}
    metrics = {}
    t0 = time.time()

    for target in ALL_TARGETS:
        gbr = GradientBoostingRegressor(
            n_estimators=250,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            random_state=42,
        )
        gbr.fit(X_train, y_train[target])
        pred = gbr.predict(X_test)

        metrics[target] = {
            "r2": round(float(r2_score(y_test[target], pred)), 4),
            "mae": round(float(mean_absolute_error(y_test[target], pred)), 1),
            "mape": round(float(mean_absolute_percentage_error(y_test[target], pred)) * 100, 2),
        }
        models[target] = gbr
        if verbose:
            print(f"{target:45s} R2={metrics[target]['r2']:.3f}  "
                  f"MAPE={metrics[target]['mape']:.2f}%")

    # Global feature importance (averaged across all component models)
    importance_matrix = np.array([models[t].feature_importances_ for t in ALL_TARGETS])
    avg_importance = importance_matrix.mean(axis=0)
    feature_importance = dict(zip(feature_cols, [round(float(v), 4) for v in avg_importance]))
    feature_importance = dict(
        sorted(feature_importance.items(), key=lambda kv: kv[1], reverse=True)
    )

    # Dataset-level stats used by the app for benchmarking / percentile insights
    dataset_stats = {
        "total_weight_by_type": df.groupby("Boiler_Type")["Total_Estimated_Weight_kg"]
                                  .describe()[["mean", "std", "min", "max"]].round(1)
                                  .to_dict(orient="index"),
        "capacity_range": [int(df["Capacity_MW"].min()), int(df["Capacity_MW"].max())],
        "steam_flow_range": [int(df["Steam_Flow_TPH"].min()), int(df["Steam_Flow_TPH"].max())],
        "pressure_range": [int(df["Rated_Pressure_kgcm2"].min()), int(df["Rated_Pressure_kgcm2"].max())],
        "temperature_range": [int(df["Rated_Steam_Temperature_C"].min()), int(df["Rated_Steam_Temperature_C"].max())],
        "n_rows": len(df),
    }

    bundle = {
        "models": models,
        "feature_cols": feature_cols,
        "boiler_types": boiler_types,
        "category_groups": CATEGORY_GROUPS,
        "all_targets": ALL_TARGETS,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "dataset_stats": dataset_stats,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
    }

    if verbose:
        avg_r2 = np.mean([m["r2"] for m in metrics.values()])
        avg_mape = np.mean([m["mape"] for m in metrics.values()])
        print(f"\nTrained {len(ALL_TARGETS)} models in {time.time()-t0:.1f}s")
        print(f"Average R2 across components: {avg_r2:.3f}")
        print(f"Average MAPE across components: {avg_mape:.2f}%")

    return bundle


def main():
    bundle = train_bundle(verbose=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved bundle -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
