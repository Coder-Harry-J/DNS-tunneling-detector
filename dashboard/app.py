"""
DNS Tunneling Detection Dashboard — Flask Backend
Loads both Tuned and Baseline/Pre-tuned models (CART, SVM, ANN) and serves REST API.
"""
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

# ─── Path Setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
DATA_DIR = Path(__file__).resolve().parent

app = Flask(__name__)

# Feature schema (18 columns used by trained models)
FEATURE_SCHEMA = [
    'query_length', 'domain_entropy', 'A_frequency', 'PTR_frequency',
    'AAAA_frequency', 'rr_count', 'ttl_mean', 'ttl_variance',
    'unique_country_count', 'unique_ttl_count', 'reverse_dns_known',
    'rr_type_A', 'rr_type_NONE', 'rr_type_PTR', 'rr_type_TXT',
    'entropy_x_length', 'tunneling_rr_score', 'network_diversity'
]

MODELS_TUNED = {}
MODELS_BASELINE = {}
SCALER = None
RAW_COLS = None
SCENARIOS = []


def load_artifacts():
    global MODELS_TUNED, MODELS_BASELINE, SCALER, RAW_COLS, SCENARIOS
    print("[INIT] Loading models (both Tuned and Baseline)...")
    
    # 1. Final Tuned Models (Stage 10 GridSearchCV)
    MODELS_TUNED = {
        "CART": joblib.load(PROCESSED / "cart_tuned_model.joblib"),
        "SVM":  joblib.load(PROCESSED / "svm_tuned_model.joblib"),
        "ANN":  joblib.load(PROCESSED / "ann_tuned_model.joblib"),
    }
    
    # 2. Pre-Tuned Baseline Models (Stage 8 & Defaults)
    MODELS_BASELINE = {
        "CART": joblib.load(PROCESSED / "cart_baseline_model.joblib"),
        "SVM":  joblib.load(PROCESSED / "svm_baseline_model.joblib"),
        "ANN":  joblib.load(PROCESSED / "ann_baseline_model.joblib"),
    }

    SCALER = joblib.load(PROCESSED / "preprocessor_scaler.joblib")
    RAW_COLS = list(SCALER.feature_names_in_)

    scenarios_path = DATA_DIR / "scenarios_data.json"
    if scenarios_path.exists():
        with open(scenarios_path, "r", encoding="utf-8") as f:
            SCENARIOS = json.load(f)
            
    print(f"  Loaded Tuned models: {list(MODELS_TUNED.keys())}")
    print(f"  Loaded Baseline models: {list(MODELS_BASELINE.keys())}")
    print(f"  Loaded scenarios: {len(SCENARIOS)}")


def evaluate_feature_vector(feat_df: pd.DataFrame, mode: str = "tuned") -> dict:
    """Evaluate a single 18-feature row across all three models in specified mode."""
    models_dict = MODELS_BASELINE if mode == "baseline" else MODELS_TUNED
    X = feat_df[FEATURE_SCHEMA].astype(float)
    results = {}
    for name, model in models_dict.items():
        prob = float(model.predict_proba(X)[0][1])
        # Use probability threshold of 0.5 for consistent calibration
        pred = 1 if prob >= 0.5 else 0
        results[name] = {
            "prediction": pred,
            "label": "Malicious" if pred == 1 else "Benign",
            "probability": round(prob * 100, 2),
            "confidence": round(max(prob, 1 - prob) * 100, 2),
            "mode": mode
        }
    return results


def scale_raw_input(raw: dict) -> pd.DataFrame:
    """Scale raw physical inputs and apply Stage 7 feature engineering."""
    row = {col: 0.0 for col in RAW_COLS}
    row.update(raw)
    df_raw = pd.DataFrame([row])[RAW_COLS]
    df_scaled = pd.DataFrame(SCALER.transform(df_raw), columns=RAW_COLS)
    # Stage 7 interaction signals
    df_scaled['entropy_x_length'] = df_scaled['domain_entropy'] * df_scaled['query_length']
    df_scaled['tunneling_rr_score'] = df_scaled['rr_type_TXT'] + df_scaled['rr_type_NONE'] - df_scaled['rr_type_A']
    df_scaled['network_diversity'] = (
        df_scaled['unique_country_count'] +
        df_scaled['unique_asn_count'] +
        df_scaled['distinct_domains_count']
    )
    return df_scaled[FEATURE_SCHEMA]


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", scenarios=SCENARIOS)


@app.route("/api/health")
def api_health():
    return jsonify({
        "status": "ok",
        "tuned_models": list(MODELS_TUNED.keys()),
        "baseline_models": list(MODELS_BASELINE.keys()),
        "scenarios_count": len(SCENARIOS)
    })


@app.route("/api/scenarios")
def api_scenarios():
    return jsonify(SCENARIOS)


@app.route("/api/comparison")
def api_comparison():
    """Returns the comprehensive benchmark comparison between Baseline and Tuned models."""
    comp_file = PROCESSED / "baseline_vs_tuned_full_comparison.csv"
    if comp_file.exists():
        df = pd.read_csv(comp_file)
        return jsonify(df.to_dict(orient="records"))
    return jsonify([])


@app.route("/api/predict/scenario/<scenario_id>", methods=["GET", "POST"])
def predict_scenario(scenario_id):
    mode = request.args.get("mode", "tuned").lower()
    scenario = next((s for s in SCENARIOS if s["id"] == scenario_id), None)
    if scenario is None:
        return jsonify({"error": f"Scenario '{scenario_id}' not found"}), 404
    try:
        df_feat = pd.DataFrame([scenario["features"]])
        models_eval = evaluate_feature_vector(df_feat, mode=mode)
        return jsonify({
            "models": models_eval,
            "mode": mode,
            "key_features": scenario.get("display_features", scenario["features"]),
            "scenario": {k: v for k, v in scenario.items() if k not in ("features", "display_features")}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict/custom", methods=["POST"])
def predict_custom():
    mode = request.args.get("mode", "tuned").lower()
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "No JSON payload received"}), 400
    try:
        df_scaled = scale_raw_input(data)
        models_eval = evaluate_feature_vector(df_scaled, mode=mode)
        key_features = {
            "query_length": float(data.get("query_length", 14.0)),
            "domain_entropy": float(data.get("domain_entropy", 2.5)),
            "entropy_x_length": round(float(data.get("query_length", 14.0)) * float(data.get("domain_entropy", 2.5)), 2),
            "tunneling_rr_score": 1.0 if data.get("rr_type_NONE", 0) > 0 or data.get("rr_type_TXT", 0) > 0 else -1.0,
            "network_diversity": float(data.get("unique_country_count", 2.0)),
            "rr_type_NONE": float(data.get("rr_type_NONE", 0.0)),
            "rr_count": float(data.get("rr_count", 4.0)),
            "ttl_mean": float(data.get("ttl_mean", 120.0)),
        }
        return jsonify({
            "models": models_eval,
            "mode": mode,
            "key_features": key_features,
            "scenario": {
                "id": "custom", "label": "Custom Query",
                "category": "custom", "description": "Custom parameter probe",
                "icon": "⚙️"
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/run_all", methods=["GET", "POST"])
def run_all_scenarios():
    """Run all 10 scenarios in specified mode (tuned or baseline)."""
    mode = request.args.get("mode", "tuned").lower()
    results = []
    for scenario in SCENARIOS:
        try:
            df_feat = pd.DataFrame([scenario["features"]])
            models_eval = evaluate_feature_vector(df_feat, mode=mode)
            results.append({
                "models": models_eval,
                "mode": mode,
                "key_features": scenario.get("display_features", scenario["features"]),
                "scenario": {k: v for k, v in scenario.items() if k not in ("features", "display_features")}
            })
        except Exception as e:
            results.append({"error": str(e), "scenario": scenario})
    return jsonify(results)


def load_models():
    """Alias for launcher scripts."""
    load_artifacts()


if __name__ == "__main__":
    load_artifacts()
    print("\n=======================================================")
    print("  DNS Tunneling NIDS Dashboard (Tuned + Baseline)")
    print("  Open in browser: http://127.0.0.1:5000")
    print("=======================================================\n")
    app.run(debug=False, port=5000, host="0.0.0.0")
