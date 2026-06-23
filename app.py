import gradio as gr
import pandas as pd
import joblib
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty
from mp_api.client import MPRester

# ---- Load trained model and feature list (saved from Colab) ----
model = joblib.load("stability_model_app.pkl")
feature_names = joblib.load("model_features_app.pkl")

featurizer = ElementProperty.from_preset("magpie")

# Common substitution candidates by chemical family (simple periodic-table neighbors)
SUBSTITUTE_MAP = {
    "Co": ["Ni", "Fe", "Mn", "Zn", "Mg"],
    "W": ["Mo", "Cr", "V", "Ti", "S"],
    "Fe": ["Co", "Ni", "Mn", "Cr"],
    "Ni": ["Co", "Fe", "Cu", "Zn"],
    "Ti": ["Zr", "V", "Sn"],
    "Si": ["Ge", "Sn", "C"],
}


def predict_stability(formula):
    if not formula or not formula.strip():
        return "Enter a chemical formula, e.g. CoWO4", ""

    try:
        df = pd.DataFrame({"formula": [formula.strip()]})
        df = StrToComposition(target_col_id="composition").featurize_dataframe(
            df, col_id="formula", ignore_errors=True
        )
        if df["composition"].isna().any():
            return f"Could not parse formula: {formula}", ""

        df = featurizer.featurize_dataframe(df, col_id="composition", ignore_errors=True)
        X = df[feature_names]

        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        confidence = max(proba) * 100

        label = "Likely STABLE" if pred else "Likely UNSTABLE"
        icon = "\u2713" if pred else "\u2717"

        result = f"{icon} {label}\n\nConfidence: {confidence:.1f}%"
        detail = (
            f"Prediction based on chemistry-only features (Magpie composition "
            f"descriptors) — no structural data used, so this works for any "
            f"formula, including ones not yet in any database.\n\n"
            f"Model: XGBoost, trained on 30,000 real materials from the "
            f"Materials Project, cross-validated accuracy ~75%."
        )
        return result, detail

    except Exception as e:
        return f"Error processing formula: {str(e)}", ""


def explore_substitutions(formula, api_key):
    if not formula or not formula.strip():
        return "Enter a chemical formula, e.g. CoWO4"
    if not api_key or not api_key.strip():
        return "Enter your Materials Project API key to search the database"

    formula = formula.strip()

    # Find which known element in the formula has substitution candidates
    target_element = None
    for el in SUBSTITUTE_MAP:
        if el in formula:
            target_element = el
            break

    if not target_element:
        return (
            f"No known substitution candidates configured for elements in '{formula}'.\n"
            f"Currently supports substitutions around: {', '.join(SUBSTITUTE_MAP.keys())}"
        )

    candidates = SUBSTITUTE_MAP[target_element]
    rows = []

    try:
        with MPRester(api_key.strip()) as mpr:
            for sub_el in candidates:
                new_formula = formula.replace(target_element, sub_el, 1)
                docs = mpr.materials.summary.search(
                    formula=new_formula,
                    fields=["material_id", "formula_pretty", "band_gap",
                            "energy_above_hull", "formation_energy_per_atom"]
                )
                if len(docs) == 0:
                    rows.append({
                        "Formula": new_formula, "Found": "No (gap!)",
                        "Bandgap (eV)": "-", "Stability (E above hull)": "-",
                        "Formation Energy": "-"
                    })
                else:
                    best = min(docs, key=lambda d: d.energy_above_hull)
                    rows.append({
                        "Formula": new_formula, "Found": "Yes",
                        "Bandgap (eV)": f"{best.band_gap:.3f}",
                        "Stability (E above hull)": f"{best.energy_above_hull:.4f}",
                        "Formation Energy": f"{best.formation_energy_per_atom:.3f}"
                    })
    except Exception as e:
        return f"Error querying Materials Project: {str(e)}"

    result_df = pd.DataFrame(rows)
    return result_df.to_string(index=False)


with gr.Blocks(title="Materials Discovery Toolkit") as demo:
    gr.Markdown("# Materials Discovery Toolkit")
    gr.Markdown(
        "Built by John Valan Tony A | M.Sc Physics | "
        "[GitHub Repository](https://github.com/johnvalantony/Materials-data-analysis)"
    )

    with gr.Tabs():
        with gr.TabItem("Stability Predictor"):
            gr.Markdown(
                "Enter any chemical formula and predict whether it is likely "
                "thermodynamically stable — using a machine learning model "
                "trained purely on chemistry, not lookup. Works even on "
                "formulas that don't exist in any database yet."
            )
            with gr.Row():
                formula_input = gr.Textbox(
                    label="Chemical Formula", placeholder="e.g. CoWO4, Fe2O3, SiO2"
                )
            predict_btn = gr.Button("Predict Stability", variant="primary")
            with gr.Row():
                result_output = gr.Textbox(label="Prediction", lines=3)
                detail_output = gr.Textbox(label="Details", lines=5)

            predict_btn.click(
                fn=predict_stability, inputs=formula_input,
                outputs=[result_output, detail_output]
            )

        with gr.TabItem("Substitution Explorer"):
            gr.Markdown(
                "Enter a known material formula and an element within it will "
                "be substituted with chemically similar elements. Each "
                "resulting formula is checked against the real Materials "
                "Project database. Requires your own free Materials Project "
                "API key from next-gen.materialsproject.org."
            )
            with gr.Row():
                sub_formula_input = gr.Textbox(
                    label="Base Formula", placeholder="e.g. CoWO4"
                )
                api_key_input = gr.Textbox(
                    label="Materials Project API Key", type="password"
                )
            explore_btn = gr.Button("Explore Substitutions", variant="primary")
            sub_result_output = gr.Textbox(label="Substitution Results", lines=10)

            explore_btn.click(
                fn=explore_substitutions, inputs=[sub_formula_input, api_key_input],
                outputs=sub_result_output
            )

demo.launch()
