import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import time

# ------------------------------------------------------------
# ----------  Hybrid Model Class ------------------------------
# ------------------------------------------------------------
class HybridModel:
    """
    Matches your saved hybrid_model.pkl which contains:
        vectorizer, lr, xgb, cat
    """

    def __init__(self, vectorizer=None, lr=None, xgb=None, cat=None,
                 w_lr=0.3, w_xgb=0.4, w_cat=0.3):

        self.vectorizer = vectorizer
        self.lr = lr
        self.xgb = xgb
        self.cat = cat

        total = w_lr + w_xgb + w_cat
        self.w_lr = w_lr / total
        self.w_xgb = w_xgb / total
        self.w_cat = w_cat / total

    def predict_proba(self, df):
        if isinstance(df, pd.DataFrame):
            texts = df["text"].astype(str)
        else:
            texts = pd.Series(df).astype(str)

        X_tfidf = self.vectorizer.transform(texts)

        prob_lr = self.lr.predict_proba(X_tfidf)[:, 1]
        prob_xgb = self.xgb.predict_proba(X_tfidf)[:, 1]
        prob_cat = self.cat.predict_proba(pd.DataFrame({"text": texts}))[:, 1]

        final = (
            self.w_lr * prob_lr +
            self.w_xgb * prob_xgb +
            self.w_cat * prob_cat
        )
        return final


# ------------------------------------------------------------
# ---------------------- Load Model ---------------------------
# ------------------------------------------------------------
raw = joblib.load("model/hybrid_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

hybrid = HybridModel(
    vectorizer=raw.vectorizer,
    lr=raw.lr,
    xgb=raw.xgb,
    cat=raw.cat
)

xgb_model = raw.xgb  # for SHAP


# ------------------------------------------------------------
# --------------------- Streamlit UI --------------------------
# ------------------------------------------------------------
st.set_page_config(
    page_title="AI vs Human Text Detector",
    layout="centered",
    page_icon="🤖"
)

st.markdown("""
    <style>
        .main-title {font-size:40px !important; font-weight:700; color:black; text-align:center;}
        .subtitle {text-align:center; font-size:18px; color:gray;}
        .result-box {border-radius:15px; padding:25px; background-color:#1E1E1E; color:white; text-align:center;
                     font-size:22px; font-weight:bold; margin-bottom:15px;}
        .metric-box {text-align:center; border:1px solid #333; border-radius:10px; padding:10px; background-color:#121212;
                     margin:5px; color:white;}
        .progress-text {font-size:16px; color:#00BFFF; text-align:center; font-weight:600;}
        .stTabs [data-baseweb="tab-list"] button {font-size:18px; font-weight:600;}
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>🤖 AI vs Human Text Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Single Text Detection and Batch Evaluation with Explainability</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📝 Single Text Detection", "📂 File Upload & Metrics"])


# ------------------------------------------------------------
# ---------------------- TAB 1: Single ------------------------
# ------------------------------------------------------------
with tab1:

    user_text = st.text_area("✍️ Enter text to analyze:", height=200)

    if st.button("🔍 Detect"):

        if user_text.strip() == "":
            st.warning("Please enter some text before detecting.")

        else:
            prob_ai = hybrid.predict_proba(pd.DataFrame({"text": [user_text]}))[0]
            prob_human = 1 - prob_ai

            label = "🤖 AI-Generated Text" if prob_ai > 0.5 else "🧠 Human-Written Text"
            color = "#FF4B4B" if prob_ai > 0.5 else "#00C851"

            st.markdown(
                f"<div class='result-box' style='background-color:{color};'>{label}</div>",
                unsafe_allow_html=True
            )

            st.write(f"### Confidence: {prob_ai:.2%}")

            # Confidence Bar
            fig, ax = plt.subplots(figsize=(6, 1.5))
            categories = ['Human 🧠', 'AI 🤖']
            values = [prob_human * 100, prob_ai * 100]
            colors = ['#00C851', '#FF4B4B']

            bars = ax.barh(categories, values, color=colors)
            ax.set_xlim(0, 100)

            for bar, val in zip(bars, values):
                ax.text(val + 1, bar.get_y() + bar.get_height() / 2, f"{val:.2f}%", va='center')

            ax.set_xlabel("Probability (%)")
            st.pyplot(fig)

            # Explanation
            if prob_ai > 0.8:
                st.info("🧩 Strong AI-like style detected.")
            elif prob_ai < 0.2:
                st.success("✍️ Text appears naturally human-written.")
            else:
                st.warning("🤝 Mixed signals — may be human-edited AI.")

            # ------------------------------------------------------------
            # ------------------------ SHAP ------------------------------
            # ------------------------------------------------------------
            if xgb_model:
                st.subheader("🧠 SHAP Explainability")

                feature_names = hybrid.vectorizer.get_feature_names_out()

                explainer = shap.Explainer(xgb_model, feature_names=feature_names)
                X_single = hybrid.vectorizer.transform([user_text])
                shap_values = explainer(X_single)

                fig_shap, ax_shap = plt.subplots(figsize=(8, 5))
                shap.plots.bar(shap_values[0], show=False)
                st.pyplot(fig_shap)


# ------------------------------------------------------------
# -------------------- TAB 2: Batch Eval ----------------------
# ------------------------------------------------------------
with tab2:

    uploaded = st.file_uploader("📤 Upload CSV (must have text + label columns)", type=["csv"])

    if uploaded:

        df = pd.read_csv(uploaded)

        possible_text_cols = ["Data", "data", "Text", "text"]
        possible_label_cols = ["Labels", "Label", "labels", "label"]

        text_col = next((c for c in df.columns if c in possible_text_cols), None)
        label_col = next((c for c in df.columns if c in possible_label_cols), None)

        if not text_col or not label_col:
            st.error(
                f"❌ Missing required columns.\n\n"
                f"Expected Text column options: {possible_text_cols}\n"
                f"Expected Label column options: {possible_label_cols}\n\n"
                f"Found: {list(df.columns)}"
            )

        else:
            st.success(f"Detected ➤ Text: `{text_col}` | Label: `{label_col}`")

            preds = []
            probs = []
            total = len(df)

            progress = st.progress(0)
            status = st.empty()

            for i, t in enumerate(df[text_col]):
                p = hybrid.predict_proba(pd.DataFrame({"text": [str(t)]}))[0]
                probs.append(p)
                preds.append(1 if p > 0.5 else 0)

                progress.progress(int((i + 1) / total * 100))
                status.markdown(f"<p class='progress-text'>Processing {i+1}/{total}</p>",
                                unsafe_allow_html=True)
                time.sleep(0.01)

            df["Predicted"] = preds
            df["Confidence_AI"] = probs

            # ---- METRICS ----
            y_true = df[label_col]
            y_pred = df["Predicted"]
            y_prob = df["Confidence_AI"]

            report = classification_report(y_true, y_pred, output_dict=True)

            accuracy = accuracy_score(y_true, y_pred)
            roc = roc_auc_score(y_true, y_prob)
            precision = report['weighted avg']['precision']
            recall = report['weighted avg']['recall']
            f1 = report['weighted avg']['f1-score']

            st.subheader("📊 Evaluation Metrics")

            col1, col2, col3 = st.columns(3)
            col1.markdown(f"<div class='metric-box'><b>Accuracy</b><br>{accuracy*100:.2f}%</div>", unsafe_allow_html=True)
            col2.markdown(f"<div class='metric-box'><b>Precision</b><br>{precision:.4f}</div>", unsafe_allow_html=True)
            col3.markdown(f"<div class='metric-box'><b>Recall</b><br>{recall:.4f}</div>", unsafe_allow_html=True)

            col4, col5 = st.columns(2)
            col4.markdown(f"<div class='metric-box'><b>F1-Score</b><br>{f1:.4f}</div>", unsafe_allow_html=True)
            col5.markdown(f"<div class='metric-box'><b>ROC-AUC</b><br>{roc:.4f}</div>", unsafe_allow_html=True)

            # ---- Confusion Matrix ----
            st.write("### Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)

            fig, ax = plt.subplots()
            ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Human", "AI"])
            ax.set_yticklabels(["Human", "AI"])

            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j], ha='center', va='center')

            st.pyplot(fig)

            # ---- Download ----
            st.download_button(
                "📥 Download Predictions CSV",
                df.to_csv(index=False).encode("utf-8"),
                "ai_text_predictions.csv",
                "text/csv"
            )


# ------------------------------------------------------------
# -------------------------- Footer ---------------------------
# ------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Developed by <b>Team 13</b> | Hybrid Ensemble + Explainable AI</p>",
    unsafe_allow_html=True
)

