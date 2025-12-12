# =================================================================================================
# PHASE 4: MACHINE LEARNING CLASSIFICATION MODELS
# =================================================================================================

# 1. INSTALL LIBRARIES
print(" ⏳  Installing libraries...")
!pip install scikit-learn xgboost transformers torch tensorflow python-docx seaborn matplotlib xlsxwriter --quiet > /dev/null 2>&1

# 2. IMPORTS
import os
import joblib
from tensorflow.keras.models import Sequential, Model as KerasModel
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xlsxwriter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    auc,
    f1_score # Import f1_score explicitly
)
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from IPython.display import display, HTML

warnings.filterwarnings("ignore")
print(" 🚀  PHASE 4 STARTED: DIAMOND SIMPLIFIED MODE (MACRO F1 SELECTION)")
print("=" * 80)

# ==========================
#  🎨  HELPER: STYLE TABLES
# ==========================
def style_word_table(table):
    """Apply borders + blue header styling to a Word table."""
    tbl = table._tbl
    tblPr = tbl.tblPr
    tblBorders = tblPr.first_child_found_in("w:tblBorders")
    if tblBorders is None:
        tblBorders = OxmlElement("w:tblBorders")
        tblPr.append(tblBorders)
    for border in ["top", "left", "bottom", "right", "insideH", "insideV"]:
        edge = OxmlElement(f"w:{border}")
        edge.set(qn("w:val"), "single")
        edge.set(qn("w:sz"), "4")
        edge.set(qn("w:space"), "0")
        edge.set(qn("w:color"), "000000")
        tblBorders.append(edge)
    for i, row in enumerate(table.rows):
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            if i == 0:
                tc = cell._element
                tcPr = tc.get_or_add_tcPr()
                shd = OxmlElement("w:shd")
                shd.set(qn("w:val"), "clear")
                shd.set(qn("w:color"), "auto")
                shd.set(qn("w:fill"), "2E75B6")
                tcPr.append(shd)
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.bold = True
                        run.font.color.rgb = RGBColor(255, 255, 255)

# ==========================
#  💾  HELPER: SAVE MODELS
# ==========================
def save_all_models(models_dict, save_dir="models"):
    """
    Saves all ML/DL models to disk based on their type.
    Uses .keras format for Keras models.
    """
    os.makedirs(save_dir, exist_ok=True)
    for model_name, model_obj in models_dict.items():
        # Case 1 — Keras deep learning model
        if isinstance(model_obj, KerasModel):
            file_path = os.path.join(save_dir, f"{model_name}.keras")
            model_obj.save(file_path)
            print(f"[Saved] Keras model -> {file_path}")
        # Case 2 — All pickle-compatible models (Sklearn, XGBoost)
        else:
            file_path = os.path.join(save_dir, f"{model_name}.pkl")
            joblib.dump(model_obj, file_path)
            print(f"[Saved] Pickle model -> {file_path}")
    print("\nAll models saved successfully!")

#REMOVED: calculate_weighted_score helper function is no longer needed.

# ==========================
# 1. LOAD DATASET
# ==========================
target_file = "Complete_Dataset_With_Features.xlsx"
sheet_name = "All_Data"
if os.path.exists(target_file):
    print(f" ✅  Found File: {target_file}")
    try:
        df_all = pd.read_excel(target_file, sheet_name=sheet_name)
    except Exception:
        df_all = pd.read_excel(target_file)
else:
    from google.colab import files
    print(" 📂  Please upload 'Complete_Dataset_With_Features.xlsx'...")
    uploaded = files.upload()
    filename = list(uploaded.keys())[0]
    df_all = pd.read_excel(filename, sheet_name=sheet_name)

text_col = "Text After Processing" if "Text After Processing" in df_all.columns else df_all.columns[0]
orig_text_col = "Original Text" if "Original Text" in df_all.columns else text_col
label_col = "Text Type" if "Text Type" in df_all.columns else "label"
print(f" 📊  Total Rows: {len(df_all)}")

# ==========================
# 2. FEATURE PREPARATION (SMART MAPPING + L2-NORM)
# ==========================
# Normalize name of feature 98 to a single canonical name
feat98_name = "feat_098_bert_cls_l2norm"
if "feat_098_bert_cls_norm" in df_all.columns:
    print(" ℹ️  Detected feature name: 'feat_098_bert_cls_norm' -> using as L2-norm.")
    df_all.rename(columns={"feat_098_bert_cls_norm": feat98_name}, inplace=True)
elif "feat_098_bert_cls_prob" in df_all.columns:
    print(" ℹ️  Detected feature name: 'feat_098_bert_cls_prob' -> treating as L2-norm proxy.")
    df_all.rename(columns={"feat_098_bert_cls_prob": feat98_name}, inplace=True)

required_features = [
    "feat_006_multiple_elongations",
    "feat_029_semicolons",
    "feat_052_interjections",
    "feat_075_active_voice_sentences",
    feat98_name,
]

missing = [c for c in required_features if c not in df_all.columns]
if missing:
    raise ValueError(f" ❌  Missing columns: {missing}")

# Prepare X and y (keep indices for error-text export)
indices = df_all.index.values
X = df_all.loc[indices, required_features].fillna(0)
y = df_all.loc[indices, label_col]

le = LabelEncoder()
y_enc = le.fit_transform(y)  # Binary: e.g., Human=0, AI=1

# --- DYNAMIC LABEL IDENTIFICATION ---
# Identify label names and indices correctly based on data content
label_map = {index: label for index, label in enumerate(le.classes_)}
ai_label_name = None
human_label_name = None

for index, name in label_map.items():
    # Simple heuristic: look for "AI" in the label name
    if "AI" in name:
        ai_label_name = name
    else:
        human_label_name = name

# Fallback based on indices if naming is unconventional
if ai_label_name is None: ai_label_name = le.classes_[1]
if human_label_name is None: human_label_name = le.classes_[0]

print(f" ℹ️  Class Mapping: Human='{human_label_name}', AI='{ai_label_name}'")

# --- CRITICAL FIX: Update numerical ai_label based on found name ---
# This ensures subsequent calculations use the correct numerical index for the AI class.
ai_label = le.transform([ai_label_name])[0]
# -----------------------------------------------------------------


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split 70 / 15 / 15 using indices for later mapping
idx_train, idx_temp, y_train, y_temp = train_test_split(
    indices, y_enc, test_size=0.30, random_state=42, stratify=y_enc
)
idx_val, idx_test, y_val, y_test = train_test_split(
    idx_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# Split X_scaled directly with same random_state to stay aligned with y_enc
X_train_s, X_temp_s, _, _ = train_test_split(
    X_scaled, y_enc, test_size=0.30, random_state=42, stratify=y_enc
)
X_val_s, X_test_s, _, _ = train_test_split(
    X_temp_s, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

split_summary_df = pd.DataFrame(
    {
        "Subset": ["Training", "Validation", "Testing"],
        "Count": [len(idx_train), len(idx_val), len(idx_test)],
        "Ratio": ["70%", "15%", "15%"],
    }
)
print("\n 📋  (Sheet 1) Random Data Split:")
display(split_summary_df)

# ==========================
# 3. TRAINING (ALL MODELS)
# ==========================
print("\n ⚙️  Training Models...")
models = {
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42, use_label_encoder=False),
}

results_list = []
model_preds = {}
model_probs = {}

# 3.1 Traditional models (UPDATED for Macro F1)
for name, model in models.items():
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    probs = model.predict_proba(X_test_s)[:, 1]
    model_preds[name] = preds
    model_probs[name] = probs

    # Generate detailed report as a dictionary
    report_dict = classification_report(y_test, preds, output_dict=True, target_names=le.classes_, zero_division=0)

    # Extract metrics safely using the dynamic names
    ai_metrics = report_dict.get(ai_label_name, {})
    human_metrics = report_dict.get(human_label_name, {})

    results_list.append(
        {
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            # ADDED: Macro F1-Score (Average of both classes)
            "Macro F1-Score": f1_score(y_test, preds, average='macro'),
            "ROC-AUC": roc_auc_score(y_test, probs),
            # AI Metrics
            "Precision (AI)": ai_metrics.get('precision', 0),
            "Recall (AI)": ai_metrics.get('recall', 0),
            "F1-Score (AI)": ai_metrics.get('f1-score', 0),
            # Human Metrics
            "Precision (Human)": human_metrics.get('precision', 0),
            "Recall (Human)": human_metrics.get('recall', 0),
            "F1-Score (Human)": human_metrics.get('f1-score', 0),
            "Type": "Traditional",
        }
    )

# 3.2 Neural Network on tabular features
print(" 🧠  Training Neural Network...")
# Setting seeds for reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

nn_model = Sequential(
    [
        Dense(64, activation="relu", input_shape=(X_train_s.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ]
)
nn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = nn_model.fit(
    X_train_s,
    y_train,
    validation_data=(X_val_s, y_val),
    epochs=20,
    batch_size=32,
    verbose=0,
)

nn_probs = nn_model.predict(X_test_s, verbose=0).flatten()
nn_preds = (nn_probs > 0.5).astype(int)
nn_name = "Feedforward NN + BERT (768D)"
model_preds[nn_name] = nn_preds
model_probs[nn_name] = nn_probs

# Generate detailed report for NN
report_dict_nn = classification_report(y_test, nn_preds, output_dict=True, target_names=le.classes_, zero_division=0)
ai_metrics_nn = report_dict_nn.get(ai_label_name, {})
human_metrics_nn = report_dict_nn.get(human_label_name, {})

results_list.append(
    {
        "Model": nn_name,
        "Accuracy": accuracy_score(y_test, nn_preds),
        # ADDED: Macro F1-Score for NN
        "Macro F1-Score": f1_score(y_test, nn_preds, average='macro'),
        "ROC-AUC": roc_auc_score(y_test, nn_probs),
        # AI Metrics
        "Precision (AI)": ai_metrics_nn.get('precision', 0),
        "Recall (AI)": ai_metrics_nn.get('recall', 0),
        "F1-Score (AI)": ai_metrics_nn.get('f1-score', 0),
        # Human Metrics
        "Precision (Human)": human_metrics_nn.get('precision', 0),
        "Recall (Human)": human_metrics_nn.get('recall', 0),
        "F1-Score (Human)": human_metrics_nn.get('f1-score', 0),
        "Type": "Deep Learning",
    }
)

# Combine DataFrames
all_results_df = pd.DataFrame(results_list)
trad_results_df = all_results_df[all_results_df["Type"] == "Traditional"].drop(columns=["Type"])
all_models_df = all_results_df.drop(columns=["Type"])
history_df = pd.DataFrame(history.history)

# ==========================
# 4. BEST MODEL SELECTION (SIMPLIFIED: BASED ON MACRO F1-SCORE)
# ==========================
print("\n ⚖️  Selecting Best Model based on Macro F1-Score...")

# ----- SIMPLIFIED SECTION START -----
# No weights, no custom calculation function.
# Just select based on the single best metric for balance: Macro F1-Score.

target_metric = "Macro F1-Score"
best_model_idx = all_models_df[target_metric].idxmax()
best_model_name = all_models_df.loc[best_model_idx, "Model"]
best_model_score = all_models_df.loc[best_model_idx, target_metric]

print(f"\n 🏆  Best Model (based on {target_metric}): {best_model_name}")
print(f"    {target_metric}: {best_model_score:.4f}")
# ----- SIMPLIFIED SECTION END -----


# --- DISPLAY TABLES ON SCREEN (UPDATED) ---
print("\n 📋  (Sheet 2) Traditional ML Results (Full Metrics):")
display(trad_results_df)
# Sort by the new target metric
print(f"\n 📋  (Sheet 3) All Models Results (Sorted by {target_metric}):")
display(all_models_df.sort_values(by=target_metric, ascending=False))
print("\n 📋  (Sheet 4) Neural Network Training History (First 5 Epochs):")
display(history_df.head())
# ----------------------------------------


# 4.1 Best Model Detailed Report
best_report = classification_report(y_test, model_preds[best_model_name], output_dict=True, target_names=le.classes_)
best_model_report_df = pd.DataFrame(best_report).transpose()
print(f"\n 📋  (Sheet 5) Detailed Report for Best Model ({best_model_name}):")
display(best_model_report_df)

# 4.2 Feature importance
print("\n 🌟  Calculating Feature Importance...")
imp_model = models["Random Forest"]
if best_model_name in models and hasattr(models[best_model_name], "feature_importances_"):
    imp_model = models[best_model_name]

if hasattr(imp_model, "feature_importances_"):
    importances = imp_model.feature_importances_
    feat_imp_df = pd.DataFrame(
        {"Feature": required_features, "Importance": importances}
    ).sort_values(by="Importance", ascending=False)
    print("\n 📋  (Sheet 6) Feature Importance:")
    display(feat_imp_df)

    plt.figure(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=feat_imp_df, palette="viridis")
    plt.title("Feature Importance (Best Tree-Based Model)")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.show()
    plt.close()
else:
    feat_imp_df = pd.DataFrame({"Feature": required_features, "Importance": np.nan})
    print("Feature importance not available for the selected best model.")

# 4.3 Error texts for best model
print("\n 📝  Extracting Error Texts...")
test_subset = df_all.loc[idx_test].copy()
test_subset["True_Label"] = le.inverse_transform(y_test)
test_subset["Predicted_Label"] = le.inverse_transform(model_preds[best_model_name])
test_subset["Prob_AI"] = model_probs[best_model_name]
error_df = test_subset[test_subset["True_Label"] != test_subset["Predicted_Label"]].copy()
export_cols = [orig_text_col, "True_Label", "Predicted_Label", "Prob_AI"] + required_features
error_df_clean = error_df[export_cols] if not error_df.empty else pd.DataFrame(columns=export_cols)

if not error_df_clean.empty:
    print(f"Found {len(error_df_clean)} misclassified samples.")
    print(" 📋  (Sheet 7) Error Analysis Preview (First 3 samples):")
    display(error_df_clean.head(3))
else:
    print("✅ Perfect accuracy! No errors to analyze.")


# ==========================
# 5. VISUALIZATION & WORD REPORT
# ==========================
print("\n 🎨  Generating Visuals & Word Report...")
doc = Document()
doc.add_heading("Final Classification Report", 0)

# --- ADDED: Data Split Table to Word ---
doc.add_heading("1. Data Split Summary", level=1)
table = doc.add_table(rows=len(split_summary_df) + 1, cols=len(split_summary_df.columns))
style_word_table(table)
for i, col in enumerate(split_summary_df.columns):
    table.rows[0].cells[i].text = str(col)
for i, row in split_summary_df.iterrows():
    for j, val in enumerate(row):
        table.rows[i + 1].cells[j].text = str(val)

# --- ADDED: Best Model Info to Word (UPDATED MESSAGE) ---
doc.add_heading("2. Best Model Selection", level=1)
# Updated paragraph to reflect the new simple selection criteria
doc.add_paragraph(f"The best performing model, selected based on the highest Macro F1-Score (average F1-score of both classes), is: {best_model_name} with a score of {best_model_score:.4f}.")

# 5.1 Overall metrics table (UPDATED sorting)
doc.add_heading("3. Model Metrics Summary (Sorted by Macro F1-Score)", level=1)
# Sort dataframe by the new target metric
all_models_sorted = all_models_df.sort_values(by=target_metric, ascending=False)
table = doc.add_table(rows=len(all_models_sorted) + 1, cols=len(all_models_sorted.columns))
style_word_table(table)
for i, col in enumerate(all_models_sorted.columns):
    table.rows[0].cells[i].text = str(col)
for i, row in all_models_sorted.iterrows():
    for j, val in enumerate(row):
        if isinstance(val, float):
            table.rows[i + 1].cells[j].text = f"{val:.4f}"
        else:
            table.rows[i + 1].cells[j].text = str(val)

# --- ADDED: Detailed Best Model Report to Word ---
doc.add_heading("4. Best Model Detailed Report", level=1)
table = doc.add_table(rows=len(best_model_report_df) + 1, cols=len(best_model_report_df.columns) + 1)
style_word_table(table)
table.rows[0].cells[0].text = "Metric"
for i, col in enumerate(best_model_report_df.columns):
    table.rows[0].cells[i+1].text = str(col)
for i, (index, row) in enumerate(best_model_report_df.iterrows()):
    table.rows[i + 1].cells[0].text = str(index)
    for j, val in enumerate(row):
        if isinstance(val, float):
            table.rows[i + 1].cells[j+1].text = f"{val:.4f}"
        else:
            table.rows[i + 1].cells[j+1].text = str(val)


# 5.2 Performance Benchmark Plot (UPDATED to show Macro F1 instead of Weighted)
doc.add_heading("5. Performance Benchmark Plot (Key Metrics)", level=1)
# Replace Weighted_Score with Macro F1-Score in the plot
melted_df = all_models_df.melt(id_vars="Model",
                               value_vars=["Accuracy", "ROC-AUC", "F1-Score (AI)", "F1-Score (Human)", "Macro F1-Score"],
                               var_name="Metric", value_name="Score")
plt.figure(figsize=(14, 7))
sns.barplot(data=melted_df, x="Model", y="Score", hue="Metric", palette="colorblind")
plt.title("Model Performance Comparison (Key Metrics)")
plt.xlabel("Model")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.ylim(0, 1.1)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
benchmark_plot_file = "model_performance_comparison.png"
plt.savefig(benchmark_plot_file)
plt.show()
plt.close()
doc.add_picture(benchmark_plot_file, width=Inches(6))

# --- ADDED: ROC Curves Plot ---
doc.add_heading("6. ROC Curves (All Models)", level=1)
plt.figure(figsize=(10, 8))
for name, probs in model_probs.items():
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves (All Models)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
roc_plot_file = "roc_curves_all_models.png"
plt.savefig(roc_plot_file)
plt.show()
plt.close()
doc.add_picture(roc_plot_file, width=Inches(6))


# 5.3 Per-model CM + Error Confidence
doc.add_heading("7. Model Analysis (Confusion Matrix + Error Confidence)", level=1)
for name in model_preds.keys():
    doc.add_heading(f"Model: {name}", level=2)
    preds = model_preds[name]
    probs = model_probs[name]

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    cmap = "Greens" if name == best_model_name else "Blues"
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=le.classes_,
        yticklabels=le.classes_,
    )
    plt.title(f"Confusion Matrix: {name}")
    plt.tight_layout()
    cm_file = f"cm_{name.replace(' ', '_')}.png"
    plt.savefig(cm_file)
    plt.show()
    plt.close()
    doc.add_paragraph("Confusion Matrix:")
    doc.add_picture(cm_file, width=Inches(4))

    # Error Confidence
    # CRITICAL FIX: Use the correctly identified numerical ai_label here
    confs = [p if pred == ai_label else 1 - p for p, pred in zip(probs, preds)]
    errors_mask = y_test != preds
    error_confs = [c for c, e in zip(confs, errors_mask) if e]
    if error_confs:
        plt.figure(figsize=(6, 4))
        sns.histplot(error_confs, bins=10, kde=True, color="red")
        plt.title(f"Error Confidence: {name}")
        plt.xlabel("Confidence in Wrong Decision")
        plt.axvline(0.5, color="black", linestyle="--")
        plt.tight_layout()
        err_file = f"err_{name.replace(' ', '_')}.png"
        plt.savefig(err_file)
        plt.show()
        plt.close()
        doc.add_paragraph("Error Confidence Analysis:")
        doc.add_picture(err_file, width=Inches(4))
    else:
        doc.add_paragraph("No misclassified samples for this model (Accuracy 100%).")

# 5.4 Feature importance image (if available)
doc.add_heading("8. Feature Importance (Best Tree-Based Model)", level=1)
if "feature_importance.png" in os.listdir():
    doc.add_picture("feature_importance.png", width=Inches(6))
else:
    doc.add_paragraph("Feature importance not available for the selected best model.")

# 5.5 Neural Network training history
doc.add_heading("9. Neural Network Training History", level=1)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Val")
plt.title("Accuracy")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Val")
plt.title("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("nn_history.png")
plt.show()
plt.close()
doc.add_picture("nn_history.png", width=Inches(6))

doc.save("Final_Classification_Report.docx")

# ==========================
# 6. EXCEL EXPORT (EXACT STRUCTURE + EXTRAS)
# ==========================
output_excel = "Classification_Results_Complete.xlsx"
print(f"\n 💾  Saving Excel: {output_excel}...")
with pd.ExcelWriter(output_excel, engine="xlsxwriter") as writer:
    split_summary_df.to_excel(writer, sheet_name="Data Split Summary", index=False)
    trad_results_df.to_excel(writer, sheet_name="Traditional ML Results", index=False)
    # Sort by the new target metric in Excel too
    all_models_df.sort_values(by=target_metric, ascending=False).to_excel(writer, sheet_name="All Models Results", index=False)
    # Create a single-row DataFrame for the best model name and score
    pd.DataFrame({"Best Model Name": [best_model_name], f"Best {target_metric}": [best_model_score]}).to_excel(writer, sheet_name="Best Model Selection", index=False)
    best_model_report_df.to_excel(writer, sheet_name="Best Model Detailed Report", index=True)
    history_df.to_excel(writer, sheet_name="NN Training History", index=True)
    feat_imp_df.to_excel(writer, sheet_name="Feature Importance", index=False)
    if not error_df_clean.empty:
        error_df_clean.to_excel(writer, sheet_name="Error Analysis Texts", index=False)

print("\n 🎉  PHASE 4 COMPLETE! (DIAMOND SIMPLIFIED EDITION)")
print(" 📥  Downloading Report & Excel...")
from google.colab import files
files.download(output_excel)
files.download("Final_Classification_Report.docx")

# ==========================
# 7. SAVE TRAINED MODELS
# ==========================
# Create models directory (if it does not exist)
os.makedirs("models", exist_ok=True)

# Map the trained models to concise names
models_dict = {
    "naive_bayes":          models["Naive Bayes"],
    "logistic_regression":  models["Logistic Regression"],
    "random_forest":        models["Random Forest"],
    "svm":                  models["SVM"],
    "xgboost":              models["XGBoost"],
    "ffnn":                 nn_model,   # Feedforward Neural Network
}

# Save all models to disk using the same strategy as your colleague
save_all_models(models_dict, save_dir="models")

# Zip and download the models folder
print(" 📦  Zipping models folder...")
!zip -r models.zip models > /dev/null 2>&1
files.download("models.zip")