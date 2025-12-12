# =================================================================================================
# PHASE 3: FEATURE ENGINEERING & DATA SPLITTING
# =================================================================================================


# 1- Task 3.1:
# Engineer Individual Stylometric Feature Assignment (6، 29، 52، 75، 98)

# 1. INSTALL SYSTEM DEPENDENCIES AND LIBRARIES
print("⏳ Installing system dependencies and libraries...")
!apt-get install -y libcairo2 libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info > /dev/null 2>&1
!pip install scikit-learn transformers torch tqdm pandas numpy nltk openpyxl xlsxwriter python-docx weasyprint --quiet > /dev/null 2>&1

# 2. IMPORTS
import os
import gc
import re
import warnings

import numpy as np
import pandas as pd
import torch
from google.colab import files
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from docx import Document
from docx.shared import Inches, Pt
from IPython.display import HTML, display
import weasyprint

warnings.filterwarnings("ignore")

print("🚀 PHASE 3 INITIALIZED: OPTIMIZED BATCH MODE")
print("=" * 80)

# =================================================================================================
# 3. LOAD DATASET (preprocessing_results.xlsx)
# =================================================================================================

# Prefer local file to avoid repeated uploads
if os.path.exists("preprocessing_results.xlsx"):
    filename = "preprocessing_results.xlsx"
    print(f"✅ Found file locally: {filename}")
else:
    print("\n📂 Please upload 'preprocessing_results.xlsx'...")
    uploaded = files.upload()
    if len(uploaded) == 0:
        raise ValueError("❌ No file uploaded.")
    filename = list(uploaded.keys())[0]
    print(f"✅ Using uploaded file: {filename}")

# =================================================================================================
# 4. LOAD & MERGE ALL SHEETS
# =================================================================================================

print("⏳ Loading and merging sheets from Excel...")

try:
    excel = pd.ExcelFile(filename)
    frames = []

    for sheet in excel.sheet_names:
        df = pd.read_excel(filename, sheet_name=sheet)

        # Derive split name from sheet name
        split_prefix = sheet.split("_")[0]
        if split_prefix == "by":
            split_name = "by_polishing"
        elif split_prefix == "from":
            split_name = (
                "from_title_and_content" if "title_and_content" in sheet else "from_title"
            )
        else:
            split_name = split_prefix

        # Detect text type (Human vs AI)
        text_type = "Human" if "original_abstract" in sheet else "AI"

        df["Split"] = split_name
        df["Text Type"] = text_type
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)

    # Standardize column names
    if "Original" in df_all.columns:
        df_all = df_all.rename(
            columns={
                "Original": "Original Text",
                "After Preprocessing": "Text After Processing",
            }
        )

    print(f"✅ Dataset loaded successfully: {len(df_all)} rows.")

except Exception as e:
    raise ValueError(f"❌ Error while loading Excel file: {e}")

# Ensure required text column exists
if "Text After Processing" not in df_all.columns:
    raise ValueError(
        "❌ Column 'Text After Processing' not found. Please check preprocessing output."
    )

# =========================================================================================
# TASK 3.1: STYLOMETRIC FEATURE ENGINEERING (6, 29, 52, 75, 98)
# =========================================================================================

# 5. FEATURE DEFINITIONS (LIGHTWEIGHT FEATURES)

def feat_006_multiple_elongations(text):
    """
    Feature 6: Number of multiple character elongations (e.g., ههههه).
    Counts any character repeated at least 3 times consecutively.
    """
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"(.)\1{2,}", text))


def feat_029_semicolons(text):
    """
    Feature 29: Number of semicolons.
    Counts Arabic semicolon (؛) occurrences.
    """
    if not isinstance(text, str):
        return 0
    return text.count("؛")


INTERJECTIONS_AR = {
    "آه", "أوه", "واه", "وي", "واأسفي", "واحزناه", "مرحى", "أيا", "هيا", "هاه",
    "يا سلام", "ما شاء الله", "يا لها", "يا له", "برافو", "الله", "يالها", "ياللعجب",
    "آي", "أي", "ياه", "يا", "يالا", "يالله", "واها", "وا", "وا سلاماه", "وا قسطاه",
    "وا ويلاه", "ياهو", "خيبة", "ويح", "يا إلهي", "آمين", "سبحان", "ياسلام", "يا رب"
}

def feat_052_interjections(text):
    """
    Feature 52: Number of interjections.
    Counts how many tokens match a predefined list of Arabic interjections.
    """
    if not isinstance(text, str):
        return 0
    words = text.split()
    return sum(1 for w in words if w.strip() in INTERJECTIONS_AR)


def feat_075_active_voice_sentences(text):
    """
    Feature 75: Number of active voice sentences.
    Approximation: sentences that do NOT contain common passive markers.
    """
    if not isinstance(text, str):
        return 0
    sentences = re.split(r"[.!؟\n\r]+", text)
    passive_markers = ["تُ", "يُ", "أُ", "وُ", "تم ", "تمت ", "تمّ ", "تمّت "]
    active_count = 0
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if not any(pm in s for pm in passive_markers):
            active_count += 1
    return active_count

# 6. APPLY LIGHTWEIGHT FEATURES

print("\n⚙️ Extracting lightweight features (6, 29, 52, 75)...")
tqdm.pandas()

df_all["feat_006_multiple_elongations"] = df_all["Text After Processing"].progress_apply(
    feat_006_multiple_elongations
)
df_all["feat_029_semicolons"] = df_all["Text After Processing"].progress_apply(
    feat_029_semicolons
)
df_all["feat_052_interjections"] = df_all["Text After Processing"].progress_apply(
    feat_052_interjections
)
df_all["feat_075_active_voice_sentences"] = df_all[
    "Text After Processing"
].progress_apply(feat_075_active_voice_sentences)

# 7. FEATURE 98: BERT CLS L2-NORM (SEMANTIC DENSITY PROXY)

print("\n⚙️ Extracting Feature 98 (BERT CLS L2-norm) in batches...")
print("   This uses the L2-norm of the CLS embedding as a semantic density measure.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Using device: {device.upper()}")

bert_tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv2")
bert_model = AutoModel.from_pretrained("aubmindlab/bert-base-arabertv2").to(device)
bert_model.eval()

# Initialize column for feature 98 (L2-norm of CLS)
feature_98_col = "feat_098_bert_cls_l2norm"
if feature_98_col not in df_all.columns:
    df_all[feature_98_col] = 0.0

batch_size = 32  # Safe batch size for Colab
texts = df_all["Text After Processing"].astype(str).tolist()

for i in tqdm(range(0, len(texts), batch_size), desc="BERT CLS batches"):
    batch_texts = texts[i : i + batch_size]

    try:
        # Tokenize batch (truncation length can be reduced to 256 if needed)
        inputs = bert_tokenizer(
            batch_texts,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = bert_model(**inputs)
            # CLS embedding is the first token of last_hidden_state
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            # L2-norm across embedding dimension
            norms = torch.norm(cls_embeddings, p=2, dim=1).cpu().numpy()

        df_all.loc[i : i + len(batch_texts) - 1, feature_98_col] = norms

    except Exception as e:
        print(f"⚠️ Error in batch starting at index {i}: {e}")

    # Free memory after each batch
    del inputs, outputs, cls_embeddings
    torch.cuda.empty_cache()

print("✅ Feature 98 (CLS L2-norm) extraction complete.")

# =========================================================================================
# TASK 3.2: TRAIN / VALIDATION / TEST SPLIT (70 / 15 / 15, STRATIFIED BY TEXT TYPE)
# =========================================================================================

print("\n🔀 Performing 70-15-15 stratified split by 'Text Type'...")

feature_cols = [
    "feat_006_multiple_elongations",
    "feat_029_semicolons",
    "feat_052_interjections",
    "feat_075_active_voice_sentences",
    feature_98_col,
]

train_df, temp_df = train_test_split(
    df_all,
    test_size=0.30,
    random_state=42,
    stratify=df_all["Text Type"],
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["Text Type"],
)

# =========================================================================================
# REPORT GENERATION (HTML, PDF, WORD – LIGHTWEIGHT BUT SUFFICIENT)
# =========================================================================================

print("\n📝 Generating summary reports...")

# 1) Split summary DataFrame
split_summary = pd.DataFrame(
    {
        "Dataset_Part": [
            "Total Dataset",
            "Training Set (70%)",
            "Validation Set (15%)",
            "Test Set (15%)",
        ],
        "Total_Records": [len(df_all), len(train_df), len(val_df), len(test_df)],
        "Human_Count": [
            df_all["Text Type"].eq("Human").sum(),
            train_df["Text Type"].eq("Human").sum(),
            val_df["Text Type"].eq("Human").sum(),
            test_df["Text Type"].eq("Human").sum(),
        ],
        "AI_Count": [
            df_all["Text Type"].eq("AI").sum(),
            train_df["Text Type"].eq("AI").sum(),
            val_df["Text Type"].eq("AI").sum(),
            test_df["Text Type"].eq("AI").sum(),
        ],
    }
)

# Add Human ratio column (percentage)
split_summary["Human_Ratio_%"] = (
    split_summary["Human_Count"] / split_summary["Total_Records"] * 100
).round(0)

# Feature statistics
feature_stats = df_all[feature_cols].describe().round(4)
feature_stats.index.name = "Feature"  # Make first column header explicit in Excel

# 2) HTML report
html_report = f"""
<!DOCTYPE html>
<html dir="rtl" lang="ar">
<head>
    <meta charset="UTF-8">
    <title>Phase 3: Feature Engineering Report</title>
    <style>
        body {{
            font-family: 'Arial', sans-serif;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: #1F4E79;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }}
        th {{
            background-color: #f2f2f2;
        }}
    </style>
</head>
<body>
    <h1>📊 Phase 3: Feature Engineering & Data Splitting</h1>

    <h3>🔀 Dataset Split Summary (70 / 15 / 15)</h3>
    {split_summary.to_html(index=False)}

    <h3>📈 Feature Distribution Statistics</h3>
    {feature_stats.to_html()}
</body>
</html>
"""

with open("Feature_Engineering_Report.html", "w", encoding="utf-8") as f:
    f.write(html_report)

display(HTML(html_report))

# 3) PDF (optional, if WeasyPrint is correctly installed)
try:
    weasyprint.HTML(string=html_report).write_pdf("Feature_Engineering_Report.pdf")
    print("✅ PDF report generated: Feature_Engineering_Report.pdf")
except Exception as e:
    print(f"⚠️ PDF generation skipped: {e}")

# 4) Simple Word document
try:
    doc = Document()
    doc.add_heading("PHASE 3: FEATURE ENGINEERING REPORT", 0)

    p = doc.add_paragraph()
    p.add_run(f"Total Records: {len(df_all)}").bold = True

    doc.add_heading("1. Dataset Split Summary", level=1)
    table = doc.add_table(rows=len(split_summary) + 1, cols=len(split_summary.columns))
    table.style = "Light Grid Accent 1"

    # Header row
    for j, col_name in enumerate(split_summary.columns):
        cell = table.rows[0].cells[j]
        cell.text = str(col_name)
        for run in cell.paragraphs[0].runs:
            run.bold = True

    # Data rows
    for i, row in split_summary.iterrows():
        row_cells = table.add_row().cells
        for j, value in enumerate(row):
            row_cells[j].text = str(value)

    doc.add_heading("2. Feature Statistics (Descriptive)", level=1)
    stats_table = doc.add_table(
        rows=len(feature_stats) + 1, cols=len(feature_stats.columns) + 1
    )
    stats_table.style = "Light Grid Accent 1"

    # Stats header
    stats_table.rows[0].cells[0].text = "Feature"
    for j, col in enumerate(feature_stats.columns, start=1):
        stats_table.rows[0].cells[j].text = str(col)

    # Stats data
    for i, (feat_name, row) in enumerate(feature_stats.iterrows(), start=1):
        stats_table.rows[i].cells[0].text = feat_name
        for j, value in enumerate(row, start=1):
            stats_table.rows[i].cells[j].text = f"{value:.4f}"

    doc.save("Feature_Engineering_Report.docx")
    print("✅ Word report generated: Feature_Engineering_Report.docx")
except Exception as e:
    print(f"⚠️ Word report generation skipped: {e}")

# =========================================================================================
# FINAL SAVE: UNIFIED EXCEL FILE FOR PHASE 4
# =========================================================================================

print("\n💾 Saving unified Excel file with all splits and features...")

output_file = "Complete_Dataset_With_Features.xlsx"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    train_df.to_excel(writer, sheet_name="Train", index=False)
    val_df.to_excel(writer, sheet_name="Validation", index=False)
    test_df.to_excel(writer, sheet_name="Test", index=False)
    df_all.to_excel(writer, sheet_name="All_Data", index=False)
    split_summary.to_excel(writer, sheet_name="Split_Summary", index=False)
    feature_stats.to_excel(writer, sheet_name="Feature_Stats", index=True)

print(f"✅ SUCCESS! Saved main dataset file: {output_file}")

print("\n📥 Preparing downloads...")
files.download(output_file)
try:
    files.download("Feature_Engineering_Report.pdf")
except:
    pass
try:
    files.download("Feature_Engineering_Report.docx")
except:
    pass

print("\n🎯 PHASE 3 COMPLETE")



