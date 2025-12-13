# Phase 2: Data Preprocessing & Exploratory Data Analysis (EDA)

# Task 2.1:

# Design and implement an Arabic-specific text preprocessing pipeline

import re
import unicodedata
import nltk
import pandas as pd
from nltk.stem.isri import ISRIStemmer
from nltk.corpus import stopwords
from datasets import load_dataset
from tabulate import tabulate
# from google.colab import files
# from IPython.display import display

nltk.download('stopwords')

# Load dataset
ds = load_dataset("KFUPM-JRCAI/arabic-generated-abstracts")

abstract_columns = [
    "original_abstract",
    "allam_generated_abstract",
    "jais_generated_abstract",
    "llama_generated_abstract",
    "openai_generated_abstract"
]

splits = ["by_polishing", "from_title", "from_title_and_content"]

# --- Preprocessing Functions ---

# 1- Normalization
def normalize_arabic(text):

    # A- Convert all text to a standard Unicode form
    text = unicodedata.normalize('NFKC', text)

    # B- Remove punctuation, non-Arabic characters, extra spaces, and anything else that is not an Arabic letter

    # B-1. Remove Zero Width and Byte Order Mark characters
    text = re.sub(r'[\u200B\u200C\u200D\uFEFF]', '', text)

    # Convert various whitespace characters to standard space and normalize
    text = re.sub(r'[\s\u00A0\u2000-\u200A\u202F\u205F\u3000\n\r\t\f]', ' ', text)

    # B-2. Remove English characters and numbers
    text = re.sub(r'[a-zA-Z0-9]', '', text)

    # B-3. Remove Tatweel (Kashida)
    text = re.sub(r'\u0640', '', text)

    # B-4. Replace Persian characters with their Arabic equivalents
    text = text.replace('\u067E', '\u0628') # Peh -> Baa
    text = text.replace('\u0686', '\u062C') # Cheh -> Jeem
    text = text.replace('\u0698', '\u0632') # Zh -> Zaa
    text = text.replace('\u06AF', '\u0643') # Gaf -> Kaaf

    # B-5. Remove various punctuation, mathematical symbols, control characters, and other non-Arabic script characters
    text = re.sub(r"[^\u0600-\u06FF\s]", "", text)
    text = re.sub(r'[^\u0621-\u064A\s]', ' ', text)

    # B-6. Normalize multiple spaces to a single space and strip leading/trailing spaces
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub("[^-ۿ ]+", " ", text)

    # C- Dealing with the hamza, alif maqsura, and waw with hamza
    return text.replace("إ", "ا").replace("أ", "ا").replace("آ", "ا").replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")

# 2- Removing diacritics (tashkeel) {َ, ُ, ِ, ْ, ّ, ً, ٌ, ٍ}
def remove_diacritics(text):
    diacritics_pattern = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    # Replaced space with empty string to fully remove diacritics
    return diacritics_pattern.sub('', text)

# 3- Removing stop words (like prepositions, conjunctions, and articles, using NLTK stopwords)
def remove_stopwords(text):
    stop_words = set(stopwords.words("arabic"))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

# 4- Stemming using ISRI stemmer
def stem_text(text):
    stemmer = ISRIStemmer()
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

# 5- Assembling the Pipeline
def preprocess_single_text(text):
    text = normalize_arabic(text)
    text = remove_diacritics(text)
    text = remove_stopwords(text)
    text = stem_text(text)
    return text

# Create a single Excel file so that each split has its own sheet. Each sheet contains two columns to compare the original text with the processed text

# !pip install xlsxwriter

if __name__ == "__main__":
    with pd.ExcelWriter('preprocessing_results.xlsx', engine='xlsxwriter') as writer:
        for split in splits:
            num_rows = len(ds[split])
            for col in abstract_columns:
                original_texts = []
                processed_texts = []
                for i in range(num_rows):
                    raw = ds[split][i][col]
                    proc = preprocess_single_text(raw)
                    original_texts.append(raw)
                    processed_texts.append(proc)

                df = pd.DataFrame({
                    'Original': original_texts,
                    'After Preprocessing': processed_texts
                })

                sheet_name = f"{split}_{col}"
                sheet_name = sheet_name[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    print("\n All text processing results have been saved in a single Excel file named (preprocessing_results.xlsx.). Only two rows of each processing operation will be displayed in the output as a sample for reference, while the processing file will be attached to the project files \n")

    excel_file = 'preprocessing_results.xlsx'
    excel = pd.ExcelFile(excel_file)
    sheet_names = excel.sheet_names

    def pretty_style(df):
        return (
            df.style
            .set_table_styles([
                {'selector': 'th', 'props': [('text-align', 'center'), ('background', '#3a7ca5'), ('color', 'white'), ('font-size', '15px'), ('border', '2px solid #889')]}
            ])
            .set_properties(**{
                'background-color': '#f7fafd',
                'font-size': '14px',
                'border': '1px solid #aab'
            })
        )

    for sheet in sheet_names:
        df = pd.read_excel(excel_file, sheet_name=sheet)
        df_show = df.rename(columns={
            'Original': 'Original Text',
            'After Preprocessing': 'Text After Processing'
        })
        print(f"\n===== Sample of : {sheet} =====")
        # display(pretty_style(df_show.head(2)))
        print(df_show.head(2))
        print('\n' + '=' * 80 + '\n')
    # files.download('preprocessing_results.xlsx')