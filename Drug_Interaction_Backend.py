#!/usr/bin/env python
# coding: utf-8

# In[22]:


#!pip install requests
#!pip install rapidfuzz


# In[2]:
import pandas as pd
import re
import spacy
from tqdm import tqdm

# -------------------
# 1. Parameters
# -------------------
infile = r"C:\Users\slk20\Documents\Drug Interaction App\icd10OrderFiles2025_0\icd10cm_order_2025.txt"
outfile = r"C:\Users\slk20\Documents\Drug Interaction App\icd10_preprocessed.parquet"   # Use .csv if you prefer!

# -------------------
# 2. Helpers
# -------------------
def clean_text_minimal(text):
    """Lowercase and strip, but keep all punctuation for better NLP parsing."""
    return text.lower().strip()

def extract_main_disease_from_doc(doc):
    """
    Extract the main disease as the longest noun chunk text.
    Falls back to longest noun token lemma if no noun chunks found.
    """
    noun_chunks = list(doc.noun_chunks)
    if noun_chunks:
        # Return the longest noun chunk text (preserves phrase)
        return max(noun_chunks, key=lambda chunk: len(chunk.text)).text.strip()
    
    # Fallback: longest noun lemma (single word)
    disease_terms = [token.lemma_ for token in doc if token.pos_ == "NOUN" and not token.is_stop]
    if disease_terms:
        return max(disease_terms, key=len)
    
    # Last fallback: last word in the doc text
    return doc.text.split()[-1]

# -------------------
# 3. Load Raw Data
# -------------------
rows = []
with open(infile, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split(maxsplit=3)
        if len(parts) == 4:
            rows.append(parts)
        else:
            continue  # skip malformed lines

df = pd.DataFrame(rows, columns=['RowID', 'Code', 'Flag', 'FullDescription'])
df['FullDescription_only'] = df['FullDescription'].str.split(r'\s{2,}', regex=True).str[0]

# Now apply minimal cleaning to preserve punctuation for NLP
df['CleanDescription'] = df['FullDescription_only'].apply(clean_text_minimal)

# -------------------
# 4. NLP Extraction (spaCy, batched with progress bar)
# -------------------
print("Processing disease terms with spaCy, this may take a while on first run...")

nlp = spacy.load("en_core_web_sm")
descriptions = df["CleanDescription"].astype(str).tolist()
results = []
batch_size = 500

for batch_start in tqdm(range(0, len(descriptions), batch_size), desc="spaCy NLP"):
    batch = descriptions[batch_start:batch_start + batch_size]
    docs = nlp.pipe(batch, batch_size=batch_size)
    results.extend(extract_main_disease_from_doc(doc) for doc in docs)

df['GeneralDisease'] = results

# -------------------
# 5. Save Preprocessed Table (only essential columns, as Parquet)
# -------------------
df_out = df[['Code', 'GeneralDisease', 'CleanDescription']]
df_out = df_out.drop_duplicates().reset_index(drop=True)

# Save in compact, fast format
df_out.to_parquet(outfile)
print(f"\nExported {len(df_out):,} rows to: {outfile}")





#-------------------------------------------------------
# import pandas as pd
# import re
# import spacy
# from tqdm import tqdm


# # -------------------
# # 1. Parameters
# # -------------------
# infile = r"C:\Users\slk20\Documents\Drug Interaction App\icd10OrderFiles2025_0\icd10cm_order_2025.txt"
# outfile = r"C:\Users\slk20\Documents\Drug Interaction App\icd10_preprocessed.parquet"   # Use .csv if you prefer!


# # -------------------
# # 2. Helpers
# # -------------------
# def clean_text(text):
#     """Lowercase, remove punctuation and extra whitespace."""
#     text = text.lower()
#     text = re.sub(r'[^a-z0-9\s]', '', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text


# def extract_main_disease_from_doc(doc):
#     disease_terms = set()
#     for chunk in doc.noun_chunks:
#         disease_terms.add(chunk.root.lemma_)
#     for token in doc:
#         if token.pos_ == "NOUN" and not token.is_stop:
#             disease_terms.add(token.lemma_)
#     if disease_terms:
#         return max(disease_terms, key=len)
#     return doc.text.split()[-1]


# # -------------------
# # 3. Load Raw Data
# # -------------------
# rows = []
# with open(infile, 'r', encoding='utf-8') as f:
#     for line in f:
#         parts = line.strip().split(maxsplit=3)
#         if len(parts) == 4:
#             rows.append(parts)
#         else:
#             continue  # skip malformed lines


# df = pd.DataFrame(rows, columns=['RowID', 'Code', 'Flag', 'FullDescription'])
# df['FullDescription_only'] = df['FullDescription'].str.split(r'\s{2,}', regex=True).str[0]
# df['CleanDescription'] = df['FullDescription_only'].apply(clean_text)


# # -------------------
# # 4. NLP Extraction (spaCy, batched with progress bar)
# # -------------------
# print("Processing disease terms with spaCy, this may take a while on first run...")


# nlp = spacy.load("en_core_web_sm")
# descriptions = df["CleanDescription"].astype(str).tolist()
# results = []
# batch_size = 500


# for batch_start in tqdm(range(0, len(descriptions), batch_size), desc="spaCy NLP"):
#     batch = descriptions[batch_start:batch_start + batch_size]
#     docs = nlp.pipe(batch, batch_size=batch_size)
#     results.extend(extract_main_disease_from_doc(doc) for doc in docs)


# df['GeneralDisease'] = results


# # -------------------
# # 5. Save Preprocessed Table (only essential columns, as Parquet)
# # -------------------
# df_out = df[['Code', 'GeneralDisease', 'CleanDescription']]
# df_out = df_out.drop_duplicates().reset_index(drop=True)


# # Save in compact, fast format
# df_out.to_parquet(outfile)
# print(f"\nExported {len(df_out):,} rows to: {outfile}")










#------------------------------------------------------------------------------------------------------------------
# import pandas as pd
# import re
# import spacy
# from tqdm import tqdm

# # -------------------
# # 1. Parameters
# # -------------------
# infile = r"C:\Users\slk20\Documents\Drug Interaction App\icd10OrderFiles2025_0\icd10cm_order_2025.txt"
# outfile = r"C:\Users\slk20\Documents\Drug Interaction App\icd10_preprocessed.parquet"   # Use .csv if you prefer!

# # -------------------
# # 2. Helpers
# # -------------------
# def clean_text(text):
#     """Lowercase, remove punctuation and extra whitespace."""
#     text = text.lower()
#     text = re.sub(r'[^a-z0-9\s]', '', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# def extract_main_disease_from_doc(doc):
#     disease_terms = set()
#     for chunk in doc.noun_chunks:
#         disease_terms.add(chunk.root.lemma_)
#     for token in doc:
#         if token.pos_ == "NOUN" and not token.is_stop:
#             disease_terms.add(token.lemma_)
#     if disease_terms:
#         return max(disease_terms, key=len)
#     return doc.text.split()[-1]

# # -------------------
# # 3. Load Raw Data
# # -------------------
# rows = []
# with open(infile, 'r', encoding='utf-8') as f:
#     for line in f:
#         parts = line.strip().split(maxsplit=3)
#         if len(parts) == 4:
#             rows.append(parts)
#         else:
#             continue  # skip malformed lines

# df = pd.DataFrame(rows, columns=['RowID', 'Code', 'Flag', 'FullDescription'])
# df['FullDescription_only'] = df['FullDescription'].str.split(r'\s{2,}', regex=True).str[0]
# df['CleanDescription'] = df['FullDescription_only'].apply(clean_text)

# # -------------------
# # 4. NLP Extraction (spaCy, batched with progress bar)
# # -------------------
# print("Processing disease terms with spaCy, this may take a while on first run...")

# nlp = spacy.load("en_core_web_sm")
# descriptions = df["CleanDescription"].astype(str).tolist()
# results = []
# batch_size = 500

# for batch_start in tqdm(range(0, len(descriptions), batch_size), desc="spaCy NLP"):
#     batch = descriptions[batch_start:batch_start + batch_size]
#     docs = nlp.pipe(batch, batch_size=batch_size)
#     results.extend(extract_main_disease_from_doc(doc) for doc in docs)

# df['GeneralDisease'] = results

# # -------------------
# # 5. Save Preprocessed Table (only essential columns, as Parquet)
# # -------------------
# df_out = df[['Code', 'GeneralDisease', 'CleanDescription']]
# df_out = df_out.drop_duplicates().reset_index(drop=True)

# # Save in compact, fast format
# df_out.to_parquet(outfile)
# print(f"\nExported {len(df_out):,} rows to: {outfile}")

# --- If you want CSV (slower, larger), uncomment: ---
# csv_outfile = outfile.replace('.parquet', '.csv')
# df_out.to_csv(csv_outfile, index=False)
# print(f"\nExported also as CSV to: {csv_outfile}")











# ---------------------------------------------------------------------------------

# import requests
# import pandas as pd
# import re
# from rapidfuzz import process, fuzz

# from tqdm import tqdm

# # (Optional) For Jupyter tqdm visualization
# from tqdm.notebook import tqdm as tqdm_notebook
# import spacy
# nlp = spacy.load("en_core_web_sm")


# # In[3]:


# file_path = r"C:\Users\slk20\Documents\Drug Interaction App\icd10OrderFiles2025_0\icd10cm_order_2025.txt"

# rows = []
# with open(file_path, 'r', encoding='utf-8') as f:
#     for line in f:
#         # Split the line into max 4 parts: first three columns + rest as full description
#         parts = line.strip().split(maxsplit=3)  
#         if len(parts) == 4:
#             row_id, code, flag, full_description = parts
#             rows.append([row_id, code, flag, full_description])
#         else:
#             print("Skipping malformed line:", line)

# # Create DataFrame with meaningful column names
# df = pd.DataFrame(rows, columns=['RowID', 'Code', 'Flag', 'FullDescription'])

# df['FullDescription_only'] = df['FullDescription'].str.split(r'\s{2,}', regex=True).str[0]

# print(df[['FullDescription', 'FullDescription_only']].head())

# print(df.head())


# # In[4]:


# def clean_text(text):
#     # Convert to lowercase
#     text = text.lower()
#     # Remove punctuation (keep spaces and alphanumeric)
#     text = re.sub(r'[^a-z0-9\s]', '', text)
#     # Remove extra spaces
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# # Apply to your dataframe column
# df['CleanDescription'] = df['FullDescription_only'].apply(clean_text)



# print(df[['FullDescription_only', 'CleanDescription']].head())


# # In[5]:


# df_clean = df.drop({'FullDescription_only', 'FullDescription'}, axis = 1)


# df_clean.head()


# # In[6]:


# def extract_main_disease_from_doc(doc):
#     disease_terms = set()
#     # Noun chunks
#     for chunk in doc.noun_chunks:
#         disease_terms.add(chunk.root.lemma_)
#     for token in doc:
#         if token.pos_ == "NOUN" and not token.is_stop:
#             disease_terms.add(token.lemma_)
#     if disease_terms:
#         return max(disease_terms, key=len)
#     return doc.text.split()[-1]

# # Batch with nlp.pipe (progress bar with tqdm)
# descriptions = df_clean["CleanDescription"].astype(str).tolist()
# results = []
# batch_size = 500
# print("Extracting main disease terms, please wait...")

# for batch_start in tqdm(range(0, len(descriptions), batch_size)):
#     batch = descriptions[batch_start:batch_start+batch_size]
#     docs = nlp.pipe(batch, batch_size=batch_size)
#     for doc in docs:
#         results.append(extract_main_disease_from_doc(doc))

# df_clean["GeneralDisease"] = results

# # Preview results
# print(df_clean[['CleanDescription', 'GeneralDisease']].head(10))
# print(f"\nUnique extracted diseases: {sorted(set(df_clean['GeneralDisease']))[:20]} ...")


# # In[7]:


# disease_names = sorted(df_clean['GeneralDisease'].dropna().unique())

# def fuzzy_find_best_match(user_input, disease_names, threshold=75):
#     """Returns (match, score), or (None, 0) if no match above threshold."""
#     matches = process.extract(user_input, disease_names, limit=1, score_cutoff=threshold)
#     if matches:
#         return matches[0][0], matches[0][1]  # (disease_name, score)
#     return None, 0


# # In[21]:


# def fetch_drug_contraindications(disease, max_results=50):
#     url = 'https://api.fda.gov/drug/label.json'
#     params = {
#         'search': f'contraindications:{disease}',
#         'limit': max_results
#     }
#     try:
#         resp = requests.get(url, params=params, timeout=10)
#         if resp.status_code != 200:
#             if resp.status_code != 404:
#                 print(f"Error: {resp.status_code}: {resp.text}")
#             return []
#         results = resp.json().get('results', [])
#         drugs = []
#         for entry in results:
#             info = entry.get('openfda', {})
#             drugs.append({
#                 'brand_name': info.get('brand_name', [''])[0],
#                 'generic_name': info.get('generic_name', [''])[0]
#             })
#         return drugs
#     except Exception as e:
#         print(f"Exception: {e}")
#         return []

# def fuzzy_find_best_match(user_input, disease_names, threshold=75):
#     matches = process.extract(user_input, disease_names, limit=1, score_cutoff=threshold)
#     if matches:
#         return matches[0][0], matches[0][1]
#     return None, 0

# # --- USAGE ---
# user_input = input("Enter a disease (e.g., glaucoma, diabetes...): ").strip()
# match, score = fuzzy_find_best_match(user_input, disease_names)

# if not match:
#     print(f"No matches found for '{user_input}'. Please check spelling.")
# else:
#     if match.lower() != user_input.lower():
#         print(f"Did you mean '{match}'?")
#     drugs = fetch_drug_contraindications(match)
#     if drugs:
#         df = pd.DataFrame(drugs)
#         df['brand_name'] = df['brand_name'].astype(str).str.strip()
#         df['generic_name'] = df['generic_name'].astype(str).str.strip()
#         filtered = df[(df['brand_name'] != "") | (df['generic_name'] != "")].copy()
#         filtered['_brand_lower'] = filtered['brand_name'].str.lower()
#         filtered['_generic_lower'] = filtered['generic_name'].str.lower()
#         deduped = filtered.drop_duplicates(subset=['_brand_lower', '_generic_lower'])
#         deduped = deduped.drop(columns=['_brand_lower', '_generic_lower'])
#         if not deduped.empty:
#             print(f"Drugs contraindicated for '{match}':")
#             print(deduped[["brand_name", "generic_name"]].to_string(index=False))
#         else:
#             print(f"No drugs with brand or generic names found for '{match}'.")
#     else:
#         print(f"No FDA medication label lists '{match}' in its contraindications.")


# In[ ]:





# In[ ]:




