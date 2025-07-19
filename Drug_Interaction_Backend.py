#!/usr/bin/env python
# coding: utf-8

# In[22]:


#!pip install requests
#!pip install rapidfuzz


# In[2]:
import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests
from rapidfuzz import process, fuzz

# --- Configuration and File Paths ---
# You can adjust these paths if your raw data or precomputed files are elsewhere
# Note: For deployment, ensure these paths are relative to your app's root
RAW_ICD10_TXT_PATH = r"C:\Users\slk20\Documents\Drug Interaction App\icd10cm-Code Descriptions-2026\icd10cm-codes-2026.txt"
PREPROCESSED_PARQUET_PATH = "icd10_preprocessed.parquet"
EMBEDDINGS_NPY_PATH = "disease_embeddings.npy"
FAISS_INDEX_BIN_PATH = "faiss_index.bin"

# --- 1. Offline Preprocessing and Asset Generation ---
def _load_raw_icd10cm_codes(filepath):
    """Loads and preprocesses raw ICD-10-CM codes from a text file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                code, desc = parts
                # Insert decimal point after 3rd char if missing and length > 3
                if len(code) > 3 and '.' not in code:
                    code = code[:3] + '.' + code[3:]
                data.append((code, desc))
    df = pd.DataFrame(data, columns=['Code', 'Description'])
    return df

def generate_precomputed_assets(output_dir="."):
    """
    Generates and saves the preprocessed ICD-10 data, embeddings, and FAISS index.
    This function should be run ONCE locally before deploying the Streamlit app.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load and Preprocess ICD-10 raw text
    print("Step 1/4: Loading and preprocessing raw ICD-10 text data...")
    df_icd = _load_raw_icd10cm_codes(RAW_ICD10_TXT_PATH)
    print(f"Loaded {len(df_icd):,} ICD-10 codes.")

    # Save preprocessed DataFrame to parquet
    parquet_full_path = os.path.join(output_dir, PREPROCESSED_PARQUET_PATH)
    df_icd.to_parquet(parquet_full_path)
    print(f"Saved preprocessed ICD-10 data to {parquet_full_path}")

    # 2. Load BioBERT model and encode disease descriptions
    print("Step 2/4: Loading BioBERT embedding model and encoding descriptions...")
    model = SentenceTransformer('pritamdeka/S-BioBert-snli-multinli-stsb')
    disease_names = df_icd['Description'].tolist()
    disease_embeddings = model.encode(disease_names, normalize_embeddings=True)
    disease_embeddings = disease_embeddings.astype(np.float32)  # FAISS requires float32

    # 3. Build FAISS index
    print("Step 3/4: Building FAISS index...")
    dimension = disease_embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension) # Using faiss_index to avoid conflict with `index` in outer scope
    faiss_index.add(disease_embeddings)
    print(f"FAISS index built with {faiss_index.ntotal} vectors.")

    # 4. Save embeddings and FAISS index
    print("Step 4/4: Saving embeddings and FAISS index...")
    embeddings_full_path = os.path.join(output_dir, EMBEDDINGS_NPY_PATH)
    faiss_index_full_path = os.path.join(output_dir, FAISS_INDEX_BIN_PATH)

    np.save(embeddings_full_path, disease_embeddings)
    faiss.write_index(faiss_index, faiss_index_full_path)
    print(f"Saved embeddings to {embeddings_full_path}")
    print(f"Saved FAISS index to {faiss_index_full_path}")
    print("Preprocessing complete. Assets ready for app deployment.")


# --- 2. In-App Asset Loading (for Streamlit app startup) ---
# This part is for the Streamlit app to load the precomputed assets quickly
_global_data = {
    "df_icd": None,
    "disease_names": None,
    "model": None,
    "faiss_index": None
}

def load_app_assets():
    """
    Loads precomputed assets (ICD data, embeddings, FAISS index) and the SBERT model
    for use in the Streamlit app. This function should be called once by the app.
    """
    if _global_data["df_icd"] is not None: # Check if already loaded
        return _global_data["df_icd"], _global_data["disease_names"], _global_data["model"], _global_data["faiss_index"]

    curr_dir = os.path.dirname(os.path.abspath(__file__)) # Get current script's directory

    # Load preprocessed ICD-10 data
    parquet_full_path = os.path.join(curr_dir, PREPROCESSED_PARQUET_PATH)
    df = pd.read_parquet(parquet_full_path)
    disease_names_list = df['Description'].dropna().tolist()

    # Load BioBERT model (for encoding new queries)
    model = SentenceTransformer('pritamdeka/S-BioBert-snli-multinli-stsb')

    # Load precomputed FAISS index
    faiss_index_full_path = os.path.join(curr_dir, FAISS_INDEX_BIN_PATH)
    faiss_index = faiss.read_index(faiss_index_full_path)

    _global_data["df_icd"] = df
    _global_data["disease_names"] = disease_names_list
    _global_data["model"] = model
    _global_data["faiss_index"] = faiss_index

    return df, disease_names_list, model, faiss_index


# --- 3. Semantic Search and Fuzzy Matching ---
def semantic_search(query, model_obj, faiss_index_obj, disease_names_list, top_k=5, score_threshold=0.65):
    """Performs semantic search using BioBERT and FAISS."""
    query_emb = model_obj.encode([query], normalize_embeddings=True).astype(np.float32)
    distances, indices = faiss_index_obj.search(query_emb, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if dist >= score_threshold:
            results.append({'disease': disease_names_list[idx], 'score': float(dist)})
    return results

def fuzzy_find_best_match(user_input, disease_names_list, threshold=80, min_length=4):
    """Performs fuzzy matching using rapidfuzz."""
    if not user_input or not disease_names_list:
        return None, 0
    input_clean = user_input.lower().strip()
    candidate_names = [d for d in disease_names_list if len(d) >= min_length]

    # 1. Exact match
    for d in candidate_names:
        if d.lower() == input_clean:
            return d, 100

    # 2. Fuzzy match
    close_names = [d for d in candidate_names if abs(len(d) - len(input_clean)) <= 1]
    fuzzy_matches = process.extract(input_clean, close_names, scorer=fuzz.WRatio, limit=5, score_cutoff=threshold)
    if fuzzy_matches:
        best_match, best_score, _ = max(fuzzy_matches, key=lambda x: x[1])
        return best_match, best_score

    # 3. Substring match
    substr_matches = [d for d in candidate_names if input_clean in d.lower()]
    if substr_matches:
        substr_matches.sort(key=len)
        return substr_matches[0], 85
    return None, 0

def find_best_match_combined(user_input, model_obj, faiss_index_obj, disease_names_list):
    """Combines semantic and fuzzy matching for the best match."""
    sem_results = semantic_search(user_input, model_obj, faiss_index_obj, disease_names_list)
    if sem_results:
        best_sem = sem_results[0]
        # Use a semantic score threshold to determine if semantic match is good enough
        if best_sem['score'] >= 0.65: # This threshold can be adjusted
            return best_sem['disease'], best_sem['score'] * 100, 'semantic'

    fuzzy_match, fuzzy_score = fuzzy_find_best_match(user_input, disease_names_list)
    if fuzzy_match:
        # Use a fuzzy score threshold, e.g., only if fuzzy score is reasonably high
        if fuzzy_score >= 75: # This threshold can be adjusted
            return fuzzy_match, fuzzy_score, 'fuzzy'
    return None, 0, None


# --- 4. FDA Drug Contraindications Lookup ---
def fetch_drug_contraindications(disease, max_results=50):
    """Fetches drug contraindications from the OpenFDA API."""
    url = 'https://api.fda.gov/drug/label.json'
    # Broaden search to multiple fields to increase chances of finding info
    search_query = (
        f'contraindications:"{disease}"'
        f' OR warnings:"{disease}"'
        f' OR precautions:"{disease}"'
        f' OR adverse_reactions:"{disease}"'
        f' OR indications_and_usage:"{disease}"' # Sometimes disease is mentioned here for related drugs
    )
    params = {'search': search_query, 'limit': max_results}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        results = resp.json().get('results', [])
        drugs = []
        for entry in results:
            info = entry.get('openfda', {})
            # Ensure brand_name and generic_name are lists and take the first item, default to empty string
            brand_name = info.get('brand_name', [''])[0]
            generic_name = info.get('generic_name', [''])[0]
            if brand_name or generic_name: # Only add if at least one name is present
                drugs.append({
                    'brand_name': brand_name,
                    'generic_name': generic_name,
                })
        # Deduplicate results based on both brand and generic names
        df_drugs = pd.DataFrame(drugs)
        if not df_drugs.empty:
            df_drugs['brand_lower'] = df_drugs['brand_name'].str.lower().str.strip()
            df_drugs['generic_lower'] = df_drugs['generic_name'].str.lower().str.strip()
            # Handle cases where one name is empty to ensure proper deduplication
            df_drugs['brand_lower'] = df_drugs['brand_lower'].replace('', np.nan)
            df_drugs['generic_lower'] = df_drugs['generic_lower'].replace('', np.nan)
            deduplicated_drugs = df_drugs.drop_duplicates(subset=['brand_lower', 'generic_lower'], keep='first')
            return deduplicated_drugs[['brand_name', 'generic_name']].to_dict('records')
        return []
    except requests.exceptions.Timeout:
        print("API request timed out.")
        return []
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred during API fetch: {e}")
        return []

# --- Main execution for asset generation (run this script directly ONCE) ---
if __name__ == "__main__":
    print("--- Running Med Checker Backend Preprocessing ---")
    # This will generate the necessary precomputed files in the current directory
    # or the specified output_dir. Ensure RAW_ICD10_TXT_PATH is correct.
    generate_precomputed_assets(".") # Generates files in the current directory (where this script is)
    print("--- Preprocessing Complete ---")

    # Optional: Test loading assets and search functions
    print("\n--- Testing loaded assets and search functions ---")
    try:
        df_test, disease_names_test, model_test, faiss_index_test = load_app_assets()
        print(f"Assets loaded successfully. ICD codes count: {len(df_test)}")

        test_query = "systemic lupus"
        best_match, score, method = find_best_match_combined(test_query, model_test, faiss_index_test, disease_names_test)
        if best_match:
            print(f"Test query '{test_query}': Best match '{best_match}' (score: {score:.2f}, method: {method})")
            test_drugs = fetch_drug_contraindications(best_match)
            if test_drugs:
                print(f"Found {len(test_drugs)} drugs for '{best_match}'. Example: {test_drugs[0]}")
            else:
                print(f"No drugs found for '{best_match}'.")
        else:
            print(f"No match found for test query '{test_query}'.")

    except Exception as e:
        print(f"Error during test run: {e}")
        print("Ensure 'generate_precomputed_assets()' was run successfully first.")




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




