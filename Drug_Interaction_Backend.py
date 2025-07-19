#!/usr/bin/env python
# coding: utf-8

# In[22]:


#!pip install requests
#!pip install rapidfuzz


# In[2]:
import streamlit as st
import pandas as pd
import re
import spacy
from fuzzywuzzy import fuzz
from Drug_Interaction_Backend import load_app_assets, find_best_match_combined, fetch_drug_contraindications
import requests
import os

# ---------
# Load spaCy model once
nlp = spacy.load("en_core_web_sm")


# ---------
# Helper function to refine keyword extraction for API query

def refine_keyword_extraction(matched_disease, user_input):
    def normalize(text):
        return re.sub(r'[^\w\s]', '', text.lower()).strip()

    matched_norm = normalize(matched_disease)
    user_norm = normalize(user_input)

    if user_norm and user_norm in matched_norm:
        return user_norm

    user_words = user_norm.split()
    longest_span = ""
    for start in range(len(user_words)):
        for end in range(len(user_words), start, -1):
            span = " ".join(user_words[start:end])
            if span and span in matched_norm:
                if len(span) > len(longest_span):
                    longest_span = span
    if longest_span:
        return longest_span

    matched_words = matched_norm.split()
    common_words = [word for word in user_words if word in matched_words]
    if common_words:
        return " ".join(common_words)

    if user_norm:
        return user_norm

    if matched_norm:
        return max(matched_norm.split(), key=len)

    return matched_disease.lower()


# ---------
# Sentences extraction with spaCy + fuzzy matching
def extract_relevant_sentences(text, disease_term, threshold=80, max_sentences=5):
    doc = nlp(text)
    matches = []
    disease_term_lower = disease_term.lower()
    for sent in doc.sents:
        sent_text = sent.text.lower()
        if disease_term_lower in sent_text:
            matches.append(sent.text)
        else:
            score = fuzz.partial_ratio(disease_term_lower, sent_text)
            if score >= threshold:
                matches.append(sent.text)
        if len(matches) >= max_sentences:
            break
    # fallback to first 1-2 sentences if no matches
    return matches if matches else [str(s) for s in doc.sents][:2]


# ---------
# Fetch detailed label info from OpenFDA and filter relevant sentences
def fetch_drug_label_details(brand_name, disease_keyword, max_chars=1000):
    base_url = "https://api.fda.gov/drug/label.json"
    search_fields = [
        "contraindications",
        "warnings",
        "precautions",
        "adverse_reactions"
    ]
    search_query = f'openfda.brand_name:"{brand_name}" AND (' + \
                   " OR ".join([f'{field}:"{disease_keyword}"' for field in search_fields]) + ")"
    params = {
        "search": search_query,
        "limit": 1
    }
    try:
        resp = requests.get(base_url, params=params, timeout=10)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        if not results:
            return {}

        label = results[0]
        parsed_sections = {}

        for field in search_fields:
            texts = label.get(field)
            if texts:
                combined_text = " ".join(texts)
                # Use fuzzy sentence extraction function here
                relevant_snippets = extract_relevant_sentences(combined_text, disease_keyword)
                snippet = " ".join(relevant_snippets)
                if len(snippet) > max_chars:
                    snippet = snippet[:max_chars] + "..."
                parsed_sections[field.capitalize().replace("_", " ")] = snippet

        return parsed_sections

    except Exception as e:
        print(f"Error fetching label details for {brand_name}: {e}")
        return {}



# ---------------------------
# 1. Load Assets (ICD-10 Data, Model, FAISS Index) - Cached
@st.cache_resource(show_spinner=True)
def cached_load_app_assets():
    # We override loading to use relative paths for deployment
    # If Drug_Interaction_Backend.py uses relative paths internally, no action needed
    return load_app_assets()

df_icd, disease_names_list, sbert_model, faiss_index = cached_load_app_assets()

# ---------------------------
# 2. Streamlit UI
if 'user_text' not in st.session_state:
    st.session_state.user_text = ''
if 'corrected_match' not in st.session_state:
    st.session_state.corrected_match = ''

def on_input_change():
    st.session_state.corrected_match = ''

st.title("Medication Contraindication Checker (OpenFDA & Semantic ICD-10)")

st.write("Enter an illness to see medications with possible safety concerns related to it.")

st.markdown(
    "<small><em>This is for educational purposes only. Not a substitute for professional medical advice.</em></small><br><br>",
    unsafe_allow_html=True)

user_input = st.text_input(
    "Enter disease or illness:",
    value=st.session_state.user_text,
    key='user_text',
    on_change=on_input_change)

if user_input:
    match, score, method = find_best_match_combined(user_input, sbert_model, faiss_index, disease_names_list)

    if not match:
        st.warning(f"No matches found for '{user_input}'. Please check spelling or try different terms.")
        st.session_state.corrected_match = ''
    else:
        icd_code_row = df_icd[df_icd['Description'] == match]
        icd_code = icd_code_row['Code'].values[0] if not icd_code_row.empty else 'Unknown'

        if method == 'semantic':
            st.success(f"Best semantic match: **{match}** (ICD-10: {icd_code}) with similarity {score:.1f}%")
        else:
            st.info(f"Using fallback fuzzy match: **{match}** (ICD-10: {icd_code}), similarity {score:.1f}%. "
                    "Semantic match was below threshold or unavailable.")

        st.session_state.corrected_match = match

        keyword_for_api = refine_keyword_extraction(match, user_input)

        with st.spinner(f"Searching FDA medication safety info for '{keyword_for_api}'..."):
            drugs = fetch_drug_contraindications(keyword_for_api)

        if not drugs:
            st.info(f"No FDA medication labels mentioning '{keyword_for_api}' found in contraindications or related fields.")
        else:
            st.success(f"Drugs with safety concerns related to **{keyword_for_api}** (from FDA):")
            for drug in drugs:
                brand_name = drug.get("brand_name", "Unknown")
                generic_name = drug.get("generic_name", "Unknown")
                st.markdown(f"### {brand_name} ({generic_name})")

                label_info = fetch_drug_label_details(brand_name, keyword_for_api)
                if label_info:
                    for section_title, text_snippet in label_info.items():
                        st.markdown(f"**{section_title}:**")
                        st.write(text_snippet)
                else:
                    st.write("ℹ️ No detailed label safety information found mentioning this illness.")




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




