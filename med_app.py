import streamlit as st
import pandas as pd
import re
import spacy
from fuzzywuzzy import fuzz
from Drug_Interaction_Backend import load_app_assets, find_best_match_combined, fetch_drug_contraindications
import requests
import numpy as np
import html

# ---------------------------
# 1. Load Assets (ICD-10 Data, Model, FAISS Index) - Cached
# ---------------------------
@st.cache_resource(show_spinner=True)
def cached_load_app_assets():
    return load_app_assets()

df_icd, disease_names_list, sbert_model, faiss_index = cached_load_app_assets()

# ---------
# Load spaCy model once
nlp = spacy.load("en_core_web_sm")


# ---------
# Enhanced semantic-only refine_keyword_extraction

def refine_keyword_extraction(matched_disease, user_input, sbert_model=None, similarity_threshold=0.7):
    """
    Refine keyword for OpenFDA query by dynamic semantic similarity between matched disease and user input substrings.
    """
    def normalize(text):
        return re.sub(r'[^\w\s]', '', text.lower()).strip()

    matched_norm = normalize(matched_disease)
    user_norm = normalize(user_input)

    if sbert_model is None:
        # fallback to simpler existing logic
        if user_norm and user_norm in matched_norm:
            return user_norm
        return matched_norm

    matched_vec = sbert_model.encode([matched_disease])[0]

    user_words = user_norm.split()
    max_phrase_len = min(5, len(user_words))  # limit phrase length
    candidates = set()

    for length in range(max_phrase_len, 0, -1):
        for start in range(len(user_words) - length + 1):
            phrase = " ".join(user_words[start:start+length])
            candidates.add(phrase)

    candidates = list(candidates)
    candidate_vecs = sbert_model.encode(candidates)

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    sims = [cosine_sim(matched_vec, cvec) for cvec in candidate_vecs]

    best_idx = np.argmax(sims)
    best_score = sims[best_idx]

    if best_score >= similarity_threshold:
        return candidates[best_idx]

    # Fallback to longest substring of user input inside matched disease
    longest_span = ""
    for start in range(len(user_words)):
        for end in range(len(user_words), start, -1):
            span = " ".join(user_words[start:end])
            if span and span in matched_norm:
                if len(span) > len(longest_span):
                    longest_span = span
                    
    if longest_span:
        return longest_span

    return matched_norm


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
# Highlight matched keywords in text for display

def highlight_keyword(text, keyword):
    # HTML escape to avoid injection issues
    escaped_text = html.escape(text)
    escaped_keyword = html.escape(keyword)
    
    import re
    pattern = re.compile(re.escape(escaped_keyword), re.IGNORECASE)
    highlighted = pattern.sub(r'<mark>\g<0></mark>', escaped_text)
    return highlighted


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
                relevant_sentences = extract_relevant_sentences(combined_text, disease_keyword)
                
                snippet = " ".join(relevant_sentences)
                if len(snippet) > max_chars:
                    snippet = snippet[:max_chars] + "..."

                parsed_sections[field.capitalize().replace("_", " ")] = {
                    "snippet": snippet,
                    "sentences": relevant_sentences
                }

        return parsed_sections

    except Exception as e:
        print(f"Error fetching label details for {brand_name}: {e}")
        return {}


# ---------------------------
# 2. Streamlit UI
# ---------------------------
if 'user_text' not in st.session_state:
    st.session_state.user_text = ''
if 'corrected_match' not in st.session_state:
    st.session_state.corrected_match = ''

def on_input_change():
    st.session_state.corrected_match = ''

st.title("Medication Contraindication Checker (OpenFDA & Semantic ICD-10)")

st.markdown("""
<small>
<em>
<b>Disclaimer:</b> This application is intended for educational and informational purposes only. \
It is not a substitute for professional medical advice, diagnosis, or treatment. \
Always seek the advice of a qualified healthcare provider with any questions regarding a medical condition or medication. \
Reliance on any information provided by this app is solely at your own risk. \
The developers and providers of this app disclaim all liability for any damages or adverse consequences resulting from use of the information herein.
</em>
</small>
<br><br>
""", unsafe_allow_html=True)


st.write("Enter an illness to see medications with possible safety concerns related to it.")

st.markdown("<br>", unsafe_allow_html=True)


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

        keyword_for_api = refine_keyword_extraction(
            match,
            user_input,
            sbert_model=sbert_model,
            similarity_threshold=0.7
        )

        with st.spinner(f"Searching FDA medication safety info for '{keyword_for_api}'..."):
            drugs = fetch_drug_contraindications(keyword_for_api)

        if not drugs:
            st.info(f"No FDA medication labels mentioning '{keyword_for_api}' found in contraindications or related fields.")
        else:
            st.success(f"Drugs with safety concerns related to **{keyword_for_api}** (from FDA):")
            for drug in drugs:
                brand_name = drug.get("brand_name", "Unknown")
                generic_name = drug.get("generic_name", "Unknown")

                st.markdown("---")
                st.markdown(f"## {brand_name} ({generic_name})")

                label_info = fetch_drug_label_details(brand_name, keyword_for_api)
                if label_info:
                    for section_title, info in label_info.items():
                        if isinstance(info, dict):
                            with st.expander(f"{section_title}"):
                                for sentence in info.get("sentences", []):
                                    highlighted_sentence = highlight_keyword(sentence, keyword_for_api)
                                    st.markdown(highlighted_sentence, unsafe_allow_html=True)
                        else:
                            with st.expander(section_title):
                                st.write(info)
                else:
                    st.write("ℹ️ No detailed label safety information found mentioning this illness.")





# ---------------------------------------------------------------------------------------------------------
# import streamlit as st
# import pandas as pd
# import re
# import spacy
# from fuzzywuzzy import fuzz
# from Drug_Interaction_Backend import load_app_assets, find_best_match_combined, fetch_drug_contraindications
# import requests
# import numpy as np

# # ---------------------------
# # 1. Load Assets (ICD-10 Data, Model, FAISS Index) - Cached
# # ---------------------------
# @st.cache_resource(show_spinner=True)
# def cached_load_app_assets():
#     return load_app_assets()

# df_icd, disease_names_list, sbert_model, faiss_index = cached_load_app_assets()

# # ---------
# # Load spaCy model once
# nlp = spacy.load("en_core_web_sm")


# # ---------
# # Enhanced semantic-only refine_keyword_extraction

# def refine_keyword_extraction(matched_disease, user_input, sbert_model=None, similarity_threshold=0.7):
#     """
#     Refine keyword for OpenFDA query by dynamic semantic similarity between matched disease and user input substrings.
    
#     Parameters:
#         matched_disease (str): ICD-10 matched disease string.
#         user_input (str): Original user input.
#         sbert_model: Sentence transformer model (must have encode method).
#         similarity_threshold (float): Minimum cosine similarity to accept a substring.
    
#     Returns:
#         str: Best keyword or phrase to query OpenFDA.
#     """
#     def normalize(text):
#         return re.sub(r'[^\w\s]', '', text.lower()).strip()

#     matched_norm = normalize(matched_disease)
#     user_norm = normalize(user_input)

#     if sbert_model is None:
#         # fallback to simpler existing logic
#         if user_norm and user_norm in matched_norm:
#             return user_norm
#         return matched_norm

#     matched_vec = sbert_model.encode([matched_disease])[0]

#     user_words = user_norm.split()
#     max_phrase_len = min(5, len(user_words))  # limit phrase length
#     candidates = set()

#     # Generate all candidate substrings up to max length
#     for length in range(max_phrase_len, 0, -1):
#         for start in range(len(user_words) - length + 1):
#             phrase = " ".join(user_words[start:start+length])
#             candidates.add(phrase)

#     candidates = list(candidates)
#     candidate_vecs = sbert_model.encode(candidates)

#     def cosine_sim(a, b):
#         return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

#     sims = [cosine_sim(matched_vec, cvec) for cvec in candidate_vecs]

#     best_idx = np.argmax(sims)
#     best_score = sims[best_idx]

#     if best_score >= similarity_threshold:
#         return candidates[best_idx]

#     # Fallback to longest substring of user input inside matched disease
#     longest_span = ""
#     for start in range(len(user_words)):
#         for end in range(len(user_words), start, -1):
#             span = " ".join(user_words[start:end])
#             if span and span in matched_norm:
#                 if len(span) > len(longest_span):
#                     longest_span = span
                    
#     if longest_span:
#         return longest_span

#     return matched_norm


# # ---------
# # Sentences extraction with spaCy + fuzzy matching

# def extract_relevant_sentences(text, disease_term, threshold=80, max_sentences=5):
#     doc = nlp(text)
#     matches = []
#     disease_term_lower = disease_term.lower()
#     for sent in doc.sents:
#         sent_text = sent.text.lower()
#         if disease_term_lower in sent_text:
#             matches.append(sent.text)
#         else:
#             score = fuzz.partial_ratio(disease_term_lower, sent_text)
#             if score >= threshold:
#                 matches.append(sent.text)
#         if len(matches) >= max_sentences:
#             break
#     # fallback to first 1-2 sentences if no matches
#     return matches if matches else [str(s) for s in doc.sents][:2]

# # ---------
# # Fetch detailed label info from OpenFDA and filter relevant sentences

# def fetch_drug_label_details(brand_name, disease_keyword, max_chars=1000):
#     base_url = "https://api.fda.gov/drug/label.json"
#     search_fields = [
#         "contraindications",
#         "warnings",
#         "precautions",
#         "adverse_reactions"
#     ]
#     search_query = f'openfda.brand_name:"{brand_name}" AND (' + \
#                    " OR ".join([f'{field}:"{disease_keyword}"' for field in search_fields]) + ")"
#     params = {
#         "search": search_query,
#         "limit": 1
#     }
#     try:
#         resp = requests.get(base_url, params=params, timeout=10)
#         resp.raise_for_status()
#         results = resp.json().get("results", [])
#         if not results:
#             return {}

#         label = results[0]
#         parsed_sections = {}

#         for field in search_fields:
#             texts = label.get(field)
#             if texts:
#                 combined_text = " ".join(texts)
#                 relevant_snippets = extract_relevant_sentences(combined_text, disease_keyword)
#                 snippet = " ".join(relevant_snippets)
#                 if len(snippet) > max_chars:
#                     snippet = snippet[:max_chars] + "..."
#                 parsed_sections[field.capitalize().replace("_", " ")] = snippet

#         return parsed_sections

#     except Exception as e:
#         print(f"Error fetching label details for {brand_name}: {e}")
#         return {}

# # ---------------------------
# # 2. Streamlit UI
# # ---------------------------
# if 'user_text' not in st.session_state:
#     st.session_state.user_text = ''
# if 'corrected_match' not in st.session_state:
#     st.session_state.corrected_match = ''

# def on_input_change():
#     st.session_state.corrected_match = ''

# st.title("Medication Contraindication Checker (OpenFDA & Semantic ICD-10)")

# st.write("Enter an illness to see medications with possible safety concerns related to it.")

# st.markdown(
#     "<small><em>This is for educational purposes only. Not a substitute for professional medical advice.</em></small><br><br>",
#     unsafe_allow_html=True)

# user_input = st.text_input(
#     "Enter disease or illness:",
#     value=st.session_state.user_text,
#     key='user_text',
#     on_change=on_input_change)

# if user_input:
#     match, score, method = find_best_match_combined(user_input, sbert_model, faiss_index, disease_names_list)

#     if not match:
#         st.warning(f"No matches found for '{user_input}'. Please check spelling or try different terms.")
#         st.session_state.corrected_match = ''
#     else:
#         icd_code_row = df_icd[df_icd['Description'] == match]
#         icd_code = icd_code_row['Code'].values[0] if not icd_code_row.empty else 'Unknown'

#         if method == 'semantic':
#             st.success(f"Best semantic match: **{match}** (ICD-10: {icd_code}) with similarity {score:.1f}%")
#         else:
#             st.info(f"Using fallback fuzzy match: **{match}** (ICD-10: {icd_code}), similarity {score:.1f}%. "
#                     "Semantic match was below threshold or unavailable.")

#         st.session_state.corrected_match = match

#         # Use enhanced semantic refine for keyword extraction
#         keyword_for_api = refine_keyword_extraction(
#             match,
#             user_input,
#             sbert_model=sbert_model,
#             similarity_threshold=0.7
#         )

#         with st.spinner(f"Searching FDA medication safety info for '{keyword_for_api}'..."):
#             drugs = fetch_drug_contraindications(keyword_for_api)

#         if not drugs:
#             st.info(f"No FDA medication labels mentioning '{keyword_for_api}' found in contraindications or related fields.")
#         else:
#             st.success(f"Drugs with safety concerns related to **{keyword_for_api}** (from FDA):")
#             for drug in drugs:
#                 brand_name = drug.get("brand_name", "Unknown")
#                 generic_name = drug.get("generic_name", "Unknown")
#                 st.markdown(f"### {brand_name} ({generic_name})")

#                 label_info = fetch_drug_label_details(brand_name, keyword_for_api)
#                 if label_info:
#                     for section_title, text_snippet in label_info.items():
#                         st.markdown(f"**{section_title}:**")
#                         st.write(text_snippet)
#                 else:
#                     st.write("ℹ️ No detailed label safety information found mentioning this illness.")





# import os
# import streamlit as st
# import pandas as pd
# import numpy as np
# import faiss
# from sentence_transformers import SentenceTransformer
# import requests
# from rapidfuzz import process, fuzz

# # ---------------------------
# # 1. Load ICD-10 Data, Embeddings & Build FAISS Index
# # ---------------------------

# @st.cache_resource(show_spinner=True)

# def load_icd_data_and_model():
#     parquet_path = os.path.join(os.path.dirname(__file__), "icd10_preprocessed.parquet")
#     df = pd.read_parquet(parquet_path)
#     disease_names = df['Description'].dropna().tolist()

#     # Load biomedical sentence-transformer model (make sure it matches your backend)
#     model = SentenceTransformer('pritamdeka/S-BioBert-snli-multinli-stsb')

#     embeddings = model.encode(disease_names, normalize_embeddings=True)
#     embeddings = embeddings.astype(np.float32)

#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatIP(dimension)
#     index.add(embeddings)

#     return df, disease_names, model, embeddings, index


# # Load once and cache
# df, disease_names, model, disease_embeddings, faiss_index = load_icd_data_and_model()

# # ---------------------------
# # 2. Semantic + Fuzzy Matching logic
# # ---------------------------

# def semantic_search(query, top_k=5, score_threshold=0.6):
#     query_embedding = model.encode([query], normalize_embeddings=True).astype(np.float32)
#     distances, indices = faiss_index.search(query_embedding, top_k)
#     results = []
#     for dist, idx in zip(distances[0], indices[0]):
#         if dist >= score_threshold:  # Filter low similarity matches
#             results.append({
#                 'disease': disease_names[idx],
#                 'score': float(dist)
#             })
#     return results

# def fuzzy_find_best_match(user_input, disease_names, threshold=80, min_length=4):
#     if not user_input or not disease_names:
#         return None, 0

#     input_clean = user_input.lower().strip()
#     candidate_names = [d for d in disease_names if len(d) >= min_length]

#     # 1. Exact match
#     for d in candidate_names:
#         if d.lower() == input_clean:
#             return d, 100

#     # 2. Fuzzy match with threshold
#     similar_length_names = [d for d in candidate_names if abs(len(d) - len(input_clean)) <= 1]
#     fuzzy_matches = process.extract(input_clean, similar_length_names, scorer=fuzz.WRatio, limit=5, score_cutoff=threshold)
#     if fuzzy_matches:
#         best_match, best_score, _ = max(fuzzy_matches, key=lambda x: x[1])
#         return best_match, best_score

#     # 3. Substring matches anywhere
#     substr_matches = [d for d in candidate_names if input_clean in d.lower()]
#     if substr_matches:
#         substr_matches.sort(key=len)
#         return substr_matches[0], 85

#     return None, 0

# def find_best_match(user_input, disease_names):
#     # First try semantic matching
#     sem_results = semantic_search(user_input, top_k=3)
#     if sem_results:
#         # Choose best semantic match if similarity above threshold
#         best_sem = sem_results[0]
#         if best_sem['score'] >= 0.65:
#             return best_sem['disease'], best_sem['score'] * 100  # scale similarity to 0-100
#     # Fallback to fuzzy if no good semantic match
#     return fuzzy_find_best_match(user_input, disease_names)

# # ---------------------------
# # 3. FDA contraindication fetch function (unchanged)
# # ---------------------------

# @st.cache_data(show_spinner=False)
# def fetch_drug_contraindications(disease, max_results=50):
#     url = 'https://api.fda.gov/drug/label.json'
#     search_query = (
#         f'contraindications:"{disease}"'
#         f' OR warnings:"{disease}"'
#         f' OR precautions:"{disease}"'
#         f' OR adverse_reactions:"{disease}"'
#     )
#     params = {'search': search_query, 'limit': max_results}
#     try:
#         resp = requests.get(url, params=params, timeout=10)
#         if resp.status_code != 200:
#             return []
#         results = resp.json().get('results', [])
#         drugs = []
#         for entry in results:
#             info = entry.get('openfda', {})
#             drugs.append({
#                 'brand_name': info.get('brand_name', [''])[0],
#                 'generic_name': info.get('generic_name', [''])[0],
#             })
#         return drugs
#     except Exception:
#         return []

# # ---------------------------
# # 4. Streamlit UI
# # ---------------------------

# # Maintain session state for input corrections
# if 'user_text' not in st.session_state:
#     st.session_state.user_text = ''

# if 'corrected_match' not in st.session_state:
#     st.session_state.corrected_match = ''

# def on_input_change():
#     st.session_state.corrected_match = ''

# st.title("Medication Contraindication Checker (OpenFDA & Semantic ICD-10 Search)")

# st.write("Enter an illness to see medications with possible safety concerns related to it.")

# st.markdown("""
# <small><em>This is for educational purposes only. Not a substitute for professional medical advice.</em></small>
# <br><br>
# """, unsafe_allow_html=True)

# user_input = st.text_input(
#     "Enter disease or illness:",
#     value=st.session_state.user_text,
#     key='user_text',
#     on_change=on_input_change
# )

# if user_input:
#     match, score = find_best_match(user_input, disease_names)
#     if not match:
#         st.warning(f"No matches found for '{user_input}'. Please check spelling or try different terms.")
#         st.session_state.corrected_match = ''
#     else:
#         icd_code_row = df[df['Description'] == match]
#         icd_code = icd_code_row['Code'].values[0] if not icd_code_row.empty else 'Unknown'

#         if match.lower() != user_input.lower() or score < 100:
#             st.info(f"Did you mean **{match}** (ICD-10 code: {icd_code})? Showing results for closest match.")

#         st.session_state.corrected_match = match

#         with st.spinner(f"Searching FDA medication safety info for {match}..."):
#             drugs = fetch_drug_contraindications(match)

#         if drugs:
#             drug_df = pd.DataFrame(drugs)
#             drug_df['brand_name'] = drug_df['brand_name'].astype(str).str.strip()
#             drug_df['generic_name'] = drug_df['generic_name'].astype(str).str.strip()
#             filtered = drug_df[(drug_df['brand_name'] != "") | (drug_df['generic_name'] != "")]
#             filtered['_brand_lower'] = filtered['brand_name'].str.lower()
#             filtered['_generic_lower'] = filtered['generic_name'].str.lower()
#             deduped = filtered.drop_duplicates(subset=['_brand_lower', '_generic_lower'])
#             deduped = deduped.drop(columns=['_brand_lower', '_generic_lower'])
#             if not deduped.empty:
#                 st.success(f"Drugs with safety concerns related to **{match}**, please consult a doctor:")
#                 st.dataframe(deduped[["brand_name", "generic_name"]])
#             else:
#                 st.info(f"No drugs found mentioning '{match}' in warnings, contraindications, or adverse reactions.")
#         else:
#             st.info(f"No FDA medication labels mentioning '{match}' found.")





#--------------------------------------------------------------------------------------------------
# import os
# import streamlit as st
# import pandas as pd
# import rapidfuzz
# from rapidfuzz import process

# @st.cache_resource
# def load_data():
#     # Build the path to the parquet file relative to this script's location
#     parquet_path = os.path.join(os.path.dirname(__file__), "icd10_preprocessed.parquet")
#     df = pd.read_parquet(parquet_path)
#     disease_names = sorted(df['GeneralDisease'].dropna().unique())
#     return df, disease_names

# def fuzzy_find_best_match(user_input, disease_names, threshold=75):
#     if not user_input or not disease_names:
#         return None, 0
#     matches = process.extract(user_input, disease_names, limit=1, score_cutoff=threshold)
#     if matches:
#         return matches[0][0], matches[0][1]
#     return None, 0


# @st.cache_data(show_spinner=False)
# def fetch_drug_contraindications(disease, max_results=50):
#     import requests
#     url = 'https://api.fda.gov/drug/label.json'
#     params = {
#         'search': f'contraindications:{disease}',
#         'limit': max_results
#     }
#     try:
#         resp = requests.get(url, params=params, timeout=10)
#         if resp.status_code != 200:
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
#     except Exception:
#         return []


# # ---------------- STREAMLIT UI -------------------
# st.title("Medication Contraindication Checker (Information Obtained from OpenFDA API)")
# st.write("Enter an illness to see which medications adversely interact with it (FDA label contraindications).")
# st.markdown(
#     "<small><em>Disclaimer: This application is for educational purposes only.\
#           Please consult a medical professional if you believe you are having a life-threatening reaction.</em></small>",
#     unsafe_allow_html=True
# )
# df, disease_names = load_data()

# # st.write(f"disease_names type: {type(disease_names)}, count: {len(disease_names)}")
# # st.write("Sample:", disease_names[:5])


# user_input = st.text_input("Enter a disease or illness:")

# # st.write(f"user_input: '{user_input}' of type {type(user_input)}")

# if user_input:
#     match, score = fuzzy_find_best_match(user_input, disease_names)
#     if not match:
#         st.warning(f"No matches found for '{user_input}'. Please check spelling.")
#     else:
#         if match.lower() != user_input.lower():
#             st.info(f"Did you mean **{match}**? Results shown for closest match.")
#         with st.spinner(f"Searching FDA contraindications for {match}..."):
#             drugs = fetch_drug_contraindications(match)
#         if drugs:
#             drug_df = pd.DataFrame(drugs)
#             drug_df['brand_name'] = drug_df['brand_name'].astype(str).str.strip()
#             drug_df['generic_name'] = drug_df['generic_name'].astype(str).str.strip()
#             filtered = drug_df[(drug_df['brand_name'] != "") | (drug_df['generic_name'] != "")]
#             filtered['_brand_lower'] = filtered['brand_name'].str.lower()
#             filtered['_generic_lower'] = filtered['generic_name'].str.lower()
#             deduped = filtered.drop_duplicates(subset=['_brand_lower', '_generic_lower'])
#             deduped = deduped.drop(columns=['_brand_lower', '_generic_lower'])
#             if not deduped.empty:
#                 st.success(f"Drugs contraindicated for **{match}**, please consult a doctor before using:")
#                 st.dataframe(deduped[["brand_name", "generic_name"]])
#             else:
#                 st.info(f"No drugs with brand or generic names found for '{match}'.")
#         else:
#             st.info(f"No FDA medication label lists '{match}' in its contraindications.")







#-------------------------------------------------------------------------------

# import streamlit as st
# import pandas as pd
# import re
# from rapidfuzz import process, fuzz

# @st.cache_resource
# def load_data():
#     # Load the preprocessed CSV file just once, no spaCy needed!
#     # Update path to wherever you saved the CSV from backend
#     df = pd.read_csv(r"C:\Users\slk20\Documents\Drug Interaction App\icd10_preprocessed.csv")
#     disease_names = sorted(df['GeneralDisease'].dropna().unique())
#     return df, disease_names

# def fuzzy_find_best_match(user_input, disease_names, threshold=75):
#     matches = process.extract(user_input, disease_names, limit=1, score_cutoff=threshold)
#     if matches:
#         return matches[0][0], matches[0][1]
#     return None, 0

# @st.cache_data(show_spinner=False)
# def fetch_drug_contraindications(disease, max_results=50):
#     import requests
#     url = 'https://api.fda.gov/drug/label.json'
#     params = {
#         'search': f'contraindications:{disease}',
#         'limit': max_results
#     }
#     try:
#         resp = requests.get(url, params=params, timeout=10)
#         if resp.status_code != 200:
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
#     except Exception:
#         return []

# # ---------------- STREAMLIT UI -------------------
# st.title("Medication Contraindication Checker")
# st.write("Enter an illness and see medications that should not be taken if you have it (FDA label contraindications).")

# df, disease_names = load_data()

# user_input = st.text_input("Enter a disease or illness:")

# if user_input:
#     match, score = fuzzy_find_best_match(user_input, disease_names)
#     if not match:
#         st.warning(f"No matches found for '{user_input}'. Please check spelling.")
#     else:
#         if match.lower() != user_input.lower():
#             st.info(f"Did you mean **{match}**? Results shown for closest match.")
#         with st.spinner(f"Searching FDA contraindications for {match}..."):
#             drugs = fetch_drug_contraindications(match)
#         if drugs:
#             drug_df = pd.DataFrame(drugs)
#             drug_df['brand_name'] = drug_df['brand_name'].astype(str).str.strip()
#             drug_df['generic_name'] = drug_df['generic_name'].astype(str).str.strip()
#             filtered = drug_df[(drug_df['brand_name'] != "") | (drug_df['generic_name'] != "")]
#             filtered['_brand_lower'] = filtered['brand_name'].str.lower()
#             filtered['_generic_lower'] = filtered['generic_name'].str.lower()
#             deduped = filtered.drop_duplicates(subset=['_brand_lower', '_generic_lower'])
#             deduped = deduped.drop(columns=['_brand_lower', '_generic_lower'])
#             if not deduped.empty:
#                 st.success(f"Drugs contraindicated for **{match}**:")
#                 st.dataframe(deduped[["brand_name", "generic_name"]])
#             else:
#                 st.info(f"No drugs with brand or generic names found for '{match}'.")
#         else:
#             st.info(f"No FDA medication label lists '{match}' in its contraindications.")
