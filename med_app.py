import os
import streamlit as st
import pandas as pd

@st.cache_resource
def load_data():
    # Build the path to the parquet file relative to this script's location
    parquet_path = os.path.join(os.path.dirname(__file__), "icd10_preprocessed.parquet")
    df = pd.read_parquet(parquet_path)
    disease_names = sorted(df['GeneralDisease'].dropna().unique())
    return df, disease_names


def fuzzy_find_best_match(user_input, disease_names, threshold=75):
    matches = process.extract(user_input, disease_names, limit=1, score_cutoff=threshold)
    if matches:
        return matches[0][0], matches[0][1]
    return None, 0


@st.cache_data(show_spinner=False)
def fetch_drug_contraindications(disease, max_results=50):
    import requests
    url = 'https://api.fda.gov/drug/label.json'
    params = {
        'search': f'contraindications:{disease}',
        'limit': max_results
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return []
        results = resp.json().get('results', [])
        drugs = []
        for entry in results:
            info = entry.get('openfda', {})
            drugs.append({
                'brand_name': info.get('brand_name', [''])[0],
                'generic_name': info.get('generic_name', [''])[0]
            })
        return drugs
    except Exception:
        return []


# ---------------- STREAMLIT UI -------------------
st.title("Medication Contraindication Checker (Information Obtained from OpenFDA API)")
st.write("Enter an illness to see which medications adversely interact with it (FDA label contraindications).")
st.markdown(
    "<small><em>Disclaimer: This application is for educational purposes only.\
          Please consult a medical professional if you believe you are having a life-threatening reaction.</em></small>",
    unsafe_allow_html=True
)
df, disease_names = load_data()


user_input = st.text_input("Enter a disease or illness:")


if user_input:
    match, score = fuzzy_find_best_match(user_input, disease_names)
    if not match:
        st.warning(f"No matches found for '{user_input}'. Please check spelling.")
    else:
        if match.lower() != user_input.lower():
            st.info(f"Did you mean **{match}**? Results shown for closest match.")
        with st.spinner(f"Searching FDA contraindications for {match}..."):
            drugs = fetch_drug_contraindications(match)
        if drugs:
            drug_df = pd.DataFrame(drugs)
            drug_df['brand_name'] = drug_df['brand_name'].astype(str).str.strip()
            drug_df['generic_name'] = drug_df['generic_name'].astype(str).str.strip()
            filtered = drug_df[(drug_df['brand_name'] != "") | (drug_df['generic_name'] != "")]
            filtered['_brand_lower'] = filtered['brand_name'].str.lower()
            filtered['_generic_lower'] = filtered['generic_name'].str.lower()
            deduped = filtered.drop_duplicates(subset=['_brand_lower', '_generic_lower'])
            deduped = deduped.drop(columns=['_brand_lower', '_generic_lower'])
            if not deduped.empty:
                st.success(f"Drugs contraindicated for **{match}**, please consult a doctor before using:")
                st.dataframe(deduped[["brand_name", "generic_name"]])
            else:
                st.info(f"No drugs with brand or generic names found for '{match}'.")
        else:
            st.info(f"No FDA medication label lists '{match}' in its contraindications.")







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
