import os
import streamlit as st
import pandas as pd
import re
from rapidfuzz import process, fuzz


@st.cache_resource
def load_data():
    parquet_path = os.path.join(os.path.dirname(__file__), "icd10_preprocessed.parquet")
    df = pd.read_parquet(parquet_path)
    disease_names = sorted(df['Description'].dropna().unique())
    return df, disease_names

st.cache_data.clear()

def fuzzy_find_best_match(user_input, disease_names, threshold=80, min_length=4, popularity_dict=None):
    if not user_input or not disease_names:
        return None, 0

    input_clean = user_input.lower().strip()
    candidate_names = [d for d in disease_names if len(d) >= min_length]

    # 1. Exact match
    for d in candidate_names:
        if d.lower() == input_clean:
            return d, 100

    # 2. Fuzzy match with threshold
    similar_length_names = [
        d for d in candidate_names if abs(len(d) - len(input_clean)) <= 1
    ]
    fuzzy_matches = process.extract(
        input_clean,
        similar_length_names,
        scorer=fuzz.WRatio,
        limit=5,
        score_cutoff=threshold
    )
    if fuzzy_matches:
        best_match, best_score, _ = max(fuzzy_matches, key=lambda x: x[1])
        return best_match, best_score

    # 3. Substring matches anywhere
    substr_matches = [d for d in candidate_names if input_clean in d.lower()]
    if substr_matches:
        if popularity_dict:
            substr_matches.sort(key=lambda x: popularity_dict.get(x.lower(), 0), reverse=True)
        else:
            substr_matches.sort(key=len)
        return substr_matches[0], 85

    return None, 0


@st.cache_data(show_spinner=False)
def fetch_drug_contraindications(disease, max_results=50):
    import requests
    url = 'https://api.fda.gov/drug/label.json'
    # Multi-field search query excluding indications_and_usage
    search_query = (
        f'contraindications:"{disease}"'
        f' OR warnings:"{disease}"'
        f' OR precautions:"{disease}"'
        f' OR adverse_reactions:"{disease}"'
    )
    params = {
        'search': search_query,
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
                'generic_name': info.get('generic_name', [''])[0],
                # Optionally, add these if you want to show context:
                # 'contraindications': ' '.join(entry.get('contraindications', [])),
                # 'warnings': ' '.join(entry.get('warnings', [])),
                # 'precautions': ' '.join(entry.get('precautions', [])),
                # 'adverse_reactions': ' '.join(entry.get('adverse_reactions', [])),
            })
        return drugs
    except Exception:
        return []


if 'user_text' not in st.session_state:
    st.session_state.user_text = ''

if 'corrected_match' not in st.session_state:
    st.session_state.corrected_match = ''

def on_input_change():
    st.session_state.corrected_match = ''

st.title("Medication Contraindication Checker (Information Obtained from OpenFDA API)")
st.write("Enter an illness to see which medications adversely interact with it (FDA label contraindications).")
st.markdown(
    """
    <small>
      <em>
        Disclaimer: This application is for educational and informational purposes only, and does not constitute medical advice, diagnosis, or treatment. Do not rely on this application for medical decision-making. Always consult a qualified healthcare provider with any questions you may have regarding a medical condition. If you are experiencing a medical emergency, call your local emergency number or seek care immediately.<br>
        The creators and developers of this application accept no liability for any harm or loss resulting from your reliance on the information provided.
      </em>
    </small>
    <br><br>
    """,
    unsafe_allow_html=True,
)

df, disease_names = load_data()

user_input = st.text_input(
    "Enter a disease or illness:",
    value=st.session_state.user_text,
    key='user_text',
    on_change=on_input_change
)

if user_input:
    match, score = fuzzy_find_best_match(user_input, disease_names)
    if not match:
        st.warning(f"No matches found for '{user_input}'. Please check spelling or try different wording.")
        st.session_state.corrected_match = ''
    else:
        icd_code_row = df[df['Description'] == match]
        icd_code = icd_code_row['Code'].values[0] if not icd_code_row.empty else 'Unknown'

        if match.lower() != user_input.lower() or score < 100:
            st.info(f"Did you mean **{match}** (ICD-10-CM code: {icd_code})? Results shown for closest match.")
        st.session_state.corrected_match = match

        with st.spinner(f"Searching FDA medication safety information for {match}..."):
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
                st.success(f"Drugs with safety concerns related to **{match}**, please consult a doctor before use:")
                st.dataframe(deduped[["brand_name", "generic_name"]])
            else:
                st.info(f"No drugs with brand or generic names found mentioning '{match}'.")
        else:
            st.info(f"No FDA medication labels found mentioning '{match}' in contraindications, warnings, precautions, or adverse reactions.")




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
