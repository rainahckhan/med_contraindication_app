import os
import streamlit as st
import pandas as pd
import re
from rapidfuzz import process, fuzz

@st.cache_resource
def load_data():
    parquet_path = os.path.join(os.path.dirname(__file__), "icd10_preprocessed.parquet")
    df = pd.read_parquet(parquet_path)
    disease_names = sorted(df['GeneralDisease'].dropna().unique())
    return df, disease_names

st.cache_data.clear()
# def fuzzy_find_best_match(user_input, disease_names, threshold=80, min_length=4):
#     if not user_input or not disease_names:
#         return None, 0

    # input_clean = user_input.lower().strip()
    # candidate_names = [d for d in disease_names if len(d) >= min_length]

    # matches = process.extract(
    #     input_clean,
    #     candidate_names,
    #     scorer=fuzz.WRatio,
    #     limit=5,
    #     score_cutoff=threshold
    # )

    # input_len = len(input_clean)
    # for match, score, _ in matches:
    #     if len(match) >= 0.7 * input_len:
    #         return match, score

    # return None, 0



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
        # If popularity_dict provided, sort by popularity descending
        if popularity_dict:
            substr_matches.sort(key=lambda x: popularity_dict.get(x.lower(), 0), reverse=True)
        else:
            # fallback: shorter names first as proxy for popularity
            substr_matches.sort(key=len)

        return substr_matches[0], 85  # confidence score indicating fallback substring match

    # No match at all
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

# Initialize session state keys and defaults
if 'user_text' not in st.session_state:
    st.session_state.user_text = ''

if 'corrected_match' not in st.session_state:
    st.session_state.corrected_match = ''

# Callback to handle input changes and clear corrected_match
def on_input_change():
    st.session_state.corrected_match = ''  # Reset corrected match on new input

st.title("Medication Contraindication Checker (Information Obtained from OpenFDA API)")
st.write("Enter an illness to see which medications adversely interact with it (FDA label contraindications).")
st.markdown(
    "<small><em>Disclaimer: This application is for educational purposes only.\
          Please consult a medical professional if you believe you are having a life-threatening reaction.</em></small>",
    unsafe_allow_html=True
)

df, disease_names = load_data()

# Controlled text input with key and on_change callback
user_input = st.text_input(
    "Enter a disease or illness:",
    value=st.session_state.user_text,
    key='user_text',
    on_change=on_input_change
)

# Do NOT assign st.session_state.user_text = user_input here!
# Streamlit already manages this for you based on the 'key'

# if user_input:
#     match, score = fuzzy_find_best_match(user_input, disease_names)
#     if not match:
#         st.warning(f"No matches found for '{user_input}'. Please check spelling or try different wording.")
#         st.session_state.corrected_match = ''
#     else:
#         if match.lower() != user_input.lower():
#             st.info(f"Did you mean **{match}**? Results shown for closest match.")
#         st.session_state.corrected_match = match

#         with st.spinner(f"Searching FDA contraindications for {match}..."):
#             drugs = fetch_drug_contraindications(match)
if user_input:
    match, score = fuzzy_find_best_match(user_input, disease_names)
    if not match:
        st.warning(f"No matches found for '{user_input}'. Please check spelling or try different wording.")
        st.session_state.corrected_match = ''
    else:
        # Notify only if suggested match differs case-insensitive or score < 100
        if match.lower() != user_input.lower() or score < 100:
            st.info(f"Did you mean **{match}**? Results shown for closest match.")
        st.session_state.corrected_match = match

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
