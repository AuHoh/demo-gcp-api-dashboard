import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dashboard Prêt à dépenser crédit")


st.title("Tableau d'évaluation des risques pour l'accord des crédits")
st.subheader("Prédiction du modèle d'évaluation")
@st.cache_data
def get_data():
    path = '/Users/audreyhohmann/Documents/Formation/OCR/P7/df_forstream.parquet'
    return pd.read_parquet(path)
df = get_data()

#st.dataframe(df)
ID_pret = st.selectbox('Choisir ID du prêt', df['SK_ID_CURR'], help='Filtrer sur les identifiants des crédits')

# Filtrer le dataframe pour l'ID sélectionné
filtered_df = df.loc[df['SK_ID_CURR'] == ID_pret]

# Vérifier si des données correspondent à l'ID sélectionné
if not filtered_df.empty:
    st.subheader(f"Profil du client pour l'ID prêt sélectionné : {ID_pret}")
    st.write(f"Genre du client : {filtered_df['CODE_GENDER'].values[0]}")
    age_client = (filtered_df['DAYS_BIRTH'].values[0] / 365).round(0)
    st.write(f"Âge : {age_client}")
    st.write(f"Situation familiale : {filtered_df['NAME_FAMILY_STATUS'].values[0]}")
    st.write(f"Nombre d'enfants : {filtered_df['CNT_CHILDREN'].values[0]}")
    st.write(f"Emploi : {filtered_df['OCCUPATION_TYPE'].values[0]}")

    revenu = float(filtered_df['AMT_INCOME_TOTAL'].values[0])
    st.write(f"Revenus annuels du client : {revenu}")
    updated_income = st.slider("Variations du revenus annuels", 0.0, 600000.0, revenu, 5000.0)

    bien = float(filtered_df['AMT_GOODS_PRICE'].values[0])
    st.write(f"Prix du bien : {bien}")
    updated_bien = st.slider("Variations des prix du bien ", 0.0, 4000000.0, bien, 10000.0)

    credit = float(filtered_df['AMT_CREDIT'].values[0])
    st.write(f"Montant du crédit : {credit}")
    updated_credit = st.slider("Variations des montants du crédit ", 0.0, 4000000.0, credit, 10000.0)

    annuity = float(filtered_df['AMT_ANNUITY'].values[0])
    st.write(f"Montant des annuités : {credit}")
    updated_annuity = st.slider("Variations des montants des annuités ", 0.0, 230000.0, annuity, 5000.0)

    term = float("{:.2f}".format(filtered_df['CREDIT_TERM'].values[0]*100))
    st.write(f"Taux de paiement : {term}")
    updated_term = st.slider("Variations du taux de paiement ", 0.0, 40.0, term, 1.0)


else:
    st.write("Aucune donnée correspondante pour l'ID prêt sélectionné.")

