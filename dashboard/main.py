import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Dashboard Prêt à dépenser crédit")

st.title("Tableau d'évaluation des risques pour l'accord des crédits")

@st.cache_data
def get_data():
    path = '/Users/audreyhohmann/Documents/Formation/OCR/P7/df_forstream.parquet'
    return pd.read_parquet(path)
df = get_data()

#st.dataframe(df)
ID_pret = st.selectbox('Choisir ID du prêt', df['SK_ID_CURR'], help='Filtrer sur les identifiants des crédits')

# Filtrer le dataframe pour l'ID sélectionné
filtered_df = df.loc[df['SK_ID_CURR'] == ID_pret]

predict_btn = st.button('Prédire')
st.write("Prédiction du modèle d'évaluation : ")
#st.metric(label="Score de prédiction", value=0.53, delta=value-th_proba)

st.subheader('Modifications des valeurs des features pour nouvelle prédiction : ')
# Vérifier si des données correspondent à l'ID sélectionné
if not filtered_df.empty:
    with st.sidebar:
        st.subheader(f"Profil du client pour l'ID prêt sélectionné : {ID_pret}")
        st.write(f"Genre du client : {filtered_df['CODE_GENDER'].values[0]}")
        age_client = (filtered_df['DAYS_BIRTH'].values[0] / 365).round(0)
        st.write(f"Âge : {age_client}")
        st.write(f"Situation familiale : {filtered_df['NAME_FAMILY_STATUS'].values[0]}")
        st.write(f"Nombre d'enfants : {filtered_df['CNT_CHILDREN'].values[0]}")
        st.write(f"Propriétaire d'une maison ou d'un appartement (Y/N) : {filtered_df['FLAG_OWN_REALTY'].values[0]}")
        st.write(f"Propriétaire d'une voiture (Y/N) : {filtered_df['FLAG_OWN_CAR'].values[0]}")
        st.write(f"Emploi : {filtered_df['OCCUPATION_TYPE'].values[0]}")
        years_employed = (filtered_df['DAYS_EMPLOYED'].values[0] / - 365).round(0)
        st.write(f"Nombre d'années d'activité : {years_employed}")

    revenu = float(filtered_df['AMT_INCOME_TOTAL'].values[0])
    st.write(f"Revenus annuels du client ('AMT_INCOME_TOTAL') : {revenu}")
    updated_income = st.slider("Variations du revenus annuels", 0.0, 600000.0, revenu, 5000.0)

    bien = float(filtered_df['AMT_GOODS_PRICE'].values[0])
    st.write(f"Prix du bien ('AMT_GOODS_PRICE') : {bien}")
    updated_bien = st.slider("Variations des prix du bien ", 0.0, 4000000.0, bien, 10000.0)

    credit = float(filtered_df['AMT_CREDIT'].values[0])
    st.write(f"Montant du crédit ('AMT_CREDIT') : {credit}")
    updated_credit = st.slider("Variations des montants du crédit ", 0.0, 4000000.0, credit, 10000.0)

    annuity = float(filtered_df['AMT_ANNUITY'].values[0])
    st.write(f"Montant des annuités : {credit}")
    updated_annuity = st.slider("Variations des montants des annuités ('AMT_ANNUITY')", 0.0, 230000.0, annuity, 5000.0)

    term = float("{:.2f}".format(filtered_df['CREDIT_TERM'].values[0]*100))
    st.write(f"Taux de paiement : {term}")
    updated_term = st.slider("Variations du taux de paiement ", 0.0, 40.0, term, 1.0)
else:
    st.write("Aucune donnée correspondante pour l'ID prêt sélectionné.")

predict_btn = st.button('Nouvelle prédiction')

@st.cache_data
def get_data():
    path_train = '/Users/audreyhohmann/Documents/Formation/OCR/P7/df_train_forstream.parquet'
    return pd.read_parquet(path_train)
df_train = get_data()


def plot_kde(df, feature_y):
    # Créer la figure et les sous-graphiques
    fig, ax = plt.subplots(figsize=(10, 8))

    # Condition spécifique pour DAYS_BIRTH et DAYS_EMPLOYED
    if feature_y == 'DAYS_BIRTH':
        # KDE plot des prêts remboursés à temps (target == 0)
        sns.kdeplot(df.loc[df['TARGET'] == 0, feature_y] / 365, color="green", label='crédit accordé', ax=ax)

        # KDE plot des prêts non remboursés à temps (target == 1)
        sns.kdeplot(df.loc[df['TARGET'] == 1, feature_y] / 365, color="red", label='crédit refusé', ax=ax)

    elif feature_y == 'DAYS_EMPLOYED':
        # KDE plot des prêts remboursés à temps (target == 0)
        sns.kdeplot(df.loc[df['TARGET'] == 0, feature_y] / -365, color="green", label='crédit accordé', ax=ax)

        # KDE plot des prêts non remboursés à temps (target == 1)
        sns.kdeplot(df.loc[df['TARGET'] == 1, feature_y] / -365, color="red", label='crédit refusé', ax=ax)

    else:
        # KDE plot des prêts remboursés à temps (target == 0)
        sns.kdeplot(df.loc[df['TARGET'] == 0, feature_y], color="green", label='crédit accordé', ax=ax)

        # KDE plot des prêts non remboursés à temps (target == 1)
        sns.kdeplot(df.loc[df['TARGET'] == 1, feature_y], color="red", label='crédit refusé', ax=ax)


    # Ajout de la position du client
    if selected_feature_y == 'AMT_INCOME_TOTAL':
        plt.axvline(x=revenu, color='blue', linestyle='--', label='Position du client')
    elif selected_feature_y == 'AMT_CREDIT':
        plt.axvline(x=credit, color='blue', linestyle='--', label='Position du client')
    elif selected_feature_y == 'AMT_GOODS_PRICE':
        plt.axvline(x=bien, color='blue', linestyle='--', label='Position du client')
    elif selected_feature_y == 'AMT_ANNUITY':
        plt.axvline(x=annuity, color='blue', linestyle='--', label='Position du client')
    elif selected_feature_y == 'DAYS_BIRTH':
        plt.axvline(x=age_client, color='blue', linestyle='--', label='Position du client')
    elif selected_feature_y == 'DAYS_EMPLOYED':
        plt.axvline(x=years_employed, color='blue', linestyle='--', label='Position du client')

    # Configuration de l'étiquetage du graphique
    ax.set_xlabel(feature_y)
    ax.set_ylabel('Densité')
    ax.set_title('Répartition des ' + feature_y)


    # Ajout de la légende
    ax.legend()

    # Redimensionner le texte pour une meilleure lisibilité
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Afficher le graphique dans Streamlit
    st.pyplot(fig)

st.subheader("Distribution de la feature sélectionnée selon les classes du modèle d'entraînement")
# Widget selectbox pour choisir la feature y
selected_feature_y = st.selectbox('Choisir la feature', df_train[['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_ANNUITY', 'DAYS_BIRTH', 'DAYS_EMPLOYED']].columns)

# Vérifier si des données correspondent à la feature sélectionnée
if selected_feature_y in df_train.columns:
    # Appeler la fonction plot_kde avec la feature sélectionnée
    plot_kde(df_train, selected_feature_y)

else:
    st.write("La feature sélectionnée n'est pas présente dans le dataframe.")
