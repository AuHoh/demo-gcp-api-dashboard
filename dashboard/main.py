import json
import os
import pickle

from PIL import Image

import pandas as pd
import numpy as np
import requests
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

api_uri = os.getenv('API_URI', 'http://localhost:8000')

st.set_page_config(page_title="Prêt à dépenser")

st.title("Tableau d'évaluation des risques pour l'accord des crédits")


def request_prediction(api_uri, data):
    headers = {"Content-Type": "application/json"}

    response = requests.request(
        method='POST', headers=headers, url=api_uri, json=data)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()


@st.cache_data
def get_std_scaler():
    model_path = os.getenv('SCALER_PATH', 'scaler.pkl')
    with open(model_path, 'rb') as file:
        std_scaler = pickle.load(file)
    return std_scaler


@st.cache_data
def get_data_test_predict():
    path = '/Users/audreyhohmann/Documents/Formation/OCR/P7/X_test_full.parquet'
    return pd.read_parquet(path)


df_predict = get_data_test_predict()


@st.cache_data
def get_data_test():
    path = '/Users/audreyhohmann/Documents/Formation/OCR/P7/df_forstream.parquet'
    return pd.read_parquet(path)


std_scaler = get_std_scaler()

df = get_data_test()

def dowload_excel():
    # Chemin du fichier Excel existant
    col_des = '/Users/audreyhohmann/Documents/Formation/OCR/P7/colonnes_description.xlsx'

    # Lecture du fichier Excel en tant que binaire
    with open(col_des, 'rb') as fichier:
        contenu = fichier.read()

    # Téléchargement du fichier
    st.download_button(
        label='Télécharger la description des features du modèle',
        data=contenu,
        file_name='colonnes_description.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )


# st.dataframe(df)
ID_pret = st.selectbox('Choisir ID du prêt', df['SK_ID_CURR'], help='Filtrer sur les identifiants des crédits')

# Filtrer le dataframe pour l'ID sélectionné
filtered_df = df.loc[df['SK_ID_CURR'] == ID_pret]
filtered_df_predict = df_predict.loc[df_predict['SK_ID_CURR'] == ID_pret].reset_index(drop=True)

st.subheader('OPTIONNEL : Modifications des valeurs des features suivantes : ')
# Vérifier si des données correspondent à l'ID sélectionné
if not filtered_df.empty:
    with st.sidebar:
        st.header(f"Profil du client pour l'ID prêt sélectionné : {ID_pret}")
        genre = filtered_df['CODE_GENDER'].values[0]
        st.write(f"Genre du client : {genre}")
        age_client = (filtered_df['DAYS_BIRTH'].values[0] / 365).round(0)
        st.write(f"Âge : {age_client}")
        family = filtered_df['NAME_FAMILY_STATUS'].values[0]
        st.write(f"Situation familiale : {family}")
        st.write(f"Nombre d'enfants : {filtered_df['CNT_CHILDREN'].values[0]}")
        st.write(f"Propriétaire d'une maison ou d'un appartement (Y/N) : {filtered_df['FLAG_OWN_REALTY'].values[0]}")
        st.write(f"Propriétaire d'une voiture (Y/N) : {filtered_df['FLAG_OWN_CAR'].values[0]}")
        job = filtered_df['OCCUPATION_TYPE'].values[0]
        st.write(f"Emploi : {job}")
        years_employed = (filtered_df['DAYS_EMPLOYED'].values[0] / - 365).round(0)
        st.write(f"Nombre d'années d'activité : {years_employed}")
        revenu = float(filtered_df['AMT_INCOME_TOTAL'].values[0])
        st.write(f"Revenus annuels du client ('AMT_INCOME_TOTAL') : {revenu}")
        st.subheader(f"Informations sur le prêt : ")
        bien = float(filtered_df['AMT_GOODS_PRICE'].values[0])
        st.write(f"Prix du bien ('AMT_GOODS_PRICE') : {bien}")
        credit = float(filtered_df['AMT_CREDIT'].values[0])
        st.write(f"Montant du crédit ('AMT_CREDIT') : {credit}")
        annuity = float(filtered_df['AMT_ANNUITY'].values[0])
        st.write(f"Montant des annuités : {annuity}")

    updated_income = st.slider("Variations du revenus annuels", 0.0, 600000.0, revenu, 5000.0)
    updated_bien = st.slider("Variations des prix du bien ", 0.0, 4000000.0, bien, 10000.0)
    updated_credit = st.slider("Variations des montants du crédit ", 0.0, 4000000.0, credit, 10000.0)
    updated_annuity = st.slider("Variations des montants des annuités ('AMT_ANNUITY')", 0.0, 230000.0, annuity, 5000.0)


    predict_btn = st.button('Prédire')
    if predict_btn:
        unscale_filtered_df_predict = pd.DataFrame(std_scaler.inverse_transform(
            filtered_df_predict.drop(['SK_ID_CURR'], axis=1)),
            columns=filtered_df_predict.drop(['SK_ID_CURR'], axis=1).columns)

        unscale_filtered_df_predict['AMT_INCOME_TOTAL'] = revenu
        unscale_filtered_df_predict['AMT_GOODS_PRICE'] = bien
        unscale_filtered_df_predict['AMT_CREDIT'] = credit
        unscale_filtered_df_predict['AMT_ANNUITY'] = annuity

        unscale_filtered_df_predict['CREDIT_INCOME_PERCENT'] = np.divide(updated_credit, updated_income)
        unscale_filtered_df_predict['ANNUITY_INCOME_PERCENT'] = np.divide(updated_annuity, updated_income)
        unscale_filtered_df_predict['CREDIT_TERM'] = np.divide(updated_annuity, updated_credit)
        unscale_filtered_df_predict = unscale_filtered_df_predict.replace(np.inf, 0)
        filtered_df_predict = pd.DataFrame(std_scaler.transform(unscale_filtered_df_predict),
                                           columns=unscale_filtered_df_predict.columns)

        response_result = json.loads(request_prediction(f'{api_uri}/predict',
                                                        filtered_df_predict.to_dict(orient='index')[0]))

        st.subheader("Prédiction du modèle d'évaluation : ")
        st.write(response_result['prediction'])
        st.metric(label="Score de prédiction et delta par rapport au seuil optimal",
                  value=response_result['credit_score_risk']['predict_proba'],
                  delta=response_result['credit_score_risk']['predict_proba'] - response_result['credit_score_risk'][
                      'predict_th_proba'],
                  delta_color="inverse")

        graph = Image.open('/Users/audreyhohmann/Documents/Formation/OCR/P7/top50most.png')
        st.image(graph, caption ="Features contribuant le plus à l'élaboration du modèle")

    contrib_btn = st.button('Contribution des features au score client')
    if contrib_btn:
        response_result = json.loads(request_prediction(f'{api_uri}/contrib',
                                                        filtered_df_predict.drop(['SK_ID_CURR'], axis=1).to_dict(orient='index')[0]))
        shap_values = response_result['shap_values']
        df_shap_values = pd.DataFrame([shap_values],
                                      columns=filtered_df_predict.drop(['SK_ID_CURR'], axis=1).columns).T.reset_index(drop=False)
        df_shap_values.columns = ['feature', 'shap_value']
        df_shap_values = df_shap_values.sort_values('shap_value', ascending=False)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(data=df_shap_values.head(15), x='shap_value', y='feature', palette="viridis", ax=ax)
        ax.set_xlabel("Valeurs d'importance")
        ax.set_ylabel("")
        ax.set_title("Graphique des features importance local", fontsize=22)
        st.pyplot(fig)


else:
    st.write("Aucune donnée correspondante pour l'ID prêt sélectionné.")




dowload_excel()


@st.cache_data
def get_data_train():
    path_train = '/Users/audreyhohmann/Documents/Formation/OCR/P7/df_train_forstream.parquet'
    return pd.read_parquet(path_train)


df_train = get_data_train()

sns.set_theme(style="ticks", font='sans-serif', palette="Set2")


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
    ax.set_title('Répartition des ' + feature_y, fontsize=22)

    # Ajout de la légende
    ax.legend()

    # Redimensionner le texte pour une meilleure lisibilité
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Afficher le graphique dans Streamlit
    st.pyplot(fig)


def plot_relplot(df, x_feature, y_feature, hue_op=None):
    # Créer la figure et les sous-graphiques
    fig, ax = plt.subplots(figsize=(10, 8))

    if hue_op is None:
        sns.scatterplot(data=df, x=x_feature, y=y_feature, color='olive', ax=ax)
    elif hue_op == job:
        sns.scatterplot(data=df.loc[df['OCCUPATION_TYPE'] == job], x=x_feature, y=y_feature, hue='OCCUPATION_TYPE',
                        ax=ax)
    elif hue_op == genre:
        sns.scatterplot(data=df.loc[df['CODE_GENDER'] == genre], x=x_feature, y=y_feature, hue='CODE_GENDER', ax=ax)
    elif hue_op == family:
        sns.scatterplot(data=df.loc[df['NAME_FAMILY_STATUS'] == family], x=x_feature, y=y_feature,
                        hue='NAME_FAMILY_STATUS', ax=ax)
    else:
        sns.scatterplot(data=df, x=x_feature, y=y_feature, hue=hue_op, ax=ax)

    ax.set_title(f"Analyse entre les features {x_feature} et {y_feature}", x=0.5, y=1.05, fontsize=22)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    ax.legend()
    # plt.tight_layout()

    # ajout de la position du client
    if x_feature == 'AMT_INCOME_TOTAL' and y_feature == 'AMT_CREDIT':
        plt.scatter(revenu, credit, color='blue', marker='x', label='Position du client')
    elif x_feature == 'AMT_INCOME_TOTAL' and y_feature == 'AMT_GOODS_PRICE':
        plt.scatter(revenu, bien, color='blue', marker='x', label='Position du client')
    elif x_feature == 'AMT_INCOME_TOTAL' and y_feature == 'AMT_ANNUITY':
        plt.scatter(revenu, annuity, color='blue', marker='x', label='Position du client')

    # Afficher le graphique dans Streamlit
    st.pyplot(fig)


st.subheader("Distribution de la feature sélectionnée selon les classes du modèle d'entraînement")
# Widget selectbox pour choisir la feature y
selected_feature_y = st.selectbox('Choisir la feature', df_train[
    ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_ANNUITY', 'DAYS_BIRTH', 'DAYS_EMPLOYED']].columns)

# Vérifier si des données correspondent à la feature sélectionnée
if selected_feature_y in df_train.columns:
    # Appeler la fonction plot_kde avec la feature sélectionnée
    plot_kde(df_train, selected_feature_y)

else:
    st.write("La feature sélectionnée n'est pas présente dans le dataframe.")

st.subheader("Analyse bivariée entre les features quantitatives")

with st.spinner('Wait for it...'):
    selected_feature_x_relplot = st.selectbox('Choisir la feature 1', df_train[
        ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_ANNUITY']].columns, key="x_feature")
    selected_feature_y_relplot = st.selectbox('Choisir la feature 2',
                                              df_train[['AMT_CREDIT', 'AMT_GOODS_PRICE', 'AMT_ANNUITY']].columns,
                                              key="y_feature")
    selected_feature_hue = st.selectbox('Choisir la feature catégorielle (coloration des points)', ['None'] + df_train[
        ['CODE_GENDER', 'OCCUPATION_TYPE', 'NAME_FAMILY_STATUS']].columns.tolist() + [genre] + [job] + [family],
                                        index=0,
                                        key="hue_feature")

    if selected_feature_hue == 'None':
        hue_op = None
    else:
        hue_op = selected_feature_hue

    plot_relplot(df_train, selected_feature_x_relplot, selected_feature_y_relplot, hue_op)

