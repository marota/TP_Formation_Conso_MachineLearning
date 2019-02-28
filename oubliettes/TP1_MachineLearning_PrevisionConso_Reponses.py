# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext_format_version: '1.3'
#   jupytext_formats: py:light
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   language_info:
#     codemirror_mode:
#       name: ipython
#       version: 3
#     file_extension: .py
#     mimetype: text/x-python
#     name: python
#     nbconvert_exporter: python
#     pygments_lexer: ipython3
#     version: 3.5.4
# ---

# # Le problème
#
# Un tout nouveau* data challenge vient d'être lancé sur la plateforme dataScience.net pour aider RTE a faire de meilleures prévisions de conso ! 
# https://www.datascience.net/fr/challenge/33/details
#
# <img src="pictures/ChallengeConso.png" width=1000 height=60>
#
# Le premier challenge a montré que des prévisionnistes externes pouvaient faire mieux que nos meilleurs prévisionnistes à RTE... nous voilà ringardisés ! 
# Une deuxième chance nous est néanmoins offerte pour montrer que quand même on s'y connait en prévision de conso :-)
#
# Nous voilà tous réunis pour prendre ce défi à bras le corps et redorer l'image de RTE. On compte sur vous !

# ## Un outil: le Machine Learning
# Pour cela nous allons avoir recours au Machine Learning. Cela nous permettra de créer un modèle qui apprend et s'adapte au contexte sans programmer un système expert avec des "centaines" de règles en dur par de la programmation logique. Le Machine Learning nécessite toutefois de la connaissance experte dans le domaine d'intérêt pour créer des modèles pertinents et efficaces. En effet, si notre modèle embarque trop de variables peu explicatives, il sera noyé dans l'information, surapprendra sur les exemples qu'on lui a montrés, et aura du mal à généraliser en prédisant avec justesse sur de nouveaux exemples. 

# ## Une difficulté: le feature engineering
# Au-delà de la simple sélection de variables pertinentes, on fait surtout ce que l'on appelle du feature engineering avec notre expertise: on créé des variables transformées ou aggrégées, comme une consommation moyenne sur le mois précédent ou une température moyenne sur la France, pour guider l'algorithme à apprendre sur l'information la plus pertinente et synthétique.
# Cela suppose de bien connaître nos données, de les traiter et de les visualiser avec différents algorithmes au préalable.
#
# Nous allons ici voir ce que cela implique en terme de développement et d'implémentation de participer à un tel challenge, en montrant les capacités du Machine Learning sur la base de modèles "classiques", et aussi leurs limites.
#
# ## Ce que l'on va voir dans ce premier TP :
# 1) Formaliser le problème: que souhaite-t-on prédire (quel est mon Y) ? Avec quelles variables explicatives (quel est mon X) ?
#
# 2) Collecter les données: où se trouvent les données ? Quel est le format ? Comment les récupérer ?
#
# 3) Investiguer les données: visualiser des séries temporelles, faire quelques statistiques descriptives
#
# 4) Préparer les données: pour entrainer et tester un premier modèle
#
# 5) Créer et entrainer un premier modèle simple: ce sera notre baseline
#
# 6) Evaluer un modèle
#
# 7) Itérer en créant de nouveaux modèles avec de nouvelles variables explicatives
#
# 8) Jouez: créer vos propres modèles, tester sur une saison différente, tester sur une région différente, faire une prévision avec incertitudes, détecter des outliers
#
# ## To be continued
# Le deuxième TP permettra d'investiguer les modèles "Deep" avec réseaux de neurones, en montrant le moindre besoin en feature engineering et leur plus grande capacité a absorber l'information de par les représentations hiérarchiques qu'ils se créent.

# # Dimensionnement en temps
# On prévoit un une durée d'environ 2h pour ce TP1, debrief inclus :
# - 20-30 minutes pour charger et préparer les données
# - 30-40 minutes pour analyser et visualiser les données
# - 45-60 minutes pour créer, entrainer, évaluer et interpréter les modèles

# # Se familiariser avec le problème: Eco2mix
# Quand on parle de courbe de consommation France, il y a une application incontournable : eco2mix !
# Allons voir à quoi ressemblent ces courbes de consommation, pour nous faire une idée du problème et se donner quelques intuitions:
# http://www.rte-france.com/fr/eco2mix/eco2mix
# ou sur application mobile

# ## Chargement des Librairies

# +
# Exécutez la cellule ci-dessous (par exemple avec shift-entrée)
# Si vous exécuter ce notebook depuis votre PC, il faudra peut-etre installer certaines librairies avec 
# 'pip install ma_librairie'
import os  # accès aux commandes système
import datetime  # structure de données pour gérer des objets calendaires
import pandas as pd  # gérer des tables de données en python
import numpy as np  # librairie d'opérations mathématiques
import matplotlib.pyplot as plt  # tracer des visualisations
import sklearn  # librairie de machine learning
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt

from fbprophet import Prophet  # un package de series temporelles mis a disposition par facebook
import urllib3  # scrapper le web
import shutil  # move ou copier fichier
import zipfile  # compresser ou décompresser fichier

# Pour visualisation sur une carte utilsons la librairie bokeh qui fait appel a une api GoogleMaps
from bokeh.palettes import inferno
from bokeh.io import show, output_notebook
from bokeh.models import (
  GMapPlot, GMapOptions, ColumnDataSource, Circle, Range1d, PanTool, WheelZoomTool, BoxSelectTool, ColorBar, LogTicker,
    LabelSet, Label,HoverTool
)
from collections import OrderedDict
import seaborn as sns
from bokeh.models.mappers import LogColorMapper

# %matplotlib inline
# -

# # Configuration
# Choix du répertoire de travail "data_folder" dans lequel tous les fichiers csv seront entreposés

data_folder = os.path.join(os.getcwd(),"data")
# %autosave 0


# Petite vérification
print("Mon repertoire est : {}".format(data_folder))
print("Fichiers contenus dans ce répertoire :")
for file in os.listdir(data_folder):
    print(" - " + file)

# # Récupération des données
#
# Dans cette partie nous allons charger les fichiers csv nécessaires pour l'analyse, puis les convertir en data-frame python.
# Les données de base à récupérer sont :
# - Les historiques de consommation
# - Le calendrier des jours fériés
# - Les données météo, ainsi que la liste des stations
# - (Bonus) Le calendrier des jours TEMPO

# ## Récuperation des historiques de consommation

# Conversion en tant que data-frame
# Remarquez que l'on manipule un gros fichier, ce qui explique pourquoi l'exécution de cette cellule prend du temps
conso_csv = os.path.join(data_folder, "YconsoT0")
conso_df = pd.read_csv(conso_csv, sep=";", engine='c', header=0) #engine en language C et position header pour accélérer le chargement

# Regardons la tête des données
conso_df.shape

# Afficher les dimensions et le noms des colonnes de la data frame
print(conso_df.shape)
print(conso_df.columns)

# **Questions**
# - Quelles variables à prévoir ont été ici formatées ?
# - Que signifie la variable 'Consommation NAT t0' ?

# **Réponse**
#
# - La consommation des 13 régions françaises et de la consommation nationale. Tout cela sur un horizon de 32h au pas 30min.
# - Il s'agit de la consommation réalisée

# ## La composante temporelle pour une problématique de prévision
#
# <img src="pictures/clock.png" width=60 height=60>

# ### Conversion d'une colonne

# conso_Y.csv contient en particulier 2 colonnes 'date' et 'time', 
# les deux contenant des objets de type "string".
# Nous allons fusionner ces informations en une nouvelle colonne
# d'objets "ds" (dateStamp) mieux adaptés pour la manipulation de dates et d'heures
conso_df['ds'] = pd.to_datetime(conso_df['date'] + " " + conso_df['time'])

conso_df[['ds', 'date', 'time']].head(5)

# ### Crochet : petit tutoriel sur les méthodes datetime
#
# Pour manipuler des dates (effectuer des tris, des sélections, récupérer si c'est un lundi, mardi,...), il est plus efficace de passer par un objet "datetime" plutôt que de se débrouiller en manipulant des string. 
#
# La cellule ci-dessous a pour but d'illustrer comment utiliser ces objets.

# +
# datetime vers string
noel_2017_date = datetime.date(2017, 12, 25)
noel_2017_str = datetime.datetime.strftime(noel_2017_date, format="%Y-%m-%d")
print("noel_2017_date vaut : {} ; et est de type {}".format(noel_2017_date, str(type(noel_2017_date))))
print("noel_2017_str vaut : {} ; et est de type {}".format(noel_2017_str, str(type(noel_2017_str))))

# string vers datetime
starwars_day_2017_str = "2017-05-04"
starwars_day_2017_date = datetime.datetime.strptime(starwars_day_2017_str, "%Y-%m-%d")
print("starwars_day_2017_date vaut : {} ; et est de type {}".format(starwars_day_2017_date, str(type(starwars_day_2017_date))))
print("starwars_day_2017_str vaut : {} ; et est de type {}".format(starwars_day_2017_str, str(type(starwars_day_2017_str))))

# Voyager dans le temps
saint_sylvestre_2017_date = datetime.date(2017, 12, 31)
bienvenu_en_2018_date = saint_sylvestre_2017_date + datetime.timedelta(days=1)
print(bienvenu_en_2018_date)
# -

# ## Récuperation des jours fériés

jours_feries_csv = os.path.join(data_folder,"joursFeries.csv")
jours_feries_df = pd.read_csv(jours_feries_csv, sep=";")

# Pour la première colonne, les dates sont au format "string"
# Nous allons les convertir en objet "datetime" mieux adaptés pour la manipulation de dates
jours_feries_df.ds = pd.to_datetime(jours_feries_df.ds)

# Regardons la tête des données
jours_feries_df.head(8)

# ## Récupération des stations météo

stations_meteo_csv = os.path.join(data_folder, "StationsMeteoRTE.csv")

stations_meteo_df = pd.read_csv(stations_meteo_csv, sep=";")

stations_meteo_df.head(5)

# NB - Pour en savoir plus sur les poids :  
# https://clients.rte-france.com/lang/fr/visiteurs/services/actualites.jsp?id=9482&mode=detail

# ## Récupération du dataframe de météo
# <img src="pictures/weather.png" width=60 height=60>
# Comme d'habitude ! 
#
# Nos données sont dans 'meteoX_T.csv', qui est situé dans data_folder. En l'occurence on se limite aux températures comme information météo.
#
# Importez-les dans un dataframe _meteo&#95;df_
# et regardez à quoi elles ressemblent. Pensez aussi à changer les dates _string_ vers le format _datetime_

# ### Les données météo sont encryptées. 
# Pour les lire vous avez besoin d'un mot de passe qui ne peut vous être donné que dans le cadre d'un travail au sein de RTE

password=None

# +
# TODO: charger "meteoX_T.csv" dans "meteo_df"
meteo_zip = os.path.join(data_folder, "meteoX_T0_T24.zip")
zfMeteo = zipfile.ZipFile(meteo_zip)#.extractall(pwd=bytes(password,'utf-8'))
zfMeteo.setpassword(bytes(password,'utf-8'))
meteo_df = pd.read_csv(zfMeteo.open('meteoX_T0_T24'),sep=";",engine='c',header=0)
# END

# TODO: créer une colonne "ds" dans "meteo_df" comme précédemment
meteo_df['ds'] = pd.to_datetime(meteo_df['date'] + ' ' + meteo_df['time'])
# END

#transformer le type des colonnes temperatures en numerique si ce n'était pas le cas
tempCols = [col for col in meteo_df.columns if 'Th' in col]
meteo_df[tempCols] = meteo_df[tempCols].apply(pd.to_numeric)

# TODO: afficher les 5 premières lignes de "meteo_df"
meteo_df.head(5)
# END
# -

meteo_zip

meteo_df.shape

# **Questions**
# - Comment interprétez-vous les noms des colonnes ?

# ## Réduire notre problème
# Le dataframe de consommation est très lourde, et contient beaucoup d'information inutile (au moins en première approximation) pour notre problème de prévision de la consommation nationale. On va donc la simplifier.
#   
# On va se concentrer sur la consommation à l'**échelle nationale**, au **pas horaire**.

# on commence par ecarter les colonnes inutiles
consoFrance_df = conso_df[['ds', 'Consommation.NAT.t0']]

# et maintenant on ne garde que les heures pleines
minutes = consoFrance_df['ds'].dt.minute
indices_hours = np.where(minutes.values == 0.0)
consoFranceHoraire_df = consoFrance_df.loc[indices_hours]

# les index de ce sous-dataframe correspondent à celle du dataframe de base,
# et donc sont pour l'instant des multiples de 4.
# on va les réinitialiser pour avoir une dataframe "neuve"
consoFranceHoraire_df = consoFranceHoraire_df.reset_index(drop=True)  
consoFranceHoraire_df.head(5)

# On procéde de même pour le dataframe de météo pour ne garder que les heures pleines

minutes = meteo_df['ds'].dt.minute
mask = np.where(minutes.values == 0.0)
meteoHoraire_df = meteo_df.loc[mask]
meteoHoraire_df = meteoHoraire_df.reset_index(drop=True)
meteo_obs_df = meteoHoraire_df[list(meteoHoraire_df.columns[meteoHoraire_df.columns.str.endswith("Th+0")]) + ['ds']]

meteo_obs_df['ds'].head(5)

# ## Extraction des prévisions météo 24h à l'avance
# Information pleine de sens pour notre besoin.

meteo_prev_df = meteoHoraire_df[list(meteoHoraire_df.columns[meteoHoraire_df.columns.str.endswith("Th+24")]) + ['ds']]

# +
print(meteo_prev_df.shape)

meteo_prev_df.head(3)
# -

# ## Bonus : récupération de données depuis internet
#
# Dans le but d'automatiser un processus, nous pouvons implémenter une fonction qui ira chercher les dernières données mises à disposition sur internet.  
#
# Pour cet exemple nous allons considérer les jours Tempo, et (si le temps le permet en fin de TP) tester si cette information permet d'améliorer la qualité des prédictions.

# ### Manipulation à la main
#
#  - Recupérez à la main le calendrier TEMPO pour 2017-2018 :
#  http://www.rte-france.com/fr/eco2mix/eco2mix-telechargement
#  - Le déposer dans _data&#95;folder_
#  - Le dézipper
#  - Regarder les données dans excel ou autre. Notez en particulier la fin du fichier, la supprimer
#  
# Importez ces données dans un dataframe avec 'read_excel' de la librairie pandas ou autre méthode

tempo_xls = os.path.join(data_folder, "eCO2mix_RTE_tempo_2017-2018.xls")
tempo_df = pd.read_csv(tempo_xls, sep="\t", encoding="ISO-8859-1")  # ce fichier est en fait un csv et non un xls

print(tempo_df.head(5))

# ### La même chose automatisée
#
# On récupère automatiquement les informations sur Internet à partir de l'url, sans devoir les chercher à la main soi-même.

def get_tempo_data(url, data_folder, tempo_xls_zip_name):
    
    tempo_xls_zip = os.path.join(data_folder, tempo_xls_zip_name)
    
    # Récupération du fichier zip depuis internet
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    http = urllib3.PoolManager()    
    with http.request('GET', url, preload_content=False) as resp, open(tempo_xls_zip, 'wb') as out_file:
        shutil.copyfileobj(resp, out_file)
        
    with zipfile.ZipFile(tempo_xls_zip, "r") as zip_file:
        zip_file.extractall(data_folder)

    # Petite vérification
    if not os.path.isfile(tempo_xls_zip):
        print("ERROR!! {} not found in {}".format("eCO2mix_RTE_tempo_2017-2018.xls", data_folder))
        raise RuntimeError("Tempo data not uploaded :-(")

    # Import de ces données dans un dataframe
    tempo_df = pd.read_csv(tempo_xls_zip, sep="\t", encoding="ISO-8859-1")
    # Suppression du disclaimer de la dernière ligne de tempo_df, par exemple avec la méthode drop d'un dataframe
    last_row = len(tempo_df.index) - 1
    tempo_df = tempo_df.drop(tempo_df.index[last_row])

    return tempo_df

# Pour de sombres raisons de proxy ou de droit en écriture ( :-) ), la fonction "get_tempo_data" peut échouer dans le cadre de ce TP.

## Test de la fonction définie ci-dessus
""" 
url = "https://eco2mix.rte-france.com/curves/downloadCalendrierTempo?season=17-18"
tempo_xls_zip_name = "eCO2mix_RTE_tempo_2017-2018.zip"

tempo_df = get_tempo_data(url, data_folder, tempo_xls_zip_name)

print(tempo_df)
"""

# Pour les personnes intéressées par le webscrapping, jeter un oeil du côté de <a href="https://www.crummy.com/software/BeautifulSoup/bs4/doc/" title="link to google">BeautifulSoup</a>

# # Visualisation des données 
#
# La DataScience et le Machine Learning supposent de bien appréhender les données sur lesquelles nos modèles vont être entrainés. Pour se faire, il est utile de se faire quelques stats descriptives et des visualisations pour nos différentes variables.
#
# Traitant d'un problème de prévisions, on visualisera en particulier des séries temporelles.
#
# Vous allez voir:
# - échantillons de données
# - profils de courbe de consommation journaliers et saisonniers
# - visualisation de corrélation entre conso J et conso retardée
# - visualisations des stations météos
# - visualisations des séries temporelles des températures
# - calcul de corrélation sur la température entre les différentes stations météo

# ## Calcul de statistiques descriptives sur la consommation nationale
# A l'aide de la fonction _describe_.

consoFranceHoraire_df['Consommation.NAT.t0'].describe()

# ## Visualiser la consommation d'un jour particulier
# On souhaite visualiser la consommation réalisée pour un jour donné de l'historique.

def plot_load(var_load, year, month, day):
    date_cible = datetime.datetime(year=year, month=month, day=day)  # implicitement heure = minuit
    date_lendemain_cible = date_cible + datetime.timedelta(days=1)
    mask = (consoFranceHoraire_df.ds >= date_cible) & (consoFranceHoraire_df.ds <= date_lendemain_cible)   
    consoJour = consoFranceHoraire_df[mask]
    plt.plot(consoJour['ds'], consoJour[var_load], color='blue')
    plt.show()

plot_load('Consommation.NAT.t0', 2016, 12, 20)

conso_df.shape

# ## Afficher une semaine arbitraire de consommation
# On pourra modifier la fonction précédente en rajoutant le timedelta en paramètre.

def plot_load_timedelta(var_load, year, month, day, delta_days):
    date_cible = datetime.datetime(year=year, month=month, day=day)
    date_lendemain_cible = date_cible + datetime.timedelta(days=delta_days)

    conso_periode = consoFranceHoraire_df[(consoFranceHoraire_df.ds >= date_cible) 
                                      & (consoFranceHoraire_df.ds <= date_lendemain_cible)]
    plt.plot(conso_periode['ds'], conso_periode[var_load], color='blue')
    plt.show()

plot_load_timedelta('Consommation.NAT.t0', 2016, 12, 20, delta_days=7)

# ## Observation des profils de la consommation pour les mois d'hiver et les mois d'été
# Toujours dans le but d'appréhender nos données, on va regarder les profils moyens pour le smois d'été et pour ceux d'hiver. On va également observer le min et le max pour avoir une idée de la variabilité du signal.

# +
# Par commodité, on isole le mois pour après attraper les mois d'hiver et d'été
consoFranceHoraire_df['month'] = consoFranceHoraire_df['ds'].dt.month

# On isole aussi les heures
consoFranceHoraire_df['hour'] = consoFranceHoraire_df['ds'].dt.hour

# On sépare les jours de la semaine en week-end / pas week-end
# De base, la fonction datetime.weekday() renvoie 0 => Lundi, 2 => Mardi, ..., 5 => Samedi, 6 => Dimanche
# Ci-dessous, si on a un jour d ela semaine alors dans la colonne weekday on mettra 1, et 0 si c'est le week-end
consoFranceHoraire_df['weekday'] = (consoFranceHoraire_df['ds'].dt.weekday < 5).astype(int)  # conversion bool => int
# -

consoFranceHoraire_df.head(5)

# +
# On aggrège les mois d'hiver ensemble
groupedHiver = consoFranceHoraire_df[(consoFranceHoraire_df.month == 12) | 
                                     (consoFranceHoraire_df.month == 1) | 
                                     (consoFranceHoraire_df.month == 2)].groupby(['weekday', 'hour'], as_index=True)

# Idem pour les mois d'été
groupedEte = consoFranceHoraire_df[(consoFranceHoraire_df.month == 6) | 
                                   (consoFranceHoraire_df.month == 7) | 
                                   (consoFranceHoraire_df.month == 8)].groupby(['weekday', 'hour'], as_index=True)

statsHiver = groupedHiver['Consommation.NAT.t0'].aggregate([np.mean, np.min, np.max])
statsEte = groupedEte['Consommation.NAT.t0'].aggregate([np.mean, np.min, np.max])
# -

print(statsHiver)

# +
# On affiche des infos sur le profil pour les jours de la semaine
semaine = statsHiver.loc[1]  # 0 pour les jours de semaine
weekend = statsHiver.loc[0]  # 0 pour weekend

plt.plot(semaine['amin'], color='cyan')
plt.plot(semaine['mean'], color='blue')
plt.plot(semaine['amax'], color='cyan')
plt.show()
# -

# ## Lien avec la consommation passée
# A l'aide de la fonction shift, pour un point horaire cible on regarde  :
# - la consommation de l'heure précédente, 
# - du jour précédent, 
# - de la semaine précédente.
#
# On regarde ensuite si la consommation réalisé peut se deviner à partir de ces observations.

consoFranceHoraire_df['lag1H'] = consoFranceHoraire_df['Consommation.NAT.t0'].shift(1)
consoFranceHoraire_df['lag1D'] = consoFranceHoraire_df['Consommation.NAT.t0'].shift(24)
consoFranceHoraire_df['lag1W'] = consoFranceHoraire_df['Consommation.NAT.t0'].shift(24*7)

consoFranceHoraire_df.head(24 * 7 + 1)

# On regarde maintenant graphiquement si on a une belle corrélation ou non :

def plot_scatter_load(var_x):
    plt.scatter(consoFranceHoraire_df[var_x],consoFranceHoraire_df['Consommation.NAT.t0'])
    plt.title(var_x)
    plt.show()

plot_scatter_load('lag1H')
plot_scatter_load('lag1D')
plot_scatter_load('lag1W')

# ### Question
# Que pensez-vous de ces corrélations ?

# ## Visualisation des stations météo

# Regardons si les températures des stations météo sont corrélées entre elles :

# Petit rappel
print(meteo_obs_df.shape)
print(meteo_obs_df.columns)
meteo_obs_df.head(3)

# +
#matrix_correlation = meteo_obs_df.corr() #calcul d'une corrélation globale

#calcul de la corrélation en fonction de la saison

meteo_obs_df['saison'] = ((meteo_obs_df['ds'].dt.month ==1) |(meteo_obs_df['ds'].dt.month==2)|(meteo_obs_df['ds'].dt.month==12)).astype(int)*1+((meteo_obs_df['ds'].dt.month ==3 )|(meteo_obs_df['ds'].dt.month==4)|(meteo_obs_df['ds'].dt.month==5)).astype(int)*2+((meteo_obs_df['ds'].dt.month ==6 )|(meteo_obs_df['ds'].dt.month==7)|(meteo_obs_df['ds'].dt.month==8)).astype(int)*3+((meteo_obs_df['ds'].dt.month ==9) |(meteo_obs_df['ds'].dt.month==10)|(meteo_obs_df['ds'].dt.month==11)).astype(int)*4  # conversion bool => int
matrix_correlation = meteo_obs_df.groupby(['saison']).corr() 
matrix_correlation
# -

#heatMap pour un meilleur visuel
#.loc[1]=hiver
#.loc[2]=printemps
#.loc[3]=été
#.loc[4]=automne
plt.imshow(matrix_correlation.loc[4].as_matrix(),cmap='PuBu_r', interpolation='nearest')
plt.colorbar()
plt.show()

# ### Question
# - Que pensez-vous de ces corrélations ?

# ## Fusionner le jeu de données météo avec les données de consommation
#
# A l'aide de la fonction pd.merge

# ### On commence avec le réalisé

print(consoFranceHoraire_df.shape)
print(consoFranceHoraire_df.columns)
print()
print(meteo_obs_df.shape)
print(meteo_obs_df.columns)

conso_meteo = pd.merge(consoFranceHoraire_df, meteo_obs_df, on='ds')
print(conso_meteo.shape)
print(conso_meteo.columns)

# ### On passe au prévisionnel

conso_meteo_prev = pd.merge(consoFranceHoraire_df, meteo_prev_df, on='ds')

print(conso_meteo_prev.shape)
print(conso_meteo_prev.columns)

# ## Visualiser la consommation en fonction de la température de la station Paris-Montsouris
# On voudrait savoir si la consommation nationale peut s'expliquer en regardant simplement la température de la station du Parc Montsouris et en ignorant ce qui est extérieur au périphérique (Paris étant le centre du monde).
#
# Regardons le graphe de corrélation :

# Paris Montsouris est la station météo n°156
# On s'intéresse à la température Th
# On regarde le réalisé, soit +0
plt.scatter(conso_meteo['X156Th+0'], conso_meteo['Consommation.NAT.t0'], alpha=0.2)
plt.show()

# ### Question
# - Que pensez-vous de ces corrélations ? Est-ce suffisant ?

# ## Visualisation des stations météo
# Pour visualiser la localisation des stations météos qui nous sont fournies, positionnons ces stations avec leur localisation GPS sur une Google Map

nstations = stations_meteo_df.shape[0]
print(nstations)

# +
map_options = GMapOptions(lat=47.08, lng=2.39, map_type="roadmap", zoom=6)

plot = GMapPlot(x_range=Range1d(), y_range=Range1d(), map_options=map_options)
plot.title.text = "France"

# For GMaps to function, Google requires you obtain and enable an API key:
#
#     https://developers.google.com/maps/documentation/javascript/get-api-key
#
# Replace the value below with your personal API key:
plot.api_key = "AIzaSyC05Bs_e0q6KWyVHlmy0ymHMKMknyMbCm0"
tempInstant1 = meteo_obs_df.iloc[0, np.arange(0, 35)]

# nos données d'intérêt pour créer notre visualisation
data=dict(lat=stations_meteo_df['latitude'],
          lon=stations_meteo_df['longitude'],
          label=stations_meteo_df['Nom'],
          temp=tempInstant1)

source = ColumnDataSource(data)

# l'échelle de couleur pour la température
Tlow=0
Thigh=20
color_mapper = LogColorMapper(palette="Viridis256", low=Tlow, high=Thigh)

# la couleur de remplissage des cercles est fonction de la valeur de la temérature
circle = Circle(x="lon", y="lat", size=15, fill_color={'field': 'temp', 'transform': color_mapper},
                fill_alpha=0.8, line_color=None,)

# les labels que l'on souhaite afficher en passant un curseur sur une station
labels = LabelSet(x='lon', y='lat', text='label', level='glyph', x_offset=5, y_offset=5,
                  source=source, render_mode='canvas')

# on ajoute la layer
plot.add_glyph(source, circle)

# le tooltip quand on pose le curseur dessus
hover = HoverTool(tooltips= OrderedDict([
    ("index", "$index"),
    ("(xx,yy)", "(@lon, @lat)"),
    ("label", "@label"),
    ("T", "@temp")
]))

# on plot
plot.add_tools(PanTool(), WheelZoomTool(), BoxSelectTool(), hover)


color_bar = ColorBar(color_mapper=color_mapper, ticker=LogTicker(),
                 label_standoff=12, border_line_color=None, location=(0,0))
plot.add_layout(color_bar, 'right')

output_notebook()#"gmap_plot.html"
show(plot)
# -

# ## Bricolage d'un modèle prédictif naïf
#
# <img src="pictures/hommeNaif.png" width=500 height=60>

# +
datetime_a_predire = datetime.datetime.strptime("2016-12-20_14:00", "%Y-%m-%d_%H:%M")

y_true = float(consoFranceHoraire_df.loc[consoFranceHoraire_df['ds'] == datetime_a_predire]['Consommation.NAT.t0'])

print("On veut predire la consommation du {}, soit {}".format(datetime_a_predire, y_true))
# -

# ## Première idée, un modèle naïf : pour l'heure qui nous intéresse, on plaque bêtement la valeur de consommation nationale de la veille

# +
datetime_la_veille = datetime_a_predire - datetime.timedelta(days=1)
y_pred = float(consoFranceHoraire_df.loc[consoFranceHoraire_df['ds'] == datetime_la_veille]['Consommation.NAT.t0'])
pred_error = abs(y_true - y_pred)

print("Modele 1 -- pred: {}, realisee: {}, erreur: {}%".format(y_pred, y_true, pred_error/y_true * 100))
# -

# ## Deuxième idée avec de l'expertise : pareil, avec comme raffinement le fait que l'on considere maintenant l'influence de la temperature
#
# <img src="pictures/ExpertJamy.jpg" width=500 height=60>

delta_MW_par_degre = 2300  # par expertise, 
                           # on considere qu'une augmentation moyenne de 1°C 
                           # conduit à une augmentation de 2300MW de la conso nationale

# +
# Pour faire simple, on va prétendre que ce qui se passe à Paris 
# est représentatif de ce qu'il se passe en France comme delta de température
# Paris-Montsouris est la station numéro 156

temperature_Montsouris_veille = float(conso_meteo.loc[conso_meteo['ds'] == datetime_la_veille]['X156Th+0'])
temperature_Montsouris_cible = float(conso_meteo.loc[conso_meteo['ds'] == datetime_a_predire]['X156Th+0'])
delta_temp = temperature_Montsouris_cible - temperature_Montsouris_veille

delta_MW_because_temp = delta_temp * delta_MW_par_degre

# +
y_pred = float(consoFranceHoraire_df.loc[consoFranceHoraire_df['ds'] == datetime_la_veille]['Consommation.NAT.t0']) + delta_MW_because_temp
pred_error = abs(y_true - y_pred)

print("Modele 2 -- pred: {}, realisee: {}, erreur: {}%".format(y_pred, y_true, pred_error/y_true * 100))
# -

# Bon... Bien essayé mais maintenant on va être plus sérieux !

# # Préparer un jeu d'entrainement et un jeu de test
# En machine learning, il y a 2 types d'erreur que l'on peut calculer : l'erreur d'entrainement et l'erreur de test. 
#
# Pour évaluer la capacité de notre modèle à bien généraliser sur de nouvelles données, il est très important de se préserver un jeu de test indépendant de celui d'entrainement.
#
# Il faut donc segmenter notre dataset en 2 : 
# - un premier jeu servira pour l'entrainement, 
# - tandis que le second servira à mesurer les performances du modèle prédictif.

def prepareDataSetEntrainementTest(Xinput, Yconso, dateDebut, dateRupture, nbJourlagRegresseur=0):
    
    dateStart = Xinput.iloc[0]['ds']
    
    DateStartWithLag = dateStart + pd.Timedelta(str(nbJourlagRegresseur)+' days')  #si un a un regresseur avec du lag, il faut prendre en compte ce lag et commencer l'entrainement a la date de debut des donnees+ce lag
    XinputTest = Xinput[(Xinput.ds >= dateRupture)]    

    XinputTrain=Xinput[(Xinput.ds < dateRupture) & (Xinput.ds > DateStartWithLag) & (Xinput.ds > dateDebut)]
    YconsoTrain=Yconso[(Yconso.ds < dateRupture) & (Yconso.ds > DateStartWithLag) & (Yconso.ds > dateDebut)]
    YconsoTest=Yconso[(Xinput.ds >= dateRupture)]
    
    return XinputTrain, XinputTest, YconsoTrain, YconsoTest

# +
Yconso = consoFranceHoraire_df[['ds','Consommation.NAT.t0']]
# Pour la librairie Prophet que l'on va bientôt utiliser,
# il nous faut renommer "Consommation NAT t0" en "y"
Yconso.columns = ['ds','y'] 

# on souhaite un jeu de test qui commence à partir du 1er mai 2017
dateDebut = datetime.datetime(year=2013, month=1, day=7)#pour éviter les NaN dans le jeu de données
dateRupture = datetime.datetime(year=2017, month=5, day=1)#début du challenge prevision de conso

# On va commencer par un modèle autoregressif très simple, ici X=Y
# Pas de prise en compte de la météo, des variables calendaires, etc...
Xinput = Yconso 
nbJourlagRegresseur = 0  # pas de prise en compte des consommations passées pour l'instant
# -

XinputTrain, XinputTest, YconsoTrain, YconsoTest = prepareDataSetEntrainementTest(Xinput, Yconso, 
                                                                                  dateDebut, dateRupture, 
                                                                                  nbJourlagRegresseur)




print('la taille de l échantillon XinputTrain est:' + str(XinputTrain.shape))
print('la taille de l échantillon XinputTest est:' + str(XinputTest.shape))
print('la taille de l échantillon YconsoTrain est:' + str(YconsoTrain.shape))
print('la taille de l échantillon YconsoTest est:' + str(YconsoTest.shape))
print("la proportion de data d'entrainement est de:" + str(YconsoTrain.shape[0] / (YconsoTrain.shape[0] + YconsoTest.shape[0])))

# # Fonctions utilitaires

# Créons la fonction modelError qui va calculer pour un échantillon (Y, Y_hat) différents scores :
# - erreur relative moyenne (MAPE en %)
# - erreur relative max (en %)
# - rmse (en MW)
#

def modelError(Y, Yhat):

    Y = Y.reset_index(drop=True)
    
    relativeErrorsTest = np.abs((Y['y'] - Yhat) /Y['y']) 
    errorMean = np.mean(relativeErrorsTest)
    errorMax = np.max(relativeErrorsTest)
    rmse = np.sqrt(mean_squared_error(Y['y'], Yhat))
   
    return relativeErrorsTest, errorMean, errorMax, rmse

def evaluation(YTrain, YTest, YTrainHat, YTestHat):
    # Ytrain et Ytest ont deux colonnes : ds et y
    # YtrainHat et YTestHat sont des vecteurs
    ErreursTest, ErreurMoyenneTest, ErreurMaxTest, RMSETest = modelError(YTest, YTestHat)
    print("l'erreur relative moyenne de test est de:" + str(round(ErreurMoyenneTest*100,1))+"%")
    print("l'erreur relative max de test est de:" + str(round(ErreurMaxTest*100,1)) +"%")
    print('le rmse de test est de:' + str(round(RMSETest,0)))
    print()
    ErreursTest, ErreurMoyenneTest, ErreurMaxTest, RMSETest = modelError(YTrain, YTrainHat)
    print("l'erreur relative moyenne de train est de:" + str(round(ErreurMoyenneTest*100,1))+"%")
    print("l'erreur relative max de train est de:" + str(round(ErreurMaxTest*100,1)) +"%")
    print('le rmse de test est de:' + str(round(RMSETest,0))) 


def evaluation_par(X, Y, Yhat,avecJF=True):
    Y['weekday'] = Y['ds'].dt.weekday
    Y['hour'] = Y['ds'].dt.hour
    if(avecJF):
        Y['JoursFeries'] = X['JoursFeries']
    Y['APE'] = np.abs(Y['y']-Yhat)/Y['y']
    dataWD = Y[['weekday','APE']]
    groupedWD = dataWD.groupby(['weekday'], as_index=True)
    statsWD = groupedWD.aggregate([np.mean])
    dataHour = Y[['hour','APE']]
    groupedHour = dataHour.groupby(['hour'], as_index=True)
    statsHour = groupedHour.aggregate([np.mean])
    
    if(avecJF):
        dataJF=Y[['JoursFeries','APE']]
        groupedJF = dataJF.groupby(['JoursFeries'], as_index=True)
        statsJF = groupedJF.aggregate([np.mean])
    else:
        statsJF=None
    
    return statsWD,statsHour,statsJF

Xinput.head()

# # Créer un modèle avec Prophet
# Vous allez utiliser la librairie Prophet developpée par facebook.: https://research.fb.com/prophet-forecasting-at-scale/. Elle a été publiée en 2017 et permet de faire des modèles de prévision sur des séries temporelles. En particulier, ces modèles captent surtout des saisonnalités, et peuvent également tenir compte de jours particuliers comme les jours fériés. Il est possible de rajouter d'autre variables explicatives selon un modèle statistique linéaire.
#
# C'est une librairie relativement ergonomique et performante en terme de temps de calculs d'où son choix ici.
# Un des aspects intéressant également est qu'elle repose sur un language probabiliste PyStan. Il est ainsi possible de décrire des variables selon une loi dans notre modèle et d'obtenir sans plus de développement des intervalles de confiance et incertitudes.
#
# Pour un tutoriel bien fait pour comprendre et utiliser Prophet, je vous recommande le lien suivant: http://www.degeneratestate.org/posts/2017/Jul/24/making-a-prophet/
#
# La prophétie autoréalisatrice: Marc Zuckerberg futur Président ..??
#
# <img src="pictures/zuckerbergProphet.jpg" width=500 height=30>

# creer un modèle prophet avec une saisonnalité journalière et une tendance nulle pour la consommation
mTrain = Prophet(daily_seasonality=True, n_changepoints=0)  # on considere une tendance relativement constante pour la consommation sur les 4 ans

# ## Entrainer un modèle
# Notre modèle a des paramètres tels que les saisonnalités qu'il va falloir maintenant apprendre au vu de notre jeu d'entrainement. Il faut donc caler notre modèle sur ce jeu d'entrainement.

mTrain.fit(XinputTrain)

# ## Faire des prédictions
# Une fois qu'un modèle de prévision est entrainé, il ne s'avère utile que s'il est performant sur de nouvelles situations. Faisons une prévision sur notre jeu de test.

forecastTest = mTrain.predict(XinputTest)
forecastTrain = mTrain.predict(XinputTrain)

# +
# on visualise nos previsions avec incertitudes
dateavantRupture = dateRupture - pd.Timedelta('30 days')  # pour visualiser aussi les réalisations d'avant

print('on plot a partir de la date:' + str(dateavantRupture))
mTrain.history = mTrain.history[mTrain.history.ds >= dateavantRupture]  # pour demander à Prophet de ne plotter que notre période d'interet
mTrain.plot(forecastTest)

plt.show()
# -

# ## Visualiser le modèle
# Prophet dispose de méthodes de visualisation qui permettent d'interpreter le modèle appris, en particulier d'un point de vue des saisonalités.

# +
# on visualise notre modele avec ses saisonalites
mTrain.plot_components(forecastTest)

plt.show()
# -

# ## Interpreter le modèle 
# Au vu des visualisations précédentes :
# - quelles interprétations pouvez-vous faire du modèle?
# - Comment varie le comportement de la courbe de consommation?

# +
#avecJF=False#on n a pas encore considere de jours feries
#evalWD,evalHour,evalJF = evaluation_par(XinputTest,YconsoTest,forecastTest['yhat'],avecJF)
#print(str(round(evalWD*100,1)))
#print(str(round(evalHour*100,1)))
#print(str(round(evalJF*100,1)))
# -

# ## Evaluer l'erreur de prévision
# Au vu de ces previsions faites par notre modèle sur de nouvelles situations, quelle est la performance de notre modèle sur ce jeu de test ?

evaluation(YconsoTrain, YconsoTest, forecastTrain['yhat'], forecastTest['yhat'])

# on visualise nos previsions par rapport a la realité
plt.plot(YconsoTest['ds'], YconsoTest['y'], 'b')
plt.plot(forecastTest['ds'], forecastTest['yhat'], 'r')
plt.show()

# ## Enquêter autour des erreurs de prévision

# ### Evaluation en fonction du jour de semaine, de l'heure, si jour férié ou non

# ### Comment se distribue l'erreur ?

erreur_relative_test, erreur_moyenne_test, erreur_max_test, rmse = modelError(YconsoTest, forecastTest['yhat'])

num_bins = 100
plt.hist(erreur_relative_test, num_bins)
plt.show()

# ### A quel moment se trompe-t-on le plus ?

plt.plot(forecastTest['ds'], erreur_relative_test, 'r')
plt.title("erreur relative sur la periode de test")
plt.show()

# +
threshold = 0.18

mask = (erreur_relative_test >= threshold)
forecastTest['ds'].loc[mask]
# -

# ## Feature engineering
# Quelles variables explicatives peuvent nous permettre de créer un modele plus perfomant ?

# # On quitte Prophet pour d'autres modèles : RandomForest et XGBoost

# ## Preparation du X et du y

# ### On sort le y

Yconso = conso_meteo[['ds','Consommation.NAT.t0']]
Yconso.columns = ['ds', 'y']

Yconso.head(3)

# ### On s'occupe du X

Xinput = conso_meteo_prev.drop(['lag1H'],axis=1)  # variables calendaires, conso retardée, température prévue
Xinput = Xinput.drop(['Consommation.NAT.t0'],axis=1)

print(Xinput.shape)
print(Xinput.columns)

# On encode les données calendaires en one-hot encoding pour le modèle.
# Cet encodage est nécessaire pour que le modèle mathématique puisse appréhender la notion de date.

encodedWeekDay = pd.get_dummies(Xinput['weekday'],prefix="weekday")
encodedMonth = pd.get_dummies(Xinput['month'],prefix="month")
encodedHour = pd.get_dummies(Xinput['hour'],prefix="hour")

encodedWeekDay.head(3)

encodedMonth.head(3)

encodedHour.head(3)

Xinput = pd.concat([Xinput, encodedMonth, encodedWeekDay, encodedHour], axis=1)
Xinput = Xinput.drop(['month','weekday','hour'],axis=1)

print(Xinput.shape)
print(Xinput.columns)

colsToKeepWeather=[s for s in Xinput.columns.get_values() if 'Th+24' in s]
Xinput[colsToKeepWeather] = Xinput[colsToKeepWeather].shift(24)

# Récupération des prévisions météo à J+1 pour la veille
lag_colsToKeepWeather=[s+"_J_1" for s in colsToKeepWeather ]
Xinput[lag_colsToKeepWeather]=Xinput[colsToKeepWeather].shift(24)
time=pd.to_datetime(Xinput['ds'], yearfirst=True)
Xinput['posan']= time.dt.dayofyear
Xinput['FranceTh+24'] = np.dot(Xinput[colsToKeepWeather],stations_meteo_df['Poids'])
Xinput['SFranceTh+24'] = Xinput['FranceTh+24'].ewm(alpha=0.9).mean()

#Récupération des jours fériés dans Xinput
Xinput = pd.merge(Xinput,jours_feries_df,how="left", on="ds")
encodedHolidays = pd.get_dummies(Xinput[['holiday']], prefix="JF")
encodedHolidays['JoursFeries'] = encodedHolidays.sum(axis=1)
Xinput = pd.concat([Xinput, encodedHolidays], axis=1)
Xinput = Xinput.drop(['holiday'], axis=1)

#affichage de toutes les variables de base
list(Xinput) #list plutôt que print pour avoir la liste complète

XinputTrain, XinputTest, YconsoTrain, YconsoTest = prepareDataSetEntrainementTest(Xinput, 
                                                                                  Yconso, 
                                                                                  dateDebut, 
                                                                                  dateRupture, 
                                                                                  nbJourlagRegresseur)

print('shape de XinputTrain est:' + str(XinputTrain.shape[0]))
print('shape de XinputTest est:' + str(XinputTest.shape[0]))
print('shape de YconsoTrain est:' + str(YconsoTrain.shape[0]))
print('shape de YconsoTest est:' + str(YconsoTest.shape[0]))
print('la proportion de data d entrainement est de:' + str(YconsoTrain.shape[0] / (YconsoTrain.shape[0] + YconsoTest.shape[0])))

# ## Modèle RandomForest
#
# <img src="pictures/randomForestExplain.png" width=500 height=30>

from sklearn.ensemble import RandomForestRegressor

# ### Préparation des données d'entrée

# +
colsToKeepWeather = [s for s in Xinput.columns.get_values() if 'Th+24' in s]
colsToKeepMonth = [v for v in Xinput.columns.get_values() if 'month' in v]
colsToKeepWeekday = [v for v in Xinput.columns.get_values() if 'weekday' in v]
colsToKeepHour = [v for v in Xinput.columns.get_values() if 'hour' in v]
colsToKeepHolidays = [v for v in Xinput.columns.get_values() if 'JF_' in v]

colsRF = np.concatenate((['lag1D','lag1W','JoursFeries'],
                         colsToKeepWeather,colsToKeepMonth,colsToKeepWeekday,colsToKeepHour))
# -

# ### Entrainement du modèle

# La cellule peut prendre un peu de temps à exécuter
rfTrain = RandomForestRegressor(n_estimators=30, max_features=colsRF.size, n_jobs=3)
rfTrain.fit(XinputTrain[colsRF], YconsoTrain['y'])

# ### Prediction

forecastTest = rfTrain.predict(XinputTest[colsRF])
forecastTrain = rfTrain.predict(XinputTrain[colsRF])

# ### Evaluation

# +
evaluation(YconsoTrain, YconsoTest, forecastTrain, forecastTest)

# on visualise nos previsions par rapport a la realité
plt.plot(YconsoTest['ds'], YconsoTest['y'], 'b', YconsoTest['ds'], forecastTest, 'r')
plt.show()
# -

evalWD,evalHour,evalJF = evaluation_par(XinputTest,YconsoTest,forecastTest)
print(str(round(evalWD*100,1)))
print(str(round(evalHour*100,1)))
print(str(round(evalJF*100,1)))

# ## Modèle xgboost
#
# <img src="pictures/XGboost.png" width=500 height=30>

import xgboost as xgb

xgbTrain = xgb.XGBRegressor( )
xgbTrain.fit(XinputTrain[colsRF], YconsoTrain['y'])
forecastTestXGB = xgbTrain.predict(XinputTest[colsRF])
forecastTrainXGB = xgbTrain.predict(XinputTrain[colsRF])

evaluation(YconsoTrain, YconsoTest, forecastTrainXGB, forecastTestXGB)

evalWD,evalHour,evalJF = evaluation_par(XinputTest,YconsoTest,forecastTestXGB)
print(str(round(evalWD*100,1)))
print(str(round(evalHour*100,1)))
print(str(round(evalJF*100,1)))

# ## Et nos modèles naifs au fait?

forecastTestNaif1=np.array(XinputTest['lag1D'])
forecastTrainNaif1=np.array(XinputTrain['lag1D'])
evaluation(YconsoTrain, YconsoTest, forecastTrainNaif1, forecastTestNaif1)

# +
temperature_Montsouris_veille = XinputTrain['X156Th+24_J_1']
temperature_Montsouris_cible = XinputTrain['X156Th+24']
delta_temp = temperature_Montsouris_cible - temperature_Montsouris_veille

delta_MW_because_temp_train = delta_temp * delta_MW_par_degre * ((temperature_Montsouris_cible<15)*1)

temperature_Montsouris_veille = XinputTest['X156Th+24_J_1']
temperature_Montsouris_cible = XinputTest['X156Th+24']
delta_temp = temperature_Montsouris_cible - temperature_Montsouris_veille

delta_MW_because_temp_test = delta_temp * delta_MW_par_degre * ((temperature_Montsouris_cible<15)*1)

forecastTestNaif2=np.array(XinputTest['lag1D'] + delta_MW_because_temp_test)
forecastTrainNaif2=np.array(XinputTrain['lag1D']+ delta_MW_because_temp_train)
evaluation(YconsoTrain, YconsoTest, forecastTrainNaif2, forecastTestNaif2)
# -

# ### Question
# - Selon vous, pourquoi l'erreur max est significative pour tous les modèles ?
# - Comment y remédier ?

# # Bonus: à vous de jouer
#
# Bravo ! Vous avez déjà créé un premier modèle performant pour faire des prévisions sur une fenêtre glissante à horizon 24h !
#
# Maintenant à vous de mettre votre expertise pour créer de nouveaux modèles.
#
# Vous pouvez continuer à explorer le problème selon plusieurs axes:
# - créer des modèles pour les régions françaises
# - tester votre modèle sur une autre saison (l'hiver par exemple)
# - créer de nouvelles variables explicatives ? Quid de la météo et de la température? Des jours fériés ? Du feature engineering plus complexe...
# - détecter des outliers dans les données
# - etudiez les incertitudes et les possibilités offertes par PyStan
#
# Mettez-vous en 3 groupes, explorez pendant 30 minutes, et restituez.




