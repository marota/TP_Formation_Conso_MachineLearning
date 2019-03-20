# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 1.0.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # TP facultatif : Préparation du jeu de données brut

# Quand on se lance avec enthousiasme dans un nouveau projet de machine-learning, on pense avant tout au choix du modèle que l'on va utiliser. Cependant, et même s'il ne s'agit pas de l'étape la plus agréable du travail, un préalable indispensable est de réunir les données brutes et de les mettre en forme pour qu'elle puisse être ingérée par le modèle.
#
# L'objectif de ce TP est de comprendre comment, à partir de différentes sources de données, on construit nos jeux de données "propres" pour ensuite construire le meilleur modèle d'apprentissage possible pour la prévision de consommation nationale.
#
# Nos fichiers d'entrée bruts sont les suivants :
# > * YconsoT0.csv
# > * joursFeries.csv
# > * StationsMeteoRTE.csv  # Les coordonnées géographiques et les poids associés aux stations météos
# > * meteoX_T0_T24.zip
# > * eCO2mix_RTE_tempo_2017-2018.xls  # le sinformations sur les jours TEMPO
#
# Et les fichiers que l'on va créer sont :
# > * Xinput.csv  # Les entrées pour le modèle d'apprentissage
# > * Yconso.csv  # les sorties pour le modèle d'apprentissage

# ## Environnement
#
# Chargement des librairies python, et quelques éléments de configuration.

# +
# Exécutez la cellule ci-dessous (par exemple avec shift-entrée)
# Si vous exécuter ce notebook depuis votre PC, il faudra peut-etre installer certaines librairies avec 
# 'pip3 install ma_librairie'
import os  # accès aux commandes système
import datetime  # structure de données pour gérer des objets calendaires
import pandas as pd  # gérer des tables de données en python
import numpy as np  # librairie d'opérations mathématiques
import zipfile # manipulation de fichiers zip
import urllib3 # téléchargement de fichier

from bokeh.palettes import inferno
from bokeh.io import show, output_notebook
from bokeh.models import (
  GMapPlot, GMapOptions, ColumnDataSource, Circle, Range1d, PanTool, WheelZoomTool, BoxSelectTool, ColorBar, LogTicker,
    LabelSet, Label,HoverTool
)
from bokeh.models.mappers import LogColorMapper
from collections import OrderedDict

# %autosave 0

data_folder = os.path.join(os.getcwd(),"data")

print("")
print("Mon repertoire de data est : {}".format(data_folder))
print("")
print("Fichiers contenus dans ce répertoire :")
for file in os.listdir(data_folder):
    print(" - " + file)
# -

# ## Récupération des données
#
# Dans cette partie nous allons charger les fichiers csv nécessaires pour l'analyse, puis les convertir en data-frame python. 
#
# Les données de base à récupérer sont :
# - Les historiques de consommation
# - Le calendrier des jours fériés
# - Les données météo, ainsi que la liste des stations
# - Le calendrier des jours TEMPO

# ### Données de consommation
#
# Dans un premier temps on importe les données de consommation réalisée à partir du fichier "YconsoT0.csv". La date et l'heure sont données dans les deux premières colonnes, et les autres colonnes correspondent aux consommations des 12 régions françaises (hors Corse) et à la consommation nationale.
#
# Pour cela on utilise la bibliothèque **pandas** pour la manipulation de données et la fonction **read_csv**.

# #### Import depuis un csv

# Les données du csv sont importé dans un objet de type dataframe
conso_csv = os.path.join(data_folder, "YconsoT0.csv")
conso_df = pd.read_csv(conso_csv, sep=";")

# Il faut ensuite vérifier que les données sont importées correctement

# Afficher les dimensions et le noms des colonnes de la data frame
print(conso_df.shape)  # Nombre de lignes, nombre de colonnes

# Liste des colonnes de la data-frama
print(conso_df.columns)

# Affichage des premières lignes
print(conso_df.head(5))

# #### Petit détour pour gérer les dates
#
# Le fichier YconsoT0.csv contient en particulier 2 colonnes 'date' et 'time'. Celles-ci contiennent des objets de type "string" correspondant à la date et à l'heure. 
#
# <img src="pictures/clock.png" width=60 height=60>
#
# Nous allons fusionner ces informations en une nouvelle colonne d'objets de type **datetime** mieux adaptés pour la manipulation de dates et d'heures. En effet, pour manipuler des dates (effectuer des tris, des sélections, récupérer si c'est un lundi, mardi,...), il est plus efficace de passer par un objet "datetime" plutôt que de se débrouiller en manipulant des chaînes de caractères.

# On appelle "ds" (dateStamp) cette nouvelle colonne
conso_df['ds'] = pd.to_datetime(conso_df['date'] + " " + conso_df['time'])

conso_df[['ds', 'date', 'time']].head(5)

# La cellule ci-dessous a pour but d'illustrer comment utiliser ces objets.

# +
# datetime vers string
noel_2017_date = datetime.date(2017, 12, 25)
noel_2017_str = datetime.datetime.strftime(noel_2017_date, format="%Y-%m-%d")
print("noel_2017_date vaut : {}, et est de type {}".format(noel_2017_date, str(type(noel_2017_date))))
print("noel_2017_str vaut : {}, et est de type {}".format(noel_2017_str, str(type(noel_2017_str))))
print("---")

# string vers datetime
starwars_day_2017_str = "2017-05-04"
starwars_day_2017_date = datetime.datetime.strptime(starwars_day_2017_str, "%Y-%m-%d")
print("starwars_day_2017_date vaut : {}, et est de type {}".format(starwars_day_2017_date, str(type(starwars_day_2017_date))))
print("D'ailleurs, c'était le " + str(starwars_day_2017_date.weekday() + 1) + " ème jour de la semaine, où 0 correspond à lundi et 6 correspond à dimanche")
print("starwars_day_2017_str vaut : {}, et est de type {}".format(starwars_day_2017_str, str(type(starwars_day_2017_str))))
print("---")

# Voyager dans le temps
saint_sylvestre_2017_date = datetime.date(2017, 12, 31)
bienvenu_en_2018_date = saint_sylvestre_2017_date + datetime.timedelta(days=1)
print("Le 31 décembre 2017 plus un jour ça donne le {}".format(bienvenu_en_2018_date))
# -

# #### Réduction du problème : se débarrasser des données qui ne nous intéressent pas
#
# Le dataframe de consommation est volumineux, et contient beaucoup d'information inutile (au moins en première approximation) pour notre problème de prévision de la consommation nationale. On va donc le simplifier.
#
# On va se concentrer sur la consommation à l'**échelle nationale** au **pas horaire**. On va donc ne conserver que la colonne qui nous intéresse, et ne conserver que les lignes qui correspondent aux heures pleines.

# on commence par ecarter les colonnes inutiles
conso_france_df = conso_df[['ds', 'Consommation.NAT.t0']]

# +
# et maintenant on ne garde que les heures pleines
# Pour cela on va utiliser notre colonne d'objet datetime
minutes = conso_france_df['ds'].dt.minute
indices_hours = np.where(minutes.values == 0.0)

#print(conso_france_df['ds'])
#print(minutes)
#print(indices_hours)
# -

conso_france_horaire_df = conso_france_df.loc[indices_hours]
conso_france_horaire_df.head(5)

# les index de ce sous-dataframe correspondent à celle du dataframe de base,
# et donc sont pour l'instant des multiples de 4.
# on va les réinitialiser pour avoir une dataframe "neuve"
conso_france_horaire_df = conso_france_horaire_df.reset_index(drop=True)  
print(conso_france_horaire_df.head(5))
print(conso_france_horaire_df.shape)

# ### Récuperation des jours fériés
#
# Même principe

# +
jours_feries_csv = os.path.join(data_folder,"joursFeries.csv")
jours_feries_df = pd.read_csv(jours_feries_csv, sep=";")

jours_feries_df.head(5)
# -

# Pour la première colonne, les dates sont au format "string"
# Nous allons les convertir en objet "datetime" mieux adaptés pour la manipulation de dates
print("Après import du csv, la colonne ds est de type " + str(type(jours_feries_df.ds[0])))

jours_feries_df.ds = pd.to_datetime(jours_feries_df.ds)
print("maintenant, la colonne ds est de type " + str(type(jours_feries_df.ds[0])))

jours_feries_df.head(8)

# Cette dataframe fait correspondre le timestamp "2012-12-25" avec "Noël". Cependant, le timestamp "2012-12-25" correspond implicitemlent au timestamp "2012-12-25 00:00", ce qui fait que pour l'instant les timestamp "2012-12-25 00:05" ou "2012-12-25 03:00" ne sont pas identifiés comme étant aussi Noël, ce qui va poser problème plus tard.
#
# La cellule ci-dessous va étendre la dataframe des jours fériés de sorte à résoudre ce problème.

# +
timestamps_of_interest = conso_df[['ds']]
timestamps_of_interest["day"] = timestamps_of_interest["ds"].apply(lambda x: datetime.datetime.strftime(x, format="%Y-%m-%d"))

tmp_df = jours_feries_df
tmp_df["day"] = jours_feries_df["ds"].apply(lambda x: datetime.datetime.strftime(x, format="%Y-%m-%d"))

tmp_df = pd.merge(timestamps_of_interest, tmp_df, on='day', how="left", suffixes=("", "_tmp"))

jours_feries_df = tmp_df[["ds", "holiday"]]
print(jours_feries_df.loc[450:500])
# -

# ### Récupération des coordonnées géographiques des stations météo - au boulot !
#
# On va charger le csv qui à chaque station météo attribue sa longitude/latitude/poids. Pour en savoir plus sur les poids :  
# https://clients.rte-france.com/lang/fr/visiteurs/services/actualites.jsp?id=9482&mode=detail
#
# **Votre mission** :
# - Importez les données contenues dans le fichier csv *StationsMeteoRTE.csv* qui se situe dans data_folder vers un dataframe *stations_meteo_df*
# - Regardez à quoi ces données ressemblent 

# +
# Chargez les données de StationsMeteoRTE.csv vers stations_meteo_df

## TODO - START
stations_meteo_csv = os.path.join(data_folder, "StationsMeteoRTE.csv")
stations_meteo_df = pd.read_csv(stations_meteo_csv, sep=";")
## TODO - END

# +
# Affichez les 5 premières lignes de stations_meteo_df

## TODO - START
stations_meteo_df.head(5)
## TODO - END
# -

# Pour compter le nombre de stations il suffit de compter le nombre de lignes dans le data-frame
# Ceci se fait un utilisant "shape"
nb_stations = stations_meteo_df.shape[0]
print(nb_stations)

# ## Récupération du dataframe de météo
#
# <img src="pictures/weather.png" width=60 height=60>
#
# On va utiliser les mêmes fonctions que précédemment pour lire le fichier **'meteoX_T.csv'**, qui est situé dans data_folder et contient les historiques de température réalisée et prévue pour différentes stations Météo France.
#
# **Attention : Les données météo sont encryptées dans un fichier zip.**  
# Pour les lire vous avez besoin d'un mot de passe qui ne peut vous être donné que dans le cadre d'un travail au sein de RTE.

meteo_zip = os.path.join(data_folder, "meteoX_T0_T24.zip")

password = None


# +
# Cette étape peut être un peu longue car le fichier est volumineux

# Pour travailler avec les fichiers zip, on utilise la bibliothèque **zipfile**.
zipfile_meteo = zipfile.ZipFile(meteo_zip)
zipfile_meteo.setpassword(bytes(password,'utf-8'))
meteo_df = pd.read_csv(zipfile_meteo.open('meteoX_T0_T24'),sep=";",engine='c',header=0)
# -

# On se crée une colonne avec des objets timestamp pour les dates
meteo_df['ds'] = pd.to_datetime(meteo_df['date'] + ' ' + meteo_df['time'])

print(meteo_df.shape)  # (nb lignes , nb_colonnes)

print(meteo_df.head(5))

# Comme pour la consommation, on ne retient que les données des heures rondes afin de réduire la taille du problème.

# +
minutes = meteo_df['ds'].dt.minute
mask = np.where(minutes.values == 0.0)
meteo_horaire_df = meteo_df.loc[mask]

# On remet les index au propre
meteo_horaire_df = meteo_horaire_df.reset_index(drop=True)
# -


# Pour se mettre dans le cadre d'un exercice de prévision, et toujours dans l'idée de réduire la taille du problème, on ne va conserver que les températures réalisées, les températures prévues à 24h (noms de colonnes finissant par 'Th+24'), ainsi que la colonne _ds_.

colonnes_a_garder = ['ds'] + list(meteo_horaire_df.columns[meteo_horaire_df.columns.str.endswith("Th+0")]) + list(meteo_horaire_df.columns[meteo_horaire_df.columns.str.endswith("Th+24")])
meteo_prev_df = meteo_horaire_df[colonnes_a_garder]

print(meteo_prev_df.head(5))
print(meteo_prev_df.shape)

# #### Pour se faire plaisir, on visualise les données météo sur une carte

# +
columns_of_interest = [col for col in meteo_horaire_df.columns if col.endswith("Th+0")]
temperatures_realisees = meteo_horaire_df[columns_of_interest]
temperature_journee = temperatures_realisees.loc[0]

#print(temperature_journee)

# +
map_options = GMapOptions(lat=47.08, lng=2.39, map_type="roadmap", zoom=5)

plot = GMapPlot(x_range=Range1d(), y_range=Range1d(), map_options=map_options)
plot.title.text = "France"

# For GMaps to function, Google requires you obtain and enable an API key:
#
#     https://developers.google.com/maps/documentation/javascript/get-api-key
#
# Replace the value below with your personal API key:
plot.api_key = "AIzaSyC05Bs_e0q6KWyVHlmy0ymHMKMknyMbCm0"

# nos données d'intérêt pour créer notre visualisation
data = dict(lat=stations_meteo_df['latitude'],
            lon=stations_meteo_df['longitude'],
            label=stations_meteo_df['Nom'],
            temp=temperature_journee
           )

source = ColumnDataSource(data)

# l'échelle 
Tlow = 0
Thigh = 20
color_mapper = LogColorMapper(palette="Viridis256", low=Tlow, high=Thigh)

# la couleur de remplissage des cercles est fonction de la valeur de la temérature
circle = Circle(x="lon", y="lat", 
                size=15, 
                fill_color={'field': 'temp', 'transform': color_mapper},
                fill_alpha=0.8, 
                line_color=None,)

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

# ## Bonus : récupération de données depuis internet
#
# Dans le but d'automatiser un processus, nous pouvons implémenter une fonction qui va chercher les dernières données mises à disposition sur internet.  
#
# Pour l'exemple de la prévision de consommation, il serait pertinent de fournir en entrée du modèle l'information sur le type de jour Tempo. Les clients ayant souscrit à ce type de contrat sont incités à réduire leur consommations les jours BLANC et ROUGE, aussi on peut penser que cette information permettra d'améliorer la qualité des prédictions.

# ### Manipulation à la main
#
# Avant d'implémenter la version automatique, faisons une fois à la main cette manipulation.
#
#  - Recupérez à la main le calendrier TEMPO pour 2017-2018 :
#  http://www.rte-france.com/fr/eco2mix/eco2mix-telechargement
#  - Le déposer dans _data&#95;folder_
#  - Le dézipper
#  - Regarder les données dans excel ou autre. Notez en particulier la fin du fichier, la supprimer
#  
# Importez ces données dans un dataframe avec 'read_excel' de la librairie pandas ou autre méthode

tempo_xls = os.path.join(data_folder, "eCO2mix_RTE_tempo_2017-2018.xls")
tempo_df = pd.read_csv(tempo_xls, sep="\t", encoding="ISO-8859-1")  # ce fichier est en fait un csv et non un xls...

print(tempo_df.head(5))


# ### La même chose automatisée
#
# On récupère maintenant automatiquement les informations sur Internet à partir de l'url, sans devoir les chercher à la main soi-même.

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


# On teste la fonction définie ci-dessus. Parfois pour de sombres raisons de proxy la connection au serveur peut échouer. Comme ce TP porte sur le machine-learning on ne s'acharnera pas sur cette partie en cas d'échec :-)

# +
#url = "https://eco2mix.rte-france.com/curves/downloadCalendrierTempo?season=17-18"
#tempo_xls_zip_name = "eCO2mix_RTE_tempo_2017-2018.zip"

#tempo_df = get_tempo_data(url, data_folder, tempo_xls_zip_name)

#print(tempo_df)
# -

# Pour les personnes intéressées par le webscrapping, jeter un oeil du côté de <a href="https://www.crummy.com/software/BeautifulSoup/bs4/doc/" title="link to google">BeautifulSoup</a>

# ##  Fusion des données
#
# <img src="pictures/fusion.png" width=600 height=200>
#
# On va maintenant construire un dataframe unique qui regroupe toutes les données nécessaire à notre modèle de prévision. On aura ici une ligne pour chaque timestamp, et dans cette ligne à la fois notre X et notre Y pour le futur modèle de machine-learning.

# Dans un premier temps, on fusionne la consommation et la température.
merged_df = pd.merge(conso_france_horaire_df, meteo_prev_df, on = 'ds')

print(merged_df.shape)
print(merged_df.columns)

# Ensuite, on fusionne avec le calendrier des jours fériés en joignant sur la colle "ds" des timestamps.

merged_df = pd.merge(merged_df, jours_feries_df, how = "left", on = "ds")

print(merged_df.shape)
print(merged_df.columns)

# ### Calcul de la température France 32 villes 
#
# On va ajouter une colonne à notre dataframe, colonne que - par expérience/expertise - on sait pouvoir être utile pour prévoir la consommation.
#
# La température France est une moyenne pondérée de la température de 32 stations. On a donc besoin des poids de stations_meteo_df.

merged_df['FranceTh+0'] = np.dot(merged_df[list(merged_df.columns[merged_df.columns.str.endswith("Th+0")])], stations_meteo_df['Poids'])
merged_df['FranceTh+24'] = np.dot(merged_df[list(merged_df.columns[merged_df.columns.str.endswith("Th+24")])], stations_meteo_df['Poids'])


print(merged_df.shape)
print(merged_df.columns)

# ### Cohérence temporelle des données pour notre modèle de prédiction

# Prenons quelques instants pour regarder les données que l'on a pour l'instant :

merged_df

# Pour chaque point horaire, on a :
# * la consommation réalisée (au niveau national
# * la température réalisée pour chaque station météo, ainsi qu'une valeur représentative de la température moyenne à l'échelle nationale
# * une prévision de température pour 24 heures plus tard pour chaque station météo, ainsi qu'une prévision de la température moyenne France pour dans 24 heures
# * l'information si le point horaire appartient à un jour férié ou non    

# Ce que l'on veut faire, c'est mettre au point un modèle qui prédit comme Y :
# * la consommation nationale pour point_horaire_cible
# prenant en entrée un X qui comporte tout ou partie de :
# * l'information si point_horaire_cible appartient à un jour férié
# * La prévision météo pour point_horaire_cible, prévision établie 24h à l'avance
# * La consommation réalisée 24h avant point_horaire_cible
# * La température réalisée 24h avant point_horaire_cible
# Il faut simplement être vigilant sur le fait qu'il est interdit de prédire une consommation pour point horaire cible en ayant comme entrée la température réalisée pour point horaire cible (on n'est pas dans minority report)
#
# Ainsi, on va adapter _merged_df_ de sorte à ce que chaque ligne correspondent à un point horaire cible à prédire, avec en colonne :
# * le Y (la consommation nationale pour point_horaire_cible)
# * Le X (cf. ci-dessus)
#
# Ainsi, on va devoir décaler toutes les données de températures de 24 heures

merged_df[list(merged_df.columns[merged_df.columns.str.endswith("Th+0")])] = merged_df[list(merged_df.columns[merged_df.columns.str.endswith("Th+0")])].shift(24)
merged_df[list(merged_df.columns[merged_df.columns.str.endswith("Th+24")])] = merged_df[list(merged_df.columns[merged_df.columns.str.endswith("Th+24")])].shift(24)

# ## Sauvegarde du fichier 
#
# Tout d'abord on sépare les données en deux : 
# - le vecteur de consommation à prévoir : y_conso
# - La matrice des variables explicatives : X_input
#
# Sachant que plus tard notre modèle aura pour mission d'établir une correspondance _f_ telle que l'on ait du mieux possible une relation *y = f(X)*.

y_conso = merged_df[['ds', 'Consommation.NAT.t0']]
y_conso.columns = ['ds', 'y']

X_input = merged_df.drop(['Consommation.NAT.t0'], axis=1)

y_conso.to_csv("data/Yconso.csv", index = False)
X_input.to_csv("data/Xinput.csv", index = False)








