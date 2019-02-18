# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # TP facultatif : Préparation du jeu de données brut

# L'objectif de ce TP est de comprendre comment, à partir de différentes sources de données, on construit une première table.
# Cette table sera ensuite traitée pour constuire le meilleur modèle d'apprentissage possible pour la prévision de consommation nationale.
#
# ## Environnement

# +
# Exécutez la cellule ci-dessous (par exemple avec shift-entrée)
# Si vous exécuter ce notebook depuis votre PC, il faudra peut-etre installer certaines librairies avec 
# 'pip install ma_librairie'
import os  # accès aux commandes système
import datetime  # structure de données pour gérer des objets calendaires
import pandas as pd  # gérer des tables de données en python
import numpy as np  # librairie d'opérations mathématiques
import zipfile # manipulation de fichiers zip
import urllib3 # téléchargement de fichier
data_folder = os.path.join(os.getcwd(),"data")
# %autosave 0

print("Mon repertoire est : {}".format(data_folder))
print("Fichiers contenus dans ce répertoire :")
for file in os.listdir(data_folder):
    print(" - " + file)
# -

# ## Récupération des données
#
# Dans cette partie nous allons charger les fichiers csv nécessaires pour l'analyse, puis les convertir en data-frame python. Les données de base à récupérer sont :
#
# - Les historiques de consommation
# - Le calendrier des jours fériés
# - Les données météo, ainsi que la liste des stations
# - Le calendrier des jours TEMPO
#
#

# ### Données de consommation
#
# Dans un premier temps on importe les données de consommation réalisée à partir du fichier "YconsoT0". La date et l'heure sont données et les autres colonnes correspondent aux consommations des 12 régions françaises (hors Corse) et à la consommation nationale.
# Pour cela on utilise la bibliothèque **pandas** pour la manipulation de données et la fonction **read_csv**.

# Conversion en tant que data-frame
# Remarquez que l'on manipule un gros fichier, ce qui explique pourquoi l'exécution de cette cellule prend du temps
conso_csv = os.path.join(data_folder, "YconsoT0")
conso_df = pd.read_csv(conso_csv, sep=";", engine='c', header=0) #engine en language C et position header pour accélérer le chargement

# Il faut ensuite vérifier que les données sont importées correctement

# Afficher les dimensions et le noms des colonnes de la data frame
print(conso_df.shape)
print(conso_df.columns)
print(conso_df.head(5))

# ### La composante temporelle pour une problématique de prévision
#
# <img src="pictures/clock.png" width=60 height=60>
#
# Le fichier conso_Y.csv contient en particulier 2 colonnes 'date' et 'time', les deux contenant des objets de type "string" correspondant à la date et à l'heure. Nous allons fusionner ces informations en une nouvelle colonne
# d'objets "ds" (dateStamp) mieux adaptés pour la manipulation de dates et d'heures.
#

conso_df['ds'] = pd.to_datetime(conso_df['date'] + " " + conso_df['time'])

conso_df[['ds', 'date', 'time']].head(5)

# Pour manipuler des dates (effectuer des tris, des sélections, récupérer si c'est un lundi, mardi,...), il est plus efficace de passer par un objet "datetime" plutôt que de se débrouiller en manipulant des chaînes de caractères.
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

# ### Réduire notre problème
#
# Le dataframe de consommation est volumineux, et contient beaucoup d'information inutile (au moins en première approximation) pour notre problème de prévision de la consommation nationale. On va donc simplifier.
#
# On va se concentrer sur la consommation à l'échelle nationale **au pas horaire**.
#

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
print(consoFranceHoraire_df.head(5))
print(consoFranceHoraire_df.shape)

# ### Récuperation des jours fériés

jours_feries_csv = os.path.join(data_folder,"joursFeries.csv")
jours_feries_df = pd.read_csv(jours_feries_csv, sep=";")

# Pour la première colonne, les dates sont au format "string"
# Nous allons les convertir en objet "datetime" mieux adaptés pour la manipulation de dates
jours_feries_df.ds = pd.to_datetime(jours_feries_df.ds)

# Regardons la tête des données
jours_feries_df.head(8)

# ### Récupération des stations météo
#
# NB - Pour en savoir plus sur les poids :  
# https://clients.rte-france.com/lang/fr/visiteurs/services/actualites.jsp?id=9482&mode=detail

stations_meteo_csv = os.path.join(data_folder, "StationsMeteoRTE.csv")
stations_meteo_df = pd.read_csv(stations_meteo_csv, sep=";")

stations_meteo_df.head(5)

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

# ## Récupération du dataframe de météo
# <img src="pictures/weather.png" width=60 height=60>
#
# On va utiliser les mêmes fonctions que précédemment pour lire le fichier **'meteoX_T.csv'**, qui est situé dans data_folder et contient les historiques de température réalisée et prévue pour différentes stations Météo France.
#
# Importez-les dans un dataframe _meteo&#95;df_
# et regardez à quoi elles ressemblent. Pensez aussi à changer les dates _string_ vers le format _datetime_.
#
# ** Attention : Les données météo sont encryptées dans un fichier zip.** 
# Pour les lire vous avez besoin d'un mot de passe qui ne peut vous être donné que dans le cadre d'un travail au sein de RTE.
# Pour travailler avec les fichiers zip, on utilise la bibliothèque **zipfile**.

# +
#password = None
password = "FIFA_Meteo"
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
# -

# TODO: afficher les 5 premières lignes de "meteo_df"
print(meteo_df.head(5))
print(meteo_df.shape)
# END

# Comme pour la consommation, on ne retient que les données des heures rondes.

minutes = meteo_df['ds'].dt.minute
mask = np.where(minutes.values == 0.0)
meteoHoraire_df = meteo_df.loc[mask]
meteoHoraire_df = meteoHoraire_df.reset_index(drop=True)


# Pour se mettre dans le cadre d'un exercice de prévision on ne va conserver que les températures prévues à 24h (noms de colonnes finissant par 'Th+24'), ainsi que la colonne _ds_.

meteo_prev_df = meteoHoraire_df[['ds']+list(meteoHoraire_df.columns[meteoHoraire_df.columns.str.endswith("Th+24")]) ]

print(meteo_prev_df.head(5))
print(meteo_prev_df.shape)

# # Fusion des données
# <img src="pictures/fusion.png" width=600 height=200>
#
# On va maintenant construire un dataframe, issu de la fusion entre les données de consommation, de prévision de tempérture et de jours fériés.
#
# Dans un premier temps, on fusionne la consommation et la température.

Xinput = pd.merge(consoFranceHoraire_df, meteo_prev_df, on = 'ds')

print(Xinput.shape)
print(Xinput.columns)

# Ensuite, on fusionne avec le calendrier des jours fériés (**Attention : left join**)

Xinput = pd.merge(Xinput,jours_feries_df,how = "left", on = "ds")

print(Xinput.shape)
print(Xinput.columns)

# ### Calcul de la température France 32 villes 
# La température France est une moyenne pondérée de la température de 32 stations. On a donc besoin des poids de stations_meteo_df.

Xinput['FranceTh+24'] = np.dot(Xinput[list(Xinput.columns[Xinput.columns.str.endswith("Th+24")])],stations_meteo_df['Poids'])


print(Xinput.shape)
print(Xinput.columns)

# ### Cohérence temporelle des données
#
# Pour avoir la prévision de température à 24h pour l'observation t, il faut décaler les données de température de 24 pas de temps !

Xinput[list(Xinput.columns[Xinput.columns.str.endswith("Th+24")])] = Xinput[list(Xinput.columns[Xinput.columns.str.endswith("Th+24")])].shift(24)

# ## Bonus : récupération de données depuis internet
#
# Dans le but d'automatiser un processus, nous pouvons implémenter une fonction qui ira chercher les dernières données mises à disposition sur internet.  
#
# Pour cet exemple nous allons considérer les jours Tempo, et (si le temps le permet en fin de TP) tester si cette information permet d'améliorer la qualité des prédictions.
#
# ### Manipulation à la main
#
#  - Recupérez à la main le calendrier TEMPO pour 2017-2018 :
#  http://www.rte-france.com/fr/eco2mix/eco2mix-telechargement
#  - Le déposer dans _data&#95;folder_
#  - Le dézipper
#  - Regarder les données dans excel ou autre. Notez en particulier la fin du fichier, la supprimer
#  
# Importez ces données dans un dataframe avec _'read&#95;csv'_ de la librairie _pandas_ (ce fichier est en fait un csv et non un xls)

tempo_xls = os.path.join(data_folder, "eCO2mix_RTE_tempo_2017-2018.xls")
tempo_df = pd.read_csv(tempo_xls, sep="\t", encoding="ISO-8859-1")  
tempo_df['ds'] = pd.to_datetime(tempo_df.Date,dayfirst = True)

print(tempo_df.head(5))

Xinput = pd.merge(Xinput,tempo_df,how = "left", on = "ds")
print(Xinput.head(5))

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

# +
## Test de la fonction définie ci-dessus
url = "https://eco2mix.rte-france.com/curves/downloadCalendrierTempo?season=17-18"
tempo_xls_zip_name = "eCO2mix_RTE_tempo_2017-2018.zip"

tempo_df = get_tempo_data(url, data_folder, tempo_xls_zip_name)

print(tempo_df)

# -

# Pour les personnes intéressées par le webscrapping, jeter un oeil du côté de <a href="https://www.crummy.com/software/BeautifulSoup/bs4/doc/" title="link to google">BeautifulSoup</a>

# ## Sauvegarde du fichier 
#
# Tout d'abord on sépare les données en deux : 
# - le vecteur de consommation à prévoir : Yconso
# - La matrice des variables explicatives : Xinput

# +
Yconso = Xinput[['ds','Consommation.NAT.t0']]
Yconso.columns = ['ds', 'y']

Xinput = Xinput.drop(['Consommation.NAT.t0'],axis=1)
# -

Yconso.to_csv("data/Yconso.csv", index = False)
Xinput.to_csv("data/Xinput.csv", index = False)


