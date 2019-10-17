#!/usr/bin/env python
# coding: utf-8

# # Le problème : la prévision de consommation électrique
# 
# Pour garantir l'équilibre offre-demande à chaque instant et gérer l'acheminement de l'électricité, RTE construit ses propres prévisions de la consommation nationale, régionale, et locale, à différentes échéances de temps (de l'infrajournalier au pluri-annuel).
# 
# Ici on se focalise sur un problème particulier : **la prévision de la consommation électrique nationale horaire à horizon J+1** (on suppose qu'on connaît toutes les données jusqu'au jour J inclus). 
# 
# ## Les données : Eco2mix
# 
# La courbe de charge France est disponible sur eco2mix :
# http://www.rte-france.com/fr/eco2mix/eco2mix
# ou sur application mobile.
# 
# Vous pouvez naviguer sur le site pour vous familiariser avec les données sur lesquelles vous allez travailler.
# 
# ## Objectif :
# 
# Au cours de ce TP, nous allons aborder les différentes étapes nécessaires à la construction d'un modèle de prévision de consommation :
# 
# 1) Formalisation du problème: que souhaite-t-on prédire (quel est mon Y) ? Avec quelles variables explicatives (quel est mon X) ?
# 
# 2) Collecte des données: où se trouvent les données ? Quel est le format ? Comment les récupérer ? (FACULTATIF - voir TP "TP1_Preparation_donnees")
# 
# 3) Import des données et analyses descriptives : visualiser des séries temporelles, statistiques descriptives
# 
# 4) Transformation des données (feature engineering) pour entrainer et tester un premier modèle
# 
# 5) Création de prévision à dire d'expert pour servir de référence.
# 
# 6) Découpage des données : apprentissage - test
# 
# 7) Evaluer un modèle
# 
# 8) Tester des algorithmes de référence : régression linéaire, forêts aléatoires, xgboost
# 
# 9) Itérer à partir des modèles testés pour améliorer les prévisions
# 
# Nous verrons qu'une difficulté majeure réside dans la construction des "bonnes" variables explicatives ("garbage in, garbage out").
# 
# **Le notebook est parsemé de questions (<font color='green'>en vert </font>). Vous pouvez y répondre sur la feuille fournie.**
# 
# ## Méthodes de prévision considérées
# 
# Les modèles actuels reposent sur des méthodes de régression linéaire et non-linéaires. Nous étudierons ici les limites de la régression linéaire.
# 
# Pour améliorer les prévisions, nous aurons recours aux méthodes dites de Machine Learning. Ces méthodes ne dépendent pas d'une formalisation a priori du lien entre les variables explicatives X et la variable à expliquer Y. 
# Elles sont souvent moins interprétables mais peuvent être plus efficaces en prévision. Elles peuvent nécessiter plus de temps de calcul et plus de données pour cela.
# 
# Construire un bon modèle d'apprentissage nécessite en général de la connaissance experte dans le domaine d'intérêt pour créer des modèles pertinents et efficaces. 
# 
# ## To be continued : deep learning
# 
# Le deuxième TP permettra d'investiguer les modèles "Deep" avec réseaux de neurones, en montrant le moindre besoin en feature engineering et leur plus grande capacité à absorber l'information grâce aux représentations hiérarchiques qu'ils créent.

# # Environnement de travail 
# 
# Ceci est un notebook jupyter. Il permet d'exécuter du code python, d'afficher des résultats et d'écrire du texte pour décrire l'ensemble de l'étude.
# 
# <font color='red'>
#     
# **NB : L'aide de python est accessible en tapant help(nom_de_la_commande)**
# 
# </font>
# 
# ## Chargement des packages

# In[ ]:


# Exécutez la cellule ci-dessous (par exemple avec shift-entrée)
# Si vous exécuter ce notebook depuis votre PC, il faudra peut-etre installer certaines librairies avec 
# 'pip install ma_librairie'
import os  # accès aux commandes système

import datetime  # structure de données pour gérer des objets calendaires
import pytz # gestion des fuseaux horaires
from datetime import timezone
import pandas as pd  # gérer des tables de données en python
import numpy as np  # librairie d'opérations mathématiques
from math import sqrt

import sklearn  # librairie de machine learning
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

import matplotlib.pyplot as plt  # tracer des visualisations
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import zipfile  # compresser ou décompresser fichier

get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().run_line_magic('autosave', '0')


# ## Données disponibles
# 
# Choix du répertoire de travail "data_folder" dans lequel tous les fichiers csv seront entreposés. Ici le répertoire s'appelle *data*.
# 
# Ensuite on affiche les fichiers du répertoire pour vérification

# In[ ]:


data_folder = os.path.join(os.getcwd(), "data")


# In[ ]:


print("Mon repertoire est : {}".format(data_folder))
print("Fichiers contenus dans ce répertoire :")
for file in os.listdir(data_folder):
    print(" - " + file)


# # Récupération des données dans Python
# 
# Dans cette partie nous allons charger les fichiers csv nécessaires pour l'analyse, puis les convertir en data-frame python : 
# - Yconso.csv
# - Xinput.csv
# 
# Rappel : Les données brutes ont été pré-traitées à l'aide du notebook TP1_Preparation_donnees.ipynb pour obtenir ces deux fichiers.
# 
# ## import de Yconso.csv

# In[ ]:


Yconso_csv = os.path.join(data_folder, "Yconso.csv")
Yconso = pd.read_csv(Yconso_csv)
print(Yconso.head(5)) # affichage des premières lignes
print(Yconso.dtypes) # affichage du type de données pour chaque colonne


# La colonne "ds" contient la date, mais celle-ci n'est pas reconnue en tant que telle mais en tant que chaîne de caractères (https://pbpython.com/pandas_dtypes.html). On va la convertir en objet de type "datetime" plus approprié pour extraire des informations comme le jour de la semaine ou l'heure.  
# Pour plus d'information, voir le TP1_Preparation_donnees.

# In[ ]:


Yconso['ds'] = pd.to_datetime(Yconso['ds'],utc=True)
print(Yconso.dtypes)
print(Yconso.head(5))


# Visuellemement cela ne change rien.
# 
# On peut aussi afficher la dimension du DataFrame (toujours s'assurer que cela correspond aux valeurs attendues) : 

# In[ ]:


print(Yconso.shape)


# ## Import de Xinput.csv
# 
# **Attention : Les données Xinput sont encryptées dans un fichier zip. du fait de données météo**  
# Pour les lire vous avez besoin d'un mot de passe qui ne peut vous être donné que dans le cadre d'un travail au sein de RTE.
# 
# Sinon, la lecture se déroule comme pour le fichier Yconso.csv : transformation en datetime de la colonne *ds* et vérification des dimensions.

# In[ ]:


Xinput_zip = os.path.join(data_folder, "Xinput.zip")


# In[ ]:


password = ''


# In[ ]:


# Pour travailler avec les fichiers zip, on utilise la bibliothèque **zipfile**.
zipfile_xinput = zipfile.ZipFile(Xinput_zip)
zipfile_xinput.setpassword(bytes(password,'utf-8'))
Xinput = pd.read_csv(zipfile_xinput.open('Xinput.csv'),sep=",",engine='c',header=0)

Xinput['ds'] = pd.to_datetime(Xinput['ds'],utc=True)


# Vous disposez de relevés de températures en stations (voir le fichier *data/StationsMeteoRTE.csv* pour plus d'informations), d'une température France prévue (pour l'instant considéré) et réalisée (24h avant).

# In[ ]:


print("dimension de X")
print(Xinput.shape)
print("colonnes de X")
print(Xinput.columns)
print("aperçu de de X")
print(Xinput.head(35))


# 
# Dans un premier temps, nous allons travailler **uniquement avec la température France.** Les températures en stations pourront être utilisées dans la partie Bonus, par exemple.

# In[ ]:


Xinput = Xinput[['ds', 'holiday', 'Th_real_24h_avant', 'Th_prev']]


# <font color='green'>
# 
# * Quelles sont les variables disponibles (dans Xinput et Yconso)?
# * Quelles sont les dimensions (nombre d’observations et de variables) de Xinput et Yconso après lecture des fichiers csv?
# 
# </font>

# # Visualisation des données 
# 
# La DataScience et le Machine Learning supposent de bien appréhender les données sur lesquelles nos modèles vont être entrainés. Pour cela, il est utile de faire des statistiques descriptives et des visualisations de nos différentes variables.
# 
# Traitant d'un problème de prévision, on visualisera en particulier des séries temporelles.
# 
# Vous allez voir des :
# - échantillons de données
# - profils de courbe de consommation journaliers et saisonniers
# - visualisations de corrélation entre conso J et conso retardée
# 
# ## Calcul de statistiques descriptives sur la consommation nationale
# 
# A l'aide de la fonction *describe*, on calcule les indicateurs classiques. On cherche les données manquantes avec la fonction *isnull*.
# 

# In[ ]:


Yconso['ds'].describe()


# In[ ]:


Yconso['y'].describe()


# In[ ]:


Yconso['y'].isnull().sum()


# <font color='green'>
# 
# * Quelle est la valeur moyenne de la consommation horaire? son min et son max? 
# * Quelle est la période temporelle étudiée?
# 
# </font>

# ## Visualiser la consommation d'un jour particulier
# On souhaite visualiser la consommation réalisée pour un jour donné de l'historique. Pour cela on construit une fonction.

# In[ ]:


def plot_load(var_load, year, month, day):
    date_cible = pytz.utc.localize(datetime.datetime(year=year, month=month, day=day))  # implicitement heure = minuit
    date_lendemain_cible = date_cible + datetime.timedelta(days=1)
    mask = (var_load.ds >= date_cible) & (var_load.ds <= date_lendemain_cible)   
    consoJour = var_load[mask]
    plt.plot(consoJour['ds'], consoJour['y'], color='blue')
    plt.show()


# In[ ]:


plot_load(Yconso, 2016, 12, 20)


# <font color='green'>
# 
# * Afficher un jour ouvré d’hiver, un jour ouvré d’été, commenter.
# 
# </font>

# ## Afficher une semaine arbitraire de consommation
# On modifie la fonction précédente en rajoutant le timedelta en paramètre.

# In[ ]:


def plot_load_timedelta(var_load, year, month, day, delta_days):
    date_cible = pytz.utc.localize(datetime.datetime(year=year, month=month, day=day))
    date_lendemain_cible = date_cible + datetime.timedelta(days=delta_days)

    conso_periode = var_load[(var_load.ds >= date_cible) 
                                      & (var_load.ds <= date_lendemain_cible)]
    plt.plot(conso_periode['ds'], conso_periode['y'], color='blue')
    plt.show()


# In[ ]:


plot_load_timedelta(Yconso, 2016, 12, 20, delta_days=7)


# <font color='green'>
# 
# * Commenter.
# 
# </font>

# ## Observation des profils de la consommation pour les mois d'hiver et les mois d'été
# Toujours dans le but d'appréhender nos données, on va regarder les profils moyens pour les mois d'été et pour ceux d'hiver. On va également observer le min et le max pour avoir une idée de la variabilité du signal.

# In[ ]:


# Par commodité, on isole le mois pour après attraper les mois d'hiver et d'été
Xinput['month'] = Xinput['ds'].dt.month

# On isole aussi les heures
Xinput['hour'] = Xinput['ds'].dt.hour

# On sépare les jours de la semaine en week-end / pas week-end
# De base, la fonction datetime.weekday() renvoie 0 => Lundi, 2 => Mardi, ..., 5 => Samedi, 6 => Dimanche
# Ci-dessous, si on a un jour d ela semaine alors dans la colonne weekday on mettra 1, et 0 si c'est le week-end
Xinput['weekday'] = (Xinput['ds'].dt.weekday < 5).astype(int)  # conversion bool => int


# In[ ]:


Xinput.head(5)


# In[ ]:


# On aggrège les mois d'hiver ensemble
XY_df = pd.merge(Yconso, Xinput, on = 'ds')
groupedHiver = XY_df[(XY_df.month == 12) | 
                                     (XY_df.month == 1) | 
                                     (XY_df.month == 2)].groupby(['weekday', 'hour'], as_index=True)

# Idem pour les mois d'été
groupedEte = XY_df[(XY_df.month == 6) | 
                                   (XY_df.month == 7) | 
                                   (XY_df.month == 8)].groupby(['weekday', 'hour'], as_index=True)

statsHiver = groupedHiver['y'].aggregate([np.mean, np.min, np.max])
statsEte = groupedEte['y'].aggregate([np.mean, np.min, np.max])


# In[ ]:


print(statsHiver)


# In[ ]:


# On affiche des infos sur le profil pour les jours de la semaine
semaine = statsHiver.loc[1]  # 0 pour les jours de semaine
weekend = statsHiver.loc[0]  # 0 pour weekend

plt.plot(semaine['amin'], color='cyan')
plt.plot(semaine['mean'], color='blue')
plt.plot(semaine['amax'], color='cyan')
plt.show()


# ## Lien avec la consommation passée
# A l'aide de la fonction shift, pour un point horaire cible on regarde  :
# - la consommation de l'heure précédente, 
# - du jour précédent, 
# - de la semaine précédente.
# 

# In[ ]:


Xinput['lag1H'] = Yconso['y'].shift(1)
Xinput['lag1D'] = Yconso['y'].shift(24)
Xinput['lag1W'] = Yconso['y'].shift(24*7)


# In[ ]:


Xinput.head(24 * 7 + 1)


# On trace maintenant les nuages de points afin de voir s'il y a corrélation ou non :

# In[ ]:


def plot_scatter_load(var_x):
    plt.scatter(Xinput[var_x],Yconso['y'])
    plt.title(var_x)
    plt.show()


# In[ ]:


plot_scatter_load('lag1H')
plot_scatter_load('lag1D')
plot_scatter_load('lag1W')


# <font color='green'>
#     
# * Quelle(s) variable(s) vous semble(nt) pertinentes pour construire un modèle de prévision à J+1?
# 
# </font>

# ## Visualiser la consommation en fonction de la température 
# 
# On voudrait savoir si la consommation nationale peut s'expliquer en regardant simplement la température moyenne sur la France. Pour cela, on peut aussi tracer un nuage de points.

# In[ ]:


plt.scatter(Xinput['Th_prev'], Yconso['y'], alpha=0.2)
plt.show()


# <font color='green'>
#     
# * Que pensez-vous de ce nuage ? 
# * Quelles autres variables explicatives proposeriez-vous? De quels types sont-elles?
# 
# <font color='green'>

# # Construction d'un modèle prédictif naïf
# 
# <img src="pictures/hommeNaif.png" width=500 height=60>

# In[ ]:


# Pour se faire les dents on va considérer juste un point horaire
datetime_a_predire = pytz.utc.localize(datetime.datetime.strptime("2016-12-20_14:00", "%Y-%m-%d_%H:%M"))
y_true_point_horaire_cible = float(Yconso.loc[Yconso['ds'] == datetime_a_predire]['y'])

print("On veut predire la consommation du {}, soit {}".format(datetime_a_predire, y_true_point_horaire_cible))


# ## Première idée, un modèle naïf : pour l'heure qui nous intéresse, on plaque bêtement la valeur de consommation nationale de la veille

# On commence par juste notre point horaire

# In[ ]:


y_pred_modele_naif_1 = float(Xinput.loc[Xinput['ds'] == datetime_a_predire]['lag1D'])
pred_error = abs(y_true_point_horaire_cible - y_pred_modele_naif_1)

print("Modele 1 -- pred: {}, realisee: {}, erreur: {}%".format(y_pred_modele_naif_1, y_true_point_horaire_cible, pred_error/y_true_point_horaire_cible * 100))


# Voyons maintenant ce que ça donne non plus sur un unique point horaire mais sur l'ensemble des points horaires :

# In[ ]:


y_pred_modele_naif_1 = Xinput["lag1D"]

pred_error = (np.abs(Yconso["y"] - y_pred_modele_naif_1.loc[24:]) / Yconso["y"] * 100)

print(np.mean(pred_error))


# Bon c'est pas fou...

# ## Deuxième idée : modèle naïf avec de l'expertise métier 
# 
# Chez RTE, on considère qu'une augmentation moyenne de 1°C conduit à une augmentation de 2400MW de la consommation nationale pour des températures inférieures à 15°C. On propose donc comme consommation prévue la consommation de la veille, corrigée par 2400 fois l'écart à la température de la veille, si l'on n'excède pas les 15°C.
# 
# 
# <img src="pictures/ExpertJamy.jpg" width=500 height=60>

# In[ ]:


delta_MW_par_degre = 2400  
            
threshold_temperature = 15


# On commence par juste notre point horaire préféré

# In[ ]:


temperature_real_veille = float(Xinput.loc[Xinput['ds'] == datetime_a_predire]['Th_real_24h_avant'])
temperature_prevu_cible = float(Xinput.loc[Xinput['ds'] == datetime_a_predire]['Th_prev'])
delta_temp = min(threshold_temperature, temperature_prevu_cible) - min(threshold_temperature, temperature_real_veille)
delta_MW_because_temp = delta_temp * delta_MW_par_degre

y_pred_modele_naif_2 = float(Xinput.loc[Xinput['ds'] == datetime_a_predire]['lag1D']) - delta_MW_because_temp
pred_error = abs(y_true_point_horaire_cible - y_pred_modele_naif_2)

print("Modele 2 -- pred: {}, realisee: {}, erreur: {}%".format(y_pred_modele_naif_2, y_true_point_horaire_cible, pred_error/y_true_point_horaire_cible * 100))


# Et maintenant sur l'ensemble des points horaires :

# In[ ]:


y_pred = Xinput["lag1D"]

temp_prev_with_threshold = np.minimum([threshold_temperature], Xinput['Th_prev'].values)
temp_actual_with_threshold = np.minimum([threshold_temperature], Xinput['Th_real_24h_avant'].values)

delta_temp = temp_prev_with_threshold - temp_actual_with_threshold
delta_MW_because_temp = delta_temp * delta_MW_par_degre

y_pred_modele_naif_2 = Xinput["lag1D"] - delta_MW_because_temp
pred_error = (np.abs(Yconso["y"] - y_pred_modele_naif_2) / Yconso["y"] * 100)

print(np.mean(pred_error))


# Bon... Bien essayé avec ces modèles naïfs, mais maintenant on va être plus sérieux !

# # Préparer un jeu d'entrainement et un jeu de test
# En machine learning, il y a 2 types d'erreur que l'on peut calculer : l'erreur d'entrainement et l'erreur de test. 
# 
# Pour évaluer la capacité de notre modèle à bien généraliser sur de nouvelles données, il est très important de se préserver un jeu de test indépendant de celui d'entrainement.
# 
# Il faut donc segmenter notre dataset en 2 : 
# - un premier jeu servira pour l'entrainement, 
# - tandis que le second servira à mesurer les performances du modèle prédictif.

# In[ ]:


def prepareDataSetEntrainementTest(Xinput, Yconso, dateDebut, dateRupture, nbJourlagRegresseur=0):
    
    dateStart = Xinput.iloc[0]['ds']
    
    DateStartWithLag = dateStart + pd.Timedelta(str(nbJourlagRegresseur)+' days')  #si un a un regresseur avec du lag, il faut prendre en compte ce lag et commencer l'entrainement a la date de debut des donnees+ce lag
    XinputTest = Xinput[(Xinput.ds >= dateRupture)]    

    XinputTrain=Xinput[(Xinput.ds < dateRupture) & (Xinput.ds > DateStartWithLag) & (Xinput.ds > dateDebut)]
    YconsoTrain=Yconso[(Yconso.ds < dateRupture) & (Yconso.ds > DateStartWithLag) & (Yconso.ds > dateDebut)]
    YconsoTest=Yconso[(Xinput.ds >= dateRupture)]
    
    return XinputTrain, XinputTest, YconsoTrain, YconsoTest


# # Fonctions utilitaires

# Créons la fonction modelError qui va calculer pour un échantillon (Y, Y_hat) différents scores :
# - erreur relative moyenne (MAPE en %)
# - erreur relative max (en %)
# - RMSE (en MW)
# 
# Cette fonction est ensuite utilisée par les fonctions *evaluation* et *evaluation_par* qui nous permettront d'évaluer nos modèles.
# 

# In[ ]:


def modelError(Y, Yhat):

    Y = Y.reset_index(drop=True)
    
    relativeErrorsTest = np.abs((Y['y'] - Yhat) /Y['y']) 
    errorMean = np.mean(relativeErrorsTest)
    errorMax = np.max(relativeErrorsTest)
    rmse = np.sqrt(mean_squared_error(Y['y'], Yhat))
   
    return relativeErrorsTest, errorMean, errorMax, rmse


# In[ ]:


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


# In[ ]:


def evaluation_par(X, Y, Yhat,avecJF=True):
    Ytmp = Y
    Ytmp['weekday'] = Ytmp.ds.dt.weekday
    Ytmp['hour'] = Ytmp.ds.dt.hour
    if(avecJF):
        Ytmp['JoursFeries'] = X['JoursFeries']
    Ytmp['APE'] = np.abs(Ytmp['y']-Yhat)/Ytmp['y']
    dataWD = Ytmp[['weekday','APE']]
    groupedWD = dataWD.groupby(['weekday'], as_index=True)
    statsWD = groupedWD.aggregate([np.mean])
    dataHour = Ytmp[['hour','APE']]
    groupedHour = dataHour.groupby(['hour'], as_index=True)
    statsHour = groupedHour.aggregate([np.mean])
    
    if(avecJF):
        dataJF = Ytmp[['JoursFeries','APE']]
        groupedJF = dataJF.groupby(['JoursFeries'], as_index=True)
        statsJF = groupedJF.aggregate([np.mean])
    else:
        statsJF = None
    
    return statsWD, statsHour, statsJF


# # Feature engineering : préparation de Xinput
# 
# L'objectif de cette partie est d'enrichir Xinput à partir des données initiales. Il s'agit notamment d'exploiter les différentes informations calendaires disponibles.
# 
# Dans un premier temps, on supprime la consommation retardée d'une heure, non disponible pour notre exercice de prévision J+1.
# Et on vérifie le travail.
# 

# In[ ]:


Xinput = Xinput.drop(['lag1H'],axis=1)  


# In[ ]:


print(Xinput.shape)
print(Xinput.columns)


# Ensuite, on encode les données calendaires en one-hot encoding pour le modèle. Autrement dit, on construit pour chaque modalité, une variable binaire associée.
# Cet encodage est nécessaire pour que le modèle mathématique puisse appréhender la notion de date.

# In[ ]:


encodedWeekDay = pd.get_dummies(Xinput['weekday'],prefix="weekday")
encodedMonth = pd.get_dummies(Xinput['month'],prefix="month")
encodedHour = pd.get_dummies(Xinput['hour'],prefix="hour")


# In[ ]:


encodedWeekDay.head(3)


# In[ ]:


encodedMonth.head(3)


# In[ ]:


encodedHour.head(3)


# In[ ]:


Xinput = pd.concat([Xinput, encodedMonth, encodedWeekDay, encodedHour], axis=1)
Xinput = Xinput.drop(['month','weekday','hour'],axis=1)


# In[ ]:


print(Xinput.shape)
print(Xinput.columns)


# Pour les données météo, on récupère aussi la prévision effectuée pour la veille. 

# In[ ]:


colsToKeepWeather = [s for s in Xinput.columns.get_values() if 'Th_prev' in s]
lag_colsToKeepWeather = [ s + "_J_1" for s in colsToKeepWeather ]
Xinput[lag_colsToKeepWeather] = Xinput[colsToKeepWeather].shift(24)


# On récupère aussi le jour de l'année grâce à la fonction *dayofyear*, ainsi que les jours fériés. On crée une variable binaire associée à chaque jour férié.

# In[ ]:


#Récupération des jours fériés dans Xinput
time = pd.to_datetime(Xinput['ds'], yearfirst=True,utc=True)
Xinput['posan']= time.dt.dayofyear
encodedHolidays = pd.get_dummies(Xinput[['holiday']], prefix = "JF")
encodedHolidays['JoursFeries'] = encodedHolidays.sum(axis = 1)
Xinput = pd.concat([Xinput, encodedHolidays], axis = 1)
Xinput = Xinput.drop(['holiday'], axis = 1)


# On ajoute des températures seuillées, à 15°C pour l'effet chauffage, et à 18°C pour l'effet climatisation.

# In[ ]:


threshold_temperature_heat = 15
threshold_temperature_cool = 18

Xinput['temp_prev_with_threshold_heat'] = np.maximum(0, threshold_temperature_heat - Xinput['Th_prev'].values)
Xinput['temp_prev_with_threshold_cool'] = np.maximum(0, Xinput['Th_prev'].values - threshold_temperature_cool)


# In[ ]:


#affichage de toutes les variables de base
list(Xinput) #list plutôt que print pour avoir la liste complète


# Enfin, nous construisons les listes pour appeler plus rapidement les colonnes d'un même type.

# In[ ]:


colsToKeepWeather = [s for s in Xinput.columns.get_values() if 'Th_prev' in s]
colsToKeepMonth = [v for v in Xinput.columns.get_values() if 'month' in v]
colsToKeepWeekday = [v for v in Xinput.columns.get_values() if 'weekday' in v]
colsToKeepHour = [v for v in Xinput.columns.get_values() if 'hour' in v]
colsToKeepHolidays = [v for v in Xinput.columns.get_values() if 'JF_' in v]


# # Construction des jeux d'entrainement et de test
# 
# Pour éviter de construire un modèle qui apprend "par coeur" sur ses données, et qui disposerait alors d'une capacité de généralisation faible, il est d'usage courant de disposer de plusieurs jeux de données (de caractéristiques similaires). Le minimum est de construire un jeu d'entraînement, sur lequel on estime le modèle, et un jeu de test, sur lequel on va appliquer le modèle. Rapidement dit : un bon modèle est un modèle dont la capacité prédictive ne se dégrade pas trop sur le jeu test.
# 
# Pour cela, on crée la fonction *prepareDataSetEntrainementTest* qui va permettre de couper Y et Xinput en deux parties.

# In[ ]:


# on souhaite un jeu de test qui commence à partir du 1er mai 2017
dateDebut = pytz.utc.localize( datetime.datetime(year=2014, month=1, day=15))#pour éviter les NaN dans le jeu de données
dateRupture = pytz.utc.localize(datetime.datetime(year=2017, month=12, day=1))#début du challenge prevision de conso
nbJourlagRegresseur = 0


# In[ ]:


Yconso.tail()


# In[ ]:


XinputTrain, XinputTest, YconsoTrain, YconsoTest = prepareDataSetEntrainementTest(Xinput, Yconso, 
                                                                                  dateDebut, dateRupture, 
                                                                                  nbJourlagRegresseur)


# In[ ]:


print('la taille de l échantillon XinputTrain est:' + str(XinputTrain.shape))
print('la taille de l échantillon XinputTest est:' + str(XinputTest.shape))
print('la taille de l échantillon YconsoTrain est:' + str(YconsoTrain.shape))
print('la taille de l échantillon YconsoTest est:' + str(YconsoTest.shape))
print("la proportion de data d'entrainement est de:" + str(YconsoTrain.shape[0] / (YconsoTrain.shape[0] + YconsoTest.shape[0])))


# ## Test des fonctions sur le modèle expert
# 
# Recalculons les prévisions à l'aide des deux modèles experts et évaluons ces deux approches :
# 
# ### Modèle naïf 1

# In[ ]:


pred_train_naif1= XinputTrain["lag1D"]
pred_test_naif1= XinputTest["lag1D"]
evaluation(YconsoTrain, YconsoTest,  pred_train_naif1.values,pred_test_naif1.values)


# ### Modèle naïf 2

# In[ ]:


# prévision train
temp_prev_with_threshold = np.minimum([threshold_temperature], XinputTrain['Th_prev'].values)
temp_actual_with_threshold = np.minimum([threshold_temperature], XinputTrain['Th_real_24h_avant'].values)

delta_temp = temp_prev_with_threshold - temp_actual_with_threshold
delta_MW_because_temp = delta_temp * delta_MW_par_degre

pred_train_naif2 = XinputTrain["lag1D"] - delta_MW_because_temp


# prévision test
temp_prev_with_threshold = np.minimum([threshold_temperature], XinputTest['Th_prev'].values)
temp_actual_with_threshold = np.minimum([threshold_temperature], XinputTest['Th_real_24h_avant'].values)

delta_temp = temp_prev_with_threshold - temp_actual_with_threshold
delta_MW_because_temp = delta_temp * delta_MW_par_degre
pred_test_naif2 = XinputTest["lag1D"] - delta_MW_because_temp

# scores
evaluation(YconsoTrain, YconsoTest,  pred_train_naif2.values,pred_test_naif2.values)


# 
# # Régression linéaire simple
# 
# Le modèle naïf avec expertise métier a été inspiré de la forme de la courbe d'évolution de la consommation en fonction de la température en France. 
# Pour rappel:

# In[ ]:


plt.scatter(Xinput['Th_prev'], Yconso['y'], alpha=0.2)
plt.show()


# La consommation pourrait être modélisée par une fonction linéaire par morceaux de la température, avec une pente plus importante pour les températures froides que pour les températures élevées. Au lieu de fixer les gradients à 2400MW/°C et 0, ceux-ci pourraient être calibrés à partir des données.
# 

# ## Entrainer un modèle
# Notre modèle a des paramètres qu'il va falloir maintenant apprendre au vu de notre jeu d'entrainement. Il faut donc caler notre modèle sur ce jeu d'entrainement.

# In[ ]:


colsLR_simple = np.concatenate(([s for s in XinputTrain.columns.get_values() if 'temp_prev_with_' in s], colsToKeepHour, colsToKeepWeekday, colsToKeepMonth))

mTrain = linear_model.LinearRegression(fit_intercept = False)
mTrain.fit(XinputTrain[colsLR_simple], YconsoTrain[['y']])


# ## Interpréter le modèle 

# In[ ]:


coef_joli = pd.DataFrame(np.concatenate(( np.array([colsLR_simple]).T,mTrain.coef_.T),axis=1),columns = ['variable','coefficient'])
print(coef_joli)


# <font color='green'>
# 
# * Commentez les coefficients de régression obtenus. 
# * Comparez notamment les gradients obtenus avec le modèle naïf.
# 
# </font>

# ## Faire des prédictions
# Une fois qu'un modèle de prévision est entrainé, il ne s'avère utile que s'il est performant sur de nouvelles situations. Faisons une prévision sur notre jeu de test. Traçons les courbes obtenues et calculons les scores.

# In[ ]:


forecastTest = np.concatenate(mTrain.predict(XinputTest[colsLR_simple]))
forecastTrain = np.concatenate(mTrain.predict(XinputTrain[colsLR_simple]))


# In[ ]:


plt.scatter(forecastTest, YconsoTest[['y']])
plt.show()

plt.plot(YconsoTest['ds'], YconsoTest['y'], 'b', YconsoTest['ds'], forecastTest, 'r')
plt.show()


# In[ ]:


evaluation(YconsoTrain, YconsoTest, forecastTrain,  forecastTest)
evalWD,evalHour,evalJF = evaluation_par(XinputTest,YconsoTest,forecastTest,avecJF=True)
print(str(round(evalWD*100,1)))
print(str(round(evalHour*100,1)))
print(str(round(evalJF*100,1)))


# ### Comment se distribue l'erreur ?

# In[ ]:


erreur_relative_test, erreur_moyenne_test, erreur_max_test, rmse = modelError(YconsoTest, forecastTest)


# In[ ]:


num_bins = 100
plt.hist(erreur_relative_test, num_bins)
plt.show()


# ### A quel moment se trompe-t-on le plus ?

# In[ ]:


plt.plot(YconsoTest['ds'], erreur_relative_test, 'r')
plt.title("erreur relative sur la periode de test")
plt.show()


# In[ ]:


erreur_relative_test.head()


# In[ ]:


threshold = 0.18

mask = (erreur_relative_test >= threshold)
erreurs_df = pd.DataFrame(np.concatenate((YconsoTest[['ds','y']],np.array([forecastTest]).T),axis=1),columns=["date","y","prev"])
print(erreurs_df[mask])


# Au vu des résultats précédents :
# <font color= 'green'>
# 
# - que pensez-vous du modèle?
# - comment se distribue l'erreur?
# - quand se trompe-t-on le plus?
# - quelles variables explicatives ajouter?
# 
# </font>

# # Autres modèles : RandomForest et XGBoost

# ## Modèle RandomForest
# 
# <img src="pictures/randomForestExplain.png" width=500 height=30>

# ### Choix des données d'entrée

# In[ ]:


colsRF = np.concatenate((['lag1D','lag1W'],
                         colsToKeepWeather,colsToKeepMonth,colsToKeepWeekday,colsToKeepHour,colsToKeepHolidays))
list(colsRF)


# ### Entrainement du modèle

# In[ ]:


# La cellule peut prendre un peu de temps à exécuter
print(Xinput.head(20))
rfTrain = RandomForestRegressor(n_estimators=30, max_features=colsRF.size, n_jobs=3, oob_score = True, bootstrap = True)
rfTrain.fit(XinputTrain[colsRF], YconsoTrain['y'])


# <font color='green'>
# 
# * Grâce à l'aide de la fonction, déterminer les paramètres de cette méthode
# 
# </font>

# ### Prediction

# In[ ]:


forecastTest = rfTrain.predict(XinputTest[colsRF])
forecastTrain = rfTrain.predict(XinputTrain[colsRF])


# ### Evaluation

# In[ ]:


evaluation(YconsoTrain, YconsoTest, forecastTrain, forecastTest)

# on visualise nos previsions par rapport a la realité
plt.plot(YconsoTest['ds'], YconsoTest['y'], 'b', YconsoTest['ds'], forecastTest, 'r')
plt.show()

print('R^2 Training Score: {:.2f} \nOOB Score: {:.2f} \nR^2 Validation Score: {:.2f}'.format(rfTrain.score(XinputTrain[colsRF], YconsoTrain['y']), 
                                                                                             rfTrain.oob_score_,
rfTrain.score(XinputTest[colsRF], YconsoTest['y'])))


# In[ ]:


evalWD,evalHour,evalJF = evaluation_par(XinputTest,YconsoTest,forecastTest)
print(str(round(evalWD*100,1)))
print(str(round(evalHour*100,1)))
print(str(round(evalJF*100,1)))


# ## Modèle xgboost
# 
# <img src="pictures/XGboost.png" width=500 height=30>

# In[ ]:


xgbTrain = xgb.XGBRegressor(objective='reg:squarederror')
xgbTrain.fit(XinputTrain[colsRF], YconsoTrain['y'].values)
forecastTestXGB = xgbTrain.predict(XinputTest[colsRF])
forecastTrainXGB = xgbTrain.predict(XinputTrain[colsRF])


# <font color='green'>
# 
# * Grâce à l'aide de la fonction, déterminer les paramètres de cette méthode
# 
# </font>

# In[ ]:


evaluation(YconsoTrain, YconsoTest, forecastTrainXGB, forecastTestXGB)


# In[ ]:


evalWD,evalHour,evalJF = evaluation_par(XinputTest, YconsoTest, forecastTestXGB)
print(str(round(evalWD * 100,1)))
print(str(round(evalHour * 100,1)))
print(str(round(evalJF * 100,1)))


# # Bonus: à vous de jouer
# 
# Bravo ! Vous avez déjà créé un premier modèle performant pour faire des prévisions sur une fenêtre glissante à horizon 24h !
# 
# Maintenant à vous de mettre votre expertise pour créer de nouveaux modèles.
# 
# Vous pouvez continuer à explorer le problème selon plusieurs axes:
# - choix de la méthode de modélisation
# - apprendre votre modèle sur une période différente
# - créer de nouvelles variables explicatives ? Quid de la météo et de la température? Des jours fériés ? Du feature engineering plus complexe...
# - détecter des outliers dans les données
# 
# Mettez-vous en 3 groupes, explorez pendant 30 minutes, et restituez.

# In[ ]:





# In[ ]:




