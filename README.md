# Data Science × Logistic Regression

> Harry Potter and the Data Scientist

---

## Sommaire

- [Data Science × Logistic Regression](#data-science--logistic-regression)
  - [Sommaire](#sommaire)
  - [1. Avant-propos](#1-avant-propos)
  - [2. Introduction](#2-introduction)
  - [3. Objectifs](#3-objectifs)
  - [4. Instructions générales](#4-instructions-générales)
    - [4.1 Setup et Usage](#41-setup-et-usage)
  - [5. Mandatory Part](#5-mandatory-part)
    - [5.1 Data Analysis](#51-data-analysis)
    - [5.2 Data Visualization](#52-data-visualization)
      - [5.2.1 Histogram](#521-histogram)
      - [5.2.2 Scatter plot](#522-scatter-plot)
      - [5.2.3 Pair plot](#523-pair-plot)
    - [5.3 Logistic Regression](#53-logistic-regression)
  - [6. Bonus Part](#6-bonus-part)
  - [7. Soumission et évaluation par les pairs](#7-soumission-et-évaluation-par-les-pairs)

----

## 1. Avant-propos

<details>
<summary><strong>Biographie de Yann Le Cun</strong></summary>

Voici ce que dit Wikipédia à propos de Yann Le Cun, l'un des pères fondateurs de l'IA :

**Yann Le Cun** est né près de Paris, en France, en _1960_. Il est chercheur en intelligence
artificielle et en vision par ordinateur (robotique). **Il est considéré comme l'un des
inventeurs de l'apprentissage profond**.

Titulaire d'une licence de l'ESIEE Paris obtenue en 1983, il est diplômé de l'université Pierre
et Marie Curie et a obtenu un doctorat en 1987, au cours duquel il a proposé une
première version de **l'algorithme d'apprentissage par rétropropagation** couramment utilisé par les
algorithmes d'optimisation par descente de gradient pour ajuster le poids des neurones en calculant
le gradient de la fonction de perte. Il a été chercheur postdoctoral associé dans le laboratoire
de Geoffrey Hinton à l'université de Toronto de 1987 à 1988.

Depuis les années 1980, _Yann Le Cun travaille sur l'apprentissage automatique_,
la vision par ordinateur et l'apprentissage profond : la capacité d'un ordinateur à
reconnaître des représentations (images, textes, vidéos, sons) en les exposant de manière
répétée à des échantillons d'entraînement.

En 1987, Yann Le Cun rejoint l'université de Toronto, puis, en 1988, les laboratoires de
recherche d'AT&T, où il développe les méthodes d'apprentissage supervisé.

Yann Le Cun est également l'un des principaux créateurs de la technologie de compression
d'images DjVu (avec Léon Bottou et Patrick Haffner).

Yann Le Cun est professeur à l'université de New York où il a créé le
Center for Data Science. Il travaille notamment sur le développement
technologique des voitures autonomes. Il est l'auteur de nombreux ouvrages, dont
« Deep Learning » (2015) et « Neural Networks and Machine Learning » (2013).

Le 9 décembre 2013, Yann Le Cun a été invité par Mark Zuckerberg à rejoindre Facebook
pour concevoir et diriger le laboratoire d'intelligence artificielle FAIR (« Facebook Artificial Intelligence
Research ») à New York, Menlo Park et depuis 2015 à Paris, afin de travailler sur la reconnaissance d'images.
 Il avait auparavant refusé une proposition similaire de Google.

En 2016, il a été professeur invité d'informatique à la « Chaire Annuelle
Informatique et Sciences Numériques » au Collège de France à Paris.
</details>

----

## 2. Introduction

Oh non ! Depuis sa création, la célèbre école de sorciers, Poudlard, n'avait jamais connu une telle
offense. **Les forces du mal ont ensorcelé le Choixpeau**. Il ne répond plus et
est incapable de remplir son rôle qui consiste à répartir les élèves dans les différentes maisons.

La nouvelle année scolaire approche. Heureusement, le professeur McGonagall a pu
prendre des mesures dans une situation aussi stressante, car il est impossible pour Poudlard de ne pas
accueillir de nouveaux élèves... _Elle a décidé de faire appel à vous, un « data scientist » moldu qui est
capable de créer des miracles avec l'outil que tous les Moldus savent utiliser : un « ordinateur »_.

Malgré la réticence intrinsèque de nombreux sorciers, le directeur de l'école vous accueille
dans son bureau pour vous expliquer la situation. Vous êtes ici parce que son informateur a découvert
que **vous êtes capable de recréer un Choixpeau magique à l'aide de vos outils moldus**. Vous lui expliquez
que pour que vos outils « moldus » fonctionnent, vous avez besoin des données des élèves. Hésitante,
le professeur McGonagall vous remet un livre de sorts poussiéreux. Heureusement pour vous, un simple
« Digitalis ! » suffit pour que le livre se transforme en clé USB.

----

## 3. Objectifs

Dans ce projet _Data Science × Régression logistique_, vous poursuivrez votre exploration du
_Machine Learning_ en découvrant différents outils.

L'utilisation du terme « science des données » dans le titre peut être considérée par certains comme excessive.
C'est vrai. Nous ne prétendons pas vous donner toutes les bases de la science des données dans ce sujet.
Le sujet est vaste. Nous ne couvrirons que certaines bases qui, selon nous, sont utiles pour l'exploration des données
avant de les envoyer à l'algorithme d'apprentissage automatique.

Vous allez mettre en œuvre un **modèle de classification linéaire** dans le prolongement du sujet sur la régression linéaire :
une régression logistique. Nous vous encourageons également à créer une boîte à outils d'apprentissage automatique
au fur et à mesure de votre progression.

En résumé :

- Vous apprendrez à lire un ensemble de données, à le visualiser de différentes manières, ainsi qu'à sélectionner et
nettoyer les informations inutiles de vos données.
- Vous allez entraîner un modèle de régression logistique qui résoudra un problème de classification.

----

## 4. Instructions générales

**Vous pouvez utiliser le langage de votre choix**. Cependant, nous vous recommandons de choisir un langage
disposant d'une bibliothèque facilitant le tracé et le calcul des propriétés statistiques d'un ensemble de données.

> [!IMPORTANT]
> Toute méthode qui effectue tout le travail à votre place (par exemple, la méthode describe()
> d'un DataFrame pandas) sera considérée comme de la triche.

### 4.1 Setup et Usage

🔹 <strong>Étape 1 : Créer un environnement virtuel</strong>

```bash
# Créer l'environnement
python3 -m venv venv

# Activer l'environnement
source venv/bin/activate

# Desactiver l'environnement
deactivate
```

🔹 <strong>Étape 2 : Installer les dépendances</strong>

Mettre a jour pip :
```bash
pip install --upgrade pip
```


```bash
pip install -r requirements.txt
```

Vérifier l’installation :

```bash
pip list
```

Geler les versions :

```bash
pip freeze > requirements.txt
```

----

## 5. Mandatory Part

> [!NOTE]
> Il est fortement recommandé d'effectuer les étapes dans l'ordre suivant.

### 5.1 Data Analysis

> [!IMPORTANT]
> Nous allons voir quelques étapes fondamentales de l'exploration des données. Bien sûr, ce
> ne sont pas les seules techniques disponibles ni les seules étapes à suivre.
> Chaque ensemble de données et chaque problème doivent être abordés de manière unique. Vous
> trouverez certainement d'autres façons d'analyser vos données à l'avenir.

Tout d'abord, examinez les données disponibles. Regardez sous quel format elles sont présentées,
s'il existe différents types de données, les différentes plages, etc. Il est important de
vous faire une idée de votre matière première avant de commencer. Plus vous travaillerez avec les données, plus
vous développerez une intuition sur la manière dont vous pourrez les utiliser.

Dans cette partie, le professeur McGonagall vous demande de créer un programme appelé `describe.[extension]`.<br>
Ce programme prendra un ensemble de données comme paramètre. Il devra simplement afficher les informations
relatives à toutes les caractéristiques numériques, comme dans l'exemple suivant :

> [!IMPORTANT]
> Il est interdit d'utiliser toute fonction qui effectue le travail à votre place, telle que :
> count, mean, std, min, max, percentile, etc., quel que soit le
> langage que vous utilisez. Bien sûr, il est également interdit d'utiliser la
> bibliothèque describe ou toute fonction qui lui ressemble (plus ou moins)
> provenant d'une autre bibliothèque.

----

### 5.2 Data Visualization

La visualisation des données est un outil puissant pour un data scientist. Elle vous permet d'obtenir des informations
et de développer une intuition sur l'aspect de vos données. La visualisation de vos données vous permet également
de détecter des défauts ou des anomalies.

Dans cette section, **vous devez créer un ensemble de scripts**, chacun utilisant une méthode de visualisation particulière
pour répondre à une question. Il n'y a pas nécessairement une seule réponse à la
question.

#### 5.2.1 Histogram

Créez un script appelé **`histogram.[extension]`** qui affiche un histogramme répondant à la
question suivante :

    - Quel cours à Poudlard présente une répartition homogène des notes entre les quatre maisons ?

#### 5.2.2 Scatter plot

Créez un script appelé **`scatter_plot.[extension]`** qui affiche un nuage de points répondant à la question suivante :

    - Quelles sont les deux caractéristiques qui sont similaires ?

#### 5.2.3 Pair plot

Créez un script appelé **`pair_plot.[extension]`** qui affiche un graphique en paires ou un nuage de points
(selon la bibliothèque que vous utilisez).

    - À partir de cette visualisation, quelles caractéristiques allez-vous utiliser pour votre régression logistique ?

----

### 5.3 Logistic Regression

Vous arrivez à _la dernière partie_ : c**oder votre Magic Hat**. Pour ce faire, vous devez _effectuer un
multi-classificateur_ à l'aide d'une régression logistique one-vs-all (one-vs-rest).

Vous devrez créer deux programmes :
- Le premier servira à entraîner vos modèles et s'appelle **`logreg_train.[extension`**].
Il prend **dataset_train.csv** comme paramètre. _Pour la partie obligatoire, vous devez
utiliser la technique de descente de gradient afin de minimiser l'erreur_. Le programme génère
un fichier contenant les poids qui seront utilisés pour la prédiction.
- Le second doit être nommé **`logreg_predict.[extension]`**. Il prend dataset_test.csv
comme paramètre et un fichier contenant les poids entraînés par le programme précédent.

Afin d'évaluer les performances de votre classificateur, ce deuxième programme devra
générer un fichier de prédiction houses.csv formaté exactement comme suit :

```bash
$> cat houses.csv
Index,Hogwarts House
0,Gryffindor
1,Hufflepuff
2,Ravenclaw
3,Hufflepuff
4,Slytherin
5,Ravenclaw
6,Hufflepuff
[...]
```

----

## 6. Bonus Part

Il existe de nombreux bonus intéressants qui peuvent être ajoutés à ce sujet. Voici quelques
suggestions :

- Ajouter d'autres champs pour **`describe.[extension]`**
- Implémente **stochastic gradient descent**
- Implémenter d'autres algorithmes d'optimisation (GD par lots, GD par mini-lots ou autres)

> [!IMPORTANT]
> La partie bonus ne sera évaluée que si la partie obligatoire est
> PARFAITE. Parfaite signifie que la partie obligatoire a été entièrement
> réalisée et fonctionne sans aucun problème. Si vous n'avez pas satisfait à TOUTES
> les exigences obligatoires, votre partie bonus ne sera pas évaluée.

----

## 7. Soumission et évaluation par les pairs

Remettez votre devoir dans votre dépôt Git comme d'habitude. Seul le travail contenu dans votre dépôt
sera évalué lors de la soutenance. N'hésitez pas à vérifier les noms de vos
dossiers et fichiers pour vous assurer qu'ils sont corrects.

Pendant la correction, vous serez évalué sur votre travail rendu (aucune fonction ne fera
tout le travail à votre place), ainsi que sur votre capacité à présenter, expliquer et justifier vos
choix.

Votre classificateur sera évalué sur les données présentes dans dataset_test.csv. Vos réponses
seront évaluées à l'aide du score de précision de la bibliothèque Scikit-Learn. Le professeur
McGonagall estime que votre algorithme n'est comparable au Choixpeau magique que s'il obtient un
score de précision minimum de 98 %.

Il sera également important de pouvoir expliquer le fonctionnement des algorithmes d'apprentissage automatique
utilisés.
