# %% [markdown]
# <a href="https://colab.research.google.com/github/erikhhs/ads/blob/main/mini_datalab_antwoorden.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Voorspel of een tumor goedaardig is of kwaadaardig

# %% [markdown]
# ## Importeren toolset
# - De tools die we gebruiken moeten we eerst importeren voordat we ze kunnen gebruiken.
# - Voer onderstaande cel uit.

# %%
#%matplotlib inline
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
from sklearn import tree as tree_plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import plot_confusion_matrix
def plot_feature_importances_cancer(model):
    n_features = cancer.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), cancer.feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)

# %% [markdown]
# ## 1 Laden van de borstkanker dataset
# - 10 kolommen met kenmerken van de tumor.
# - 1 kolom waarin staat of de tumor goedaardig is of kwaadaardig (0 = kwaadaardig, 1 = goedaardig). Dit is vastgesteld met een punctie.
# - Voer onderstaande cel uit om de dataset te laden en een dataframe ervan te maken.

# %%
df = pd.DataFrame(cancer.data, columns = cancer.feature_names)
top10 = ['smoothness error',
 'concavity error',
 'worst smoothness',
 'area error',
 'concave points error',
 'mean texture',
 'worst area',
 'worst radius',
 'worst texture',
 'mean concave points']
df = df[top10]
df['IsBenign'] = cancer.target
display(df)
print( {n: v for n, v in zip(['kwaadaardig', 'goedaardig'], np.bincount(cancer.target))})
print('1 in kolom IsBenign betekent dat de tumor goedaardig is')

# %% [markdown]
# ### 1.1 informatie over de dataset
# - Toon de informatie van de dataset in onderstaande cel.
# - Zijn er ontbrekende waardes?
# - Welke datatypes zijn er?

# %%
df.info()

# %% [markdown]
# ## 2 Interview je opdrachtgever
# Bereid een interview voor:
# - Bestudeer de dataset.
# - Schrijf op wat nog niet duidelijk is.
# - Maak vragen waarmee je kunt achterhalen welke doelen de opdrachtgever wil behalen.
# 
# Interview de opdrachtgever en noteer hier welke doelen de opdrachtgever wil behalen:
# 
# 
# 

# %% [markdown]
# ## 3 Data analyse
# Maak grafieken waarin je het volgende laat zien:
# - Het aantal regels dat goedaardig is en het aantal dat kwaadaardig is.
# - De informatie die ieder van de 10 kenmerken geeft voor het correct voorspellen van de aard van de tumor (10 grafieken)
# - Welk kenmerk geeft de meeste informatie om te kunnen voorspellen of een tumor goed- of kwaadaardig?

# %%
sns.catplot(kind = 'count', x = 'IsBenign', data=df)

# %%
def chart_distributions(data,by):
    cols = list(data.columns)
    for col in cols:
        if col != by:
            sns.displot(hue = by , x = col , data=data, kind = 'kde', fill = True)
            plt.show()

# %%
chart_distributions(df,'IsBenign')

# %% [markdown]
# ## Splitsen in x-variabelen en y-variabele
# 
# Met de x-variabelen proberen we de y-variabele zo goed mogelijk te voorspellen.
# 
# X-variabelen: de kenmerken van de tumor (mean radius, mean texture etc.).
# 
# y-variabele: de klasse die we willen voorspellen. In dit geval kwaadaardig of goedaardig, 0 of 1.
# 
# - Splits de data in df in X en y

# %%
X = df.drop('IsBenign', axis=1)
y = df['IsBenign']

# %% [markdown]
# ## Splitsen in training data en test data
# De training data gebruiken we om het model te trainen.
# 
# De test data gebruiken we om te bepalen hoe goed ons model kan voorspellen.
# 
# - Splits de data in X_test, X_train, y_test en y_train
# 

# %%
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state = 42)
print('X_train:', 'Aantal rijen =' ,X_train.shape[0], ',','Aantal kolommen =', X_train.shape[1], )
print('y_train: ', 'Aantal rijen =', y_train.shape[0], ',','Aantal kolommen =', 1 )
print('X_test:', 'Aantal rijen =' ,X_test.shape[0], ',','Aantal kolommen =', X_test.shape[1], )
print('y_test: ', 'Aantal rijen =', y_test.shape[0], ',','Aantal kolommen =', 1 )

# %% [markdown]
# ## Trainen van het model
# - Train een beslisboom met behulp van een DecisionTreeClassifier

# %%
tree = DecisionTreeClassifier(random_state = 42)
tree.fit(X_train, y_train)

# %% [markdown]
# ## Visualisatie van het model
# - Visualeer de beslisboom.
# - Leg uit aan je opdrachtgever uit hoe deze beslisboom gebruikt kan worden in haar / zijn dagelijkse werkzaamheden.

# %%
fig = plt.figure(figsize=(140,60))
_ = tree_plt.plot_tree(tree, 
                   feature_names=cancer.feature_names,  
                   class_names=cancer.target_names,
                   filled=True, impurity = 'true', fontsize = 60)

# %% [markdown]
# ## Voorspellen met de getrainde beslisboom
# - Gebruik de beslisboom om te voorspellen wat de aard van de tumoren is. Gebruik hiervoor X_test.
# - Hoeveel procent van je voorspellingen is juist?

# %%
predictions = tree.predict(X_test)
display('Onze voorspellingen:', predictions[:])
display('De werkelijkheid:', np.array(y_test))
print("Percentage juist voorspeld: {:.2f} ".format( tree.score(X_test, y_test)*100))

# %% [markdown]
# ## Analyseren van de voorspellingen
# - Maak een confusion matrix.
# - Hoeveel kwaadaardige tumoren zitten er in de test set?
# - Hoeveel daarvan heeft het model gevonden?
# 
# De recall score wordt als volgt berekend: het aantal kwaadaardige tumoren in de test set / het aantal gevonden kwaadaardige tumoren.
# - Hoe hoog is de recall score?
# - Waarom is in deze situatie de recall score van belang?

# %%
plot_confusion_matrix(tree, X_test, y_test,display_labels=['kwaadaardig', 'goedaardig'], cmap=plt.cm.Blues, normalize=None)
plt.show()


