License CC BY-NC-ND, Valérie Roy & Thierry Parmentelat


```python
from IPython.display import HTML
HTML(filename="_static/style.html")
```

# TP on the moon

**Notions intervenant dans ce TP**

* suppression de colonnes avec `drop` sur une `DataFrame`
* suppression de colonne entièrement vide avec `dropna` sur une `DataFrame`
* accès aux informations sur la dataframe avec `info`
* valeur contenues dans une `Series` avec `unique` et `value_counts` 
* conversion d'une colonne en type numérique avec `to_numeric` et `astype` 
* accès et modification des chaînes de caractères contenues dans une colonne avec l'accesseur `str` des `Series`
* génération de la liste Python des valeurs d'une série avec `tolist`
   
**N'oubliez pas d'utiliser le help en cas de problème.**

**Répartissez votre code sur plusieurs cellules**

1. importez les librairies `pandas` et `numpy`


```python
# votre code
import numpy as np
import pandas as pd
```

2. 1. lisez le fichier de données `data/objects-on-the-moon.csv`
   2.  affichez sa taille et regardez quelques premières lignes


```python
# votre code
df = pd.read_csv('data/objects-on-the-moon.csv')
```


```python
df.index
```


```python
df.head()
```

3. 1. vous remarquez une première colonne franchement inutile  
     utiliser la méthode `drop` des dataframes pour supprimer cette colonne de votre dataframe  
     `pd.DataFrame.drop?` pour obtenir de l'aide


```python
# votre code
df.drop(df.columns[0], axis=1, inplace=True)
df
```

4. 1. appelez la méthode `info` des dataframes  
   (`non-null` signifie `non-nan` i.e. non manquant)
   1. remarquez une colonne entièrement vide


```python
# votre code
df.info()
```

5. 1. utilisez la méthode `dropna` des dataframes  
     pour supprimer *en place* les colonnes qui ont toutes leurs valeurs manquantes  
     (et pas uniquement la colonne `'Size'`)
   2. vérifiez que vous avez bien enlevé la colonne `'Size'`


```python
# votre code
df.dropna(axis=1, how='all', inplace=True)
df
```

6. 1. affichez la ligne d'`index` $88$, que remarquez-vous ?
   2. toutes ses valeurs sont manquantes  
     utilisez la méthode `dropna` des dataframes  
     pour supprimer *en place* les lignes qui ont toutes leurs valeurs manquantes
     (et pas uniquement la ligne d'index $88$)


```python
# votre code
df.loc[88]
```


```python
df.dropna(axis=0, how='all', inplace=True)
```

7. 1. utilisez l'attribut `dtypes` des dataframes pour voir le type de vos colonnes
   2. que remarquez vous sur la colonne des masses ?


```python
# votre code
df.dtypes
```

La colonne des masses n'est pas de type flottant

8. 1. la colonne des masses n'est pas de type numérique mais de type `object`  
      (ici des `str`)   
   1. utilisez la méthode `unique` des `Series`pour en regarder le contenu
   2. que remarquez vous ?


```python
# votre code
df['Mass (lb)'].unique()
```

il n'y a pas que des nombres, mais aussi des éléments de comparaison ('<12787' par ex)

9. 1. conservez la colonne `'Mass (lb)'` d'origine  
      (par exemple dans une colonne de nom `'Mass (lb) orig'`)  
   1. utilisez la fonction `pd.to_numeric` pour convernir  la colonne `'Mass (lb)'` en numérique    
   (en remplaçant  les valeurs invalides par la valeur manquante)
   1. naturellement vous vérifiez votre travail en affichant le type de la série `df['Mass (lb)']`


```python
# votre code
df['Mass (lb) orig'] = df['Mass (lb)']
df
```


```python
df['Mass (lb)'] = pd.to_numeric('Mass (lb)', errors='coerce')
df['Mass (lb)'].dtype
```

10. 1. cette solution ne vous satisfait pas, vous ne voulez perdre aucune valeur  
       (même au prix de valeurs approchées)  
    1. vous décidez vaillamment de modifier les `str` en leur enlevant les caractères `<` et `>`  
       afin de pouvoir en faire des entiers
    - *hint*  
       les `pandas.Series` formées de chaînes de caractères sont du type `pandas` `object`  
       mais elle possèdent un accesseur `str` qui permet de leur appliquer les méthodes python des `str`  
       (comme par exemple `replace`)
        ```python
        df['Mass (lb) orig'].str
        ```
        remplacer les `<` et les `>` par des '' (chaîne vide)
     3. utilisez la méthode `astype` des `Series` pour la convertir finalement en `int` 


```python
# votre code
```

11. 1. sachant `1 kg = 2.205 lb`  
   créez une nouvelle colonne `'Mass (kg)'` en convertissant les lb en kg  
   arrondissez les flottants en entiers en utilisant `astype`


```python
# votre code
```

12. 1. Quels sont les pays qui ont laissé des objets sur la lune ?
    2. Combien en ont-ils laissé en pourcentage (pas en nombre) ?  
     *hint* regardez les paramètres de `value_counts`


```python
# votre code
```

13. 1. Quel est le poid total des objets sur la lune en kg ?
    2. quel est le poids total des objets laissés par les `United States`  ?


```python
# votre code
```

14. 1. quel pays a laissé l'objet le plus léger ?  
     *hint* comme il existe une méthode `min` des séries, il existe une méthode `argmin` 


```python
# votre code
```

15. 1. y-a-t-il un Memorial sur la lune ?  
     *hint*  
     en utilisant l'accesseur `str` de la colonne `'Artificial object'`  
     regardez si une des description contient le terme `'Memorial'`
    2. quel pays qui a mis ce mémorial ?  


```python
# votre code
```

16. 1. faites la liste Python des objets sur la lune  
     *hint*  
     utilisez la méthode `tolist` des séries


```python
# votre code
```

***
