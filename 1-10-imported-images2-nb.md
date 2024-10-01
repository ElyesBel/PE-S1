---
jupytext:
  cell_metadata_json: true
  encoding: '# -*- coding: utf-8 -*-'
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

Licence CC BY-NC-ND, Valérie Roy & Thierry Parmentelat

+++

# TP images (2/2)

merci à Wikipedia et à stackoverflow

**le but de ce TP n'est pas d'apprendre le traitement d'image  
on se sert d'images pour égayer des exercices avec `numpy`  
(et parce que quand on se trompe ça se voit)**

```{code-cell} ipython3
import numpy as np
from matplotlib import pyplot as plt
```

+++ {"tags": ["framed_cell"]}

````{admonition} → **notions intervenant dans ce TP**

* sur les tableaux `numpy.ndarray`
  * `reshape()`, masques booléens, *ufunc*, agrégation, opérations linéaires
  * pour l'exercice `patchwork`:  
    on peut le traiter sans, mais l'exercice se prête bien à l'utilisation d'une [indexation d'un tableau par un tableau - voyez par exemple ceci](https://ue12-p24-numerique.readthedocs.io/en/main/1-14-numpy-optional-indexing-nb.html)
  * pour l'exercice `sepia`:  
    ici aussi on peut le faire "naivement" mais l'utilisation de `np.dot()` peut rendre le code beaucoup plus court
* pour la lecture, l'écriture et l'affichage d'images
  * utilisez `plt.imread()`, `plt.imshow()`
  * utilisez `plt.show()` entre deux `plt.imshow()` si vous affichez plusieurs images dans une même cellule

  ```{admonition} **note à propos de l'affichage**
  :class: seealso dropdown admonition-small

  * nous utilisons les fonctions d'affichage d'images de `pyplot` par souci de simplicité
  * nous ne signifions pas là du tout que ce sont les meilleures!  
    par exemple `matplotlib.pyplot.imsave` ne vous permet pas de donner la qualité de la compression  
    alors que la fonction `save` de `PIL` le permet
  * vous êtes libres d'utiliser une autre librairie comme `opencv`  
    si vous la connaissez assez pour vous débrouiller (et l'installer), les images ne sont qu'un prétexte...
  ```
````

+++

## Création d'un patchwork

+++

1. Le fichier `data/rgb-codes.txt` contient une table de couleurs:
```
AliceBlue 240 248 255
AntiqueWhite 250 235 215
Aqua 0 255 255
.../...
YellowGreen 154 205 50
```
Le nom de la couleur est suivi des 3 valeurs de ses codes `R`, `G` et `B`  
Lisez cette table en `Python` et rangez-la dans la structure qui vous semble adéquate.

```{code-cell} ipython3
# votre code
filename = 'data/rgb-codes.txt'
colors = dict()
with open(filename, 'r') as file:
    for line in file:
        colname, *l = line.split() #l'* permet de dire 'tout ce qu'il y a après (3 éléments) tu le fous dans l' (sinon y'a 4 éléments pr 2 variables)
        colors[colname] = np.array([int(e) for e in l], dtype=np.uint8)
colors
```

2. Affichez, à partir de votre structure, les valeurs rgb entières des couleurs suivantes  
`'Red'`, `'Lime'`, `'Blue'`

```{code-cell} ipython3
# votre code
print('Red :', colors['Red'], '\n', 'Lime :', colors['Lime'], '\n', 'Blue :', colors['Blue'])
```

3. Faites une fonction `patchwork` qui  

   * prend une liste de couleurs et la structure donnant le code des couleurs RGB
   * et retourne un tableau `numpy` avec un patchwork de ces couleurs  
   * (pas trop petits les patchs - on doit voir clairement les taches de couleurs  
   si besoin de compléter l'image mettez du blanc

+++

````{admonition} indices
:class: dropdown
  
* sont potentiellement utiles pour cet exo:
  * la fonction `np.indices()`
  * [l'indexation d'un tableau par un tableau](https://ue12-p24-numerique.readthedocs.io/en/main/1-14-numpy-optional-indexing-nb.html)
* aussi, ça peut être habile de couper le problème en deux, et de commencer par écrire une fonction `rectangle_size(n)` qui vous donne la taille du patchwork en fonction du nombre de couleurs  
  ```{admonition} et pour calculer la taille au plus juste
  :class: tip dropdown

  en version un peu brute, on pourrait utiliser juste la racine carrée;
  par exemple avec 5 couleurs créer un carré 3x3 - mais 3x2 c'est quand même mieux !

  voici pour vous aider à calculer le rectangle qui contient n couleurs

  n | rect | n | rect | n | rect | n | rect |
  -|-|-|-|-|-|-|-|
  1 | 1x1 | 5 | 2x3 | 9 | 3x3 | 14 | 4x4 |
  2 | 1x2 | 6 | 2x3 | 10 | 3x4 | 15 | 4x4 |
  3 | 2x2 | 7 | 3x3 | 11 | 3x4 | 16 | 4x4 |
  4 | 2x2 | 8 | 3x3 | 12 | 3x4 | 17 | 4x5 |
  ```
````

```{code-cell} ipython3
# votre code
def rectangle_size(n):
    '''donne la taille du tableau nécessaire pour afficher le bon nombre de couleurs'''
    racine = np.sqrt(n)
    ptent = np.floor(racine) #partie entière de la racine
    if ptent**2 >= n: #si n est un carré parfait ou qu'un tableau carré "par défaut" existe
        return (int(ptent), int(ptent))
    elif ptent*(ptent+1) >= n: #si on peut avoir un tableau rectangulaire
        return (int(ptent), int(ptent+1))
    else:
        return (int(ptent+1), int(ptent+1)) #dernier cas de figure : on doit prendre un tableau carré plus grand
```

```{code-cell} ipython3
for i in range(1, 18):
    print(i, rectangle_size(i))
```

*ça a l'air ok*

```{code-cell} ipython3
from matplotlib import pyplot as plt
```

```{code-cell} ipython3
"""def patchwork(liste_couleurs):
    '''retourne le patchwork demandé, en complétant avec du blanc si besoin'''
    n = len(liste_couleurs)
    reste = rectangle_size(n)[0]*rectangle_size(n)[1] - n
    pattern = np.indices(rectangle_size(n)) #on crée le tableau qui va recueillir nos couleurs
    for i in range(reste):
        liste_couleurs = np.append(liste_couleurs, 'White') #on rajoute du blanc pour compléter le tableau
    colormap = np.array(colors[elem] for elem in liste_couleurs)
    plt.imshow(colormap[pattern])"""
```

Bon, visiblement ma fonction ne marche pas puisque je n'arrive pas à faire fonctionner le reste, et j'y ai déjà passé quelque temps... j'utilise dans la suite la fonction d'un camarade

```{code-cell} ipython3
def patchwork(liste_couleurs, dico):
    n = len(liste_couleurs)
    (L, l) = rectangle_size(n) #taille du array final
    (array1, array2) = np.indices((L, l))
    array = l*array1 + array2 #pour créer l'array qui a pour valeurs 0 à n en avançant linéairement par lignes
    temp = [dico[color] for i, color in enumerate(liste_couleurs)]
    temp += [[255,255,255]]*(L*l - n)
    colormap = np.array(temp, dtype=np.uint8)
    return colormap[array]
```

4. Tirez aléatoirement une liste de couleurs et appliquez votre fonction à ces couleurs.

```{code-cell} ipython3
# votre code
li = np.random.choice(list(colors.keys()), 10)
plt.imshow(patchwork(li, colors));
```

5. Sélectionnez toutes les couleurs à base de blanc et affichez leur patchwork  
même chose pour des jaunes

```{code-cell} ipython3
# votre code
liste_blanc = [couleur for couleur in colors.keys() if 'White' in couleur]
plt.imshow(patchwork(liste_blanc, colors));
```

```{code-cell} ipython3
liste_jaune = [couleur for couleur in colors.keys() if 'Yellow' in couleur]
plt.imshow(patchwork(liste_jaune, colors));
```

6. Appliquez la fonction à toutes les couleurs du fichier  
et sauver ce patchwork dans le fichier `patchwork.png` avec `plt.imsave`

```{code-cell} ipython3
# votre code
liste_tous = [cle for cle in colors.keys()]
tab = patchwork(liste_tous, colors)
plt.imshow(tab);
```

```{code-cell} ipython3
type(tab)
```

```{code-cell} ipython3
plt.imsave('patchwork.png', tab)
```

7. Relisez et affichez votre fichier  
   attention si votre image vous semble floue c'est juste que l'affichage grossit vos pixels

vous devriez obtenir quelque chose comme ceci

```{image} media/patchwork-all.jpg
:align: center
```

```{code-cell} ipython3
# votre code
im = plt.imread('patchwork.png')
plt.imshow(im);
```

## Somme dans une image & overflow

+++

0. Lisez l'image `data/les-mines.jpg`

```{code-cell} ipython3
# votre code
im = plt.imread('data/les-mines.jpg')
plt.imshow(im);
```

1. Créez un nouveau tableau `numpy.ndarray` en sommant **avec l'opérateur `+`** les valeurs RGB des pixels de votre image

```{code-cell} ipython3
np.shape(im)
```

```{code-cell} ipython3
# votre code
tab_plus = np.empty((533, 800, 1))
tab_plus[:,:,0] = im[:,:,0]+im[:,:,1]+im[:,:,2]
plt.imshow(tab_plus);
```

2. Regardez le type de cette image-somme, et son maximum; que remarquez-vous?  
   Affichez cette image-somme; comme elle ne contient qu'un canal il est habile de l'afficher en "niveaux de gris" (normalement le résultat n'est pas terrible ...)


   ```{admonition} niveaux de gris ?
   :class: dropdown tip

   cherchez sur google `pyplot imshow cmap gray`
   ```

```{code-cell} ipython3
# votre code
type(tab_plus)
```

```{code-cell} ipython3
np.max(tab_plus)
```

c'est un flottant et non un entier : + convertirait automatiquement les entiers en flottants par défaut ?

```{code-cell} ipython3
plt.imshow(tab_plus, cmap='grey');
```

3. Créez un nouveau tableau `numpy.ndarray` en sommant mais cette fois **avec la fonction d'agrégation `np.sum`** les valeurs RGB des pixels de votre image

```{code-cell} ipython3
#votre code
tab_sum = np.empty((533, 800, 1))
tab_sum[:,:,0] = np.sum(im, axis=2)
plt.imshow(tab_sum);
```

4. Comme dans le 2., regardez son maximum et son type, et affichez la

```{code-cell} ipython3
# votre code
np.max(tab_sum)
```

pas le même max, cette fois il n'est pas plafonné à 255, pourtant on code en uint8...

```{code-cell} ipython3
type(tab_sum)
```

```{code-cell} ipython3
plt.imshow(tab_sum, cmap='grey');
```

5. Les deux images sont de qualité très différente, pourquoi cette différence ? Utilisez le help `np.sum?`

+++

La différence réside dans la gestion des débordements : ici on code en uint8 ; quand on utilise +, le résultat de l'addition lorsqu'elle dépasse 255 est "repart de 0" pour le reste, d'où quelques pixels dont le niveau de gris apparapit comme anormal.
A l'inverse, np.sum() semble posséder une fonction de gestion des débordements qui permet d'éviter ce problème (qui se transcrit sur l'image qu'on affiche, donc) ; elle semble utiliser un type plus large que uint8 pour éviter les problèmes de débordement au moment du calcul, mais revient au type souhaité après coup, et doit "saturer" à 255 tout débordement, ce qui renvoie une image moins "aléatoire" et donc d'apparence plus nette.

+++

6. Passez l'image en niveaux de gris de type entiers non-signés 8 bits  
(de la manière que vous préférez)

```{code-cell} ipython3
# votre code
tab_gris_uint8 = tab_sum.astype(np.uint8)
plt.axis('off')
plt.imshow(tab_gris_uint8, cmap='grey');
```

7. Remplacez dans l'image en niveaux de gris,  
les valeurs >= à 127 par 255 et celles inférieures par 0  
Affichez l'image avec une carte des couleurs des niveaux de gris  
vous pouvez utilisez la fonction `numpy.where`

```{code-cell} ipython3
# votre code
result = np.where(tab_gris_uint8 >= 127, 255, 0)
plt.axis('off')
plt.imshow(result, cmap='grey');
```

8. avec la fonction `numpy.unique`  
regardez les valeurs différentes que vous avez dans votre image en noir et blanc

```{code-cell} ipython3
# votre code
val_differentes, count = np.unique(result, return_counts=True)
print(val_differentes, count)
```

## Image en sépia

+++

Pour passer en sépia les valeurs R, G et B d'un pixel  
(encodées ici sur un entier non-signé 8 bits)  

1. on transforme les valeurs `R`, `G` et `B` par la transformation  
`0.393 * R + 0.769 * G + 0.189 * B`  
`0.349 * R + 0.686 * G + 0.168 * B`  
`0.272 * R + 0.534 * G + 0.131 * B`  
(attention les calculs doivent se faire en flottants pas en uint8  
pour ne pas avoir, par exemple, 256 devenant 0)  
1. puis on seuille les valeurs qui sont plus grandes que `255` à `255`
1. naturellement l'image doit être ensuite remise dans un format correct  
(uint8 ou float entre 0 et 1)

+++

````{tip}
jetez un coup d'oeil à la fonction `np.dot` 
qui est si on veut une généralisation du produit matriciel

dont voici un exemple d'utilisation:
````

```{code-cell} ipython3
# exemple de produit de matrices avec `numpy.dot`
# le help(np.dot) dit: dot(A, B)[i,j,k,m] = sum(A[i,j,:] * B[k,:,m])

i, j, k, m, n = 2, 3, 4, 5, 6
A = np.arange(i*j*k).reshape(i, j, k)
B = np.arange(m*k*n).reshape(m, k, n)

C = A.dot(B)
# or C = np.dot(A, B)

print(f"en partant des dimensions {A.shape} et {B.shape}")
print(f"on obtient un résultat de dimension {C.shape}")
print(f"et le nombre de termes dans chaque `sum()` est {A.shape[-1]} == {B.shape[-2]}")
```

**Exercice**

+++

1. Faites une fonction qui prend en argument une image RGB et rend une image RGB sépia  
la fonction `numpy.dot` peut être utilisée si besoin, voir l'exemple ci-dessus

```{code-cell} ipython3
# votre code
def rgb_to_sepia(im):
    sepia_mat = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]]) #matrice de transformation en sépia
    sepia_im = np.dot(im[...,:3], sepia_mat.T) #on transpose la matrice sepia_mat pour faire le bon produit matriciel
    sepia_im = np.clip(sepia_im, 0, 255) #on s'assure que les valeurs sont dans [0, 255]
    return sepia_im.astype(np.uint8) #on les convertit en uint8
```

2. Passez votre patchwork de couleurs en sépia  
Lisez le fichier `patchwork-all.jpg` si vous n'avez pas de fichier perso

```{code-cell} ipython3
# votre code
patchwork = plt.imread('patchwork.png')
plt.imshow(patchwork)
plt.title('Patchwork originel')
plt.axis('off');
```

```{code-cell} ipython3
patchwork_s = rgb_to_sepia(patchwork)
plt.imshow(patchwork_s)
plt.title('Patchwork en sepia')
plt.axis('off');
```

Cela m'a l'air bizarre mais je ne sais pas ce qui était attendu (ça a l'air de fonctionner sur les-mines.jpg pourtant)

+++

3. Passez l'image `data/les-mines.jpg` en sépia

```{code-cell} ipython3
# votre code
lesmines_s = rgb_to_sepia(plt.imread('data/les-mines.jpg'))
plt.imshow(lesmines_s);
```

## Exemple de qualité de compression

+++

1. Importez la librairie `Image`de `PIL` (pillow)  
(vous devez peut être installer PIL dans votre environnement)

```{code-cell} ipython3
# votre code
from PIL import Image
```

2. Quelle est la taille du fichier `data/les-mines.jpg` sur disque ?

```{code-cell} ipython3
file = "data/les-mines.jpg"
```

```{code-cell} ipython3
# votre code
#j'imagine qu'on ne souhaite pas utiliser le module os
with open(file, 'rb') as im: #on lit en binaire
    contenu = im.read() #on lit le fichier en entier, le curseur arrive donc à la fin
    taille = len(contenu)
print('la taille du fichier est', taille, 'octets')
print(533*800)
```

3. Lisez le fichier 'data/les-mines.jpg' avec `Image.open` et avec `plt.imread`

```{code-cell} ipython3
# votre code
Image.open(file)
```

```{code-cell} ipython3
image = plt.imshow(plt.imread(file))
plt.axis('off');
```

4. Vérifiez que les valeurs contenues dans les deux objets sont proches

```{code-cell} ipython3
# votre code
im_pil = Image.open(file)
im_pil_array = np.array(im_pil)
im_plt = plt.imread(file) #on fait en sorte d'avoir 2 arrays
sont_proches = np.isclose(im_pil_array, im_plt, atol=0.1) #on regarde la "proximité" en faisant varier la tolérance
print(np.all(sont_proches)) #on vérifie si tous les points sont proches
```

5. Sauvez (toujours avec de nouveaux noms de fichiers)  
l'image lue par `imread` avec `plt.imsave`  
l'image lue par `Image.open` avec `save` et une `quality=100`  
(`save` s'applique à l'objet créé par `Image.open`)

```{code-cell} ipython3
# votre code
plt.imsave('imageplt.png', im_plt)
im_pil.save('imagepil.png', quality=100)
```

6. Quelles sont les tailles de ces deux fichiers sur votre disque ?  
Que constatez-vous ?

```{code-cell} ipython3
# votre code
with open('imageplt.png', 'rb') as im: #on lit en binaire
    contenu = im.read() #on lit le fichier en entier, le curseur arrive donc à la fin
    taille = len(contenu)
print('la taille du fichier est', taille, 'octets')
```

```{code-cell} ipython3
with open('imagepil.png', 'rb') as im: #on lit en binaire
    contenu = im.read() #on lit le fichier en entier, le curseur arrive donc à la fin
    taille = len(contenu)
print('la taille du fichier est', taille, 'octets')
```

l'image PLT prend relativement moins de place que celle plt ; alors même qu'elle semble mieux définie

+++

7. Relisez les deux fichiers créés et affichez avec `plt.imshow` leur différence

```{code-cell} ipython3
# votre code
imageplt = plt.imread('imageplt.png')
imagepil = plt.imread('imagepil.png')
plt.imshow(imageplt)
plt.axis('off')
plt.show()
plt.imshow(imagepil)
plt.axis('off')
plt.show();
```

j'ai du mal à voir la différence...
