# Projet Kaggle What's cooking

Projet des chefs

## JSON Parser

Comment le Parser va marcher 

exemple d'entrée JSON

```
{
 "id": 24717,
 "cuisine": "indian",
 "ingredients": [
     "tumeric",
     "vegetable stock",
     "tomatoes",
     "garam masala",
     "naan",
     "red lentils",
     "red chili peppers",
     "onions",
     "spinach",
     "sweet potatoes"
 ]
 },
```

##Traitement de la BDD

J'ai directement travaillé sur du pandas.
create_lexicon te crée les deux listes qui répertories tous les types de cuisines et d'ingrédients.
create_arrays te crée au final deux listes de liste et pas deux listes de tableaux parce que
j'avais envie de tester le rn rapidement mais faudrait le changer pour que ca soit plus rapide

A noter que le json test ne contient pas de target donc les listes tests utilisées pour tester
le réseaux proviennent enfait du train qui est partitionné à proportion de 80/20.

##RN

4 couches

rate te permet de gerer la proportion de dataset que tu veux utiliser pour les tests

et après code classique de RN

40% accuracy, 20% data : 1000,1000,500,500 dropout=0.9

Globalement ce qui à l'air d'être important c'est le nombre de neurones par couches, exemple :
RN 5 couches : 2000*5 moins performant sur les 10 premieres epochs qu'un RN 3 couches
3000/3000/2000 sur l'ensemble des données (7 eme epoch 57% je suis pas allé plus loin c'était long et je
voulais test d'autres choses)



