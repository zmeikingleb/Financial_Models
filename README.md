# Financial Models – Projet EDP (Janvier 2026)

Ce projet présente un **outil de résolution d'équations aux dérivées partielles (EDP)** appliquées à la finance. L’objectif est de permettre la simulation et le pricing de différents modèles financiers à travers une interface professionnelle Python, visualisable dans **Streamlit** (Excel/Jupyter pour présentation graphique uniquement).

## Objectif

L’objet du projet est de présenter un outil capable de résoudre des EDP compatibles avec **l’équation fondamentale de la finance**, en utilisant l’algorithme de **Thomas** et la méthode générique présentée dans les slides du chapitre *"EDP – Application à la finance"*.

## Interface

- Réalisée en **Python** avec interface Streamlit professionnelle.  
- Tous les calculs sont effectués en Python ; seuls les graphiques peuvent être présentés dans Excel ou Jupyter.  
- L’application se lance avec :  
  `bash
  streamlit run app.py`

## Modèles Implémentés

Au minimum, les modèles suivants sont implémentés :

1. Black & Scholes (BS) : 

- Pricing d’options Vanilla et Exotiques (Knockout / Barrier).

 ### Visualisation :

- Heatmaps de V(t,x)

- Courbe V(t=0,x)

- Surface V(t,x)

**Possibilité de modifier les hyperparamètres : taux, volatilité, theta de Crank-Nicolson, Nt, Nx, maturité, etc.**

2. Vasicek :

- Comparaison de différentes valeurs de theta avec les mêmes visualisations que BS.

**Possibilité de faire varier tous les hyperparamètres.**

3. CIR (Cox-Ingersoll-Ross)

- Similaire à Vasicek : visualisations et variations des hyperparamètres.



**Projet réalisé individuellement.**

Tous les calculs sont faits en **Python.**
