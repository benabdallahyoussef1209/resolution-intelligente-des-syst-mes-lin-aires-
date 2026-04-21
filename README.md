# 🔢 Résolution de Systèmes Linéaires (Gauss-Seidel, LU, Cramer)

## 📌 Description
Ce projet implémente plusieurs méthodes numériques pour résoudre des systèmes linéaires \( Ax = b \).  
Le programme choisit automatiquement la méthode la plus adaptée selon les propriétés de la matrice.

---

## 🚀 Méthodes implémentées

Le projet contient 3 méthodes principales :

### 1️⃣ Cramer
Utilisé pour les petits systèmes (n ≤ 3).

### 2️⃣ Décomposition LU
Décomposition de la matrice A en L et U, puis résolution en deux étapes :
- Ly = b (substitution avant)
- Ux = y (substitution arrière)

### 3️⃣ Gauss-Seidel
Méthode itérative efficace pour :
- matrices à diagonale dominante
- matrices symétriques définies positives

---

## 🧠 Choix automatique de la méthode

Le programme sélectionne automatiquement :

- ✔️ Gauss-Seidel si :
  - matrice diagonale dominante
  - ou matrice symétrique définie positive

- ✔️ Cramer si n ≤ 3

- ✔️ LU sinon

---
## 🚀 Live Demo
[https://ton-app.streamlit.app](https://hik8xgpjde7uqwf3wer7ut.streamlit.app/)

## 📊 Exemple de données

```python
A = [[2, 1, 0, 0],
     [1, 3, 1, 0],
     [0, 1, 4, 1],
     [0, 0, 1, 2]]

b = [4, 10, 18, 11]
▶️ Exécution
python main.py
🛠️ Technologies utilisées
Python 🐍
NumPy 🔢
📌 Fonctionnalités
Détection automatique de la meilleure méthode
Résolution de systèmes linéaires
Implémentation de méthodes numériques classiques
Vérification des propriétés de la matrice
👤 Auteur : Ben Abdallah Youssef

Projet académique – méthodes numériques / algèbre linéaire
