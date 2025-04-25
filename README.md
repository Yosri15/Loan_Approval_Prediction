# Loan Approval Prediction System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.2.2-orange)
![Pandas](https://img.shields.io/badge/Pandas-2.0.3-red)

Un système de Machine Learning pour prédire l'approbation de prêts bancaires en fonction des caractéristiques des clients.

## 📌 Project Overview
Ce projet vise à :
- Prédire la probabilité d'approbation d'un prêt bancaire
- Gérer le déséquilibre des classes avec SMOTE
- Comparer les performances de différents modèles (Random Forest, Régression Logistique, etc.)
- **Score AUC final** : 0.85

## 🗂️ Dataset
- **Source** : https://www.kaggle.com/competitions/playground-series-s4e10/data
- **Features clés** :
  - `loan_amnt` : Montant du prêt
  - `loan_int_rate` : Taux d'intérêt
  - `person_emp_length` : Ancienneté professionnelle
  - `person_income` : Revenu annuel
  - Variables catégorielles : `loan_intent`, `person_home_ownership`, etc.
- **Target** : `loan_status` (0 = Refusé, 1 = Approuvé)

## ⚙️ Methodology
1. **Prétraitement** :
   - Gestion des valeurs manquantes (imputation par moyenne)
   - Encodage des variables catégorielles (`LabelEncoder`)
   - Normalisation des données (`StandardScaler`)

2. **Modélisation** :
   ```python
   Models testés :
   - Logistic Regression
   - Random Forest (meilleur performeur)
   - Decision Tree
   - K-Nearest Neighbors
