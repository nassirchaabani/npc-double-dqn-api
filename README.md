# NPC Double DQN API

Projet personnel autour du comportement d'un NPC dans un petit GridWorld. Un Double DQN apprend à atteindre une cible en évitant un obstacle. L'état utilise les positions relatives de la cible et de l'obstacle afin de mieux généraliser aux placements aléatoires. Le modèle entraîné peut ensuite être évalué et exposé par une API FastAPI.

## Prérequis

- Windows, Linux ou macOS
- Python 3.10 (le projet est volontairement fixé sur cette version)

## Installation sous Windows

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Il n'est pas nécessaire d'activer l'environnement virtuel. Cette méthode évite aussi les restrictions PowerShell liées à `Activate.ps1`.

## Utilisation

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m src.train --episodes 2000
.\.venv\Scripts\python.exe -m src.evaluate --episodes 500
.\.venv\Scripts\python.exe -m uvicorn src.api:app --reload
```

L'entraînement crée `artifacts/npc_dqn.pt`. Un modèle issu de l'ancienne représentation d'état ne doit pas être réutilisé avec cette version. Tant que le nouveau fichier n'existe pas, `/predict` répond volontairement avec le statut 503 au lieu de simuler une prédiction.

Endpoints : `/health`, `/predict`, `/metrics`. Les résultats d'entraînement ne sont pas inclus dans ce dépôt : ils doivent être mesurés lors d'une exécution réelle.

## Résultats

Le modèle a été entraîné pendant 2 000 épisodes puis évalué, sans exploration, sur 500 configurations générées avec des graines différentes de celles de l'entraînement.

| Mesure | Résultat |
|---|---:|
| Taux de réussite | 95,4 % |
| Récompense moyenne | 9,31 |
| Épisodes d'évaluation | 500 |

Une première version fondée sur les positions absolues atteignait 69,6 % de réussite. L'utilisation de positions relatives, d'un Double DQN, d'un réseau cible périodique et d'un signal de progression a porté le taux à 95,4 %. Ce résultat concerne uniquement cet environnement de taille 5 × 5 et ne constitue pas une comparaison avec d'autres algorithmes.

## Limites

- environnement volontairement simple, avec une cible et un obstacle ;
- entraînement et inférence sur CPU ;
- absence de comparaison expérimentale avec PPO ou une politique aléatoire ;
- configuration cloud présente comme base de travail, mais aucun déploiement AWS n'est revendiqué.

