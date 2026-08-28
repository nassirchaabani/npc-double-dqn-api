# NPC RL MLOps

[![Tests](https://github.com/nassirchaabani/npc-double-dqn-api/actions/workflows/ci.yml/badge.svg)](https://github.com/nassirchaabani/npc-double-dqn-api/actions/workflows/ci.yml)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

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

## Résultats mesurés et baselines

Les trois politiques ont été évaluées sur les mêmes 500 épisodes (graines 10 000 à 10 499). Le DQN classique et le Double DQN ont chacun été entraînés pendant 2 000 épisodes avec la graine 42.

| Politique | Taux de réussite | Récompense moyenne |
| --- | ---: | ---: |
| Aléatoire | 40,4 % | -2,29 |
| DQN classique | 94,8 % | 9,28 |
| Double DQN | 95,4 % | 9,31 |

Le Double DQN gagne **55 points de pourcentage** face à la politique aléatoire. Son avantage sur le DQN classique est limité à **0,6 point** dans ce protocole ; il ne constitue donc pas, à lui seul, une preuve de supériorité générale.

Pour reproduire la comparaison :

~~~powershell
.\.venv\Scripts\python.exe -m src.train --episodes 2000 --algorithm dqn --output artifacts/npc_classic_dqn.pt
.\.venv\Scripts\python.exe -m src.compare_baselines --episodes 500
~~~