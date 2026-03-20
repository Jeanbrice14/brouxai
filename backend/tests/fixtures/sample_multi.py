"""Générateur de fixtures multi-fichiers pour les tests BrouxAI.

Usage :
    python tests/fixtures/sample_multi.py

Génère dans tests/fixtures/ :
    - transactions.csv  (50 lignes) : id, date, client_id, produit, montant, qte
    - clients_multi.csv (15 lignes) : id, nom, region, segment, ca_cumul

Les client_id de transactions correspondent aux id de clients_multi.
"""

from __future__ import annotations

import os
import random
from datetime import date, timedelta

import pandas as pd

random.seed(42)

FIXTURES_DIR = os.path.dirname(os.path.abspath(__file__))

# ── clients_multi.csv ─────────────────────────────────────────────────────────

CLIENTS = [
    ("CLI001", "Dupont SA", "Nord", "PME", 145000),
    ("CLI002", "Martin & Fils", "Ouest", "ETI", 320000),
    ("CLI003", "Bernard Tech", "Est", "PME", 98000),
    ("CLI004", "Petit Commerce", "Ouest", "TPE", 42000),
    ("CLI005", "Grand Groupe IDF", "Ile-de-France", "ETI", 870000),
    ("CLI006", "Lopez Distribution", "Sud", "PME", 215000),
    ("CLI007", "Moreau Industries", "Nord", "ETI", 560000),
    ("CLI008", "Simon Conseil", "Sud", "PME", 133000),
    ("CLI009", "Laurent BTP", "Est", "TPE", 67000),
    ("CLI010", "Fontaine Retail", "Ile-de-France", "PME", 288000),
    ("CLI011", "Rousseau Agro", "Ouest", "TPE", 39000),
    ("CLI012", "Vincent Pharma", "Nord", "ETI", 720000),
    ("CLI013", "Leroy Digital", "Est", "PME", 195000),
    ("CLI014", "David Energie", "Sud", "ETI", 480000),
    ("CLI015", "Garnier Services", "Ile-de-France", "TPE", 55000),
]

clients_df = pd.DataFrame(CLIENTS, columns=["id", "nom", "region", "segment", "ca_cumul"])

# ── transactions.csv ──────────────────────────────────────────────────────────

PRODUITS = ["PROD-A42", "PROD-B17", "PROD-C05", "PROD-D91", "PROD-E33"]
CLIENT_IDS = [c[0] for c in CLIENTS]
START_DATE = date(2024, 1, 1)

transactions = []
for i in range(1, 51):
    tx_date = START_DATE + timedelta(days=random.randint(0, 364))
    client_id = random.choice(CLIENT_IDS)
    produit = random.choice(PRODUITS)
    montant = round(random.uniform(500, 50000), 2)
    qte = random.randint(1, 30)
    transactions.append(
        {
            "id": f"TX{i:04d}",
            "date": tx_date.isoformat(),
            "client_id": client_id,
            "produit": produit,
            "montant": montant,
            "qte": qte,
        }
    )

transactions_df = pd.DataFrame(transactions)

# ── Sauvegarde ────────────────────────────────────────────────────────────────

clients_path = os.path.join(FIXTURES_DIR, "clients_multi.csv")
transactions_path = os.path.join(FIXTURES_DIR, "transactions.csv")

clients_df.to_csv(clients_path, index=False)
transactions_df.to_csv(transactions_path, index=False)

print(f"clients_multi.csv    : {len(clients_df)} lignes -> {clients_path}")
print(f"transactions.csv     : {len(transactions_df)} lignes -> {transactions_path}")

# Vérification de la jointure
coverage = transactions_df["client_id"].isin(clients_df["id"]).mean()
print(f"Taux de couverture client_id->id : {coverage:.0%}")
