#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module de collecte de données pour l'ETF personnalisé.
Ce script récupère les données de marché et fondamentales
pour les actifs potentiels de l'ETF.
"""

import os
import yaml
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
import logging
import requests
from concurrent.futures import ThreadPoolExecutor
import json
import time
from tqdm import tqdm

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_collection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DataCollector:
    """Classe pour collecter des données de marché et fondamentales."""
    
    def __init__(self, config_path="./config/config.yaml"):
        """
        Initialise le collecteur de données.
        
        Args:
            config_path (str): Chemin vers le fichier de configuration YAML.
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.data_dir = Path("./data")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Création des répertoires s'ils n'existent pas
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Récupération des paramètres de sélection d'actifs
        self.markets = self.config["asset_selection"]["markets"]
        self.sectors = self.config["asset_selection"]["sectors"]
        self.filters = self.config["asset_selection"]["filters"]
        
        # Paramètres de backtesting
        self.start_date = self.config["backtesting"]["start_date"]
        self.end_date = self.config["backtesting"]["end_date"]
        
        # Source de données
        self.market_data_source = self.config["data_sources"]["market_data"]
        self.fundamental_data_source = self.config["data_sources"]["fundamental_data"]
        self.esg_data_source = self.config["data_sources"]["esg_data"]
        
        # API keys
        self.api_keys = self.config["data_sources"]["api_keys"]
        
    def _load_config(self):
        """
        Charge la configuration depuis le fichier YAML.
        
        Returns:
            dict: Configuration chargée.
        """
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {e}")
            raise
            
    def get_market_tickers(self):
        """
        Récupère la liste des tickers par marché et secteur.
        
        Returns:
            dict: Dictionnaire des tickers par marché.
        """
        logger.info("Récupération des tickers par marché et secteur...")
        
        market_tickers = {}
        
        # Mapping des marchés vers les indices ou ETFs représentatifs
        market_etfs = {
            "Brazil": "EWZ",
            "Russia": "RSX",
            "India": "INDA",
            "China": "MCHI",
            "South Africa": "EZA",
            "Mexico": "EWW",
            "Indonesia": "EIDO",
            "Turkey": "TUR"
        }
        
        # Mapping des secteurs vers des mots-clés pour filtrage
        sector_keywords = {
            "Technology": ["technology", "software", "hardware", "semiconductor"],
            "Telecommunications": ["telecom", "communication", "network"],
            "Consumer Electronics": ["electronics", "consumer", "devices"],
            "Fintech": ["fintech", "payment", "blockchain", "digital banking"],
            "E-commerce": ["ecommerce", "e-commerce", "online retail", "marketplace"]
        }
        
        # Pour chaque marché, récupérer les composants des ETFs représentatifs
        for market in self.markets:
            etf_ticker = market_etfs.get(market)
            if not etf_ticker:
                logger.warning(f"Pas de ticker ETF trouvé pour le marché: {market}")
                continue
                
            try:
                # Utiliser yfinance pour obtenir les holdings de l'ETF
                etf = yf.Ticker(etf_ticker)
                holdings = etf.get_holdings()
                
                if holdings is None or holdings.empty:
                    logger.warning(f"Pas de holdings trouvés pour l'ETF {etf_ticker} ({market})")
                    continue
                
                # Filtrer par secteur
                market_stocks = []
                for _, row in holdings.iterrows():
                    ticker = row.get("Ticker")
                    sector = row.get("Sector", "")
                    
                    # Vérifier si le secteur correspond à l'un des secteurs cibles
                    sector_match = False
                    for target_sector, keywords in sector_keywords.items():
                        if any(keyword.lower() in sector.lower() for keyword in keywords):
                            sector_match = True
                            break
                            
                    if sector_match and ticker:
                        market_stocks.append(ticker)
                
                market_tickers[market] = market_stocks
                logger.info(f"Récupéré {len(market_stocks)} tickers pour {market}")
                
            except Exception as e:
                logger.error(f"Erreur lors de la récupération des tickers pour {market}: {e}")
        
        # Sauvegarder les tickers dans un fichier JSON
        output_path = self.raw_dir / "market_tickers.json"
        with open(output_path, 'w') as f:
            json.dump(market_tickers, f, indent=2)
            
        logger.info(f"Tickers sauvegardés dans {output_path}")
        return market_tickers
    
    def get_historical_prices(self, tickers, start_date=None, end_date=None):
        """
        Récupère les données historiques de prix pour une liste de tickers.
        
        Args:
            tickers (list): Liste des tickers.
            start_date (str, optional): Date de début au format YYYY-MM-DD.
            end_date (str, optional): Date de fin au format YYYY-MM-DD.
            
        Returns:
            pd.DataFrame: DataFrame avec les prix historiques.
        """
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date
            
        logger.info(f"Récupération des prix historiques du {start_date} au {end_date} pour {len(tickers)} tickers")
        
        # Diviser les tickers en lots de 50 (limite yfinance)
        batch_size = 50
        all_data = []
        
        for i in range(0, len(tickers), batch_size):
            batch_tickers = tickers[i:i+batch_size]
            try:
                logger.info(f"Téléchargement du lot {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}")
                data = yf.download(batch_tickers, start=start_date, end=end_date, 
                                   group_by='ticker', auto_adjust=True)
                
                # Restructurer les données si un seul ticker
                if len(batch_tickers) == 1:
                    ticker = batch_tickers[0]
                    data = pd.DataFrame({ticker: data['Close']})
                else:
                    # Extraire uniquement les prix de clôture
                    data = data.xs('Close', axis=1, level=1)
                    
                all_data.append(data)
                
                # Attendre un peu pour ne pas surcharger l'API
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Erreur lors du téléchargement des prix pour le lot {i//batch_size + 1}: {e}")
        
        # Fusionner tous les lots
        if all_data:
            prices_df = pd.concat(all_data, axis=1)
            
            # Sauvegarder les données
            output_path = self.raw_dir / "historical_prices.parquet"
            prices_df.to_parquet(output_path)
            logger.info(f"Prix historiques sauvegardés dans {output_path}")
            
            return prices_df
        else:
            logger.warning("Aucune donnée de prix n'a été récupérée")
            return pd.DataFrame()
    
    def get_fundamental_data(self, tickers):
        """
        Récupère les données fondamentales pour une liste de tickers.
        
        Args:
            tickers (list): Liste des tickers.
            
        Returns:
            pd.DataFrame: DataFrame avec les données fondamentales.
        """
        logger.info(f"Récupération des données fondamentales pour {len(tickers)} tickers")
        
        # Cette fonction est un exemple simplifié
        # Dans un cas réel, on utiliserait une API comme SimFin, Alpha Vantage, etc.
        
        # Créer un DataFrame vide avec des colonnes pour les métriques fondamentales
        columns = [
            'Ticker', 'MarketCap', 'ProfitMargin', 'DebtToEquity', 
            'ROE', 'ROA', 'CurrentRatio', 'QuickRatio', 'Revenue', 
            'RevenueGrowth', 'EPS', 'EPSGrowth', 'DividendYield'
        ]
        fundamentals_df = pd.DataFrame(columns=columns)
        
        # Pour chaque ticker, essayer de récupérer les données fondamentales
        for ticker in tqdm(tickers, desc="Récupération des données fondamentales"):
            try:
                stock = yf.Ticker(ticker)
                
                # Récupérer les informations générales
                info = stock.info
                
                # Récupérer les métriques financières
                financials = stock.financials
                
                # Construire une ligne de données fondamentales
                row = {
                    'Ticker': ticker,
                    'MarketCap': info.get('marketCap', np.nan),
                    'ProfitMargin': info.get('profitMargins', np.nan),
                    'DebtToEquity': info.get('debtToEquity', np.nan),
                    'ROE': info.get('returnOnEquity', np.nan),
                    'ROA': info.get('returnOnAssets', np.nan),
                    'CurrentRatio': info.get('currentRatio', np.nan),
                    'QuickRatio': info.get('quickRatio', np.nan),
                    'Revenue': info.get('totalRevenue', np.nan),
                    'RevenueGrowth': info.get('revenueGrowth', np.nan),
                    'EPS': info.get('trailingEPS', np.nan),
                    'EPSGrowth': info.get('earningsGrowth', np.nan),
                    'DividendYield': info.get('dividendYield', np.nan)
                }
                
                # Ajouter la ligne au DataFrame
                fundamentals_df = pd.concat([fundamentals_df, pd.DataFrame([row])], ignore_index=True)
                
                # Attendre un peu pour ne pas surcharger l'API
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Erreur lors de la récupération des données fondamentales pour {ticker}: {e}")
        
        # Sauvegarder les données
        output_path = self.raw_dir / "fundamental_data.parquet"
        fundamentals_df.to_parquet(output_path)
        logger.info(f"Données fondamentales sauvegardées dans {output_path}")
        
        return fundamentals_df
    
    def get_esg_data(self, tickers):
        """
        Récupère les données ESG pour une liste de tickers.
        
        Args:
            tickers (list): Liste des tickers.
            
        Returns:
            pd.DataFrame: DataFrame avec les données ESG.
        """
        logger.info(f"Récupération des données ESG pour {len(tickers)} tickers")
        
        # Cette fonction est un placeholder - dans un cas réel, on utiliserait 
        # une API spécialisée comme Refinitiv, MSCI, Sustainalytics, etc.
        
        # Créer un DataFrame simulé avec des scores ESG aléatoires
        esg_df = pd.DataFrame({
            'Ticker': tickers,
            'ESG_Score': np.random.uniform(30, 90, size=len(tickers)),
            'Environmental_Score': np.random.uniform(20, 95, size=len(tickers)),
            'Social_Score': np.random.uniform(20, 95, size=len(tickers)),
            'Governance_Score': np.random.uniform(20, 95, size=len(tickers)),
            'Controversy_Score': np.random.uniform(0, 5, size=len(tickers))
        })
        
        # Sauvegarder les données
        output_path = self.raw_dir / "esg_data.parquet"
        esg_df.to_parquet(output_path)
        logger.info(f"Données ESG sauvegardées dans {output_path}")
        
        return esg_df
    
    def get_liquidity_data(self, tickers, start_date=None, end_date=None):
        """
        Récupère les données de liquidité (volume, spread) pour une liste de tickers.
        
        Args:
            tickers (list): Liste des tickers.
            start_date (str, optional): Date de début au format YYYY-MM-DD.
            end_date (str, optional): Date de fin au format YYYY-MM-DD.
            
        Returns:
            pd.DataFrame: DataFrame avec les données de liquidité.
        """
        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date
            
        logger.info(f"Récupération des données de liquidité du {start_date} au {end_date} pour {len(tickers)} tickers")
        
        # Récupérer les volumes de trading
        volumes_data = []
        batch_size = 50
        
        for i in range(0, len(tickers), batch_size):
            batch_tickers = tickers[i:i+batch_size]
            try:
                logger.info(f"Téléchargement des volumes pour le lot {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}")
                data = yf.download(batch_tickers, start=start_date, end=end_date, 
                                  group_by='ticker', auto_adjust=True)
                
                # Restructurer les données si un seul ticker
                if len(batch_tickers) == 1:
                    ticker = batch_tickers[0]
                    volume_data = pd.DataFrame({ticker: data['Volume']})
                else:
                    # Extraire uniquement les volumes
                    volume_data = data.xs('Volume', axis=1, level=1)
                    
                volumes_data.append(volume_data)
                
                # Attendre un peu pour ne pas surcharger l'API
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Erreur lors du téléchargement des volumes pour le lot {i//batch_size + 1}: {e}")
        
        # Fusionner tous les lots
        if volumes_data:
            volumes_df = pd.concat(volumes_data, axis=1)
            
            # Calculer les métriques de liquidité
            # 1. Volume moyen quotidien
            avg_daily_volume = volumes_df.mean()
            
            # 2. Créer un DataFrame avec les métriques de liquidité
            liquidity_df = pd.DataFrame({
                'Ticker': volumes_df.columns,
                'AvgDailyVolume': avg_daily_volume.values,
                'VolumeStability': volumes_df.std() / volumes_df.mean(),  # Coefficient de variation
            })
            
            # Sauvegarder les données
            output_path = self.raw_dir / "liquidity_data.parquet"
            liquidity_df.to_parquet(output_path)
            logger.info(f"Données de liquidité sauvegardées dans {output_path}")
            
            return liquidity_df
        else:
            logger.warning("Aucune donnée de liquidité n'a été récupérée")
            return pd.DataFrame()
    
    def combine_all_data(self):
        """
        Combine toutes les données collectées pour créer un ensemble de données unique.
        
        Returns:
            pd.DataFrame: DataFrame avec toutes les données combinées.
        """
        logger.info("Combinaison de toutes les données collectées...")
        
        try:
            # Charger les différents ensembles de données
            fundamental_path = self.raw_dir / "fundamental_data.parquet"
            esg_path = self.raw_dir / "esg_data.parquet"
            liquidity_path = self.raw_dir / "liquidity_data.parquet"
            
            if not all(path.exists() for path in [fundamental_path, esg_path, liquidity_path]):
                logger.error("Certains fichiers de données sont manquants")
                return None
            
            fundamental_df = pd.read_parquet(fundamental_path)
            esg_df = pd.read_parquet(esg_path)
            liquidity_df = pd.read_parquet(liquidity_path)
            
            # Fusionner les données
            combined_df = pd.merge(fundamental_df, esg_df, on='Ticker', how='left')
            combined_df = pd.merge(combined_df, liquidity_df, on='Ticker', how='left')
            
            # Appliquer les filtres de sélection des actifs
            filtered_df = combined_df[
                (combined_df['MarketCap'] >= self.filters['min_market_cap']) &
                (combined_df['AvgDailyVolume'] >= self.filters['min_daily_volume']) &
                (combined_df['ProfitMargin'] >= self.filters['min_profit_margin']) &
                (combined_df['DebtToEquity'] <= self.filters['max_debt_to_equity']) &
                (combined_df['ESG_Score'] >= self.filters['esg_minimum_score'])
            ]
            
            # Sauvegarder les données combinées
            output_path = self.processed_dir / "filtered_assets.parquet"
            filtered_df.to_parquet(output_path)
            logger.info(f"Données combinées et filtrées sauvegardées dans {output_path}")
            
            return filtered_df
            
        except Exception as e:
            logger.error(f"Erreur lors de la combinaison des données: {e}")
            return None
            
    def run_data_collection(self):
        """
        Exécute le processus complet de collecte de données.
        
        Returns:
            pd.DataFrame: DataFrame avec toutes les données combinées et filtrées.
        """
        logger.info("Démarrage du processus de collecte de données...")
        
        # 1. Récupérer les tickers par marché
        market_tickers = self.get_market_tickers()
        
        # 2. Créer une liste unique de tickers
        all_tickers = []
        for market, tickers in market_tickers.items():
            all_tickers.extend(tickers)
        all_tickers = list(set(all_tickers))  # Supprimer les doublons
        
        # 3. Récupérer les prix historiques
        self.get_historical_prices(all_tickers)
        
        # 4. Récupérer les données fondamentales
        self.get_fundamental_data(all_tickers)
        
        # 5. Récupérer les données ESG
        self.get_esg_data(all_tickers)
        
        # 6. Récupérer les données de liquidité
        self.get_liquidity_data(all_tickers)
        
        # 7. Combiner toutes les données
        filtered_assets = self.combine_all_data()
        
        logger.info("Processus de collecte de données terminé")
        return filtered_assets


if __name__ == "__main__":
    # Exécuter la collecte de données
    collector = DataCollector()
    filtered_assets = collector.run_data_collection()
    
    if filtered_assets is not None:
        print(f"Nombre d'actifs après filtrage: {len(filtered_assets)}")
        print("\nTop 10 actifs par capitalisation boursière:")
        print(filtered_assets.sort_values('MarketCap', ascending=False).head(10)[['Ticker', 'MarketCap', 'ESG_Score', 'AvgDailyVolume']])