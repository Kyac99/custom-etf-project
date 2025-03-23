#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module de sélection et pondération des actifs pour l'ETF personnalisé.
Ce script implémente différentes méthodes de sélection et de pondération
des actifs pour construire l'ETF.
"""

import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import json

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("asset_selection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AssetSelector:
    """Classe pour la sélection et la pondération des actifs pour l'ETF."""
    
    def __init__(self, config_path="./config/config.yaml"):
        """
        Initialise le sélecteur d'actifs.
        
        Args:
            config_path (str): Chemin vers le fichier de configuration YAML.
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.data_dir = Path("./data")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Paramètres de sélection
        self.market_constraints = self.config["weighting"]["constraints"]["max_weight_per_country"]
        self.sector_constraints = self.config["weighting"]["constraints"]["max_weight_per_sector"]
        self.asset_constraints = self.config["weighting"]["constraints"]["max_weight_per_asset"]
        
        # Méthode de pondération
        self.weighting_method = self.config["weighting"]["method"]
        
        # Paramètres smart beta
        if self.weighting_method == "smart_beta":
            self.factor_weights = {factor["name"]: factor["weight"] 
                                  for factor in self.config["weighting"]["smart_beta"]["factors"]}
    
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
    
    def load_filtered_assets(self):
        """
        Charge les données des actifs filtrés.
        
        Returns:
            pd.DataFrame: DataFrame avec les actifs filtrés.
        """
        filepath = self.processed_dir / "filtered_assets.parquet"
        if not filepath.exists():
            logger.error(f"Fichier des actifs filtrés introuvable: {filepath}")
            return None
            
        return pd.read_parquet(filepath)
    
    def load_historical_prices(self):
        """
        Charge les données historiques de prix.
        
        Returns:
            pd.DataFrame: DataFrame avec les prix historiques.
        """
        filepath = self.raw_dir / "historical_prices.parquet"
        if not filepath.exists():
            logger.error(f"Fichier des prix historiques introuvable: {filepath}")
            return None
            
        return pd.read_parquet(filepath)
    
    def calculate_momentum(self, prices_df, window=252):
        """
        Calcule le momentum pour chaque actif.
        
        Args:
            prices_df (pd.DataFrame): DataFrame avec les prix historiques.
            window (int): Fenêtre de calcul en jours.
            
        Returns:
            pd.Series: Series avec les scores de momentum.
        """
        # Calculer les rendements
        returns = prices_df.pct_change()
        
        # Calculer le momentum (rendement cumulé sur la période)
        if len(returns) >= window:
            momentum = (1 + returns.iloc[-window:]).prod() - 1
        else:
            momentum = (1 + returns).prod() - 1
            
        return momentum
    
    def calculate_volatility(self, prices_df, window=252):
        """
        Calcule la volatilité pour chaque actif.
        
        Args:
            prices_df (pd.DataFrame): DataFrame avec les prix historiques.
            window (int): Fenêtre de calcul en jours.
            
        Returns:
            pd.Series: Series avec les scores de volatilité.
        """
        # Calculer les rendements
        returns = prices_df.pct_change()
        
        # Calculer la volatilité (écart-type des rendements)
        if len(returns) >= window:
            volatility = returns.iloc[-window:].std() * np.sqrt(252)  # Annualiser
        else:
            volatility = returns.std() * np.sqrt(252)
            
        return volatility
    
    def calculate_value(self, filtered_assets):
        """
        Calcule le score de valeur pour chaque actif.
        
        Args:
            filtered_assets (pd.DataFrame): DataFrame avec les données des actifs.
            
        Returns:
            pd.Series: Series avec les scores de valeur.
        """
        # Utiliser les ratios financiers comme indicateurs de valeur
        # Par exemple, un ratio P/E bas indique une valeur potentiellement élevée
        
        # Pour simplifier, on utilise l'inverse du P/E (E/P) comme indicateur de valeur
        # Plus E/P est élevé, plus l'action est "value"
        
        # Supposons que nous avons un champ EPS dans nos données fondamentales
        if 'EPS' in filtered_assets.columns and 'MarketCap' not in filtered_assets.columns:
            filtered_assets['E/P'] = filtered_assets['EPS'] / filtered_assets['MarketCap']
            value_score = filtered_assets['E/P']
        else:
            # Si les données nécessaires ne sont pas disponibles, utiliser un score aléatoire
            logger.warning("Données nécessaires pour le calcul du score de valeur indisponibles, utilisation de valeurs aléatoires")
            value_score = pd.Series(np.random.random(len(filtered_assets)), index=filtered_assets.index)
            
        return value_score
    
    def calculate_quality(self, filtered_assets):
        """
        Calcule le score de qualité pour chaque actif.
        
        Args:
            filtered_assets (pd.DataFrame): DataFrame avec les données des actifs.
            
        Returns:
            pd.Series: Series avec les scores de qualité.
        """
        # Utiliser plusieurs indicateurs pour mesurer la qualité
        quality_indicators = []
        
        # 1. Rentabilité (ROE)
        if 'ROE' in filtered_assets.columns:
            quality_indicators.append(filtered_assets['ROE'])
            
        # 2. Marge bénéficiaire
        if 'ProfitMargin' in filtered_assets.columns:
            quality_indicators.append(filtered_assets['ProfitMargin'])
            
        # 3. Faible niveau d'endettement (inverse du ratio d'endettement)
        if 'DebtToEquity' in filtered_assets.columns:
            quality_indicators.append(1 / (filtered_assets['DebtToEquity'] + 0.01))  # Éviter division par zéro
            
        # 4. Stabilité des revenus
        if 'RevenueGrowth' in filtered_assets.columns:
            quality_indicators.append(filtered_assets['RevenueGrowth'])
            
        # Si aucun indicateur n'est disponible, utiliser un score aléatoire
        if not quality_indicators:
            logger.warning("Données nécessaires pour le calcul du score de qualité indisponibles, utilisation de valeurs aléatoires")
            return pd.Series(np.random.random(len(filtered_assets)), index=filtered_assets.index)
            
        # Normaliser chaque indicateur
        scaler = MinMaxScaler()
        normalized_indicators = []
        
        for indicator in quality_indicators:
            # Remplacer les valeurs infinies et NaN
            indicator = indicator.replace([np.inf, -np.inf], np.nan)
            indicator = indicator.fillna(indicator.mean())
            
            # Normaliser
            normalized = scaler.fit_transform(indicator.values.reshape(-1, 1)).flatten()
            normalized_indicators.append(normalized)
            
        # Combinér les indicateurs normalisés
        quality_score = np.mean(normalized_indicators, axis=0)
        return pd.Series(quality_score, index=filtered_assets.index)
    
    def calculate_factor_scores(self, filtered_assets, prices_df):
        """
        Calcule les scores pour chaque facteur de smart beta.
        
        Args:
            filtered_assets (pd.DataFrame): DataFrame avec les données des actifs.
            prices_df (pd.DataFrame): DataFrame avec les prix historiques.
            
        Returns:
            dict: Dictionnaire avec les scores pour chaque facteur.
        """
        factor_scores = {}
        
        # Filtrer les prix historiques pour n'inclure que les actifs sélectionnés
        selected_tickers = filtered_assets['Ticker'].tolist()
        filtered_prices = prices_df[selected_tickers]
        
        # 1. Momentum
        momentum_scores = self.calculate_momentum(filtered_prices)
        factor_scores['momentum'] = momentum_scores
        
        # 2. Value
        value_scores = self.calculate_value(filtered_assets)
        factor_scores['value'] = value_scores
        
        # 3. Quality
        quality_scores = self.calculate_quality(filtered_assets)
        factor_scores['quality'] = quality_scores
        
        # 4. Low Volatility (inverse de la volatilité)
        volatility = self.calculate_volatility(filtered_prices)
        low_vol_scores = 1 / (volatility + 0.001)  # Éviter division par zéro
        factor_scores['low_volatility'] = low_vol_scores
        
        return factor_scores
    
    def apply_market_cap_weighting(self, filtered_assets):
        """
        Applique une pondération par capitalisation boursière.
        
        Args:
            filtered_assets (pd.DataFrame): DataFrame avec les données des actifs.
            
        Returns:
            pd.DataFrame: DataFrame avec les pondérations.
        """
        if 'MarketCap' not in filtered_assets.columns:
            logger.error("Données de capitalisation boursière non disponibles")
            return None
            
        # Calculer les poids proportionnellement à la capitalisation boursière
        total_market_cap = filtered_assets['MarketCap'].sum()
        weights = filtered_assets['MarketCap'] / total_market_cap
        
        return weights
    
    def apply_equal_weighting(self, filtered_assets):
        """
        Applique une pondération égale à tous les actifs.
        
        Args:
            filtered_assets (pd.DataFrame): DataFrame avec les données des actifs.
            
        Returns:
            pd.DataFrame: DataFrame avec les pondérations.
        """
        n_assets = len(filtered_assets)
        weights = pd.Series(1.0 / n_assets, index=filtered_assets.index)
        
        return weights
    
    def apply_smart_beta_weighting(self, filtered_assets, prices_df):
        """
        Applique une pondération smart beta basée sur plusieurs facteurs.
        
        Args:
            filtered_assets (pd.DataFrame): DataFrame avec les données des actifs.
            prices_df (pd.DataFrame): DataFrame avec les prix historiques.
            
        Returns:
            pd.DataFrame: DataFrame avec les pondérations.
        """
        # Calculer les scores pour chaque facteur
        factor_scores = self.calculate_factor_scores(filtered_assets, prices_df)
        
        # Normaliser les scores de chaque facteur
        normalized_scores = {}
        for factor, scores in factor_scores.items():
            # Remplacer les valeurs infinies et NaN
            scores = scores.replace([np.inf, -np.inf], np.nan)
            scores = scores.fillna(scores.mean())
            
            # Normaliser
            scaler = MinMaxScaler()
            normalized = scaler.fit_transform(scores.values.reshape(-1, 1)).flatten()
            normalized_scores[factor] = pd.Series(normalized, index=scores.index)
        
        # Calculer un score composite pondéré par les poids des facteurs
        composite_score = pd.Series(0.0, index=filtered_assets.index)
        for factor, weight in self.factor_weights.items():
            if factor in normalized_scores:
                composite_score += normalized_scores[factor] * weight
                
        # Calculer les poids basés sur le score composite
        weights = composite_score / composite_score.sum()
        
        return weights
    
    def apply_fundamental_weighting(self, filtered_assets):
        """
        Applique une pondération basée sur des facteurs fondamentaux.
        
        Args:
            filtered_assets (pd.DataFrame): DataFrame avec les données des actifs.
            
        Returns:
            pd.DataFrame: DataFrame avec les pondérations.
        """
        # Pour cet exemple, utiliser le chiffre d'affaires comme facteur fondamental
        if 'Revenue' not in filtered_assets.columns:
            logger.error("Données de chiffre d'affaires non disponibles")
            return None
            
        # Calculer les poids proportionnellement au chiffre d'affaires
        total_revenue = filtered_assets['Revenue'].sum()
        weights = filtered_assets['Revenue'] / total_revenue
        
        return weights
    
    def apply_constraints(self, weights, filtered_assets):
        """
        Applique les contraintes de pondération maximale.
        
        Args:
            weights (pd.Series): Series avec les pondérations initiales.
            filtered_assets (pd.DataFrame): DataFrame avec les données des actifs.
            
        Returns:
            pd.Series: Series avec les pondérations contraintes.
        """
        # Copier les poids initiaux
        constrained_weights = weights.copy()
        
        # 1. Limiter le poids par actif
        excess_weight = constrained_weights[constrained_weights > self.asset_constraints] - self.asset_constraints
        constrained_weights[constrained_weights > self.asset_constraints] = self.asset_constraints
        
        # Redistribuer l'excès de poids si nécessaire
        if excess_weight.sum() > 0:
            # Redistribuer aux actifs qui ne dépassent pas la contrainte
            eligible_indices = constrained_weights[constrained_weights < self.asset_constraints].index
            if len(eligible_indices) > 0:
                redistribution = excess_weight.sum() / len(eligible_indices)
                constrained_weights[eligible_indices] += redistribution
        
        # 2. Normaliser pour s'assurer que la somme est égale à 1
        constrained_weights = constrained_weights / constrained_weights.sum()
        
        # 3. Vérifier et ajuster les contraintes par marché (pays)
        # Cette étape nécessiterait une information sur le pays de chaque actif
        # Pour cet exemple, supposons que nous avons cette information
        
        # 4. Vérifier et ajuster les contraintes par secteur
        # Cette étape nécessiterait une information sur le secteur de chaque actif
        # Pour cet exemple, supposons que nous avons cette information
        
        return constrained_weights
    
    def select_and_weight_assets(self):
        """
        Sélectionne et pondère les actifs pour l'ETF.
        
        Returns:
            pd.DataFrame: DataFrame avec les actifs sélectionnés et leurs pondérations.
        """
        # 1. Charger les actifs filtrés
        filtered_assets = self.load_filtered_assets()
        if filtered_assets is None:
            return None
            
        # 2. Charger les prix historiques
        prices_df = self.load_historical_prices()
        if prices_df is None:
            return None
            
        # 3. Appliquer la méthode de pondération choisie
        weights = None
        if self.weighting_method == "market_cap":
            weights = self.apply_market_cap_weighting(filtered_assets)
        elif self.weighting_method == "equal_weight":
            weights = self.apply_equal_weighting(filtered_assets)
        elif self.weighting_method == "smart_beta":
            weights = self.apply_smart_beta_weighting(filtered_assets, prices_df)
        elif self.weighting_method == "fundamental":
            weights = self.apply_fundamental_weighting(filtered_assets)
        else:
            logger.error(f"Méthode de pondération non supportée: {self.weighting_method}")
            return None
            
        if weights is None:
            logger.error("Échec du calcul des pondérations")
            return None
            
        # 4. Appliquer les contraintes de pondération
        constrained_weights = self.apply_constraints(weights, filtered_assets)
        
        # 5. Créer le DataFrame final avec les actifs sélectionnés et leurs pondérations
        filtered_assets['Weight'] = constrained_weights
        
        # 6. Trier par pondération décroissante
        portfolio = filtered_assets.sort_values('Weight', ascending=False)
        
        # 7. Sauvegarder le portefeuille
        output_path = self.processed_dir / "etf_portfolio.parquet"
        portfolio.to_parquet(output_path)
        logger.info(f"Portefeuille de l'ETF sauvegardé dans {output_path}")
        
        # 8. Créer un résumé du portefeuille au format JSON
        summary = {
            'name': self.config['etf']['name'],
            'description': self.config['etf']['description'],
            'weighting_method': self.weighting_method,
            'number_of_assets': len(portfolio),
            'creation_date': datetime.now().strftime('%Y-%m-%d'),
            'top_holdings': portfolio.head(10)[['Ticker', 'Weight']].to_dict('records'),
            'sector_allocation': {}, # À compléter si les données de secteur sont disponibles
            'country_allocation': {}, # À compléter si les données de pays sont disponibles
        }
        
        summary_path = self.processed_dir / "etf_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Résumé de l'ETF sauvegardé dans {summary_path}")
        
        return portfolio


if __name__ == "__main__":
    # Exécuter la sélection et la pondération des actifs
    selector = AssetSelector()
    portfolio = selector.select_and_weight_assets()
    
    if portfolio is not None:
        print(f"Nombre d'actifs dans le portefeuille: {len(portfolio)}")
        print("\nTop 10 actifs par pondération:")
        print(portfolio.head(10)[['Ticker', 'Weight', 'MarketCap', 'ESG_Score']])
        print(f"\nSomme des pondérations: {portfolio['Weight'].sum():.4f}")