#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Module d'analyse des coûts et de l'impact sur la liquidité pour l'ETF personnalisé.
Ce script évalue l'impact des coûts de gestion, des coûts de transaction,
et analyse l'influence de l'ETF sur la liquidité du marché cible.
"""

import os
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cost_liquidity_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CostLiquidityAnalyzer:
    """Classe pour l'analyse des coûts et de l'impact sur la liquidité de l'ETF."""
    
    def __init__(self, config_path="./config/config.yaml"):
        """
        Initialise l'analyseur de coûts et de liquidité.
        
        Args:
            config_path (str): Chemin vers le fichier de configuration YAML.
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.data_dir = Path("./data")
        self.processed_dir = self.data_dir / "processed"
        self.raw_dir = self.data_dir / "raw"
        self.results_dir = Path("./results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Paramètres de l'ETF
        self.management_fee = self.config["etf"]["management_fee"]
        self.initial_capital = self.config["backtesting"]["initial_capital"]
        self.bid_ask_spread = self.config["backtesting"]["bid_ask_spread"]
        self.broker_commission = self.config["backtesting"]["broker_commission"]
        self.rebalancing_frequency = self.config["etf"]["rebalancing"]["frequency"]
        
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
    
    def load_backtest_results(self):
        """
        Charge les résultats du backtest.
        
        Returns:
            pd.DataFrame: DataFrame avec les résultats du backtest.
        """
        filepath = self.results_dir / "backtest_results.parquet"
        if not filepath.exists():
            logger.error(f"Fichier des résultats du backtest introuvable: {filepath}")
            return None
            
        return pd.read_parquet(filepath)
    
    def load_portfolio(self):
        """
        Charge le portefeuille de l'ETF.
        
        Returns:
            pd.DataFrame: DataFrame avec le portefeuille.
        """
        filepath = self.processed_dir / "etf_portfolio.parquet"
        if not filepath.exists():
            logger.error(f"Fichier de portefeuille introuvable: {filepath}")
            return None
            
        return pd.read_parquet(filepath)
    
    def load_liquidity_data(self):
        """
        Charge les données de liquidité.
        
        Returns:
            pd.DataFrame: DataFrame avec les données de liquidité.
        """
        filepath = self.raw_dir / "liquidity_data.parquet"
        if not filepath.exists():
            logger.error(f"Fichier des données de liquidité introuvable: {filepath}")
            return None
            
        return pd.read_parquet(filepath)
    
    def analyze_management_fees(self, backtest_results):
        """
        Analyse l'impact des frais de gestion sur la performance.
        
        Args:
            backtest_results (pd.DataFrame): DataFrame avec les résultats du backtest.
            
        Returns:
            dict: Résultats de l'analyse des frais de gestion.
        """
        logger.info("Analyse de l'impact des frais de gestion...")
        
        # Calculer la valeur finale sans frais de gestion
        daily_fee = self.management_fee / 252
        days = len(backtest_results)
        
        # Simuler la performance sans frais de gestion
        portfolio_values = backtest_results['Portfolio_Value'].values
        portfolio_returns = backtest_results['Portfolio_Return'].fillna(0).values
        
        # Recalculer les rendements sans frais de gestion
        no_fee_returns = portfolio_returns + daily_fee
        no_fee_values = [self.initial_capital]
        
        for i in range(1, len(no_fee_returns)):
            no_fee_values.append(no_fee_values[i-1] * (1 + no_fee_returns[i]))
        
        backtest_results['No_Fee_Value'] = no_fee_values
        
        # Calculer l'impact total des frais
        final_value_with_fees = portfolio_values[-1]
        final_value_no_fees = no_fee_values[-1]
        fees_impact = final_value_no_fees - final_value_with_fees
        fees_impact_percentage = fees_impact / final_value_no_fees
        
        # Calculer les frais annuels moyens en valeur absolue
        years = days / 252
        annual_fees = fees_impact / years
        
        results = {
            'final_value_with_fees': final_value_with_fees,
            'final_value_no_fees': final_value_no_fees,
            'fees_impact': fees_impact,
            'fees_impact_percentage': fees_impact_percentage,
            'annual_fees': annual_fees,
            'management_fee_rate': self.management_fee
        }
        
        return results
    
    def analyze_transaction_costs(self, backtest_results):
        """
        Analyse l'impact des coûts de transaction sur la performance.
        
        Args:
            backtest_results (pd.DataFrame): DataFrame avec les résultats du backtest.
            
        Returns:
            dict: Résultats de l'analyse des coûts de transaction.
        """
        logger.info("Analyse de l'impact des coûts de transaction...")
        
        # Calculer les coûts de transaction totaux
        total_transaction_costs = backtest_results['Transaction_Cost'].sum()
        
        # Calculer les coûts de transaction en pourcentage de la valeur finale
        final_portfolio_value = backtest_results['Portfolio_Value'].iloc[-1]
        transaction_costs_percentage = total_transaction_costs / final_portfolio_value
        
        # Calculer le turnover moyen
        turnover_values = backtest_results['Turnover'][backtest_results['Turnover'] > 0]
        avg_turnover = turnover_values.mean() if len(turnover_values) > 0 else 0
        
        # Calculer le nombre de rebalancements
        num_rebalancings = len(turnover_values)
        
        # Calculer le rendement sans coûts de transaction
        portfolio_values = backtest_results['Portfolio_Value'].values
        transaction_costs = backtest_results['Transaction_Cost'].values
        
        # Recalculer les valeurs de portefeuille sans coûts de transaction
        no_cost_values = [self.initial_capital]
        for i in range(1, len(portfolio_values)):
            # Ajouter les coûts de transaction à la valeur du portefeuille
            no_cost_values.append(portfolio_values[i] + sum(transaction_costs[:i+1]))
        
        backtest_results['No_Transaction_Cost_Value'] = no_cost_values
        
        # Calculer l'impact total des coûts de transaction
        final_value_with_costs = portfolio_values[-1]
        final_value_no_costs = no_cost_values[-1]
        transaction_costs_impact = final_value_no_costs - final_value_with_costs
        transaction_costs_impact_percentage = transaction_costs_impact / final_value_no_costs
        
        results = {
            'total_transaction_costs': total_transaction_costs,
            'transaction_costs_percentage': transaction_costs_percentage,
            'avg_turnover': avg_turnover,
            'num_rebalancings': num_rebalancings,
            'transaction_costs_impact': transaction_costs_impact,
            'transaction_costs_impact_percentage': transaction_costs_impact_percentage,
            'bid_ask_spread': self.bid_ask_spread,
            'broker_commission': self.broker_commission,
            'rebalancing_frequency': self.rebalancing_frequency
        }
        
        return results
    
    def analyze_liquidity_impact(self, portfolio, liquidity_data):
        """
        Analyse l'impact de l'ETF sur la liquidité du marché.
        
        Args:
            portfolio (pd.DataFrame): DataFrame avec le portefeuille.
            liquidity_data (pd.DataFrame): DataFrame avec les données de liquidité.
            
        Returns:
            dict: Résultats de l'analyse de liquidité.
        """
        logger.info("Analyse de l'impact sur la liquidité du marché...")
        
        # Fusionner le portefeuille avec les données de liquidité
        portfolio_liquidity = pd.merge(
            portfolio, 
            liquidity_data, 
            left_on='Ticker', 
            right_on='Ticker', 
            how='left'
        )
        
        # Calculer la part de l'ETF dans le volume quotidien moyen
        etf_market_impact = portfolio_liquidity.apply(
            lambda row: (row['Weight'] * self.initial_capital) / row['AvgDailyVolume'] 
            if row['AvgDailyVolume'] > 0 else np.nan, 
            axis=1
        )
        
        portfolio_liquidity['Market_Impact'] = etf_market_impact
        
        # Statistiques sur l'impact du marché
        avg_market_impact = etf_market_impact.mean()
        max_market_impact = etf_market_impact.max()
        min_market_impact = etf_market_impact.min()
        
        # Identifier les actifs avec un impact de marché significatif (>5%)
        high_impact_assets = portfolio_liquidity[portfolio_liquidity['Market_Impact'] > 0.05]
        
        # Calculer le temps estimé pour entrer/sortir de positions
        # (supposons qu'on ne peut pas prendre plus de 10% du volume quotidien sans impact significatif)
        max_daily_participation = 0.10  # 10% du volume quotidien
        entry_exit_days = etf_market_impact / max_daily_participation
        
        portfolio_liquidity['Entry_Exit_Days'] = entry_exit_days
        
        # Résumé des résultats
        results = {
            'avg_market_impact': avg_market_impact,
            'max_market_impact': max_market_impact,
            'min_market_impact': min_market_impact,
            'high_impact_assets_count': len(high_impact_assets),
            'high_impact_assets': high_impact_assets['Ticker'].tolist(),
            'avg_entry_exit_days': entry_exit_days.mean(),
            'max_entry_exit_days': entry_exit_days.max(),
            'total_liquidity_score': (1 / entry_exit_days).sum()  # Score de liquidité global
        }
        
        # Sauvegarder les résultats détaillés
        output_path = self.results_dir / "liquidity_analysis.parquet"
        portfolio_liquidity.to_parquet(output_path)
        logger.info(f"Analyse de liquidité détaillée sauvegardée dans {output_path}")
        
        return results
    
    def analyze_all_costs(self):
        """
        Effectue une analyse complète des coûts et de la liquidité.
        
        Returns:
            dict: Résultats de l'analyse complète.
        """
        # 1. Charger les données nécessaires
        backtest_results = self.load_backtest_results()
        portfolio = self.load_portfolio()
        liquidity_data = self.load_liquidity_data()
        
        if backtest_results is None or portfolio is None or liquidity_data is None:
            logger.error("Données manquantes pour l'analyse complète des coûts et de la liquidité")
            return None
        
        # 2. Analyser les frais de gestion
        management_fees_results = self.analyze_management_fees(backtest_results)
        
        # 3. Analyser les coûts de transaction
        transaction_costs_results = self.analyze_transaction_costs(backtest_results)
        
        # 4. Analyser l'impact sur la liquidité
        liquidity_impact_results = self.analyze_liquidity_impact(portfolio, liquidity_data)
        
        # 5. Combiner tous les résultats
        combined_results = {
            'management_fees': management_fees_results,
            'transaction_costs': transaction_costs_results,
            'liquidity_impact': liquidity_impact_results,
            'total_costs': {
                'management_fees_value': management_fees_results['fees_impact'],
                'transaction_costs_value': transaction_costs_results['total_transaction_costs'],
                'total_cost_value': management_fees_results['fees_impact'] + transaction_costs_results['total_transaction_costs'],
                'total_cost_percentage': (management_fees_results['fees_impact'] + transaction_costs_results['total_transaction_costs']) / backtest_results['Portfolio_Value'].iloc[-1],
                'cost_performance_ratio': (management_fees_results['fees_impact'] + transaction_costs_results['total_transaction_costs']) / (backtest_results['Portfolio_Value'].iloc[-1] - self.initial_capital) if backtest_results['Portfolio_Value'].iloc[-1] > self.initial_capital else np.nan
            }
        }
        
        # 6. Sauvegarder les résultats
        output_path = self.results_dir / "cost_liquidity_analysis.json"
        with open(output_path, 'w') as f:
            # Conversion des valeurs numpy en types Python standard pour JSON
            def convert_numpy(obj):
                if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.ndarray,)):
                    return obj.tolist()
                return obj
            
            json_results = {k: {k2: convert_numpy(v2) for k2, v2 in v.items()} for k, v in combined_results.items()}
            json.dump(json_results, f, indent=2)
            
        logger.info(f"Analyse complète des coûts et de la liquidité sauvegardée dans {output_path}")
        
        # 7. Générer des visualisations
        self.generate_cost_liquidity_charts(backtest_results, combined_results, portfolio, liquidity_data)
        
        return combined_results
    
    def generate_cost_liquidity_charts(self, backtest_results, analysis_results, portfolio, liquidity_data):
        """
        Génère des graphiques pour visualiser l'analyse des coûts et de la liquidité.
        
        Args:
            backtest_results (pd.DataFrame): DataFrame avec les résultats du backtest.
            analysis_results (dict): Résultats de l'analyse des coûts et de la liquidité.
            portfolio (pd.DataFrame): DataFrame avec le portefeuille.
            liquidity_data (pd.DataFrame): DataFrame avec les données de liquidité.
        """
        # Configurer le style des graphiques
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("viridis")
        
        # 1. Graphique comparatif de la valeur du portefeuille avec/sans frais
        plt.figure(figsize=(12, 6))
        plt.plot(backtest_results['Date'], backtest_results['Portfolio_Value'], label='Avec frais de gestion')
        plt.plot(backtest_results['Date'], backtest_results['No_Fee_Value'], label='Sans frais de gestion')
        
        if 'No_Transaction_Cost_Value' in backtest_results.columns:
            plt.plot(backtest_results['Date'], backtest_results['No_Transaction_Cost_Value'], label='Sans coûts de transaction')
        
        plt.title(f"Impact des Frais sur la Valeur du Portefeuille\nFrais de Gestion: {self.management_fee:.2%}, Impact Total: {analysis_results['management_fees']['fees_impact_percentage']:.2%}")
        plt.xlabel('Date')
        plt.ylabel('Valeur du Portefeuille')
        plt.legend()
        plt.grid(True)
        
        # Sauvegarder le graphique
        plt.savefig(self.results_dir / "fees_impact.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Graphique des coûts de transaction par rebalancement
        transaction_costs = backtest_results[backtest_results['Transaction_Cost'] > 0]['Transaction_Cost']
        transaction_dates = backtest_results[backtest_results['Transaction_Cost'] > 0]['Date']
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(transaction_costs)), transaction_costs)
        plt.xticks(range(len(transaction_costs)), [d.strftime('%Y-%m-%d') for d in transaction_dates], rotation=45)
        plt.title(f"Coûts de Transaction par Rebalancement\nCoûts Totaux: {analysis_results['transaction_costs']['total_transaction_costs']:,.2f}")
        plt.xlabel('Date de Rebalancement')
        plt.ylabel('Coût de Transaction')
        plt.grid(True, axis='y')
        
        # Sauvegarder le graphique
        plt.savefig(self.results_dir / "transaction_costs.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Graphique du turnover par rebalancement
        turnover = backtest_results[backtest_results['Turnover'] > 0]['Turnover']
        turnover_dates = backtest_results[backtest_results['Turnover'] > 0]['Date']
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(turnover)), turnover * 100)  # En pourcentage
        plt.xticks(range(len(turnover)), [d.strftime('%Y-%m-%d') for d in turnover_dates], rotation=45)
        plt.title(f"Turnover par Rebalancement\nTurnover Moyen: {analysis_results['transaction_costs']['avg_turnover']:.2%}")
        plt.xlabel('Date de Rebalancement')
        plt.ylabel('Turnover (%)')
        plt.grid(True, axis='y')
        
        # Sauvegarder le graphique
        plt.savefig(self.results_dir / "turnover.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Graphique de l'impact sur la liquidité
        # Fusionner le portefeuille avec les données de liquidité
        portfolio_liquidity = pd.merge(
            portfolio, 
            liquidity_data, 
            left_on='Ticker', 
            right_on='Ticker', 
            how='left'
        )
        
        # Calculer l'impact sur le marché (% du volume quotidien)
        portfolio_liquidity['Market_Impact'] = (portfolio_liquidity['Weight'] * self.initial_capital) / portfolio_liquidity['AvgDailyVolume']
        
        # Trier par impact sur le marché
        portfolio_liquidity = portfolio_liquidity.sort_values('Market_Impact', ascending=False)
        
        # Prendre les 20 plus grands impacts
        top_impacts = portfolio_liquidity.head(20)
        
        plt.figure(figsize=(14, 10))
        bars = plt.barh(top_impacts['Ticker'], top_impacts['Market_Impact'] * 100)
        
        # Colorer les barres en fonction de l'impact (rouge pour élevé, vert pour faible)
        for i, bar in enumerate(bars):
            if top_impacts.iloc[i]['Market_Impact'] > 0.10:
                bar.set_color('red')
            elif top_impacts.iloc[i]['Market_Impact'] > 0.05:
                bar.set_color('orange')
            else:
                bar.set_color('green')
                
        plt.title(f"Impact de l'ETF sur la Liquidité du Marché\nImpact Moyen: {analysis_results['liquidity_impact']['avg_market_impact']:.2%}")
        plt.xlabel('Impact (% du Volume Quotidien Moyen)')
        plt.ylabel('Ticker')
        plt.grid(True, axis='x')
        plt.axvline(x=5, color='orange', linestyle='--', label='Seuil d\'Attention (5%)')
        plt.axvline(x=10, color='red', linestyle='--', label='Seuil Critique (10%)')
        plt.legend()
        
        # Sauvegarder le graphique
        plt.savefig(self.results_dir / "liquidity_impact.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Graphique du temps d'entrée/sortie estimé
        entry_exit_days = portfolio_liquidity['Market_Impact'] / 0.10  # 10% du volume quotidien
        
        plt.figure(figsize=(14, 10))
        bars = plt.barh(portfolio_liquidity['Ticker'].head(20), entry_exit_days.head(20))
        
        # Colorer les barres en fonction du temps (rouge pour long, vert pour court)
        for i, bar in enumerate(bars):
            if entry_exit_days.iloc[i] > 5:
                bar.set_color('red')
            elif entry_exit_days.iloc[i] > 2:
                bar.set_color('orange')
            else:
                bar.set_color('green')
                
        plt.title(f"Temps Estimé pour Entrer/Sortir des Positions\nTemps Moyen: {entry_exit_days.mean():.2f} jours")
        plt.xlabel('Jours Nécessaires (10% max du volume quotidien)')
        plt.ylabel('Ticker')
        plt.grid(True, axis='x')
        plt.axvline(x=1, color='green', linestyle='--', label='1 jour')
        plt.axvline(x=3, color='orange', linestyle='--', label='3 jours')
        plt.axvline(x=5, color='red', linestyle='--', label='5 jours')
        plt.legend()
        
        # Sauvegarder le graphique
        plt.savefig(self.results_dir / "entry_exit_time.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Graphique en camembert de la répartition des coûts
        management_fees = analysis_results['management_fees']['fees_impact']
        transaction_costs = analysis_results['transaction_costs']['total_transaction_costs']
        
        plt.figure(figsize=(10, 8))
        plt.pie([management_fees, transaction_costs], 
                labels=['Frais de Gestion', 'Coûts de Transaction'],
                autopct='%1.1f%%',
                startangle=90,
                explode=(0.1, 0))
        plt.title(f"Répartition des Coûts Totaux: {management_fees + transaction_costs:,.2f}")
        
        # Sauvegarder le graphique
        plt.savefig(self.results_dir / "cost_breakdown.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Graphiques d'analyse des coûts et de la liquidité générés dans le dossier {self.results_dir}")


if __name__ == "__main__":
    # Exécuter l'analyse des coûts et de la liquidité
    analyzer = CostLiquidityAnalyzer()
    results = analyzer.analyze_all_costs()
    
    if results is not None:
        print("\n=== Résumé de l'Analyse des Coûts et de la Liquidité ===")
        print(f"Impact des Frais de Gestion: {results['management_fees']['fees_impact_percentage']:.2%}")
        print(f"Impact des Coûts de Transaction: {results['transaction_costs']['transaction_costs_impact_percentage']:.2%}")
        print(f"Impact Moyen sur la Liquidité: {results['liquidity_impact']['avg_market_impact']:.2%}")
        print(f"Coûts Totaux: {results['total_costs']['total_cost_value']:,.2f}")
        print(f"Coûts en % de la Valeur Finale: {results['total_costs']['total_cost_percentage']:.2%}")