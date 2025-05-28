"""
Fairness Metrics for Healthcare AI Models
"""

import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Dict, List, Tuple

class FairnessMetrics:
    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, protected_attributes: Dict[str, np.ndarray]):
        """
        Initialize fairness metrics calculator
        
        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            protected_attributes: Dictionary mapping demographic attributes to their values
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.protected_attributes = protected_attributes

    def demographic_parity(self, attribute: str) -> Dict[str, float]:
        """
        Calculate demographic parity (difference in prediction rates across groups)
        """
        groups = np.unique(self.protected_attributes[attribute])
        pred_rates = {}
        
        for group in groups:
            mask = self.protected_attributes[attribute] == group
            pred_rate = np.mean(self.y_pred[mask])
            pred_rates[str(group)] = pred_rate
            
        return {
            'group_rates': pred_rates,
            'disparity': max(pred_rates.values()) - min(pred_rates.values())
        }

    def equal_opportunity(self, attribute: str) -> Dict[str, float]:
        """
        Calculate equal opportunity (true positive rates across groups)
        """
        groups = np.unique(self.protected_attributes[attribute])
        tpr_rates = {}
        
        for group in groups:
            mask = self.protected_attributes[attribute] == group
            tn, fp, fn, tp = confusion_matrix(self.y_true[mask], self.y_pred[mask]).ravel()
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            tpr_rates[str(group)] = tpr
            
        return {
            'group_rates': tpr_rates,
            'disparity': max(tpr_rates.values()) - min(tpr_rates.values())
        }

    def equalized_odds(self, attribute: str) -> Dict[str, Dict[str, float]]:
        """
        Calculate equalized odds (true positive and false positive rates across groups)
        """
        groups = np.unique(self.protected_attributes[attribute])
        metrics = {'tpr': {}, 'fpr': {}}
        
        for group in groups:
            mask = self.protected_attributes[attribute] == group
            tn, fp, fn, tp = confusion_matrix(self.y_true[mask], self.y_pred[mask]).ravel()
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            metrics['tpr'][str(group)] = tpr
            metrics['fpr'][str(group)] = fpr
        
        return {
            'true_positive_rates': metrics['tpr'],
            'false_positive_rates': metrics['fpr'],
            'tpr_disparity': max(metrics['tpr'].values()) - min(metrics['tpr'].values()),
            'fpr_disparity': max(metrics['fpr'].values()) - min(metrics['fpr'].values())
        }

    def evaluate_all_metrics(self) -> Dict[str, Dict]:
        """
        Calculate all fairness metrics for all protected attributes
        """
        results = {}
        for attribute in self.protected_attributes.keys():
            results[attribute] = {
                'demographic_parity': self.demographic_parity(attribute),
                'equal_opportunity': self.equal_opportunity(attribute),
                'equalized_odds': self.equalized_odds(attribute)
            }
        return results

    def generate_fairness_report(self) -> str:
        """
        Generate a human-readable report of fairness metrics
        """
        results = self.evaluate_all_metrics()
        report = []
        
        report.append("Fairness Evaluation Report")
        report.append("=" * 50)
        
        for attribute, metrics in results.items():
            report.append(f"\nProtected Attribute: {attribute}")
            report.append("-" * 30)
            
            # Demographic Parity
            dp = metrics['demographic_parity']
            report.append("\nDemographic Parity:")
            for group, rate in dp['group_rates'].items():
                report.append(f"  Group {group}: {rate:.3f}")
            report.append(f"  Overall Disparity: {dp['disparity']:.3f}")
            
            # Equal Opportunity
            eo = metrics['equal_opportunity']
            report.append("\nEqual Opportunity:")
            for group, rate in eo['group_rates'].items():
                report.append(f"  Group {group}: {rate:.3f}")
            report.append(f"  Overall Disparity: {eo['disparity']:.3f}")
            
            # Equalized Odds
            eodds = metrics['equalized_odds']
            report.append("\nEqualized Odds:")
            report.append("  True Positive Rates:")
            for group, rate in eodds['true_positive_rates'].items():
                report.append(f"    Group {group}: {rate:.3f}")
            report.append("  False Positive Rates:")
            for group, rate in eodds['false_positive_rates'].items():
                report.append(f"    Group {group}: {rate:.3f}")
            report.append(f"  TPR Disparity: {eodds['tpr_disparity']:.3f}")
            report.append(f"  FPR Disparity: {eodds['fpr_disparity']:.3f}")
            
        return "\n".join(report) 