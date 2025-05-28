"""
Model Evaluation Script
"""

import os
import yaml
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from src.evaluation.fairness_metrics import FairnessMetrics

class ModelEvaluator:
    def __init__(self, config_path='config/model_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def evaluate_model_performance(self, model, X_test, y_test, output_dir):
        """Evaluate model performance and generate visualizations"""
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Calculate Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
        plt.close()
        
        # Plot Precision-Recall curve
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, 'pr_curve.png'))
        plt.close()
        
        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'precision': precision.tolist(),
            'recall': recall.tolist()
        }
    
    def evaluate_fairness(self, y_true, y_pred, protected_attributes, output_dir):
        """Evaluate model fairness"""
        metrics = FairnessMetrics(y_true, y_pred, protected_attributes)
        report = metrics.generate_fairness_report()
        
        # Save fairness report
        with open(os.path.join(output_dir, 'fairness_report.txt'), 'w') as f:
            f.write(report)
        
        # Calculate and plot group-wise performance
        for attr, values in protected_attributes.items():
            groups = np.unique(values)
            group_metrics = {}
            
            for group in groups:
                mask = values == group
                group_metrics[group] = {
                    'size': np.sum(mask),
                    'accuracy': np.mean(y_true[mask] == y_pred[mask])
                }
            
            # Plot group-wise accuracy
            plt.figure(figsize=(10, 6))
            groups = list(group_metrics.keys())
            accuracies = [m['accuracy'] for m in group_metrics.values()]
            
            plt.bar(groups, accuracies)
            plt.title(f'Accuracy by {attr}')
            plt.ylabel('Accuracy')
            plt.ylim([0, 1])
            
            plt.savefig(os.path.join(output_dir, f'accuracy_by_{attr}.png'))
            plt.close()
            
            # Save group metrics
            with open(os.path.join(output_dir, f'group_metrics_{attr}.json'), 'w') as f:
                json.dump(group_metrics, f, indent=2)
        
        return metrics.evaluate_all_metrics()
    
    def generate_summary_report(self, model_metrics, fairness_metrics, output_dir):
        """Generate a comprehensive evaluation report"""
        report = []
        report.append("Model Evaluation Summary")
        report.append("=" * 50)
        
        # Model Performance
        report.append("\nModel Performance Metrics:")
        report.append(f"ROC AUC: {model_metrics['roc_auc']:.3f}")
        report.append(f"PR AUC: {model_metrics['pr_auc']:.3f}")
        
        # Fairness Metrics
        report.append("\nFairness Metrics Summary:")
        for attribute, metrics in fairness_metrics.items():
            report.append(f"\n{attribute}:")
            report.append(f"- Demographic Parity Disparity: {metrics['demographic_parity']['disparity']:.3f}")
            report.append(f"- Equal Opportunity Disparity: {metrics['equal_opportunity']['disparity']:.3f}")
            report.append(f"- TPR Disparity: {metrics['equalized_odds']['tpr_disparity']:.3f}")
            report.append(f"- FPR Disparity: {metrics['equalized_odds']['fpr_disparity']:.3f}")
        
        # Save report
        with open(os.path.join(output_dir, 'evaluation_summary.txt'), 'w') as f:
            f.write('\n'.join(report))
        
        return '\n'.join(report) 