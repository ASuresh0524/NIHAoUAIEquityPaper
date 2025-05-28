"""
Data Analysis Script for Healthcare AI Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import yaml

class HealthcareDataAnalyzer:
    def __init__(self, config_path='config/model_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def load_and_analyze(self, data_path):
        """Load and perform comprehensive data analysis"""
        data = pd.read_csv(data_path)
        
        analysis = {
            'basic_stats': self.compute_basic_statistics(data),
            'demographic_distribution': self.analyze_demographics(data),
            'feature_correlations': self.analyze_correlations(data),
            'bias_analysis': self.analyze_potential_bias(data)
        }
        
        return analysis
    
    def compute_basic_statistics(self, data):
        """Compute basic statistical measures"""
        stats = {}
        
        # Numerical features
        numerical_features = self.config['data']['numerical_features']
        stats['numerical'] = {
            'summary': data[numerical_features].describe(),
            'missing_values': data[numerical_features].isnull().sum(),
            'skewness': data[numerical_features].skew()
        }
        
        # Categorical features
        categorical_features = self.config['data']['categorical_features']
        stats['categorical'] = {
            'value_counts': {col: data[col].value_counts() for col in categorical_features},
            'missing_values': data[categorical_features].isnull().sum()
        }
        
        return stats
    
    def analyze_demographics(self, data):
        """Analyze demographic distributions and intersectionality"""
        demographic_features = self.config['fairness']['protected_attributes']
        analysis = {}
        
        # Single feature distributions
        for feature in demographic_features:
            dist = data[feature].value_counts(normalize=True)
            analysis[f'{feature}_distribution'] = dist
            
            # Create distribution plot
            plt.figure(figsize=(10, 6))
            dist.plot(kind='bar')
            plt.title(f'Distribution of {feature}')
            plt.ylabel('Proportion')
            plt.tight_layout()
            plt.savefig(f'docs/results/figures/{feature}_distribution.png')
            plt.close()
        
        # Intersectional analysis
        if len(demographic_features) >= 2:
            cross_tab = pd.crosstab(
                data[demographic_features[0]], 
                data[demographic_features[1]], 
                normalize='all'
            )
            analysis['intersectional'] = cross_tab
            
            # Create heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(cross_tab, annot=True, fmt='.2%', cmap='YlOrRd')
            plt.title('Intersectional Demographics')
            plt.tight_layout()
            plt.savefig('docs/results/figures/intersectional_demographics.png')
            plt.close()
        
        return analysis
    
    def analyze_correlations(self, data):
        """Analyze feature correlations"""
        numerical_features = self.config['data']['numerical_features']
        
        # Compute correlation matrix
        corr_matrix = data[numerical_features].corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlations')
        plt.tight_layout()
        plt.savefig('docs/results/figures/feature_correlations.png')
        plt.close()
        
        return corr_matrix
    
    def analyze_potential_bias(self, data):
        """Analyze potential biases in the dataset"""
        target = self.config['data']['target_column']
        protected_attributes = self.config['fairness']['protected_attributes']
        bias_analysis = {}
        
        for attr in protected_attributes:
            # Compute outcome rates by group
            group_rates = data.groupby(attr)[target].mean()
            
            # Compute statistical significance
            groups = data[attr].unique()
            p_values = {}
            
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    group1 = groups[i]
                    group2 = groups[j]
                    
                    stat, p_val = stats.ttest_ind(
                        data[data[attr] == group1][target],
                        data[data[attr] == group2][target]
                    )
                    
                    p_values[f'{group1}_vs_{group2}'] = p_val
            
            bias_analysis[attr] = {
                'group_rates': group_rates,
                'statistical_tests': p_values,
                'max_disparity': group_rates.max() - group_rates.min()
            }
            
            # Create group rate plot
            plt.figure(figsize=(10, 6))
            group_rates.plot(kind='bar')
            plt.title(f'Outcome Rates by {attr}')
            plt.ylabel('Rate')
            plt.tight_layout()
            plt.savefig(f'docs/results/figures/bias_{attr}.png')
            plt.close()
        
        return bias_analysis
    
    def generate_analysis_report(self, analysis):
        """Generate a comprehensive analysis report"""
        report = []
        report.append("# Healthcare Data Analysis Report\n")
        
        # Basic Statistics
        report.append("## Basic Statistics\n")
        report.append("### Numerical Features\n")
        report.append(analysis['basic_stats']['numerical']['summary'].to_markdown())
        
        report.append("\n### Missing Values\n")
        report.append(pd.Series(analysis['basic_stats']['numerical']['missing_values']).to_markdown())
        
        # Demographic Analysis
        report.append("\n## Demographic Analysis\n")
        for feature, dist in analysis['demographic_distribution'].items():
            if not feature.endswith('_distribution'):
                continue
            report.append(f"\n### {feature.replace('_distribution', '')}\n")
            report.append(dist.to_markdown())
        
        # Correlation Analysis
        report.append("\n## Feature Correlations\n")
        report.append("See feature_correlations.png for visualization")
        
        # Bias Analysis
        report.append("\n## Potential Bias Analysis\n")
        for attr, bias_stats in analysis['bias_analysis'].items():
            report.append(f"\n### {attr}\n")
            report.append(f"Maximum Disparity: {bias_stats['max_disparity']:.3f}\n")
            report.append("Group Rates:\n")
            report.append(bias_stats['group_rates'].to_markdown())
        
        # Save report
        with open('docs/results/data_analysis_report.md', 'w') as f:
            f.write('\n'.join(report))
        
        return '\n'.join(report)

if __name__ == '__main__':
    analyzer = HealthcareDataAnalyzer()
    analysis = analyzer.load_and_analyze('data/raw/sample_healthcare_data.csv')
    analyzer.generate_analysis_report(analysis) 