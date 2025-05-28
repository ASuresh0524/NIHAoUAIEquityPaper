"""
Tests for fairness metrics implementation
"""

import numpy as np
import pytest
from src.evaluation.fairness_metrics import FairnessMetrics

def test_demographic_parity():
    # Create synthetic test data
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    protected_attributes = {
        'gender': np.array(['M', 'M', 'M', 'M', 'F', 'F', 'F', 'F'])
    }
    
    metrics = FairnessMetrics(y_true, y_pred, protected_attributes)
    results = metrics.demographic_parity('gender')
    
    # Check if prediction rates are equal for both groups
    assert abs(results['group_rates']['M'] - results['group_rates']['F']) < 1e-6
    assert results['disparity'] < 1e-6

def test_equal_opportunity():
    # Create synthetic test data with known true positive rates
    y_true = np.array([1, 1, 1, 1, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 1, 1, 0, 0])
    protected_attributes = {
        'race': np.array(['A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])
    }
    
    metrics = FairnessMetrics(y_true, y_pred, protected_attributes)
    results = metrics.equal_opportunity('race')
    
    # Both groups should have perfect true positive rates
    assert results['group_rates']['A'] == 1.0
    assert results['group_rates']['B'] == 1.0
    assert results['disparity'] == 0.0

def test_equalized_odds():
    # Create synthetic test data
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    protected_attributes = {
        'gender': np.array(['M', 'M', 'M', 'M', 'F', 'F', 'F', 'F'])
    }
    
    metrics = FairnessMetrics(y_true, y_pred, protected_attributes)
    results = metrics.equalized_odds('gender')
    
    # Check if TPR and FPR are equal for both groups
    assert results['tpr_disparity'] == 0.0
    assert results['fpr_disparity'] == 0.0

def test_fairness_report_generation():
    # Create synthetic test data
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])
    protected_attributes = {
        'gender': np.array(['M', 'M', 'F', 'F']),
        'race': np.array(['A', 'B', 'A', 'B'])
    }
    
    metrics = FairnessMetrics(y_true, y_pred, protected_attributes)
    report = metrics.generate_fairness_report()
    
    # Check if report contains all necessary sections
    assert 'Demographic Parity' in report
    assert 'Equal Opportunity' in report
    assert 'Equalized Odds' in report
    assert 'gender' in report
    assert 'race' in report

def test_edge_cases():
    # Test with all same predictions
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 1])
    protected_attributes = {
        'gender': np.array(['M', 'M', 'F', 'F'])
    }
    
    metrics = FairnessMetrics(y_true, y_pred, protected_attributes)
    results = metrics.evaluate_all_metrics()
    
    # Check if metrics handle the case properly
    assert results['gender']['demographic_parity']['disparity'] == 0.0
    
    # Test with no positive predictions
    y_pred = np.array([0, 0, 0, 0])
    metrics = FairnessMetrics(y_true, y_pred, protected_attributes)
    results = metrics.evaluate_all_metrics()
    
    # Check if metrics handle the case properly
    assert 'gender' in results

if __name__ == '__main__':
    pytest.main([__file__]) 