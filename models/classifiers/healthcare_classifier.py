"""
Healthcare Classifier Model Implementation
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import yaml
import numpy as np

class HealthcareClassifier:
    def __init__(self, config_path='config/model_config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)['training']['classifier']
        
        self.model = RandomForestClassifier(
            n_estimators=self.config['n_estimators'],
            max_depth=self.config['max_depth'],
            random_state=self.config['random_state']
        )
        
    def train(self, X_train, y_train):
        """Train the classifier"""
        self.model.fit(X_train, y_train)
        
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'detailed_report': classification_report(y_test, y_pred)
        }
        
        return metrics
    
    def save_model(self, save_path):
        """Save the trained model"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(self.model, save_path)
        
    @classmethod
    def load_model(cls, model_path):
        """Load a trained model"""
        instance = cls()
        instance.model = joblib.load(model_path)
        return instance
    
    def feature_importance(self):
        """Get feature importance scores"""
        return {
            'importance_scores': self.model.feature_importances_,
            'std': np.std([tree.feature_importances_ 
                          for tree in self.model.estimators_], axis=0)
        } 