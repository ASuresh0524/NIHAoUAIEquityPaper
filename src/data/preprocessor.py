"""
Data Preprocessing Module for Healthcare Data
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, List

class HealthcareDataPreprocessor:
    def __init__(self, categorical_features: List[str] = None, numerical_features: List[str] = None):
        """
        Initialize the preprocessor
        
        Args:
            categorical_features: List of categorical column names
            numerical_features: List of numerical column names
        """
        self.categorical_features = categorical_features or []
        self.numerical_features = numerical_features or []
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the preprocessor to the data
        """
        # Fit label encoders for categorical features
        for feature in self.categorical_features:
            le = LabelEncoder()
            le.fit(data[feature].astype(str))
            self.label_encoders[feature] = le
            
        # Fit scaler for numerical features
        if self.numerical_features:
            self.scaler.fit(data[self.numerical_features])
    
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Transform the data using fitted preprocessors
        """
        processed_data = []
        
        # Transform categorical features
        for feature in self.categorical_features:
            le = self.label_encoders[feature]
            encoded = le.transform(data[feature].astype(str))
            processed_data.append(encoded.reshape(-1, 1))
            
        # Transform numerical features
        if self.numerical_features:
            scaled = self.scaler.transform(data[self.numerical_features])
            processed_data.append(scaled)
            
        return np.hstack(processed_data)
    
    def fit_transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        Fit and transform the data
        """
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform_categorical(self, data: np.ndarray, feature_idx: int) -> np.ndarray:
        """
        Inverse transform encoded categorical features
        """
        feature = self.categorical_features[feature_idx]
        return self.label_encoders[feature].inverse_transform(data)
    
    def inverse_transform_numerical(self, data: np.ndarray) -> np.ndarray:
        """
        Inverse transform scaled numerical features
        """
        return self.scaler.inverse_transform(data)

class HealthcareDataLoader:
    def __init__(self, data_path: str):
        """
        Initialize the data loader
        
        Args:
            data_path: Path to the healthcare dataset
        """
        self.data_path = data_path
        
    def load_data(self) -> pd.DataFrame:
        """
        Load and perform initial cleaning of healthcare data
        """
        # Load data
        data = pd.read_csv(self.data_path)
        
        # Basic cleaning
        data = data.dropna()  # Remove missing values
        
        return data
    
    def split_features_labels(self, data: pd.DataFrame, 
                            label_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split data into features and labels
        """
        X = data.drop(columns=[label_column])
        y = data[label_column]
        return X, y
    
    def get_demographic_features(self, data: pd.DataFrame, 
                               demographic_columns: List[str]) -> Dict[str, np.ndarray]:
        """
        Extract demographic features for fairness evaluation
        """
        return {col: data[col].values for col in demographic_columns}

def create_synthetic_healthcare_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Create synthetic healthcare dataset for testing
    """
    np.random.seed(42)
    
    # Generate synthetic data
    data = {
        'age': np.random.normal(50, 15, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'race': np.random.choice(['White', 'Black', 'Asian', 'Hispanic'], n_samples),
        'blood_pressure': np.random.normal(120, 20, n_samples),
        'heart_rate': np.random.normal(80, 10, n_samples),
        'diabetes': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Add some correlations
    df.loc[df['age'] > 60, 'diabetes'] = np.random.choice([0, 1], sum(df['age'] > 60), p=[0.7, 0.3])
    
    return df 