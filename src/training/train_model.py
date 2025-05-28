"""
Main training script for healthcare AI model
"""

import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict

from src.data.preprocessor import HealthcareDataLoader, HealthcareDataPreprocessor
from src.models.healthcare_gan import HealthcareGAN
from src.evaluation.fairness_metrics import FairnessMetrics

def parse_args():
    parser = argparse.ArgumentParser(description='Train healthcare AI model with fairness constraints')
    parser.add_argument('--data_path', type=str, required=True, help='Path to healthcare dataset')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save outputs')
    parser.add_argument('--gan_epochs', type=int, default=100, help='Number of epochs for GAN training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--synthetic_samples', type=int, default=1000, help='Number of synthetic samples to generate')
    return parser.parse_args()

def setup_output_directory(base_dir: str) -> str:
    """Create timestamped output directory"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(base_dir, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def train_gan(data: np.ndarray, args) -> HealthcareGAN:
    """Train GAN model"""
    input_dim = 100  # Noise dimension
    hidden_dim = 128
    output_dim = data.shape[1]
    
    gan = HealthcareGAN(input_dim, hidden_dim, output_dim)
    
    # Convert data to torch tensors and create dataloader
    from torch.utils.data import DataLoader, TensorDataset
    import torch
    
    tensor_data = torch.FloatTensor(data)
    dataset = TensorDataset(tensor_data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Train GAN
    gan.train(dataloader, args.gan_epochs)
    return gan

def evaluate_fairness(y_true: np.ndarray, y_pred: np.ndarray, 
                     protected_attributes: Dict[str, np.ndarray],
                     output_dir: str) -> None:
    """Evaluate and save fairness metrics"""
    metrics = FairnessMetrics(y_true, y_pred, protected_attributes)
    report = metrics.generate_fairness_report()
    
    # Save report
    with open(os.path.join(output_dir, 'fairness_report.txt'), 'w') as f:
        f.write(report)
    
    # Save detailed metrics
    results = metrics.evaluate_all_metrics()
    pd.DataFrame(results).to_csv(os.path.join(output_dir, 'fairness_metrics.csv'))

def main():
    args = parse_args()
    output_dir = setup_output_directory(args.output_dir)
    
    # Load and preprocess data
    loader = HealthcareDataLoader(args.data_path)
    data = loader.load_data()
    
    # Define features
    categorical_features = ['gender', 'race']
    numerical_features = ['age', 'blood_pressure', 'heart_rate']
    demographic_features = ['gender', 'race']
    
    # Preprocess data
    preprocessor = HealthcareDataPreprocessor(
        categorical_features=categorical_features,
        numerical_features=numerical_features
    )
    
    X, y = loader.split_features_labels(data, 'diabetes')
    X_processed = preprocessor.fit_transform(X)
    
    # Get demographic information for fairness evaluation
    protected_attrs = loader.get_demographic_features(data, demographic_features)
    
    # Train GAN
    gan = train_gan(X_processed, args)
    
    # Generate synthetic data
    synthetic_data = gan.generate_samples(args.synthetic_samples)
    
    # Save synthetic data
    synthetic_df = pd.DataFrame(synthetic_data, columns=categorical_features + numerical_features)
    synthetic_df.to_csv(os.path.join(output_dir, 'synthetic_data.csv'), index=False)
    
    # Train classifier on augmented dataset
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    
    # Combine original and synthetic data
    X_augmented = np.vstack([X_processed, synthetic_data])
    y_augmented = np.concatenate([y, np.random.choice([0, 1], args.synthetic_samples)])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_augmented, y_augmented, test_size=0.2, random_state=42
    )
    
    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    # Evaluate fairness
    evaluate_fairness(y_test, y_pred, protected_attrs, output_dir)
    
    print(f"Training completed. Results saved to {output_dir}")

if __name__ == '__main__':
    main() 