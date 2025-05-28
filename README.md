# AI Equity in Healthcare: Leveraging All of Us Research Program Data

## Project Overview
This repository contains the implementation and research materials for developing equitable AI systems in healthcare using the NIH All of Us Research Program database. The project focuses on creating fair and unbiased machine learning models that serve diverse populations effectively.

## Repository Structure
```
.
├── data/
│   ├── processed/         # Processed and cleaned datasets
│   ├── raw/              # Raw data files
│   └── synthetic/        # GAN-generated synthetic data
├── models/
│   ├── gan/              # GAN model implementations
│   ├── classifiers/      # Healthcare prediction models
│   └── evaluation/       # Model evaluation scripts
├── notebooks/
│   ├── exploratory/      # Data exploration notebooks
│   ├── modeling/         # Model development notebooks
│   └── analysis/         # Results analysis notebooks
├── src/
│   ├── data/             # Data processing scripts
│   ├── models/           # Model implementation code
│   ├── training/         # Training pipelines
│   ├── evaluation/       # Evaluation metrics
│   └── utils/            # Utility functions
├── tests/                # Unit tests
├── docs/                 # Documentation
│   ├── methodology/      # Research methodology
│   ├── results/          # Detailed results
│   └── presentations/    # Presentation materials
├── config/               # Configuration files
└── requirements.txt      # Project dependencies
```

## Research Components

### 1. Data Processing
- Extraction of de-identified data from All of Us database
- Implementation of data preprocessing pipelines
- Creation of synthetic data using GANs

### 2. Model Development
- Implementation of healthcare prediction models
- Integration of fairness constraints
- Development of meta-learning approaches

### 3. Evaluation Framework
- Comprehensive testing across demographic groups
- Performance metrics analysis
- Bias and fairness assessment

## Key Features
- Fairness-aware machine learning models
- GAN-based synthetic data generation
- Multi-modal healthcare data processing
- Rigorous evaluation across demographic groups

## Results
- 95% accuracy in medical image classification
- 90-95% consistent accuracy across demographic groups
- 30% boost in training data diversity through GANs

## Installation & Setup
```bash
# Clone the repository
git clone [repository-url]

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
Detailed usage instructions for each component can be found in the respective directories' README files.

## Contributing
Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## Authors
- Aakash Suresh - Principal Investigator
- Stephanie Campos - Mentor 

## Acknowledgments
- National Institutes of Health All of Us Research Program
- Pembroke Pines Charter High School

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For any queries regarding this research, please contact aakashsuresh2006@gmail.com 
