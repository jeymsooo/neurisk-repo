# Muscle Injury Risk Prediction System

This project implements a muscle injury risk prediction system for basketball players using EMG data collected from MyoWare 2.0 and ESP32. The system leverages synthetic dataset generation, real-time EMG feature extraction, and machine learning models to predict the risk of muscle injuries.

## Project Structure

```
muscle-injury-risk-prediction
├── data
│   ├── raw                # Directory for raw EMG data
│   └── processed          # Directory for processed datasets
├── src
│   ├── data_generation
│   │   └── generate_synthetic_emg.py  # Synthetic EMG dataset generation
│   ├── feature_extraction
│   │   └── emg_features.py            # Real-time EMG feature extraction
│   ├── model
│   │   ├── train_model.py              # Model training pipeline
│   │   └── utils.py                    # Utility functions for model training
│   ├── prediction
│   │   └── predictor.py                # Prediction logic
│   ├── interface
│   │   └── streamlit_app.py            # Streamlit user interface
│   └── config.py                       # Configuration settings
├── requirements.txt                     # Project dependencies
└── README.md                            # Project documentation
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd muscle-injury-risk-prediction
pip install -r requirements.txt
```

## Usage

1. **Data Collection**: Collect raw EMG data using MyoWare 2.0 and ESP32. Store the data in the `data/raw` directory.

2. **Synthetic Data Generation**: Use the `generate_synthetic_emg.py` script to create synthetic EMG datasets. This simulates realistic EMG signals for calves, hamstrings, and quadriceps.

3. **Feature Extraction**: Run the `emg_features.py` script to extract features from the raw EMG signals. This includes applying filters and computing metrics such as RMS, MAV, ZC, SSC, and WL.

4. **Model Training**: Train the machine learning models using the `train_model.py` script. This will apply SMOTE for class balancing and save the trained models.

5. **Prediction**: Use the `predictor.py` script to load the trained models and predict injury risk based on user input and extracted EMG features.

6. **Streamlit Interface**: Launch the Streamlit app using the `streamlit_app.py` script. This provides a user-friendly interface for inputting demographic information and simulating or capturing EMG features.

## Features

- Synthetic EMG dataset generation
- Real-time EMG feature extraction
- Machine learning model training with overfitting protection
- Injury risk prediction based on EMG features
- User-friendly Streamlit interface

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.