import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_squared_error
import logging

# Set up logging configuration
logging.basicConfig(filename='training_log.log', level=logging.INFO)

def main():
    try:
        # Load dataset
        data = pd.read_csv('data/aqi_data.csv')
        logging.info('Data loaded successfully.')

        # Preprocess data
        X = data.drop('AQI', axis=1)
        y = data['AQI']

        # Feature normalization using Min-Max Scaler
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Save the scaler for future use
        joblib.dump(scaler, 'scaler.pkl')
        logging.info('Scaler saved successfully.')

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        logging.info('Train-test split done.')

        # Train the model (Random Forest Regressor)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        logging.info('Model trained successfully.')

        # Cross-validation
        cv_scores = cross_val_score(model, X_scaled, y, cv=5)
        logging.info(f'Cross-validation scores: {cv_scores}')

        # Predictions
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        logging.info(f'Mean Squared Error: {mse}')

        # Calibration analysis
        calibrated_model = CalibratedClassifierCV(model)
        calibrated_model.fit(X_train, y_train)
        logging.info('Calibration done.')

        # Save model
        joblib.dump(model, 'aqi_model.pkl')
        logging.info('Model saved successfully.')

    except Exception as e:
        logging.error(f'Error occurred: {e}')

if __name__ == '__main__':
    main()