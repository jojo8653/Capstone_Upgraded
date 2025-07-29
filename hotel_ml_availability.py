import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
import streamlit as st

warnings.filterwarnings('ignore')

class HotelAvailabilityPredictor:
    """
    Advanced ML system for real-time hotel availability prediction
    Combines LSTM, LightGBM, and Random Forest for ensemble predictions
    """
    
    def __init__(self):
        self.lstm_model = None
        self.lightgbm_model = None
        self.rf_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.sequence_length = 14  # 14 days for LSTM sequences
        
    def create_time_series_features(self, df):
        """Create comprehensive time series features for ML models"""
        df = df.copy()
        
        # Ensure arrival_date is datetime
        if 'arrival_date' not in df.columns:
            df['arrival_date'] = pd.to_datetime(
                df['arrival_date_year'].astype(str) + '-' + 
                df['arrival_date_month'].str[:3] + '-' + 
                df['arrival_date_day_of_month'].astype(str),
                format='%Y-%b-%d', errors='coerce'
            )
        
        # Sort by date for time series
        df = df.sort_values('arrival_date').reset_index(drop=True)
        
        # Date-based features
        df['year'] = df['arrival_date'].dt.year
        df['month'] = df['arrival_date'].dt.month
        df['day_of_week'] = df['arrival_date'].dt.dayofweek
        df['day_of_year'] = df['arrival_date'].dt.dayofyear
        df['week_of_year'] = df['arrival_date'].dt.isocalendar().week
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_holiday_season'] = ((df['month'].isin([12, 1])) | 
                                  ((df['month'] == 7) | (df['month'] == 8))).astype(int)
        
        # Seasonal features
        df['season'] = df['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        # Cyclical encoding for temporal features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Hotel and booking features
        df['total_nights'] = df['stays_in_week_nights'] + df['stays_in_weekend_nights']
        df['total_guests'] = df[['adults','children','babies']].sum(axis=1)
        df['guest_density'] = df['total_guests'] / (df['total_nights'] + 1)
        df['booking_lead_category'] = pd.cut(df['lead_time'], 
                                           bins=[-1, 7, 30, 90, float('inf')], 
                                           labels=['Last_Minute', 'Short_Term', 'Medium_Term', 'Long_Term'])
        
        # Price features
        df['price_per_guest'] = df['adr'] / df['total_guests']
        df['revenue'] = df['adr'] * df['total_nights']
        
        # Create availability target (inverse of bookings)
        # Higher values = more availability
        df_daily = self._create_daily_availability_target(df)
        
        return df_daily
    
    def _create_daily_availability_target(self, df):
        """Create daily availability metrics as prediction target"""
        
        # Group by hotel, country, and date to calculate daily metrics
        daily_stats = df.groupby(['hotel', 'country', 'arrival_date']).agg({
            'is_canceled': ['mean', 'count'],
            'adr': ['mean', 'std'],
            'total_guests': 'sum',
            'total_nights': 'mean',
            'lead_time': 'mean'
        }).reset_index()
        
        # Flatten column names
        daily_stats.columns = ['hotel', 'country', 'arrival_date', 
                              'cancellation_rate', 'booking_count',
                              'avg_adr', 'adr_std', 'total_guests', 
                              'avg_nights', 'avg_lead_time']
        
        # Calculate availability score (0-1, higher = more available)
        daily_stats['availability_score'] = (
            daily_stats['cancellation_rate'] * 0.4 +  # More cancellations = more availability
            (1 - np.clip(daily_stats['booking_count'] / daily_stats['booking_count'].quantile(0.95), 0, 1)) * 0.6
        )
        
        # Add hotel capacity estimate (based on max bookings seen)
        hotel_capacity = df.groupby(['hotel', 'country'])['total_guests'].sum().reset_index()
        hotel_capacity.columns = ['hotel', 'country', 'estimated_capacity']
        daily_stats = daily_stats.merge(hotel_capacity, on=['hotel', 'country'], how='left')
        
        # Calculate occupancy rate
        daily_stats['occupancy_rate'] = np.clip(daily_stats['total_guests'] / daily_stats['estimated_capacity'], 0, 1)
        daily_stats['availability_score'] = np.clip(1 - daily_stats['occupancy_rate'], 0, 1)
        
        return daily_stats
    
    def prepare_features(self, df):
        """Prepare features for ML models"""
        
        # Create time series features
        df_processed = self.create_time_series_features(df)
        
        # Select feature columns
        categorical_features = ['hotel', 'country', 'season', 'booking_lead_category']
        numerical_features = ['month', 'day_of_week', 'is_weekend', 'is_holiday_season',
                             'month_sin', 'month_cos', 'day_sin', 'day_cos',
                             'avg_adr', 'adr_std', 'avg_nights', 'avg_lead_time',
                             'booking_count', 'cancellation_rate']
        
        # Handle categorical encoding
        for col in categorical_features:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df_processed[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df_processed[col].astype(str))
            else:
                df_processed[f'{col}_encoded'] = self.label_encoders[col].transform(df_processed[col].astype(str))
        
        # Feature columns for models
        self.feature_columns = numerical_features + [f'{col}_encoded' for col in categorical_features]
        
        return df_processed
    
    def create_lstm_sequences(self, data, target_col='availability_score'):
        """Create sequences for LSTM model"""
        
        # Group by hotel and create sequences
        sequences = []
        targets = []
        
        for (hotel, country), group in data.groupby(['hotel', 'country']):
            group = group.sort_values('arrival_date').reset_index(drop=True)
            
            if len(group) < self.sequence_length + 1:
                continue
                
            for i in range(len(group) - self.sequence_length):
                seq = group[self.feature_columns].iloc[i:i+self.sequence_length].values
                target = group[target_col].iloc[i+self.sequence_length]
                
                sequences.append(seq)
                targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model for time series prediction"""
        
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')  # Availability score between 0-1
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_models(self, df, test_size=0.2, validation_split=0.2):
        """Train all three models: LSTM, LightGBM, and Random Forest"""
        
        st.info("🔄 Preparing features for ML training...")
        df_processed = self.prepare_features(df)
        
        # Prepare data for traditional ML models (LightGBM, RF)
        X = df_processed[self.feature_columns].fillna(0)
        y = df_processed['availability_score'].fillna(0.5)
        
        # Split data temporally (important for time series)
        split_date = df_processed['arrival_date'].quantile(0.8)
        train_mask = df_processed['arrival_date'] <= split_date
        
        X_train, X_test = X[train_mask], X[~train_mask]
        y_train, y_test = y[train_mask], y[~train_mask]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        st.info("🌳 Training Random Forest model...")
        # Train Random Forest
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_train, y_train)
        
        st.info("🚀 Training LightGBM model...")
        # Train LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        self.lightgbm_model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50)]
        )
        
        st.info("🔗 Training LSTM model...")
        # Prepare LSTM data
        lstm_data = df_processed[df_processed['arrival_date'] <= split_date]
        X_lstm, y_lstm = self.create_lstm_sequences(lstm_data)
        
        if len(X_lstm) > 0:
            # Build and train LSTM
            self.lstm_model = self.build_lstm_model((self.sequence_length, len(self.feature_columns)))
            
            # Scale LSTM features
            X_lstm_reshaped = X_lstm.reshape(-1, len(self.feature_columns))
            X_lstm_scaled = self.scaler.transform(X_lstm_reshaped)
            X_lstm_scaled = X_lstm_scaled.reshape(-1, self.sequence_length, len(self.feature_columns))
            
            # Train LSTM
            history = self.lstm_model.fit(
                X_lstm_scaled, y_lstm,
                epochs=50,
                batch_size=32,
                validation_split=validation_split,
                verbose=0,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
                ]
            )
        
        # Evaluate models
        results = self._evaluate_models(X_test_scaled, y_test, df_processed[~train_mask])
        
        return results
    
    def _evaluate_models(self, X_test_scaled, y_test, test_data):
        """Evaluate all trained models"""
        
        results = {}
        
        # Random Forest predictions
        if self.rf_model:
            rf_pred = self.rf_model.predict(self.scaler.inverse_transform(X_test_scaled))
            results['Random Forest'] = {
                'MAE': mean_absolute_error(y_test, rf_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
                'R²': r2_score(y_test, rf_pred)
            }
        
        # LightGBM predictions
        if self.lightgbm_model:
            lgb_pred = self.lightgbm_model.predict(self.scaler.inverse_transform(X_test_scaled))
            results['LightGBM'] = {
                'MAE': mean_absolute_error(y_test, lgb_pred),
                'RMSE': np.sqrt(mean_squared_error(y_test, lgb_pred)),
                'R²': r2_score(y_test, lgb_pred)
            }
        
        # LSTM predictions (if enough data)
        if self.lstm_model and len(test_data) >= self.sequence_length:
            X_lstm_test, y_lstm_test = self.create_lstm_sequences(test_data)
            if len(X_lstm_test) > 0:
                X_lstm_reshaped = X_lstm_test.reshape(-1, len(self.feature_columns))
                X_lstm_scaled = self.scaler.transform(X_lstm_reshaped)
                X_lstm_scaled = X_lstm_scaled.reshape(-1, self.sequence_length, len(self.feature_columns))
                
                lstm_pred = self.lstm_model.predict(X_lstm_scaled).flatten()
                results['LSTM'] = {
                    'MAE': mean_absolute_error(y_lstm_test, lstm_pred),
                    'RMSE': np.sqrt(mean_squared_error(y_lstm_test, lstm_pred)),
                    'R²': r2_score(y_lstm_test, lstm_pred)
                }
        
        return results
    
    def predict_availability(self, hotel_data, prediction_date=None):
        """Predict real-time availability using ensemble of models"""
        
        if prediction_date is None:
            prediction_date = datetime.now().date()
        
        # Prepare features for prediction
        pred_data = self.prepare_features(hotel_data)
        latest_data = pred_data[pred_data['arrival_date'] <= pd.to_datetime(prediction_date)]
        
        if len(latest_data) == 0:
            return {'availability_score': 0.5, 'confidence': 'low', 'model_predictions': {}}
        
        X_pred = latest_data[self.feature_columns].fillna(0).iloc[-1:].values
        X_pred_scaled = self.scaler.transform(X_pred)
        
        predictions = {}
        
        # Random Forest prediction
        if self.rf_model:
            predictions['Random Forest'] = float(self.rf_model.predict(X_pred)[0])
        
        # LightGBM prediction
        if self.lightgbm_model:
            predictions['LightGBM'] = float(self.lightgbm_model.predict(X_pred)[0])
        
        # LSTM prediction (if we have enough historical data)
        if self.lstm_model and len(latest_data) >= self.sequence_length:
            X_lstm = latest_data[self.feature_columns].iloc[-self.sequence_length:].values
            X_lstm = X_lstm.reshape(1, self.sequence_length, len(self.feature_columns))
            X_lstm_scaled = self.scaler.transform(X_lstm.reshape(-1, len(self.feature_columns)))
            X_lstm_scaled = X_lstm_scaled.reshape(1, self.sequence_length, len(self.feature_columns))
            
            predictions['LSTM'] = float(self.lstm_model.predict(X_lstm_scaled)[0][0])
        
        # Ensemble prediction (weighted average)
        if predictions:
            weights = {'Random Forest': 0.3, 'LightGBM': 0.4, 'LSTM': 0.3}
            ensemble_score = sum(predictions[model] * weights.get(model, 0.33) 
                               for model in predictions) / len(predictions)
            
            # Confidence based on model agreement
            pred_values = list(predictions.values())
            confidence = 'high' if np.std(pred_values) < 0.1 else 'medium' if np.std(pred_values) < 0.2 else 'low'
            
            return {
                'availability_score': np.clip(ensemble_score, 0, 1),
                'confidence': confidence,
                'model_predictions': predictions,
                'model_agreement': np.std(pred_values)
            }
        
        return {'availability_score': 0.5, 'confidence': 'low', 'model_predictions': {}}
    
    def get_feature_importance(self):
        """Get feature importance from trained models"""
        
        importance_data = {}
        
        if self.rf_model and len(self.feature_columns) > 0:
            rf_importance = dict(zip(self.feature_columns, self.rf_model.feature_importances_))
            importance_data['Random Forest'] = rf_importance
        
        if self.lightgbm_model:
            lgb_importance = dict(zip(self.feature_columns, self.lightgbm_model.feature_importance()))
            importance_data['LightGBM'] = lgb_importance
        
        return importance_data
    
    def save_models(self, filepath_prefix='hotel_availability_models'):
        """Save trained models"""
        
        if self.rf_model:
            joblib.dump(self.rf_model, f'{filepath_prefix}_rf.pkl')
        
        if self.lightgbm_model:
            self.lightgbm_model.save_model(f'{filepath_prefix}_lgb.txt')
        
        if self.lstm_model:
            self.lstm_model.save(f'{filepath_prefix}_lstm.h5')
        
        # Save preprocessors
        joblib.dump(self.scaler, f'{filepath_prefix}_scaler.pkl')
        joblib.dump(self.label_encoders, f'{filepath_prefix}_encoders.pkl')
        joblib.dump(self.feature_columns, f'{filepath_prefix}_features.pkl')
    
    def load_models(self, filepath_prefix='hotel_availability_models'):
        """Load trained models"""
        
        try:
            self.rf_model = joblib.load(f'{filepath_prefix}_rf.pkl')
        except:
            pass
        
        try:
            self.lightgbm_model = lgb.Booster(model_file=f'{filepath_prefix}_lgb.txt')
        except:
            pass
        
        try:
            self.lstm_model = tf.keras.models.load_model(f'{filepath_prefix}_lstm.h5')
        except:
            pass
        
        try:
            self.scaler = joblib.load(f'{filepath_prefix}_scaler.pkl')
            self.label_encoders = joblib.load(f'{filepath_prefix}_encoders.pkl')
            self.feature_columns = joblib.load(f'{filepath_prefix}_features.pkl')
        except:
            pass 