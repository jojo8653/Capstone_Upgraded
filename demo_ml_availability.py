#!/usr/bin/env python3
"""
Demo script for testing ML-powered hotel availability prediction
"""

import pandas as pd
import numpy as np
from hotel_ml_availability import HotelAvailabilityPredictor
from datetime import datetime, timedelta

def load_sample_data():
    """Load sample hotel booking data for testing"""
    try:
        df = pd.read_csv('hotel_bookings.csv')
        print(f"✅ Loaded {len(df)} booking records")
        return df
    except FileNotFoundError:
        print("❌ hotel_bookings.csv not found. Creating sample data...")
        
        # Create synthetic data for testing
        np.random.seed(42)
        n_records = 1000
        
        sample_data = {
            'hotel': np.random.choice(['Hotel A', 'Hotel B', 'Hotel C', 'Hotel D'], n_records),
            'country': np.random.choice(['PRT', 'GBR', 'FRA', 'ESP', 'USA'], n_records),
            'arrival_date_year': np.random.choice([2022, 2023], n_records),
            'arrival_date_month': np.random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], n_records),
            'arrival_date_day_of_month': np.random.randint(1, 29, n_records),
            'stays_in_week_nights': np.random.randint(0, 7, n_records),
            'stays_in_weekend_nights': np.random.randint(0, 3, n_records),
            'adults': np.random.randint(1, 5, n_records),
            'children': np.random.randint(0, 3, n_records),
            'babies': np.random.randint(0, 2, n_records),
            'is_canceled': np.random.choice([0, 1], n_records, p=[0.7, 0.3]),
            'lead_time': np.random.randint(0, 365, n_records),
            'adr': np.random.normal(100, 30, n_records),
            'market_segment': np.random.choice(['Direct', 'Corporate', 'Online TA', 'Groups'], n_records)
        }
        
        df = pd.DataFrame(sample_data)
        print(f"✅ Created {len(df)} synthetic booking records")
        return df

def test_ml_predictor():
    """Test the ML availability predictor"""
    
    print("🚀 Testing ML Hotel Availability Predictor")
    print("=" * 50)
    
    # Load data
    df = load_sample_data()
    
    # Initialize predictor
    predictor = HotelAvailabilityPredictor()
    
    # Train models
    print("\n🔄 Training ML models...")
    try:
        results = predictor.train_models(df)
        
        print("\n📊 Training Results:")
        for model, metrics in results.items():
            print(f"\n{model}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        # Test predictions
        print("\n🔮 Testing Predictions...")
        
        # Test on a subset of hotels
        unique_hotels = df[['hotel', 'country']].drop_duplicates().head(3)
        
        for _, hotel_info in unique_hotels.iterrows():
            hotel_data = df[(df['hotel'] == hotel_info['hotel']) & 
                          (df['country'] == hotel_info['country'])]
            
            if len(hotel_data) > 0:
                prediction = predictor.predict_availability(hotel_data)
                
                print(f"\n🏨 {hotel_info['hotel']} ({hotel_info['country']}):")
                print(f"  Availability Score: {prediction['availability_score']:.3f}")
                print(f"  Confidence: {prediction['confidence']}")
                print(f"  Model Predictions: {prediction.get('model_predictions', {})}")
        
        # Feature importance
        print("\n🎯 Feature Importance:")
        importance_data = predictor.get_feature_importance()
        
        for model_name, importances in importance_data.items():
            print(f"\n{model_name} - Top 5 Features:")
            sorted_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
            for feature, importance in sorted_features:
                print(f"  {feature}: {importance:.4f}")
        
        # Save models
        print("\n💾 Saving trained models...")
        predictor.save_models('demo_models')
        print("✅ Models saved successfully!")
        
        # Test loading
        print("\n📂 Testing model loading...")
        new_predictor = HotelAvailabilityPredictor()
        new_predictor.load_models('demo_models')
        
        # Test prediction with loaded models
        if len(unique_hotels) > 0:
            hotel_info = unique_hotels.iloc[0]
            hotel_data = df[(df['hotel'] == hotel_info['hotel']) & 
                          (df['country'] == hotel_info['country'])]
            
            prediction = new_predictor.predict_availability(hotel_data)
            print(f"✅ Loaded model prediction: {prediction['availability_score']:.3f}")
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ml_predictor() 