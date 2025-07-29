# HotelOptix Disaster Response Tool - Deployment Guide

## 🚀 Quick Start

### Option 1: Basic Mode (Streamlit Cloud)
```bash
pip install -r requirements.txt
streamlit run hotel_disaster_cloud.py
```

**Features Available:**
- ✅ Disaster simulation
- ✅ Emergency rebooking (traditional algorithms)
- ✅ Worker relocation
- ✅ Financial analytics
- ❌ ML predictions (not available)

### Option 2: Full ML Mode (Local Development)
```bash
pip install -r requirements_ml.txt
streamlit run hotel_disaster_cloud.py
```

**Features Available:**
- ✅ All basic features
- ✅ LSTM, LightGBM, Random Forest models
- ✅ Real-time availability predictions
- ✅ Model training & evaluation
- ✅ Feature importance analysis

## 📋 Requirements

### Basic Mode (`requirements.txt`)
- pandas, numpy, streamlit
- matplotlib, seaborn
- ~50MB total

### Full ML Mode (`requirements_ml.txt`)
- All basic requirements
- scikit-learn, lightgbm, tensorflow
- ~500MB total (includes TensorFlow)

## 🔧 Troubleshooting

### "ModuleNotFoundError" on Streamlit Cloud
- **Solution**: Use `requirements.txt` (basic mode)
- **Cause**: Streamlit Cloud has memory/dependency limits

### Local ML Installation Issues
```bash
# Try installing dependencies individually:
pip install tensorflow==2.13.0
pip install lightgbm==4.0.0
pip install scikit-learn==1.3.0
```

### Memory Issues with Large Models
- The app automatically falls back to basic mode if ML dependencies aren't available
- No manual intervention required

## 🌐 Streamlit Cloud Deployment

1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Use `requirements.txt` (not `requirements_ml.txt`)
4. App will run in basic mode automatically

## 🎯 Data Requirements

Place `hotel_bookings.csv` in the same directory as the app. The app will create sample data if the file is missing.

## 📊 ML Model Files

When ML models are trained locally, they save as:
- `hotel_availability_models_rf.pkl`
- `hotel_availability_models_lgb.txt` 
- `hotel_availability_models_lstm.h5`

These files can be shared between environments. 