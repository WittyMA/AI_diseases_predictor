# Multi-Modal Disease Prediction System - Deployment Guide

## Overview
This is a complete multi-modal AI-powered disease prediction system that combines machine learning models for tabular data analysis and deep learning models for medical image analysis. The system includes:

- **Backend API**: Flask-based REST API with prediction endpoints
- **Frontend UI**: React-based web interface with modern design
- **ML Models**: Support for diabetes, heart disease, kidney disease, liver disease, breast cancer, COVID-19, and chest X-ray analysis
- **Multi-modal Integration**: Framework for combining different data types

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │   ML Models     │
│   (React)       │◄──►│   (Flask)       │◄──►│   (Scikit/TF)   │
│                 │    │                 │    │                 │
│ - User Interface│    │ - Prediction    │    │ - Tabular Data  │
│ - Form Handling │    │   Endpoints     │    │ - Image Analysis│
│ - Results Display│   │ - Model Loading │    │ - Preprocessing │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Prerequisites

### System Requirements
- Python 3.11+
- Node.js 20+
- npm/pnpm
- 4GB+ RAM
- 10GB+ storage

### Dependencies
- Flask
- TensorFlow
- Scikit-learn
- Pandas, NumPy
- React
- Tailwind CSS
- Lucide React Icons

## Project Structure

```
disease_prediction_system/
├── backend/
│   └── disease_prediction_api/
│       ├── src/
│       │   ├── main.py                 # Flask application entry point
│       │   ├── routes/
│       │   │   ├── prediction.py       # Prediction API endpoints
│       │   │   └── user.py            # User management routes
│       │   ├── models/                # Database models
│       │   └── static/                # Built frontend files
│       ├── venv/                      # Python virtual environment
│       └── requirements.txt           # Python dependencies
├── frontend/
│   └── disease_prediction_ui/
│       ├── src/
│       │   ├── components/            # React components
│       │   │   ├── HomePage.jsx
│       │   │   ├── DiabetesPrediction.jsx
│       │   │   ├── HeartDiseasePrediction.jsx
│       │   │   ├── CovidSymptomsPrediction.jsx
│       │   │   ├── ChestXrayPrediction.jsx
│       │   │   ├── MultimodalPrediction.jsx
│       │   │   ├── AboutPage.jsx
│       │   │   └── Header.jsx
│       │   ├── hooks/                 # Custom React hooks
│       │   └── App.jsx               # Main React application
│       ├── package.json              # Node.js dependencies
│       └── dist/                     # Built frontend files
├── models/                           # Trained ML models directory
├── notebooks/                        # Jupyter notebooks for training
└── data/                            # Dataset storage
```

## Installation & Setup

### 1. Backend Setup

```bash
# Navigate to backend directory
cd disease_prediction_system/backend/disease_prediction_api

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables (optional)
export FLASK_ENV=development
export FLASK_DEBUG=1
```

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd disease_prediction_system/frontend/disease_prediction_ui

# Install dependencies
pnpm install  # or npm install

# Build for production
pnpm run build  # or npm run build

# Copy built files to Flask static directory
cp -r dist/* ../../backend/disease_prediction_api/src/static/
```

### 3. Model Setup

Create a `models/` directory and place your trained models:

```bash
mkdir -p models/
# Place your trained models here:
# - diabetes_logistic_regression_model.pkl
# - heart_disease_random_forest_model.pkl
# - kidney_disease_gradient_boosting_model.pkl
# - liver_disease_logistic_regression_model.pkl
# - breast_cancer_svm_model.pkl
# - covid19_symptoms_random_forest_model.pkl
# - pneumonia_detection_model.h5
# - covid19_detection_model.h5
# And corresponding preprocessors
```

## Local Development

### Running the Backend
```bash
cd backend/disease_prediction_api
source venv/bin/activate
python src/main.py
```
The API will be available at `http://localhost:5000`

### Running the Frontend (Development)
```bash
cd frontend/disease_prediction_ui
pnpm run dev --host
```
The frontend will be available at `http://localhost:5173`

### Full Stack (Production Mode)
```bash
# Build frontend and copy to Flask static
cd frontend/disease_prediction_ui
pnpm run build
cp -r dist/* ../../backend/disease_prediction_api/src/static/

# Run Flask server
cd ../../backend/disease_prediction_api
source venv/bin/activate
python src/main.py
```
Access the full application at `http://localhost:5000`

## API Endpoints

### Health Check
- `GET /api/health` - System health status

### Prediction Endpoints
- `POST /api/predict/diabetes` - Diabetes prediction
- `POST /api/predict/heart_disease` - Heart disease prediction
- `POST /api/predict/covid_symptoms` - COVID-19 symptoms assessment
- `POST /api/predict/chest_xray` - Chest X-ray analysis
- `POST /api/predict/multimodal` - Multi-modal prediction (future)

### Model Status
- `GET /api/models/status` - Check loaded models status

## Deployment Options

### 1. Cloudflare Workers/Pages
```bash
# Install Wrangler CLI
npm install -g wrangler

# Login to Cloudflare
wrangler login

# Deploy frontend to Cloudflare Pages
cd frontend/disease_prediction_ui
wrangler pages deploy dist

# For backend, use Cloudflare Workers with Python runtime
# Or deploy to a cloud provider that supports Flask
```

### 2. Heroku Deployment
```bash
# Install Heroku CLI and login
heroku login

# Create Heroku app
heroku create your-disease-prediction-app

# Set Python buildpack
heroku buildpacks:set heroku/python

# Deploy
git add .
git commit -m "Deploy disease prediction system"
git push heroku main
```

### 3. Docker Deployment
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY backend/disease_prediction_api/ .
RUN pip install -r requirements.txt

EXPOSE 5000
CMD ["python", "src/main.py"]
```

```bash
# Build and run Docker container
docker build -t disease-prediction-system .
docker run -p 5000:5000 disease-prediction-system
```

### 4. AWS/GCP/Azure
- Use their respective app services (Elastic Beanstalk, App Engine, App Service)
- Upload the backend directory
- Configure environment variables
- Set up load balancing and auto-scaling as needed

## Environment Variables

Create a `.env` file in the backend directory:

```env
FLASK_ENV=production
FLASK_DEBUG=0
SECRET_KEY=your-secret-key-here
MODEL_PATH=/path/to/models
CORS_ORIGINS=https://yourdomain.com
```

## Model Training

Use the provided Jupyter notebooks to train your models:

1. `notebooks/diabetes_prediction.ipynb`
2. `notebooks/heart_disease_prediction.ipynb`
3. `notebooks/kidney_disease_prediction.ipynb`
4. `notebooks/liver_disease_prediction.ipynb`
5. `notebooks/breast_cancer_prediction.ipynb`
6. `notebooks/covid19_symptoms_prediction.ipynb`
7. `notebooks/chest_xray_pneumonia_detection.ipynb`
8. `notebooks/chest_xray_covid19_detection.ipynb`
9. `notebooks/multimodal_integration.ipynb`

## Data Requirements

### Tabular Data Features

**Diabetes Prediction:**
- Pregnancies, Glucose, BloodPressure, SkinThickness
- Insulin, BMI, DiabetesPedigreeFunction, Age

**Heart Disease Prediction:**
- age, sex, cp, trestbps, chol, fbs, restecg
- thalach, exang, oldpeak, slope, ca, thal

**COVID-19 Symptoms:**
- Age, Gender, Contact, Fever, Tiredness, Dry_Cough
- Difficulty_in_Breathing, Sore_Throat, Pains, etc.

### Image Data
- Chest X-rays in JPEG/PNG format
- Recommended size: 224x224 or 150x150 pixels
- Grayscale or RGB format supported

## Security Considerations

1. **Data Privacy**: No data is stored permanently
2. **HTTPS**: Use SSL certificates in production
3. **CORS**: Configure appropriate origins
4. **Input Validation**: All inputs are validated
5. **Rate Limiting**: Implement API rate limiting
6. **Authentication**: Add user authentication if needed

## Monitoring & Logging

```python
# Add to main.py for logging
import logging
logging.basicConfig(level=logging.INFO)

# Add health monitoring endpoints
@app.route('/health')
def health_check():
    return {'status': 'healthy', 'timestamp': datetime.now().isoformat()}
```

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Check model file paths
   - Verify model compatibility with current libraries
   - Ensure sufficient memory

2. **CORS Issues**
   - Configure Flask-CORS properly
   - Check frontend API endpoint URLs

3. **Build Failures**
   - Clear node_modules and reinstall
   - Check for missing dependencies
   - Verify Node.js version compatibility

4. **Memory Issues**
   - Reduce model size or use model quantization
   - Implement model lazy loading
   - Use cloud services with more memory

### Performance Optimization

1. **Model Optimization**
   - Use model quantization
   - Implement model caching
   - Use GPU acceleration if available

2. **Frontend Optimization**
   - Enable code splitting
   - Implement lazy loading
   - Optimize images and assets

3. **Backend Optimization**
   - Use gunicorn for production
   - Implement Redis caching
   - Use CDN for static assets

## Support & Maintenance

### Regular Updates
- Update dependencies regularly
- Retrain models with new data
- Monitor system performance
- Update security patches

### Backup Strategy
- Backup trained models
- Version control all code
- Document model training procedures
- Maintain deployment scripts

## License & Disclaimer

This system is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.

## Contact Information

For technical support or questions about deployment, please refer to the documentation or contact the development team.

---

**Last Updated**: December 2024
**Version**: 1.0.0

