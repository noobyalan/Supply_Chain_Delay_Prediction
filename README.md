# Supply Chain Delay Prediction System

## Project Background and Objectives

In today's globalized supply chain networks, delivery delays can disrupt operations, strain customer relationships, and impact bottom lines. This project develops a machine learning-based system to predict order delivery delays, enabling proactive risk management for supply chain teams.

Using historical order data, we've built a predictive model that identifies high-risk shipments before delays occur. The system combines data processing pipelines, a trained XGBoost classifier, and a web API to provide actionable insights for logistics planning.

## Technical Framework

The system follows a three-tier architecture:

- **Data Layer**: PostgreSQL database storing order records, with utilities for CSV data initialization
- **Model Layer**: XGBoost classification model integrated with preprocessing pipelines (encoding, scaling)
- **Service Layer**: FastAPI-based web service for real-time predictions, with MLflow for experiment tracking

## Setup Instructions

### Prerequisites

- Python 3.8+
- PostgreSQL 12+
- pip (Python package manager)

### Environment Setup

1. Clone the repository and navigate to the project directory

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure database connection in `src/config.py`:
   ```python
   PG_CONFIG = {
       "user": "your_username",
       "password": "your_password",
       "host": "localhost",
       "port": 5432,
       "db": "supply_chain_db"
   }
   ```

4. Initialize the database with sample data:
   ```bash
   python scripts/init_db.py
   ```

## Usage Guide

### Model Training

Run the end-to-end training pipeline:
```bash
python src/training_pipeline.py
```

This script will:
- Start an MLflow tracking server
- Load data from PostgreSQL
- Preprocess features (encoding categorical variables, scaling numerical features)
- Perform hyperparameter tuning using cross-validation
- Train the final model and register it with MLflow

### Starting the Prediction Service

Launch the API server:
```bash
uvicorn api.main:app --reload
```

The service will be available at `http://127.0.0.1:8000` with these endpoints:
- `/`: Service information
- `/docs`: Interactive API documentation
- `/health`: Service status check
- `/predict`: Accepts order details and returns delay prediction

### Making Predictions

Send a POST request to `/predict` with order data in JSON format:
```json
{
    "ID": 12345,
    "Warehouse_block": "A",
    "Mode_of_Shipment": "Flight",
    "Customer_care_calls": 3,
    "Customer_rating": 4,
    "Cost_of_the_Product": 150.50,
    "Prior_purchases": 5,
    "Product_importance": "high",
    "Gender": "M",
    "Discount_offered": 10.0,
    "Weight_in_gms": 2000
}
```

The response will contain a binary prediction (0 = on time, 1 = delayed).

## Project Structure

```
Supply_Chain_Delay_Prediction/
├── src/                      # Core functionality
│   ├── data_loader.py        # Data retrieval utilities
│   ├── feature_engineering.py # Feature processing pipelines
│   ├── model.py              # Prediction model definition
│   ├── training_pipeline.py  # End-to-end training workflow
│   ├── database_io.py        # Database interaction
│   ├── utils.py              # Helper functions
│   └── config.py             # Configuration parameters
├── api/                      # Web service
│   └── main.py               # FastAPI application
├── tests/                    # Test suite
│   ├── test_data_loader.py   # Data loading tests
│   ├── test_feature_engineering.py # Feature processing tests
│   └── test_model.py         # Model training tests
├── scripts/                  # Utility scripts
│   ├── init_db.py            # Database initialization
│   └── start_mlflow_server.sh # MLflow server startup
├── requirements.txt          # Dependencies
├── pytest.ini                # Test configuration
└── LICENSE                   # MIT License
```

## Testing

Run the test suite to verify system components:
```bash
pytest
```

Tests validate data loading, feature processing, and model training functionality.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
