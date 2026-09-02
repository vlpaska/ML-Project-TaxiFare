# NYC Taxi Fare & Trip Duration Prediction

**Machine Learning applied to large-scale urban mobility data**

Machine Learning Project | 2024

---

## Project Overview

This project explores more than **11 million New York City Yellow Taxi trips** to understand urban mobility patterns and develop machine learning models capable of predicting taxi fares and trip duration.

Using trip characteristics such as distance, pickup and drop-off locations, passenger count, and temporal information, the project develops an end-to-end data science pipeline covering:

- Large-scale data preprocessing
- Exploratory Data Analysis (EDA)
- Geospatial and temporal feature engineering
- Outlier detection and treatment
- Regression modeling
- Model comparison and evaluation
- Interactive fare estimation

The project demonstrates how historical transportation data can be transformed into predictive tools for estimating the cost and duration of taxi journeys.

---

## Objectives

The project initially explores three potential applications:

1. **Taxi fare prediction**
2. **Trip duration prediction**
3. **Pickup-density analysis**

The implemented machine learning pipeline focuses primarily on the first two objectives.

---

## Dataset

The project uses the **New York City Yellow Taxi Trip Dataset** collected by the New York City Taxi & Limousine Commission (TLC).

The analysis focuses on:

**February 2016**

The original dataset contains approximately:

- **11.38 million trips**
- **19 variables**

Features include:

- Pickup and drop-off timestamps
- Pickup and drop-off coordinates
- Trip distance
- Passenger count
- Fare amount
- Rate code
- Payment information
- Additional trip-related charges

The original dataset is not included in this repository.

---

## Machine Learning Pipeline

```text
NYC Yellow Taxi Data
        ↓
Data Cleaning
        ↓
Outlier Detection & Treatment
        ↓
Feature Engineering
        ↓
Exploratory Data Analysis
        ↓
Train / Test Split
        ↓
Regression Models
        ├── Linear Regression
        ├── Decision Tree
        └── Random Forest
        ↓
Model Evaluation
        ↓
Interactive Fare Estimator
```

---

## 1. Data Cleaning & Preprocessing

The raw dataset was cleaned before exploratory analysis and modeling.

The preprocessing pipeline includes:

- Data-type validation
- Datetime conversion
- Removal of irrelevant variables
- Duplicate detection and removal
- Removal of negative fare values
- Removal of trips with zero passengers
- Geographic filtering
- Outlier detection using the Interquartile Range (IQR)
- Outlier capping for selected numerical variables
- Memory optimization for large-scale processing

### Geographic Filtering

Pickup and drop-off coordinates were restricted to the New York City geographical area in order to remove invalid or inconsistent observations.

This was particularly important because erroneous latitude and longitude values can significantly affect geospatial features and predictive models.

---

## 2. Feature Engineering

Several additional variables were created from the original dataset.

### Trip Duration

Trip duration was calculated from pickup and drop-off timestamps:

```python
df["trip_duration_minutes"] = (
    df["tpep_dropoff_datetime"] -
    df["tpep_pickup_datetime"]
).dt.total_seconds() / 60
```

### Estimated Speed

Average trip speed was derived from distance and duration:

```python
df["speed"] = (
    df["trip_distance"] /
    (df["trip_duration_minutes"] / 60)
)
```

### Temporal Features

Additional time-based variables include:

- Pickup hour
- Drop-off hour
- Day of the week
- Weekday vs. weekend
- Time-of-day category
- Cyclical hour encoding

These variables allow the models to capture temporal patterns in taxi demand and trip characteristics.

---

## 3. Exploratory Data Analysis

Exploratory analysis was conducted to understand the structure and behavior of NYC taxi trips before model training.

The analysis investigates:

- Passenger-count distributions
- Trip-distance distributions
- Fare distributions
- Trip-duration distributions
- Average driving speed
- Pickup and drop-off activity
- Weekday vs. weekend patterns
- Time-of-day patterns
- Geographic coordinates
- Relationships between distance, duration, and fare

The analysis shows that taxi activity and fare characteristics vary according to trip distance, location, and time.

---

## 4. Taxi Fare Prediction

The first machine learning task predicts the **fare amount** of a taxi trip.

Features include:

- Passenger count
- Trip distance
- Pickup longitude and latitude
- Drop-off longitude and latitude
- Pickup hour
- Day-of-week information

Multiple regression approaches were evaluated.

### Linear Regression

Linear Regression provides a simple and interpretable baseline.

Model performance:

| Metric | Result |
|---|---:|
| MAE | **$1.34** |
| RMSE | **$1.98** |
| R² | **0.9003** |

The model explains approximately **90% of the variance in taxi fares**.

### Decision Tree Regressor

A Decision Tree was then evaluated to capture potentially non-linear relationships between trip characteristics and fare amount.

The Decision Tree slightly improved performance compared with the Linear Regression baseline while maintaining a relatively simple model structure.

---

## 5. Trip Duration Prediction

The second machine learning task predicts **trip duration**.

The models use:

- Passenger count
- Trip distance
- Pickup coordinates
- Drop-off coordinates
- Pickup hour
- Day-of-week information

Two tree-based approaches were compared.

### Decision Tree

| Metric | Result |
|---|---:|
| MAE | **3.21 min** |
| RMSE | **4.36 min** |
| R² | **0.7136** |

### Random Forest

| Metric | Result |
|---|---:|
| MAE | **3.00 min** |
| RMSE | **4.58 min** |
| R² | **0.7801** |

The Random Forest achieved the highest R² and lowest Mean Absolute Error, explaining approximately **78% of the variance in trip duration**.

---

## 6. Interactive Fare Estimation

The final section explores an interactive fare-prediction interface using **ipywidgets**.

Users can modify trip characteristics and obtain an estimated taxi fare from the trained model.

This demonstrates how the machine learning pipeline could form the basis of a user-facing application for estimating taxi costs before a journey.

---

## Technologies

### Data Analysis
- Python
- pandas
- NumPy

### Machine Learning
- scikit-learn
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

### Data Visualization
- Matplotlib
- Seaborn

### Interactive Tools
- ipywidgets
- Jupyter Notebook

### Techniques
- Feature Engineering
- IQR Outlier Detection
- Geospatial Filtering
- Cyclical Time Encoding
- Train/Test Splitting
- Regression Analysis
- Model Evaluation

---

## Model Evaluation

Models were evaluated using standard regression metrics:

**Mean Absolute Error (MAE)**  
Measures the average absolute difference between predicted and actual values.

**Root Mean Squared Error (RMSE)**  
Penalizes larger prediction errors more heavily.

**R² Score**  
Measures the proportion of variance in the target variable explained by the model.

Using multiple metrics provides a more complete evaluation than relying on a single performance measure.

---

## Repository Structure

```text
nyc-taxi-machine-learning/
│
├── Taxi_Fare_ML_Project.ipynb
├── README.md
└── .gitignore
```

The notebook contains the complete workflow from data preprocessing and exploratory analysis to machine learning and interactive fare estimation.

---

## Limitations & Future Improvements

Several extensions could improve the project:

- Hyperparameter optimization
- Cross-validation
- Integration of real-time traffic data
- Weather information
- More advanced geospatial features
- Pickup-density forecasting
- Gradient boosting models such as XGBoost or LightGBM
- Model deployment through a web application or API

Traffic and weather data would be particularly useful for improving trip-duration predictions, as these factors are not captured directly by the available trip variables.

---

## Author

**Victoria Pascarella**

Machine Learning Project — 2024

---

## Disclaimer

This project was developed for educational and portfolio purposes.

The original NYC Taxi dataset is not redistributed through this repository.
