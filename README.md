# Texas Residential Real Estate Price Prediction

## Project Overview

This project focuses on predicting residential property prices in Texas using machine learning techniques. The dataset contains detailed information about real estate properties including size, number of bedrooms and bathrooms, property type, garage capacity, construction year, and textual property descriptions.

The project combines traditional structured features with Natural Language Processing (NLP) techniques to extract additional information from property descriptions, resulting in a more accurate price prediction system.

## Objectives

* Analyze Texas residential real estate data.
* Handle missing values using advanced preprocessing techniques.
* Extract hidden information from textual descriptions using NLP.
* Engineer meaningful features for price prediction.
* Compare multiple machine learning models.
* Optimize model performance using hyperparameter tuning.

## Dataset

The dataset contains residential property listings with the following features:

| Feature        | Description                  |
| -------------- | ---------------------------- |
| Price          | Property listing price       |
| Sqft           | Property area in square feet |
| Beds           | Number of bedrooms           |
| Baths          | Number of bathrooms          |
| Baths Full     | Number of full bathrooms     |
| Garage         | Garage capacity              |
| Year Built     | Construction year            |
| Stories        | Number of stories            |
| Type           | Property type                |
| Text           | Property description         |
| Price Per SqFt | Price per square foot        |

## Exploratory Data Analysis (EDA)

Several visualizations were performed:

* Distribution analysis
* Outlier detection using boxplots
* Price vs. square footage relationships
* Property type comparisons
* Bedroom and bathroom analysis
* Story count influence on pricing
* Correlation exploration between numerical features

Libraries used:

```python
Matplotlib
Seaborn
Plotly
```

## Data Preprocessing

### Missing Value Handling

* Filled missing values using:

  * Mean Imputation
  * Mode Imputation
  * KNN Imputer

### Feature Cleaning

* Corrected missing story values using property subtype mappings.
* Reconstructed missing Price_Per_SqFt values.
* Removed unnecessary columns.
* Converted numerical fields to appropriate data types.

## Natural Language Processing (NLP)

Property descriptions were analyzed using:

### SpaCy

Used to extract:

* Number of bedrooms
* Number of bathrooms

from textual property descriptions.

Example:

```text
Beautiful home with four bedrooms and three bathrooms.
```

Extracted as:

```text
Beds = 4
Baths = 3
```

### TF-IDF Vectorization

Text descriptions were transformed into numerical features using:

```python
TfidfVectorizer(max_features=5000)
```

## ⚙ Feature Engineering

### Structured Features

* Property size
* Bedrooms
* Bathrooms
* Garage
* Stories
* Year built
* Property type

### Encoded Features

Property types were converted using:

```python
pd.get_dummies()
```

### Scaling

Robust scaling was applied:

```python
RobustScaler()
```

### Final Feature Matrix

Combined:

* TF-IDF textual features
* Structured numerical features

using:

```python
scipy.sparse.hstack()
```

---

## Models Used

### 1. Ridge Regression

```python
Ridge(alpha=50)
```

Used as a strong linear baseline model.

### 2. Support Vector Regression (SVR)

Hyperparameter tuning performed using:

```python
GridSearchCV
```

Parameters searched:

```python
C
Kernel
Gamma
Epsilon
```

Kernels tested:

* Linear
* Polynomial
* RBF

## 📈 Model Evaluation

Evaluation metric:

```python
R² Score
```

Formula:

R² measures how well the model explains variance in housing prices.

Higher values indicate better predictive performance.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* SpaCy
* SciPy
* Matplotlib
* Seaborn
* Plotly

## Future Improvements

* XGBoost Regressor
* LightGBM
* CatBoost
* Deep Learning Regression Models
* Advanced Text Embeddings (BERT)
* Model Deployment using Streamlit or FastAPI

## Results

The combination of structured property features and NLP-based textual information significantly improved price prediction performance.

The tuned SVR model achieved the best overall predictive accuracy compared to baseline regression methods.
