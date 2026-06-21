# House Price Prediction Analysis

An end-to-end Machine Learning project analyzing and predicting house prices using a dataset of 545 residential properties. This project covers data exploration, cleaning, pre-processing (encoding, handling nulls/duplicates), regression modeling, and performance visualization.

---

## Dataset Overview
The analysis uses the dataset [Housing.csv](file:///c:/Users/arka2/Downloads/HousePricePrediction_Arka%20Roy/Housing.csv), which contains **545 rows** and **13 columns** detailing various attributes of residential properties:

- **Target Variable**:
  - `price`: Sale price of the house.
- **Predictive Features**:
  - `area`: Total surface area of the property (in sq ft).
  - `bedrooms`: Number of bedrooms.
  - `bathrooms`: Number of bathrooms.
  - `stories`: Number of floors/stories.
  - `parking`: Number of parking spaces.
  - `mainroad` *(Categorical)*: Property is located on a main road (yes/no).
  - `guestroom` *(Categorical)*: Property has a guest room (yes/no).
  - `basement` *(Categorical)*: Property has a basement (yes/no).
  - `hotwaterheating` *(Categorical)*: Property has hot water heating (yes/no).
  - `airconditioning` *(Categorical)*: Property has air conditioning (yes/no).
  - `prefarea` *(Categorical)*: Property is in a preferred neighborhood (yes/no).
  - `furnishingstatus` *(Categorical)*: Furnishing level (furnished, semi-furnished, unfurnished).

---

## Project Pipeline

The step-by-step process is documented inside the Jupyter notebook: [analysis.ipynb](file:///c:/Users/arka2/Downloads/HousePricePrediction_Arka%20Roy/analysis.ipynb).

### 1. Data Exploration & Quality Check
*   Loaded data via `pandas` and checked dimensions (`545 rows, 13 columns`).
*   Identified target variable (`price`) and descriptive features.
*   Verified null/missing values across all columns (none found in raw data).

### 2. Data Preprocessing & Cleaning
*   **Imputation**: Verified null counts and established standard median/mode imputation fallback strategies.
*   **Deduplication**: Checked and removed duplicate entries.
*   **Encoding**: Converted yes/no categorical binary variables and multi-class categorical features (`furnishingstatus`) using one-hot encoding (`pd.get_dummies(drop_first=True)`).

### 3. Machine Learning Modeling
*   Split data into **80% training** (436 samples) and **20% test** (109 samples) sets with a controlled random state (`42`) for reproducibility.
*   Trained and evaluated two regression algorithms:
    1.  **Linear Regression** (Baseline regression model).
    2.  **Random Forest Regressor** (Ensemble learner with 100 decision trees).

### 4. Metrics & Evaluation
Models were compared using key regression metrics:
*   **Mean Absolute Error (MAE)**: Average magnitude of prediction errors.
*   **Root Mean Squared Error (RMSE)**: Standard deviation of residuals.
*   **Coefficient of Determination ($R^2$ Score)**: Proportion of variance explained by features.

---

## Model Performance & Comparison

| Metric | Linear Regression | Random Forest | Better Model |
| :--- | :---: | :---: | :---: |
| **MAE** | **970,043.40** | 1,021,546.04 | **Linear Regression** |
| **RMSE** | **1,324,506.96** | 1,400,565.97 | **Linear Regression** |
| **$R^2$ Score** | **65.29%** | 61.19% | **Linear Regression** |

### Key Insight
On this specific dataset, **Linear Regression outperformed the Random Forest Regressor**.
*   **Reasoning**: The dataset is relatively small (545 rows) and has highly linear features (like `area`, `bathrooms`, and `stories`) directly driving price increases. Linear regression models are less susceptible to overfitting on small sample sizes compared to complex tree-based ensemble models, which might struggle with sparsity and hyperparameter settings on a smaller scale.

---

## Visualizations

The generated plots are saved in the [charts/](file:///c:/Users/arka2/Downloads/HousePricePrediction_Arka%20Roy/charts) directory:

### Chart 1: Distribution of House Prices
Shows the frequency distribution of property prices, showing a right-skewed normal distribution peaking around 3.5M–4.5M.
*   [View Plot Image](file:///c:/Users/arka2/Downloads/HousePricePrediction_Arka%20Roy/charts/Chart%201.png)

### Chart 2: Correlation Heatmap
Provides insight into which encoded features hold the strongest positive or negative correlation with house price (e.g., `area`, `bathrooms`, and `airconditioning_yes` exhibit strong positive correlation).
*   [View Plot Image](file:///c:/Users/arka2/Downloads/HousePricePrediction_Arka%20Roy/charts/Chart%202.png)

### Chart 3: Actual vs Predicted Prices Scatter Plot
Compares predicted prices from Linear Regression against the actual validation set prices, illustrating the model's accuracy and variance.
*   [View Plot Image](file:///c:/Users/arka2/Downloads/HousePricePrediction_Arka%20Roy/charts/Chart%203.png)

---

## Repository Structure
```
HousePricePrediction_Arka Roy/
│
├── Housing.csv               # Raw Dataset
├── analysis.ipynb            # Jupyter Notebook with code & cell outputs
├── summary.pdf               # Exported PDF report summarizing findings
│
├── charts/                   # Saved visualization PNG files
│   ├── Chart 1.png
│   ├── Chart 2.png
│   └── Chart 3.png
└── README.md                 # Project README documentation
```

A compiled PDF report containing all findings and charts is also available in the root folder: [summary.pdf](file:///c:/Users/arka2/Downloads/HousePricePrediction_Arka%20Roy/summary.pdf).

---

## How to Run the Analysis

### Prerequisites
Make sure you have Python installed along with the required libraries:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Execution
Open the Jupyter notebook and run all cells:
```bash
jupyter notebook analysis.ipynb
```

---
*Created by Arka Roy*
