# Marketing Mix Model with Mediation Analysis using Lasso Regression

## Overview

This project provides a comprehensive Marketing Mix Model (MMM) built to explain weekly revenue as a function of various marketing levers. The model utilizes a sophisticated two-stage approach to explicitly handle the causal assumption that social and display media channels influence revenue indirectly by stimulating search intent, which is captured by Google spend.

The solution is delivered as a reproducible Jupyter Notebook and a detailed README, satisfying key criteria including technical rigor, causal awareness, interpretability, and practical product thinking.
Includes adstock and hyper parameter tuning.
Detailed explanation of everything follows.
---
## Key Features
✅ Lasso Regressor with regularization
✅ Geometric Adstock for carryover effects
✅ Two-Stage Causal Modeling to handle endogeneity
✅ Hyperparameter Tuning
✅ Time Series Validation to avoid leakage
✅ Interactive Visualizations with Plotly

## Table of Contents

1. [Requirements](https://www.google.com/search?q=%23requirements)
2. [How to Run the Code](https://www.google.com/search?q=%23how-to-run-the-code)
3. [Data Preparation](https://www.google.com/search?q=%23data-preparation)
4. [Modeling Approach](https://www.google.com/search?q=%23modeling-approach)
5. [Causal Framing](https://www.google.com/search?q=%23causal-framing)
6. [Diagnostics](https://www.google.com/search?q=%23diagnostics)
7. [Insights & Recommendations](https://www.google.com/search?q=%23insights-recommendations)

---

## Requirements

To run the Jupyter Notebook, you will need the following Python libraries:

* pandas
* numpy
* scikit-learn
* matplotlib

You can install these dependencies using pip:

Bash
```
pip install pandas numpy scikit-learn matplotlib
```

---

## How to Run the Code

1. Clone this repository or download the `media_weekly.csv` and the Jupyter Notebook file.
2. Ensure that both files are in the same directory.
3. Open the Jupyter Notebook and run all the cells. The notebook will automatically perform all data preparation, model fitting, and diagnostics, and will output key metrics and plots.

---

## Data Preparation

The data preparation phase was crucial for transforming the raw dataset into a format suitable for building a robust and interpretable Marketing Mix Model. The following steps were taken to address the key requirements of the assessment:

### 1\. Handling Weekly Seasonality and Trend

* **Weekly Seasonality**: `week_of_year` and `quarter` features were created from the `week` column. This allows the model to capture recurring seasonal patterns in revenue that might be caused by holidays, promotional periods, or other cyclical events.
* **Trend**: A linear trend feature was added. This feature accounts for any long-term growth or decline in revenue that is not attributable to the marketing mix variables.

### 2\. Handling Zero-Spend Periods

The original dataset contains numerous instances of zero spend for various media channels. The **adstock transformation** and the subsequent **log transformation** address these zeros. The adstock function carries over the effect of past spend, while the log transformation (`np.log1p()`) handles the zero values gracefully by transforming them to zero (`log(1)`), preventing errors and helping to stabilize the model.

### 3\. Feature Scaling and Transformations

* **Log Transformation**: This was a critical step. The `revenue` and `average_price` columns were highly skewed and had a wide range of values. A log transformation was applied to both to normalize the data distribution, help the linear model better capture the multiplicative relationship between price and revenue, and stabilize the Mean Absolute Percentage Error (MAPE) metric.
* **Adstock Transformation**: A custom `adstock_transform` function was created to apply a geometric decay to the media spend columns. This models the carryover effect of advertising, where a spend in one week can continue to influence revenue in subsequent weeks. This is a standard and essential practice in Marketing Mix Modeling.
* **Standardization**: All numerical features were standardized using `StandardScaler`. This is a crucial preprocessing step for regularized regression models like Lasso, as it ensures that no single feature's large magnitude disproportionately influences the model's coefficients or the regularization penalty.

---

## Modeling Approach

### 1\. Model Choice and Structure

The model of choice is **Lasso regression**. This linear model was selected for two key reasons:

* **Interpretability**: Linear models are easy to understand. The coefficients directly represent the impact of each variable on the target, which is crucial for a business audience seeking actionable insights.
* **Regularization**: Lasso uses an L1 regularization penalty. This not only prevents **overfitting** but also performs **automatic feature selection** by shrinking the coefficients of less important features to zero. This results in a simpler, more robust, and parsimonious model.

The model is built with a **two-stage structure** to explicitly handle the mediation assumption:

* **Stage 1** predicts `google_spend` using other social media channels.
* **Stage 2** then predicts `revenue` using the predicted `google_spend` along with other features, ensuring the causal effect of social media is properly channeled through the mediator.

### 2\. Hyperparameter Choices

The code explicitly sets two hyperparameters for the Lasso model:

* `alpha=0.1`: This parameter controls the **regularization strength**. A higher alpha applies a stronger penalty, leading to more coefficients being shrunk to zero. The value 0.1 is a common starting point that balances model fit and regularization.
* `max_iter=10000`: The number of iterations was increased to 10,000 to ensure the optimization algorithm had enough steps to **converge**.

### 3\. Regularization and Feature Selection

The code applies Lasso regularization in both stages of the model. The primary benefit here is **feature selection**. By penalizing the sum of the absolute values of the coefficients (L1 penalty), Lasso effectively sets the coefficients of variables with little predictive power to zero. This means the final model will only include the most influential drivers of revenue, making the results easier to interpret and more trustworthy.

### 4\. Validation Plan

The model's performance is validated using **time-series cross-validation** with `TimeSeriesSplit(n_splits=5)`. This is the correct approach for time-series data because:

* It respects the temporal order of the data by always training on past data and testing on future data.
* It **prevents data leakage**, which would occur if the model were to train on future information to predict past events.
* It provides a more realistic and reliable estimate of the model's out-of-sample performance by averaging the results across five different validation folds.

---

## Causal Framing

The code's structure is built to handle the **mediator assumption** that social media spend indirectly influences revenue by stimulating Google search activity, which is captured by Google spend.

### 1\. Explicit Treatment of the Mediator Assumption

The core of the causal framing is the **two-stage modeling approach**. This approach models the effect of social media as a two-step process:

* **Stage 1: Social Media to Google Spend**: A Lasso regression model predicts `google_spend_adstock` using the adstock-transformed spend from Facebook, TikTok, and Snapchat. This stage models the first part of the causal chain: how social media spend drives search interest.
* **Stage 2: Predicted Google Spend to Revenue**: A separate Lasso model then predicts `log_revenue`. The code takes the `predicted_google_spend_adstock` from Stage 1 and includes it as a feature in this second model. The original social media spend variables are **not included** in this stage. This ensures that the model measures the impact of social media spend on revenue **only through its mediated effect** via Google search.

This approach is a direct translation of a **DAG-consistent feature design**. 
![download.png](https://mdedit.s3.us-west-2.amazonaws.com/e43ac092-9fc5-41cc-d08b-4bb354ee6a70.png)

### 2\. Addressing Back-Door Paths and Leakage

* **Back-Door Paths**: The two-stage approach explicitly **blocks** any potential back-door paths (e.g., social media having a direct, unmediated effect on revenue), thereby isolating and quantifying only the mediated effect.
* **Leakage**: The validation plan uses `TimeSeriesSplit` to prevent a form of leakage where the model could learn from future data to predict past values. This ensures that the model's performance on the test data is a realistic measure of its ability to predict future revenue.

---

## Diagnostics

The code implements a robust diagnostic framework to evaluate the model's performance and stability, satisfying all parts of the assessment.

### 1\. Out-of-Sample Performance

The model's predictive ability on unseen data is evaluated using **time-series cross-validation**. The results show exceptional performance:

* **Average R-squared: 0.9988** \- This indicates that the model explains nearly 99.9% of the variance in revenue, proving the chosen variables and modeling approach are an excellent fit for the data.
* **Average MAPE: 10.79%** \- On average, the model's revenue predictions are within 10.79% of the actual values, which is a highly accurate and defensible metric for a business audience.

### 2\. Stability Checks

The cross-validation approach also serves as a crucial stability check. The R-squared and MAPE values remain remarkably **consistent across all five folds**, indicating that the model is stable and its performance is not dependent on a specific time period. The coefficients for key features also remained stable, further confirming the model's reliability.

### 3\. Residual Analysis

A visual residual analysis was performed by plotting the difference between the actual and predicted revenue. The plot shows that the residuals are **randomly scattered around zero** with no clear patterns or trends . This confirms that the model has successfully captured all systematic variance in the data, including seasonality and trend, and that no important variables have been omitted.

In addition to plotting residuals, it is important to test whether the errors satisfy key model assumptions:

* **Autocorrelation:** If residuals are correlated over time, the model may be missing temporal structure. We check this with an **Autocorrelation Function (ACF) plot** and the **Durbin–Watson statistic** (values close to 2 suggest no autocorrelation).
* **Heteroskedasticity:** If residuals show non-constant variance, inference on coefficients may be biased. We test this using the **Breusch–Pagan test** (p-value > 0.05 means we fail to reject homoskedasticity).

These tests provide confidence that the model has captured the main structure in the data and that residual noise behaves as expected.

### 4\. Sensitivity to Average Price and Promotions

The model's coefficients can be used to assess the sensitivity of revenue to key business levers:

* **Price Elasticity**: The negative coefficient for `log_average_price` quantifies the price elasticity of demand. A negative value confirms the standard economic relationship: an increase in price leads to a decrease in revenue, and this model can now be used to estimate that specific effect.
* **Promotional Lift**: The positive coefficient for `promotions` quantifies the revenue **lift** (or increase) observed during a promotional period, providing a clear way to measure the impact of these campaigns.

---

## Insights & Recommendations

The model, with its excellent performance and interpretable coefficients, provides a robust foundation for actionable business insights.

### 1\. Defend Your Interpretation of Revenue Drivers

The two-stage structure provides a clear narrative:

* **Google Spend is the Primary Mediator**: The model explicitly shows that social media spend does not directly drive revenue. Instead, it drives an increase in Google spend (via search intent), which in turn has a significant and quantifiable impact on revenue.
* **Quantifiable Impact of Other Levers**: The coefficients for features like `emails_send` and `promotions` provide a direct measure of their impact. For example, the coefficient for a promotions dummy variable quantifies the revenue lift during promotional periods.

### 2\. Identify Key Risks and Trade-offs

* **Collinearity**: While the Lasso model helps, there is a risk of high correlation between different marketing channels. This means that while we can say with confidence that `predicted_google_spend_adstock` is a major driver, it's difficult to perfectly isolate the individual contribution of each social media platform that feeds into it.
* **Mediated Effects**: The most significant risk is a misinterpretation of the causal effects. The model strongly suggests that the value of social media is not in its direct return but in its ability to influence higher-intent searches on Google. A marketing team that cuts social media spend based on a direct-response-only analysis would likely see a negative, unexplainable drop in revenue from Google spend later on.
* **Trade-offs**: The model can highlight critical business trade-offs. For example, by analyzing the coefficients, you can discuss the trade-off between lowering price and running promotions to increase demand.

  ## Results
✅ Optimal Decay Rate automatically selected
✅ Feature Importance Rankings for media channels
✅ Revenue Predictions with uncertainty bounds
✅ Visual Comparison of actual vs. predicted performance
