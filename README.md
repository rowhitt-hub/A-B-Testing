# A/B Testing Analysis of Marketing Campaign

## Overview

This project analyzes the effectiveness of two marketing strategies using **A/B testing**. The goal is to determine whether displaying a **regular advertisement (Ad)** or a **Public Service Announcement (PSA)** results in higher user conversion rates.

Using statistical hypothesis testing and exploratory analysis, the notebook evaluates whether the observed difference in conversion rates between the two groups is statistically significant.

---

## Dataset

The analysis uses the dataset:

`marketing_AB.csv`

Each row represents a unique user exposed to one of two experimental groups.

### Key Columns

| Column       | Description                                                          |
| ------------ | -------------------------------------------------------------------- |
| user id      | Unique identifier for each user                                      |
| test group   | The group the user belongs to (`ad` or `psa`)                        |
| converted    | Binary variable indicating whether the user converted (1) or not (0) |
| most ads day | Day of the week when the user saw the most advertisements            |

---

## Workflow

### 1. Data Loading and Cleaning

The dataset is loaded using **pandas** and cleaned to remove unnecessary columns.
A duplicate check on `user id` ensures that each observation corresponds to a unique user, preserving the validity of the A/B experiment.

---

### 2. Exploratory Data Analysis

Basic conversion statistics are computed for each group.

Metrics calculated:

* Total number of users in each group
* Number of conversions
* Conversion rate

This provides a preliminary comparison between the **Ad** and **PSA** groups.

---

### 3. Statistical Hypothesis Testing

Two statistical tests are used to determine whether the difference in conversion rates is significant.

#### Chi-Square Test of Independence

A contingency table is constructed to test whether **conversion outcome depends on the test group**.

**Null Hypothesis**

Conversion is independent of the test group.

**Result**

The chi-square test returns a p-value ≈ 0, indicating a statistically significant relationship.

---

#### Two-Proportion Z-Test

A two-proportion z-test directly compares the conversion probabilities of the two groups.

**Null Hypothesis**

The conversion rate of the Ad group equals that of the PSA group.

**Result**

The p-value ≈ 0 indicates strong evidence that the conversion rates differ significantly.

---

### 4. Confidence Interval Estimation

95% confidence intervals are calculated for the conversion rate of each group. These intervals provide a range of plausible values for the true population conversion rates.

Results:

Ad Group:
0.02513 – 0.02596

PSA Group:
0.01616 – 0.01955

The intervals do not overlap, further reinforcing the statistical significance of the difference.

---

### 5. Visualization

Two visualizations are generated:

**Conversion Rate Comparison**

* Bar chart showing conversion rates of both groups
* Includes 95% confidence interval error bars

**Conversion Rate by Day of Week**

* Displays how conversion varies depending on the day users saw the most ads
* Helps identify behavioral patterns related to time of exposure

---

## Key Results

| Group | Users   | Conversions | Conversion Rate |
| ----- | ------- | ----------- | --------------- |
| Ad    | 564,577 | 14,423      | 2.55%           |
| PSA   | 23,524  | 420         | 1.79%           |

Statistical tests show **p-value ≈ 0**, indicating that the difference between the two groups is highly significant.

### Final Observation

Despite the absolute difference appearing small (~0.77 percentage points), the extremely large sample size confirms that the **Ad campaign significantly outperforms the PSA campaign in driving conversions**.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* SciPy
* Statsmodels

---

## How to Run

1. Install dependencies:

```
pip install pandas numpy matplotlib seaborn scipy statsmodels
```

2. Place the dataset `marketing_AB.csv` in the working directory.

3. Run the notebook:

```
jupyter notebook A_B Testing.ipynb
```

---

## Conclusion

The analysis demonstrates that displaying advertisements significantly increases conversion rates compared to public service announcements. The statistical evidence strongly supports choosing the **Ad strategy** when the goal is maximizing conversions.
