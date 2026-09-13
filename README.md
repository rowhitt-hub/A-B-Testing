# Marketing A/B Test Dashboard

## Business Question
Does the **ad** group convert significantly better than the **psa** group in the Kaggle Marketing A/B Testing dataset?

## Hypothesis
- **Null hypothesis (H0):** Conversion rate is the same for `ad` and `psa` groups.
- **Alternative hypothesis (H1):** Conversion rate differs between `ad` and `psa` groups.

## Data
- Source: [Kaggle Marketing A/B Testing](https://www.kaggle.com/datasets/faviovaz/marketing-ab-testing)
- File: `marketing_AB.csv` (or `marketing_ab.csv`)
- Relevant columns:
  - `test_group`: `ad` or `psa`
  - `converted`: 1 if converted, 0 otherwise
  - `total_ads`: number of ads seen (continuous metric)

If the dataset is not present locally, `data_gen.py` creates a synthetic fallback with the same schema.

## Decision Framework
1. **Design first:** Use the sample size calculator in the sidebar to determine the required sample size for a desired lift, alpha, and power.
2. **Run the test:** Choose the metric (conversion rate or total ads).
3. **Interpret results:**
   - If `p-value >= alpha`: **Need more data / don't ship** — no statistically significant difference.
   - If `p-value < alpha` and the confidence interval is entirely above the practical lift: **Ship** — meaningful positive lift.
   - If `p-value < alpha` and the confidence interval is entirely below negative practical lift: **Don't ship** — significant negative impact.
   - Otherwise: statistically significant but not practically meaningful — **need more data/context**.

## Results Column
The results table displayed in the dashboard includes a `result` column that describes the quantitative outcome:
- `control` / `treatment`: labels for each group.
- `Difference (treat - control)`: shows the decision string, e.g., `"Ship: statistically significant and practically meaningful positive lift."`

## Sequential Testing / Peeking Problem
The dashboard includes a simulation that demonstrates why repeatedly checking p-values and stopping when `p < 0.05` inflates the false positive rate. With 5 peeks, the observed false positive rate can be much higher than the nominal 5%, often 15–25%. This shows why pre-registered stopping rules or alpha-spending approaches are needed in real experiments.

## How to Run
1. Install dependencies:
   ```bash
   pip install pandas scipy statsmodels streamlit
2. Download the Kaggle dataset and place marketing_AB.csv in the project directory.

3. Run the Streamlit app:

```bash
  streamlit run app.py
