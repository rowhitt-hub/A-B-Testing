import pandas as pd
import numpy as np
import os


def generate_synthetic_data(n=10000, seed=42):
    
    rng = np.random.default_rng(seed)
    n_ad = n // 2
    n_psa = n - n_ad

    user_id = np.arange(1, n + 1)
    test_group = np.array(['ad'] * n_ad + ['psa'] * n_psa)
    rng.shuffle(test_group)

    # Slightly higher conversion for 'ad' group, similar to real data patterns
    converted = np.where(
        test_group == 'ad',
        rng.binomial(1, 0.025, n),
        rng.binomial(1, 0.019, n)
    )

    total_ads = np.where(
        test_group == 'ad',
        rng.poisson(lam=8, size=n),
        rng.poisson(lam=5, size=n)
    )

    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    hours = list(range(24))
    most_ads_day = rng.choice(days, size=n)
    most_ads_hour = rng.choice(hours, size=n)

    df = pd.DataFrame({
        'user_id': user_id,
        'test_group': test_group,
        'converted': converted,
        'total_ads': total_ads,
        'most_ads_day': most_ads_day,
        'most_ads_hour': most_ads_hour
    })
    return df


def load_marketing_ab_data(path='marketing_ab.csv'):
    """
    Load the Kaggle Marketing A/B Testing dataset.
    If not found locally, uses synthetic fallback data.
    """
    candidates = [path, 'marketing_AB.csv', 'marketing_ab.csv']

    for candidate in candidates:
        if os.path.exists(candidate):
            df = pd.read_csv(candidate)

            rename_map = {
                'user id': 'user_id',
                'test group': 'test_group',
                'converted': 'converted',
                'total ads': 'total_ads',
                'most ads day': 'most_ads_day',
                'most ads hour': 'most_ads_hour'
            }
            df = df.rename(columns=rename_map)

            # Keep only relevant columns if present
            cols = ['user_id', 'test_group', 'converted', 'total_ads',
                    'most_ads_day', 'most_ads_hour']
            available = [c for c in cols if c in df.columns]
            return df[available]

    print("Kaggle marketing_ab.csv not found. Using synthetic fallback data.")
    return generate_synthetic_data()


if __name__ == "__main__":
    data = load_marketing_ab_data()
    print(data.head())
    print(data.info())