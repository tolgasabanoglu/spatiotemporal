import pandas as pd
from scipy.stats import ttest_ind

def perform_ab_test(df, group_col='group', metric_col='avg_stress'):
    """
    Perform A/B test on metric_col between two groups defined in group_col.
    
    Parameters:
        df (pd.DataFrame): DataFrame with the data.
        group_col (str): Name of the column defining groups (e.g. 'A' and 'B').
        metric_col (str): Column to compare between groups.
    
    Returns:
        dict: Test results including group means, t-statistic, p-value.
    """
    # Check groups exist
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not found in DataFrame.")
    
    groups = df[group_col].unique()
    if len(groups) != 2:
        raise ValueError(f"Group column must have exactly 2 unique values, found: {groups}")
    
    group_a = df[df[group_col] == groups[0]][metric_col].dropna()
    group_b = df[df[group_col] == groups[1]][metric_col].dropna()

    # Perform two-sided t-test assuming unequal variances (Welch's t-test)
    t_stat, p_val = ttest_ind(group_a, group_b, equal_var=False)

    results = {
        'group_a': groups[0],
        'group_b': groups[1],
        'mean_a': group_a.mean(),
        'mean_b': group_b.mean(),
        't_statistic': t_stat,
        'p_value': p_val,
        'n_a': len(group_a),
        'n_b': len(group_b),
    }

    return results


if __name__ == "__main__":
    # Example usage: load weekly_df from CSV
    # Replace with your actual data loading logic
    df = pd.read_csv('path_to_weekly_df.csv')

    # If 'group' column does not exist, create a random split for demonstration
    if 'group' not in df.columns:
        import numpy as np
        np.random.seed(42)
        df['group'] = np.random.choice(['A', 'B'], size=len(df))

    metric_to_test = 'avg_stress'  # or any metric you want to test
    results = perform_ab_test(df, group_col='group', metric_col=metric_to_test)

    print(f"A/B Test results comparing '{metric_to_test}' between groups '{results['group_a']}' and '{results['group_b']}':")
    print(f"Mean {results['group_a']}: {results['mean_a']:.3f} ({results['n_a']} samples)")
    print(f"Mean {results['group_b']}: {results['mean_b']:.3f} ({results['n_b']} samples)")
    print(f"t-statistic: {results['t_statistic']:.3f}")
    print(f"p-value: {results['p_value']:.4f}")

    if results['p_value'] < 0.05:
        print("Result is statistically significant (reject null hypothesis)")
    else:
        print("Result is NOT statistically significant (fail to reject null hypothesis)")
