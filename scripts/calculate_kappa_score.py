import pandas as pd
from sklearn.metrics import cohen_kappa_score

def load_data(filepath):
    """Load the data from CSV file."""
    return pd.read_csv(filepath, header=None, names=['annotator_1', 'annotator_2', 'category'])

def calculate_kappa_scores_by_category(df):
    """Calculate Cohen's Kappa scores for each category using quadratic weights."""
    categories = df['category'].unique()
    results_kappa = {}
    for category in categories:
        subset = df[df['category'] == category]
        kappa = cohen_kappa_score(subset['annotator_1'], subset['annotator_2'], weights='quadratic')
        results_kappa[category] = kappa
    return results_kappa

def calculate_overall_kappa_scores(df):
    """Calculate overall Cohen's Kappa scores using quadratic weights."""
    kappa = cohen_kappa_score(df['annotator_1'], df['annotator_2'], weights='quadratic')
    return kappa

def print_results(results_kappa):
    """Print Cohen's Kappa scores for each category."""
    if isinstance(results_kappa, dict):
        print('\n'.join([f"'{category}': {score}" for category, score in results_kappa.items()]))
    elif isinstance(results_kappa, float):
        print(f"'overall': {results_kappa}")

if __name__ == '__main__':
    df = load_data('data/ccnet/human_evaluation.csv')
    print("Weighted Kappa Scores:")
    print_results(calculate_overall_kappa_scores(df))
    print_results(calculate_kappa_scores_by_category(df))