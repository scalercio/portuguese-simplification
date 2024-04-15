import pandas as pd
from sklearn.metrics import cohen_kappa_score

def load_data(filepath):
    """Load the data from CSV file."""
    return pd.read_csv(filepath, header=None, names=['annotator_1', 'annotator_2', 'category'])

def calculate_kappa_scores(df):
    """Calculate Cohen's Kappa scores for each category using quadratic weights."""
    categories = df['category'].unique()
    results_kappa = {}
    for category in categories:
        subset = df[df['category'] == category]
        kappa = cohen_kappa_score(subset['annotator_1'], subset['annotator_2'], weights='quadratic')
        results_kappa[category] = kappa
    return results_kappa

def print_results(results_kappa, title=""):
    """Print Cohen's Kappa scores for each category."""
    print(f"{title}\n" + '\n'.join([f"'{category}': {kappa}" for category, kappa in results_kappa.items()]) + "\n")

if __name__ == '__main__':
    df = load_data('data/ccnet/human_evaluation.csv')
    print_results(calculate_kappa_scores(df), "Weighted Kappa Scores:")