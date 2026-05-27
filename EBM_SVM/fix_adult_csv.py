import pandas as pd

# Proper Adult dataset column names
columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

# Read without headers
df = pd.read_csv('data/adult.csv', header=None, names=columns, skipinitialspace=True)

print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst 3 rows:")
print(df.head(3))
print(f"\nTarget distribution:")
print(df['income'].value_counts())

# Save corrected file
df.to_csv('data/adult.csv', index=False)
print("\n✓ Fixed: adult.csv - headers added!")
