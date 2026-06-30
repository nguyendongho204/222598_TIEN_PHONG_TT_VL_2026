import pandas as pd
from pathlib import Path

data_dir = Path('D:/Nam4/ThucTap/EBM_SVM/data')

for name in ['sonar', 'ionosphere', 'yeast', 'ecoli', 'car', 'mushroom', 'balance', 'spambase', 'heart', 'vehicle']:
    f = data_dir / f'{name}.csv'
    if f.exists():
        df = pd.read_csv(f)
        print(f'{name+".csv":<25} shape={str(df.shape):<12} target_col={df.columns[-1]:<10} classes={df.iloc[:,-1].nunique()} dtype={df.iloc[:,-1].dtype}')
