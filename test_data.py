import pandas as pd
from pathlib import Path

data_path = Path(r'e:\Supervised Project\data\newdata.csv')
df = pd.read_csv(data_path)
print('Columns:', list(df.columns))
print('Shape:', df.shape)
print('Missing:')
print(df.isnull().sum())
print('\nSample:')
print(df.head(3))

# Test column name mapping (same logic as preprocess())
col_map = {}
for c in df.columns:
    cl = c.lower().replace(' ','').replace('(','').replace(')','').replace('$','').replace('-','')
    if 'customerid' in cl or ('customer' in cl and 'id' in cl):
        col_map[c] = 'CustomerID'
    elif 'gender' in cl or 'sex' in cl:
        col_map[c] = 'Gender'
    elif cl == 'age':
        col_map[c] = 'Age'
    elif 'annualincome' in cl or ('income' in cl and 'annual' in cl):
        col_map[c] = 'Annual Income (k$)'
    elif 'spendingscore' in cl or 'spending' in cl:
        col_map[c] = 'Spending Score'

df.rename(columns=col_map, inplace=True)
print('\nRenamed columns:', list(df.columns))
print('All required cols present:', all(c in df.columns for c in ['CustomerID','Gender','Age','Annual Income (k$)','Spending Score']))
print('TEST PASSED')
