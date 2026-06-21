from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


DATA_RAW = ROOT / 'ml' / 'data' / 'raw'
DATA_PROCESSED = ROOT / 'ml' / 'data' / 'processed'
print(DATA_RAW / 'HR-Employee-Attrition.csv')
 
FIGURES = ROOT / 'ml' / 'reports' / 'figures'

TARGET_COL = 'Attrition'
