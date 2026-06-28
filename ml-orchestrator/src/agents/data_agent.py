import pandas as pd
import numpy as np

try:
    from ..core.logger import ColoredLogger
except ImportError:  # pragma: no cover - fallback for direct script execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from core.logger import ColoredLogger


class DataAgent:
    def __init__(self, config):
        self.config = config
        self.logger = ColoredLogger()
    
    def load_and_inspect(self, file_path):
        self.logger.section("DATA LOADING & INSPECTION")
        
        self.logger.info(f"Loading: {file_path}")
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported format")
        
        self.logger.success(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        
        report = self._inspect(df)
        return df, report
    
    def _inspect(self, df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        missing = df.isnull().sum()
        duplicates = df.duplicated().sum()
        
        report = {
            'shape': df.shape,
            'numeric_cols': numeric_cols,
            'categorical_cols': categorical_cols,
            'missing_values': missing[missing > 0].to_dict(),
            'duplicates': duplicates
        }
        
        self.logger.info(f"Numeric columns: {len(numeric_cols)}")
        self.logger.info(f"Categorical columns: {len(categorical_cols)}")
        
        if missing.sum() > 0:
            self.logger.warning(f"Missing values: {missing.sum()} cells")
        
        if duplicates > 0:
            self.logger.warning(f"Duplicates: {duplicates} rows")
        
        return report