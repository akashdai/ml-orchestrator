import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

try:
    from imblearn.over_sampling import SMOTE
except ImportError:  # pragma: no cover - optional dependency may be absent
    SMOTE = None

try:
    from ..core.logger import ColoredLogger
except ImportError:  # pragma: no cover - fallback for direct script execution
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from core.logger import ColoredLogger


class PreprocessingAgent:
    def __init__(self, config):
        self.config = config
        self.logger = ColoredLogger()
        self.transformers = {}

    def preprocess(self, df, target_col=None, task_type=None):
        self.logger.section("DATA PREPROCESSING")

        df_clean = df.copy()
        df_clean = self._handle_missing(df_clean, target_col)
        df_clean = df_clean.drop_duplicates()
        df_clean = self._encode_categorical(df_clean, target_col)

        if target_col:
            df_clean = self._remove_outliers(df_clean, target_col)
            df_clean = self._scale_features(df_clean, target_col)
        else:
            df_clean = self._scale_all(df_clean)

        if task_type == 'classification':
            df_clean = self._handle_imbalance(df_clean, target_col)

        self.logger.success(f"Preprocessing: {df.shape} → {df_clean.shape}")
        return df_clean

    def _handle_missing(self, df, target_col):
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in num_cols:
            num_cols.remove(target_col)

        if num_cols:
            imputer = SimpleImputer(strategy='median')
            df[num_cols] = imputer.fit_transform(df[num_cols])

        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        if target_col and target_col in cat_cols:
            cat_cols.remove(target_col)

        if cat_cols:
            imputer = SimpleImputer(strategy='most_frequent')
            df[cat_cols] = imputer.fit_transform(df[cat_cols])

        if target_col and df[target_col].isnull().sum() > 0:
            df = df.dropna(subset=[target_col])

        self.logger.info("Missing values handled")
        return df

    def _encode_categorical(self, df, target_col):
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        if target_col and target_col in cat_cols:
            cat_cols.remove(target_col)

        for col in cat_cols:
            if df[col].nunique() <= 10:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
            else:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        if cat_cols:
            self.logger.info(f"Encoded {len(cat_cols)} categorical columns")
        return df

    def _remove_outliers(self, df, target_col):
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in num_cols:
            num_cols.remove(target_col)

        initial_len = len(df)
        for col in num_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower) & (df[col] <= upper)]

        removed = initial_len - len(df)
        if removed > 0:
            self.logger.info(f"Removed {removed} outliers")
        return df

    def _scale_features(self, df, target_col):
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in num_cols:
            num_cols.remove(target_col)

        if num_cols:
            scaler = StandardScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])
            self.transformers['scaler'] = scaler
            self.logger.info("Features scaled")
        return df

    def _scale_all(self, df):
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            scaler = StandardScaler()
            df[num_cols] = scaler.fit_transform(df[num_cols])
        return df

    def _handle_imbalance(self, df, target_col):
        if SMOTE is None:
            self.logger.warning("SMOTE unavailable; skipping class balancing")
            return df

        X = df.drop(columns=[target_col])
        y = df[target_col]

        counts = y.value_counts()
        if counts.min() / counts.max() < 0.5:
            self.logger.warning("Imbalanced - applying SMOTE")
            smote = SMOTE(random_state=42)
            X, y = smote.fit_resample(X, y)
            df = pd.concat([X, y], axis=1)
            self.logger.success(f"Balanced: {len(df)} samples")
        return df
