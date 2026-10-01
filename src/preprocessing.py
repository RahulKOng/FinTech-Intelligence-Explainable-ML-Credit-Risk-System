import numpy as np
import pandas as pd

class CreditDataPreprocessor:
    def __init__(self, target_col=None, id_cols=None, ignore_cols=None):
        self.target_col = target_col
        self.id_cols = id_cols or ['loan_id', 'id', 'Id', 'ID', 'member_id', 'Unnamed: 0']
        self.ignore_cols = ignore_cols or []
        
        self.numeric_cols = []
        self.categorical_cols = []
        self.medians = {}
        self.means = {}
        self.stds = {}
        self.modes = {}
        self.category_mappings = {}
        self.feature_names = []
        self.is_fitted = False

    def auto_detect_target(self, df):
        candidates = ['is_default', 'default', 'loan_status', 'SeriousDlqin2yrs', 'target', 'bad_loan', 'Risk', 'Class']
        for cand in candidates:
            if cand in df.columns:
                return cand
        # Fallback: check binary columns with name containing default/bad/status
        for col in df.columns:
            if any(term in col.lower() for term in ['default', 'bad', 'target', 'status', 'risk']):
                if df[col].nunique() == 2:
                    return col
        # Fallback to last column
        return df.columns[-1]

    def fit(self, df, target_col=None):
        if target_col is not None:
            self.target_col = target_col
        elif self.target_col is None:
            self.target_col = self.auto_detect_target(df)

        cols_to_drop = [c for c in self.id_cols if c in df.columns] + [c for c in self.ignore_cols if c in df.columns]
        feature_df = df.drop(columns=[c for c in cols_to_drop if c != self.target_col], errors='ignore')
        
        if self.target_col in feature_df.columns:
            X = feature_df.drop(columns=[self.target_col])
        else:
            X = feature_df

        self.numeric_cols = []
        self.categorical_cols = []
        
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                if X[col].nunique() <= 4 and X[col].dtype == object:
                    self.categorical_cols.append(col)
                else:
                    self.numeric_cols.append(col)
            else:
                self.categorical_cols.append(col)

        # Store statistics for numerical
        for col in self.numeric_cols:
            median_val = float(X[col].median(skipna=True))
            if np.isnan(median_val):
                median_val = 0.0
            self.medians[col] = median_val
            
            mean_val = float(X[col].mean(skipna=True))
            std_val = float(X[col].std(skipna=True))
            if np.isnan(std_val) or std_val == 0.0:
                std_val = 1.0
            self.means[col] = mean_val
            self.stds[col] = std_val

        # Store statistics and one-hot categories for categoricals
        self.feature_names = []
        # Numerical features come first
        self.feature_names.extend(self.numeric_cols)

        for col in self.categorical_cols:
            mode_val = X[col].mode(dropna=True)
            self.modes[col] = mode_val.iloc[0] if len(mode_val) > 0 else 'Unknown'
            # Top categories (up to 10 to avoid high dimensionality)
            top_cats = list(X[col].value_counts().nlargest(10).index)
            self.category_mappings[col] = top_cats
            for cat in top_cats:
                self.feature_names.append(f"{col}_{cat}")
            self.feature_names.append(f"{col}_other")

        self.is_fitted = True
        return self

    def transform(self, df):
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted yet.")

        n_samples = len(df)
        X_out = np.zeros((n_samples, len(self.feature_names)), dtype=np.float64)

        # Fill numerical features
        col_idx = 0
        for col in self.numeric_cols:
            if col in df.columns:
                series = pd.to_numeric(df[col], errors='coerce').fillna(self.medians[col])
            else:
                series = pd.Series([self.medians[col]] * n_samples)
            # Standardize: (x - mean) / std
            standardized = (series.values - self.means[col]) / self.stds[col]
            X_out[:, col_idx] = standardized
            col_idx += 1

        # Fill categorical one-hot features
        for col in self.categorical_cols:
            top_cats = self.category_mappings[col]
            if col in df.columns:
                series = df[col].astype(str).fillna(str(self.modes[col]))
            else:
                series = pd.Series([str(self.modes[col])] * n_samples)

            matched_any = np.zeros(n_samples, dtype=bool)
            for cat in top_cats:
                matches = (series.values == str(cat))
                X_out[:, col_idx] = matches.astype(float)
                matched_any |= matches
                col_idx += 1
            # Other category
            X_out[:, col_idx] = (~matched_any).astype(float)
            col_idx += 1

        # Extract target if present
        y_out = None
        if self.target_col in df.columns:
            y_raw = df[self.target_col]
            # Standardize binary target: 1 for default / bad / charged off, 0 for good / fully paid
            if pd.api.types.is_numeric_dtype(y_raw):
                y_out = (y_raw.values > 0).astype(int)
            else:
                bad_terms = ['default', 'charged off', 'late', 'bad', 'yes', 'true', '1']
                y_out = y_raw.astype(str).str.lower().apply(
                    lambda x: 1 if any(t in x for t in bad_terms) else 0
                ).values

        return X_out, y_out

    def get_summary_stats(self, df):
        summary = {
            'total_rows': int(len(df)),
            'total_columns': int(len(df.columns)),
            'target_column': self.target_col,
            'features': []
        }
        
        y_present = self.target_col in df.columns
        if y_present:
            y_raw = df[self.target_col]
            if pd.api.types.is_numeric_dtype(y_raw):
                y_vals = (y_raw > 0).astype(int)
            else:
                bad_terms = ['default', 'charged off', 'late', 'bad', 'yes', 'true', '1']
                y_vals = y_raw.astype(str).str.lower().apply(
                    lambda x: 1 if any(t in x for t in bad_terms) else 0
                )
            summary['default_count'] = int(y_vals.sum())
            summary['non_default_count'] = int((1 - y_vals).sum())
            summary['default_rate'] = float(round(y_vals.mean() * 100, 2))
        else:
            summary['default_count'] = None
            summary['non_default_count'] = None
            summary['default_rate'] = None

        for col in df.columns:
            missing_cnt = int(df[col].isna().sum())
            missing_pct = round(missing_cnt / len(df) * 100, 2)
            dtype_str = 'numeric' if pd.api.types.is_numeric_dtype(df[col]) else 'categorical'
            
            f_info = {
                'name': col,
                'type': dtype_str,
                'missing_count': missing_cnt,
                'missing_pct': missing_pct,
                'unique_values': int(df[col].nunique())
            }
            if dtype_str == 'numeric':
                f_info['mean'] = float(round(df[col].mean(skipna=True), 2)) if not pd.isna(df[col].mean()) else 0.0
                f_info['min'] = float(round(df[col].min(skipna=True), 2)) if not pd.isna(df[col].min()) else 0.0
                f_info['max'] = float(round(df[col].max(skipna=True), 2)) if not pd.isna(df[col].max()) else 0.0
                f_info['std'] = float(round(df[col].std(skipna=True), 2)) if not pd.isna(df[col].std()) else 0.0
            else:
                top_v = df[col].mode(dropna=True)
                f_info['top_value'] = str(top_v.iloc[0]) if len(top_v) > 0 else 'N/A'

            summary['features'].append(f_info)

        return summary
