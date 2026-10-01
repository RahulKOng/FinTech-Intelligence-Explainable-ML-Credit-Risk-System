import os
import json
import pandas as pd
import numpy as np

try:
    from src.preprocessing import CreditDataPreprocessor
    from src.models import LogisticScorecardModel, GradientBoostedRiskTree, compute_metrics
    from src.explainability import ExplainabilityEngine
    from src.scoring import CreditScoringEngine
except ImportError:
    from preprocessing import CreditDataPreprocessor
    from models import LogisticScorecardModel, GradientBoostedRiskTree, compute_metrics
    from explainability import ExplainabilityEngine
    from scoring import CreditScoringEngine

class CreditRiskPipeline:
    def __init__(self, target_col=None, model_type='logistic'):
        self.target_col = target_col
        self.model_type = model_type
        self.preprocessor = None
        self.model = None
        self.explainability_engine = None
        self.scoring_engine = CreditScoringEngine()
        self.results = {}

    def run(self, df_or_path, test_size=0.25, threshold=0.20):
        if isinstance(df_or_path, str):
            if not os.path.exists(df_or_path):
                raise FileNotFoundError(f"File not found: {df_or_path}")
            df = pd.read_csv(df_or_path)
        else:
            df = df_or_path.copy()

        # Step 1: Preprocessing & Data Profiling
        self.preprocessor = CreditDataPreprocessor(target_col=self.target_col)
        self.preprocessor.fit(df)
        data_summary = self.preprocessor.get_summary_stats(df)

        X, y = self.preprocessor.transform(df)
        if y is None:
            raise ValueError("Target column could not be determined or is missing in the dataset.")

        # Train-test split (stratified by default)
        np.random.seed(42)
        indices = np.arange(len(df))
        np.random.shuffle(indices)
        
        split_idx = int(len(df) * (1.0 - test_size))
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]

        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        df_test = df.iloc[test_idx].reset_index(drop=True)

        # Step 2: Model Training
        feature_names = self.preprocessor.feature_names
        
        # Train primary model
        if self.model_type == 'tree':
            self.model = GradientBoostedRiskTree(n_estimators=35, learning_rate=0.1)
        else:
            self.model = LogisticScorecardModel(l2_reg=1.5)
            
        self.model.fit(X_train, y_train, feature_names=feature_names)

        # Train baseline comparison model
        alt_model = GradientBoostedRiskTree(n_estimators=25) if self.model_type == 'logistic' else LogisticScorecardModel(l2_reg=1.5)
        alt_model.fit(X_train, y_train, feature_names=feature_names)

        # Step 3: Evaluation Metrics
        test_probs = self.model.predict_proba(X_test)
        alt_probs = alt_model.predict_proba(X_test)

        metrics_primary = compute_metrics(y_test, test_probs, threshold=threshold)
        metrics_alt = compute_metrics(y_test, alt_probs, threshold=threshold)

        # Step 4: Explainability Engine
        self.explainability_engine = ExplainabilityEngine(
            model=self.model,
            preprocessor=self.preprocessor,
            background_X=X_train
        )
        global_importance, shap_test = self.explainability_engine.get_global_importance(X_test)

        # Instance-level explanations for sample loans (e.g. 8 loans)
        sample_explanations = []
        for i in range(min(8, len(df_test))):
            row_dict = df_test.iloc[i].to_dict()
            inst_exp = self.explainability_engine.explain_instance(
                instance_row_raw=row_dict,
                instance_X_transformed=X_test[i:i+1],
                top_k=6
            )
            # Add applicant metadata
            loan_id_val = row_dict.get('loan_id', row_dict.get('id', f"APPL-{i+1001}"))
            applicant_score = self.scoring_engine.prob_to_score(inst_exp['predicted_probability'])
            grade, grade_label, grade_color = self.scoring_engine.score_to_grade(applicant_score)
            
            sample_explanations.append({
                'loan_id': str(loan_id_val),
                'credit_score': int(applicant_score),
                'risk_grade': grade,
                'risk_grade_label': grade_label,
                'grade_color': grade_color,
                'actual_outcome': 'Default' if y_test[i] == 1 else 'Non-Default',
                'explanation': inst_exp,
                'applicant_data': {k: v for k, v in row_dict.items() if k != self.preprocessor.target_col}
            })

        # Step 5: Credit Scoring & Financial Portfolio Simulation
        loan_amnts = df_test['loan_amnt'].values if 'loan_amnt' in df_test.columns else (
            df_test['credit_amount'].values if 'credit_amount' in df_test.columns else np.full(len(test_probs), 15000.0)
        )
        int_rates = df_test['int_rate'].values if 'int_rate' in df_test.columns else (
            df_test['installment_rate'].values * 3.5 if 'installment_rate' in df_test.columns else np.full(len(test_probs), 12.5)
        )

        portfolio_eval = self.scoring_engine.evaluate_portfolio(
            probs=test_probs,
            loan_amounts=loan_amnts,
            int_rates=int_rates,
            threshold=threshold
        )

        # Compile final structured output
        self.results = {
            'data_summary': data_summary,
            'model_info': {
                'primary_model': 'Logistic Regression Scorecard (Interpretable Log-Odds)' if self.model_type == 'logistic' else 'Gradient Boosted Decision Trees',
                'comparison_model': 'Gradient Boosted Decision Trees' if self.model_type == 'logistic' else 'Logistic Scorecard',
                'num_features': len(feature_names),
                'train_samples': len(train_idx),
                'test_samples': len(test_idx),
                'decision_threshold': threshold
            },
            'performance': {
                'primary': metrics_primary,
                'comparison': metrics_alt
            },
            'global_explainability': {
                'feature_importance': global_importance[:12],
                'coefficients': self.model.get_feature_coefficients()[:12] if hasattr(self.model, 'get_feature_coefficients') else []
            },
            'instance_explanations': sample_explanations,
            'portfolio_impact': portfolio_eval
        }
        return self.results
