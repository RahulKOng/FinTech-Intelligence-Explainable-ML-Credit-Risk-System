import numpy as np

# Regulatory Reason Code Mappings compliant with ECOA / FCRA standards
REGULATORY_REASON_CODES = {
    'fico_score': 'Credit score (FICO) below lending threshold',
    'dti': 'High debt-to-income (DTI) ratio relative to income',
    'revol_util': 'Excessive revolving credit line utilization',
    'delinq_2yrs': 'History of past delinquency on existing obligations',
    'pub_rec': 'Public derogatory records or collections reported',
    'annual_inc': 'Insufficient verifiable annual income for debt burden',
    'loan_amnt': 'Requested principal exceeds risk exposure limits',
    'term_months': 'Extended term length increases cumulative default exposure',
    'installment': 'Monthly payment obligation exceeds disposable cash flow',
    'open_acc': 'Too many or too few active credit lines',
    'total_acc': 'Limited depth of overall credit accounts history',
    'emp_length_years': 'Short duration of current employment stability',
    'duration_months': 'Loan duration is long relative to collateral/profile',
    'credit_amount': 'Credit amount exceeds standard risk tiers',
    'installment_rate': 'High installment percentage of disposable income',
    'checking_status': 'Negative or low average checking account balance',
    'credit_history': 'Prior non-payment or delayed credit payment history',
    'RevolvingUtilization': 'Revolving line utilization is too high',
    'NumberOfTime30-59DaysPastDue': 'Frequent 30-59 day delinquent payment occurrences',
    'NumberOfTimes90DaysLate': 'Severe delinquency (90+ days past due) on file',
    'DebtRatio': 'Overall debt burden ratio exceeds risk limits',
    'MonthlyIncome': 'Insufficient monthly cash flow'
}

class ExplainabilityEngine:
    def __init__(self, model, preprocessor, background_X):
        self.model = model
        self.preprocessor = preprocessor
        self.background_X = background_X
        self.feature_names = preprocessor.feature_names
        self.background_mean = np.mean(background_X, axis=0)
        self.background_prob = float(np.mean(model.predict_proba(background_X)))
        
        # Determine model type
        self.is_linear = hasattr(model, 'weights') and hasattr(model, 'bias')

    def compute_shap_values(self, X_samples):
        n_samples, n_features = X_samples.shape
        shap_values = np.zeros((n_samples, n_features), dtype=np.float64)
        
        if self.is_linear:
            # Exact Shapley values for linear / logistic scorecard in log-odds space
            # phi_j = w_j * (x_j - E[x_j])
            w = self.model.weights
            for i in range(n_samples):
                shap_values[i, :] = w * (X_samples[i, :] - self.background_mean)
        else:
            # Sampling / Permutation Shapley approximation for non-linear tree models
            n_bg = min(len(self.background_X), 50)
            bg_subset = self.background_X[np.random.choice(len(self.background_X), n_bg, replace=False)]
            
            for i in range(n_samples):
                x_inst = X_samples[i:i+1]
                # Fast feature importance marginal contribution
                base_prob = np.mean(self.model.predict_proba(bg_subset))
                for f_idx in range(n_features):
                    # Replace feature f_idx in background with instance value
                    bg_pert = bg_subset.copy()
                    bg_pert[:, f_idx] = x_inst[0, f_idx]
                    prob_with = np.mean(self.model.predict_proba(bg_pert))
                    shap_values[i, f_idx] = prob_with - base_prob

        return shap_values

    def get_global_importance(self, X_eval):
        shap_vals = self.compute_shap_values(X_eval)
        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
        
        records = []
        for idx, (name, val) in enumerate(zip(self.feature_names, mean_abs_shap)):
            # Determine directionality (correlation with target risk)
            sample_feats = X_eval[:, idx]
            sample_shaps = shap_vals[:, idx]
            corr = np.corrcoef(sample_feats, sample_shaps)[0, 1] if np.std(sample_feats) > 1e-6 else 0.0
            impact_direction = 'Positive (Increases Risk)' if corr > 0.05 else ('Negative (Reduces Risk)' if corr < -0.05 else 'Neutral / Non-linear')
            
            records.append({
                'feature_index': idx,
                'feature': name,
                'importance': float(round(val, 5)),
                'direction': impact_direction
            })
            
        records.sort(key=lambda x: x['importance'], reverse=True)
        return records, shap_vals

    def explain_instance(self, instance_row_raw, instance_X_transformed, top_k=6):
        shap_vals = self.compute_shap_values(instance_X_transformed[0:1])[0]
        pred_prob = float(self.model.predict_proba(instance_X_transformed[0:1])[0])
        
        # Sort features by absolute contribution
        sorted_indices = np.argsort(-np.abs(shap_vals))
        
        waterfall_steps = []
        adverse_actions = []
        
        base_val = self.background_prob
        current_running = base_val
        
        for idx in sorted_indices[:top_k]:
            fname = self.feature_names[idx]
            val = float(shap_vals[idx])
            
            # Extract raw human value if available
            raw_val_str = "N/A"
            if isinstance(instance_row_raw, dict) and fname in instance_row_raw:
                raw_val_str = str(instance_row_raw[fname])
            elif hasattr(instance_row_raw, 'get') and instance_row_raw.get(fname) is not None:
                raw_val_str = str(instance_row_raw.get(fname))
                
            waterfall_steps.append({
                'feature': fname,
                'contribution': float(round(val, 4)),
                'abs_contribution': float(round(abs(val), 4)),
                'direction': 'Risk Increased' if val > 0 else 'Risk Decreased',
                'raw_value': raw_val_str
            })
            
            # If feature strongly increases default risk, add to Adverse Action Notice
            if val > 0.01:
                # Find matching root feature name
                root_name = fname.split('_')[0] if '_' in fname else fname
                reason_desc = REGULATORY_REASON_CODES.get(fname, REGULATORY_REASON_CODES.get(root_name, f"Adverse risk factor: {fname}"))
                adverse_actions.append({
                    'factor': fname,
                    'reason_code': reason_desc,
                    'impact_magnitude': float(round(val, 4)),
                    'observed_value': raw_val_str
                })

        return {
            'predicted_probability': round(pred_prob, 4),
            'baseline_probability': round(base_val, 4),
            'risk_status': 'High Risk / Denied' if pred_prob >= 0.25 else 'Low-to-Medium Risk / Approved',
            'waterfall_decomposition': waterfall_steps,
            'adverse_action_notices': adverse_actions[:4]
        }

    def simulate_what_if(self, base_transformed_row, feature_modifications):
        """Simulates counterfactual changes on an applicant's features."""
        X_sim = base_transformed_row.copy()
        for feat_name, new_val in feature_modifications.items():
            if feat_name in self.preprocessor.numeric_cols:
                idx = self.preprocessor.numeric_cols.index(feat_name)
                # Standardize new value
                mean_v = self.preprocessor.means[feat_name]
                std_v = self.preprocessor.stds[feat_name]
                X_sim[0, idx] = (float(new_val) - mean_v) / std_v

        sim_prob = float(self.model.predict_proba(X_sim)[0])
        return {
            'simulated_probability': round(sim_prob, 4),
            'simulated_risk_status': 'High Risk / Denied' if sim_prob >= 0.25 else 'Low Risk / Approved'
        }
