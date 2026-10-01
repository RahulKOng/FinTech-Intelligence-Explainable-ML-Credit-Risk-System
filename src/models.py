import numpy as np
from scipy.optimize import minimize

def compute_roc_auc(y_true, y_prob):
    # Sort by probability descending
    idx = np.argsort(-y_prob)
    y_true_sorted = y_true[idx]
    
    n_pos = np.sum(y_true_sorted == 1)
    n_neg = np.sum(y_true_sorted == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5, [0, 1], [0, 1]
    
    tpr = [0.0]
    fpr = [0.0]
    
    accum_tp = 0
    accum_fp = 0
    
    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            accum_tp += 1
        else:
            accum_fp += 1
        tpr.append(accum_tp / n_pos)
        fpr.append(accum_fp / n_neg)
        
    # Trapezoidal rule for AUC
    auc = float(np.trapz(tpr, fpr))
    return max(0.0, min(1.0, auc)), fpr, tpr

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    
    total = len(y_true)
    accuracy = float((tp + tn) / total) if total > 0 else 0.0
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    auc, fpr, tpr = compute_roc_auc(y_true, y_prob)
    
    # Brier score (mean squared error of probabilities)
    brier_score = float(np.mean((y_prob - y_true)**2))
    
    # Log loss
    eps = 1e-15
    clipped_prob = np.clip(y_prob, eps, 1 - eps)
    log_loss = float(-np.mean(y_true * np.log(clipped_prob) + (1 - y_true) * np.log(1 - clipped_prob)))
    
    return {
        'threshold': float(threshold),
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'specificity': round(specificity, 4),
        'f1_score': round(f1, 4),
        'auc_roc': round(auc, 4),
        'brier_score': round(brier_score, 4),
        'log_loss': round(log_loss, 4),
        'confusion_matrix': {
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        },
        'roc_curve': {
            'fpr': [round(x, 4) for x in fpr[::max(1, len(fpr)//50)]],
            'tpr': [round(y, 4) for y in tpr[::max(1, len(tpr)//50)]]
        }
    }

class LogisticScorecardModel:
    def __init__(self, l2_reg=1.0):
        self.l2_reg = l2_reg
        self.weights = None
        self.bias = 0.0
        self.feature_names = []
        self.is_fitted = False

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -25.0, 25.0)))

    def fit(self, X, y, feature_names=None):
        n_samples, n_features = X.shape
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(n_features)]
        
        # Loss function: Negative log-likelihood + L2 regularization
        def loss_and_grad(params):
            w = params[:-1]
            b = params[-1]
            
            z = np.dot(X, w) + b
            p = self._sigmoid(z)
            eps = 1e-15
            p_clipped = np.clip(p, eps, 1.0 - eps)
            
            # Binary cross-entropy
            bce = -np.mean(y * np.log(p_clipped) + (1.0 - y) * np.log(1.0 - p_clipped))
            reg = 0.5 * self.l2_reg * np.sum(w**2) / n_samples
            loss = bce + reg
            
            # Gradient
            diff = (p - y) / n_samples
            grad_w = np.dot(X.T, diff) + (self.l2_reg * w) / n_samples
            grad_b = np.sum(diff)
            
            grad = np.concatenate([grad_w, [grad_b]])
            return loss, grad

        init_params = np.zeros(n_features + 1)
        res = minimize(
            loss_and_grad,
            init_params,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 250, 'ftol': 1e-7}
        )

        self.weights = res.x[:-1]
        self.bias = res.x[-1]
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_feature_coefficients(self):
        records = []
        for name, w in zip(self.feature_names, self.weights):
            records.append({
                'feature': name,
                'weight': float(round(w, 4)),
                'abs_weight': float(round(abs(w), 4)),
                'odds_ratio': float(round(np.exp(w), 4)),
                'risk_impact': 'Increases Risk' if w > 0 else 'Decreases Risk'
            })
        records.sort(key=lambda x: x['abs_weight'], reverse=True)
        return records

class SimpleDecisionStump:
    """Fast decision stump for gradient boosting."""
    def __init__(self):
        self.feature_idx = 0
        self.threshold = 0.0
        self.left_val = 0.0
        self.right_val = 0.0

    def fit(self, X, residuals):
        n_samples, n_features = X.shape
        best_loss = float('inf')
        
        # Subsample candidate features for speed
        n_sub = min(n_features, max(8, int(np.sqrt(n_features) * 2)))
        feat_candidates = np.random.choice(n_features, n_sub, replace=False)

        for f_idx in feat_candidates:
            col = X[:, f_idx]
            # Quantiles as threshold candidates
            thresholds = np.percentile(col, [15, 30, 50, 70, 85])
            for th in thresholds:
                left_mask = col <= th
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                l_val = np.mean(residuals[left_mask])
                r_val = np.mean(residuals[right_mask])
                
                preds = np.where(left_mask, l_val, r_val)
                loss = np.sum((residuals - preds)**2)
                
                if loss < best_loss:
                    best_loss = loss
                    self.feature_idx = f_idx
                    self.threshold = float(th)
                    self.left_val = float(l_val)
                    self.right_val = float(r_val)

        return self

    def predict(self, X):
        return np.where(X[:, self.feature_idx] <= self.threshold, self.left_val, self.right_val)

class GradientBoostedRiskTree:
    """Ensemble Gradient Boosted Trees for non-linear credit risk interactions."""
    def __init__(self, n_estimators=40, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.trees = []
        self.base_pred = 0.0
        self.feature_names = []
        self.is_fitted = False

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -25.0, 25.0)))

    def fit(self, X, y, feature_names=None):
        n_samples, n_features = X.shape
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(n_features)]
        
        # Initial log-odds: log(p / (1 - p))
        p0 = np.clip(np.mean(y), 1e-4, 1.0 - 1e-4)
        self.base_pred = np.log(p0 / (1.0 - p0))
        
        raw_preds = np.full(n_samples, self.base_pred, dtype=np.float64)
        self.trees = []

        for _ in range(self.n_estimators):
            p = self._sigmoid(raw_preds)
            # Pseudo-residuals (negative gradient of binary cross-entropy)
            residuals = y - p
            
            stump = SimpleDecisionStump()
            stump.fit(X, residuals)
            
            update = stump.predict(X)
            raw_preds += self.learning_rate * update
            self.trees.append(stump)

        self.is_fitted = True
        return self

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        raw_preds = np.full(len(X), self.base_pred, dtype=np.float64)
        for stump in self.trees:
            raw_preds += self.learning_rate * stump.predict(X)
        return self._sigmoid(raw_preds)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def get_feature_importances(self):
        counts = {i: 0 for i in range(len(self.feature_names))}
        for t in self.trees:
            counts[t.feature_idx] = counts.get(t.feature_idx, 0) + 1
        
        total = sum(counts.values()) or 1
        records = []
        for i, name in enumerate(self.feature_names):
            importance = counts[i] / total
            records.append({
                'feature': name,
                'importance': float(round(importance, 4))
            })
        records.sort(key=lambda x: x['importance'], reverse=True)
        return records
