import numpy as np

class CreditScoringEngine:
    def __init__(self, base_score=720, pdo=35, base_odds=19.0):
        """
        Calibrated FICO Scorecard mapping:
        - Base odds of 19:1 corresponds to a 5% baseline default rate -> Score 720 (Prime).
        - PDO (Points to Double the Odds) = 35 points:
            - Doubling the odds (to 38:1, ~2.5% default) adds +35 points -> Score 755 (Super Prime).
            - Halving the odds (to 9.5:1, ~9.5% default) subtracts -35 points -> Score 685 (Standard).
            - 20% default rate (~4:1 odds) -> Score ~650 (Near Subprime).
            - 50% default rate (1:1 odds) -> Score ~580 (High Risk).
        """
        self.base_score = base_score
        self.pdo = pdo
        self.base_odds = base_odds
        self.factor = pdo / np.log(2.0)
        self.offset = base_score - self.factor * np.log(base_odds)

    def prob_to_score(self, prob):
        """Converts default probability into FICO-scale credit score (300 to 850)."""
        eps = 1e-6
        p = np.clip(prob, eps, 1.0 - eps)
        # Good/Bad odds: (1 - p) / p
        odds = (1.0 - p) / p
        score = self.offset + self.factor * np.log(odds)
        return np.clip(np.round(score), 300, 850).astype(int)

    def score_to_grade(self, score):
        if score >= 750:
            return 'A', 'Prime (Excellent)', '#10b981'
        elif score >= 700:
            return 'B', 'Near Prime (Good)', '#3b82f6'
        elif score >= 650:
            return 'C', 'Standard (Fair)', '#f59e0b'
        elif score >= 600:
            return 'D', 'Subprime (Moderate Risk)', '#f97316'
        elif score >= 550:
            return 'E', 'Deep Subprime (High Risk)', '#ef4444'
        else:
            return 'HR', 'High Risk (Severe Risk)', '#991b1b'

    def evaluate_portfolio(self, probs, loan_amounts=None, int_rates=None, threshold=0.20, lgd=0.55):
        n = len(probs)
        if loan_amounts is None:
            loan_amounts = np.full(n, 15000.0)
        if int_rates is None:
            int_rates = np.full(n, 12.0)

        scores = self.prob_to_score(probs)
        approved = probs <= threshold
        
        n_approved = int(np.sum(approved))
        approval_rate = float(n_approved / n) if n > 0 else 0.0
        
        # Financial portfolio simulation
        approved_amounts = loan_amounts[approved]
        approved_probs = probs[approved]
        approved_rates = int_rates[approved]
        
        total_originated = float(np.sum(approved_amounts))
        expected_interest = float(np.sum(approved_amounts * (approved_rates / 100.0)))
        # Expected Loss = PD * EAD * LGD
        expected_defaults_loss = float(np.sum(approved_probs * approved_amounts * lgd))
        net_expected_profit = float(expected_interest - expected_defaults_loss)
        
        # Risk tier distribution
        grades_count = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0, 'HR': 0}
        for s in scores:
            g, _, _ = self.score_to_grade(s)
            grades_count[g] += 1

        # Cutoff simulation curve across thresholds
        cutoff_curve = []
        for th in np.linspace(0.05, 0.45, 17):
            appr = probs <= th
            appr_vol = float(np.sum(loan_amounts[appr]))
            int_rev = float(np.sum(loan_amounts[appr] * (int_rates[appr] / 100.0)))
            def_loss = float(np.sum(probs[appr] * loan_amounts[appr] * lgd))
            profit = int_rev - def_loss
            cutoff_curve.append({
                'threshold': float(round(th, 2)),
                'approval_rate': float(round(np.mean(appr) * 100, 1)),
                'origination_volume': float(round(appr_vol, 2)),
                'expected_loss': float(round(def_loss, 2)),
                'net_profit': float(round(profit, 2))
            })

        return {
            'threshold_used': float(threshold),
            'total_applications': n,
            'approved_count': n_approved,
            'approval_rate_pct': round(approval_rate * 100, 2),
            'total_originated_volume': round(total_originated, 2),
            'expected_interest_revenue': round(expected_interest, 2),
            'expected_default_loss': round(expected_defaults_loss, 2),
            'net_expected_profit': round(net_expected_profit, 2),
            'roi_pct': round((net_expected_profit / total_originated * 100), 2) if total_originated > 0 else 0.0,
            'grade_distribution': grades_count,
            'cutoff_optimization_curve': cutoff_curve
        }
