import unittest
import pandas as pd
import numpy as np

try:
    from src.preprocessing import CreditDataPreprocessor
    from src.models import LogisticScorecardModel, GradientBoostedRiskTree, compute_metrics
    from src.explainability import ExplainabilityEngine
    from src.scoring import CreditScoringEngine
    from src.pipeline import CreditRiskPipeline
except ImportError:
    from preprocessing import CreditDataPreprocessor
    from models import LogisticScorecardModel, GradientBoostedRiskTree, compute_metrics
    from explainability import ExplainabilityEngine
    from scoring import CreditScoringEngine
    from pipeline import CreditRiskPipeline

class TestCreditRiskSystem(unittest.TestCase):
    def setUp(self):
        self.lc_path = '/working_dir/c_f089660531aacfaf/data/lending_club_sample.csv'
        self.df = pd.read_csv(self.lc_path)

    def test_preprocessor(self):
        prep = CreditDataPreprocessor()
        prep.fit(self.df)
        self.assertEqual(prep.target_col, 'is_default')
        X, y = prep.transform(self.df)
        self.assertEqual(X.shape[0], len(self.df))
        self.assertEqual(len(y), len(self.df))
        self.assertFalse(np.isnan(X).any())

    def test_logistic_model(self):
        prep = CreditDataPreprocessor()
        prep.fit(self.df)
        X, y = prep.transform(self.df)
        
        model = LogisticScorecardModel(l2_reg=1.0)
        model.fit(X[:800], y[:800], feature_names=prep.feature_names)
        
        probs = model.predict_proba(X[800:])
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())
        metrics = compute_metrics(y[800:], probs)
        self.assertGreater(metrics['auc_roc'], 0.60)
        print(f"Logistic Scorecard AUC: {metrics['auc_roc']:.4f}")

    def test_tree_model(self):
        prep = CreditDataPreprocessor()
        prep.fit(self.df)
        X, y = prep.transform(self.df)
        
        model = GradientBoostedRiskTree(n_estimators=15)
        model.fit(X[:800], y[:800], feature_names=prep.feature_names)
        
        probs = model.predict_proba(X[800:])
        self.assertTrue((probs >= 0.0).all() and (probs <= 1.0).all())
        metrics = compute_metrics(y[800:], probs)
        self.assertGreater(metrics['auc_roc'], 0.55)
        print(f"Gradient Boosted Risk Tree AUC: {metrics['auc_roc']:.4f}")

    def test_explainability(self):
        pipe = CreditRiskPipeline(model_type='logistic')
        res = pipe.run(self.df, test_size=0.2)
        
        self.assertIn('global_explainability', res)
        self.assertIn('instance_explanations', res)
        self.assertGreater(len(res['instance_explanations']), 0)
        
        first_applicant = res['instance_explanations'][0]
        self.assertIn('waterfall_decomposition', first_applicant['explanation'])
        self.assertIn('credit_score', first_applicant)
        self.assertTrue(300 <= first_applicant['credit_score'] <= 850)
        print(f"Sample Applicant Credit Score: {first_applicant['credit_score']}, Grade: {first_applicant['risk_grade']}")

    def test_scoring_and_portfolio(self):
        scorer = CreditScoringEngine()
        score = scorer.prob_to_score(np.array([0.05, 0.20, 0.60]))
        self.assertTrue(score[0] > score[1] > score[2])
        print(f"Scores for 5%, 20%, 60% PD: {score}")

if __name__ == '__main__':
    unittest.main()
