import numpy as np
import pandas as pd
import os

np.random.seed(42)

def generate_lending_club_data(n=1200):
    loan_id = np.arange(10001, 10001 + n)
    loan_amnt = np.random.choice([5000, 7500, 10000, 12000, 15000, 20000, 25000, 30000, 35000], size=n, 
                                 p=[0.1, 0.1, 0.18, 0.15, 0.17, 0.12, 0.08, 0.06, 0.04])
    term = np.random.choice([36, 60], size=n, p=[0.72, 0.28])
    
    annual_inc = np.random.lognormal(mean=11.1, sigma=0.45, size=n)
    annual_inc = np.round(np.clip(annual_inc, 24000, 260000), -2)
    
    dti = np.random.gamma(shape=4.0, scale=4.5, size=n)
    dti = np.round(np.clip(dti, 2.0, 42.0), 2)
    
    fico_score = np.random.normal(loc=695, scale=45, size=n)
    fico_score = np.round(np.clip(fico_score, 580, 840)).astype(int)
    
    revol_util = np.random.beta(a=2.5, b=2.5, size=n) * 100.0
    revol_util = np.round(np.clip(revol_util, 2.0, 99.0), 1)
    
    delinq_2yrs = np.random.choice([0, 1, 2, 3], size=n, p=[0.82, 0.12, 0.04, 0.02])
    pub_rec = np.random.choice([0, 1, 2], size=n, p=[0.88, 0.09, 0.03])
    open_acc = np.random.poisson(lam=11, size=n)
    open_acc = np.clip(open_acc, 2, 32)
    total_acc = open_acc + np.random.poisson(lam=12, size=n)
    
    home_ownership = np.random.choice(['MORTGAGE', 'RENT', 'OWN'], size=n, p=[0.48, 0.41, 0.11])
    emp_length = np.random.choice([0, 1, 2, 3, 5, 7, 10], size=n, p=[0.08, 0.08, 0.11, 0.12, 0.18, 0.15, 0.28])
    purpose = np.random.choice(['debt_consolidation', 'credit_card', 'home_improvement', 'small_business', 'major_purchase'],
                                size=n, p=[0.55, 0.22, 0.10, 0.06, 0.07])
    
    # Calculate interest rate tied to risk factors
    base_rate = 7.5 + (850 - fico_score) * 0.06 + (dti * 0.15) + (revol_util * 0.05) + (delinq_2yrs * 1.5)
    int_rate = np.round(np.clip(base_rate + np.random.normal(0, 1.2, n), 5.5, 28.5), 2)
    
    # Installment formula
    monthly_r = (int_rate / 100.0) / 12.0
    installment = np.round(loan_amnt * (monthly_r * (1 + monthly_r)**term) / ((1 + monthly_r)**term - 1), 2)
    
    # Logistic model for actual default probability
    # Higher DTI, lower FICO, higher Revol Util, higher delinq, longer term increase default risk
    log_odds = (
        -2.8
        + 0.055 * dti
        - 0.014 * (fico_score - 690)
        + 0.022 * (revol_util - 50)
        + 0.55 * delinq_2yrs
        + 0.40 * pub_rec
        + 0.000015 * (loan_amnt - 15000)
        + 0.35 * (term == 60)
        - 0.000008 * (annual_inc - 70000)
        + 0.45 * (purpose == 'small_business')
    )
    prob_default = 1.0 / (1.0 + np.exp(-log_odds))
    # Add slight stochasticity
    is_default = (np.random.rand(n) < prob_default).astype(int)
    
    df = pd.DataFrame({
        'loan_id': loan_id,
        'loan_amnt': loan_amnt,
        'term_months': term,
        'int_rate': int_rate,
        'installment': installment,
        'annual_inc': annual_inc,
        'dti': dti,
        'fico_score': fico_score,
        'revol_util': revol_util,
        'open_acc': open_acc,
        'total_acc': total_acc,
        'delinq_2yrs': delinq_2yrs,
        'pub_rec': pub_rec,
        'emp_length_years': emp_length,
        'home_ownership': home_ownership,
        'purpose': purpose,
        'is_default': is_default
    })
    return df

def generate_german_credit_data(n=1000):
    duration = np.random.choice([6, 12, 18, 24, 30, 36, 48], size=n, p=[0.1, 0.25, 0.2, 0.2, 0.1, 0.1, 0.05])
    credit_amount = np.round(np.random.gamma(shape=2.5, scale=1200, size=n) + 500, -1)
    installment_rate = np.random.choice([1, 2, 3, 4], size=n, p=[0.15, 0.25, 0.35, 0.25])
    age = np.random.randint(19, 72, size=n)
    existing_credits = np.random.choice([1, 2, 3, 4], size=n, p=[0.65, 0.28, 0.05, 0.02])
    checking_status = np.random.choice(['<0 DM', '0-200 DM', '>=200 DM', 'none'], size=n, p=[0.28, 0.27, 0.06, 0.39])
    credit_history = np.random.choice(['critical', 'delay', 'existing_paid', 'all_paid', 'no_credits'], size=n, p=[0.3, 0.09, 0.5, 0.06, 0.05])
    employment = np.random.choice(['<1 year', '1-4 years', '4-7 years', '>=7 years', 'unemployed'], size=n, p=[0.17, 0.34, 0.17, 0.25, 0.07])
    
    log_odds = -1.2 + 0.03 * duration + 0.00015 * credit_amount + 0.2 * installment_rate - 0.025 * (age - 35)
    log_odds += np.where(checking_status == '<0 DM', 0.8, np.where(checking_status == 'none', -0.5, 0.0))
    prob = 1.0 / (1.0 + np.exp(-log_odds))
    default = (np.random.rand(n) < prob).astype(int)
    
    df = pd.DataFrame({
        'duration_months': duration,
        'credit_amount': credit_amount,
        'installment_rate': installment_rate,
        'age_years': age,
        'existing_credits': existing_credits,
        'checking_status': checking_status,
        'credit_history': credit_history,
        'employment': employment,
        'is_default': default
    })
    return df

def generate_give_me_credit_data(n=1000):
    utilization = np.round(np.random.beta(2, 5, size=n), 4)
    age = np.random.randint(21, 80, size=n)
    num_30_59_late = np.random.choice([0, 1, 2, 3], size=n, p=[0.85, 0.1, 0.03, 0.02])
    debt_ratio = np.round(np.random.gamma(2, 0.2, size=n), 3)
    monthly_income = np.round(np.random.lognormal(8.5, 0.5, size=n), -2)
    num_open_credit = np.random.poisson(8, size=n)
    num_90_late = np.random.choice([0, 1, 2], size=n, p=[0.92, 0.06, 0.02])
    num_real_estate = np.random.choice([0, 1, 2, 3], size=n, p=[0.3, 0.45, 0.2, 0.05])
    
    log_odds = -3.2 + 2.5 * utilization - 0.02 * (age - 45) + 0.8 * num_30_59_late + 1.2 * num_90_late + 0.8 * debt_ratio
    prob = 1.0 / (1.0 + np.exp(-log_odds))
    default = (np.random.rand(n) < prob).astype(int)
    
    df = pd.DataFrame({
        'RevolvingUtilization': utilization,
        'Age': age,
        'NumberOfTime30-59DaysPastDue': num_30_59_late,
        'DebtRatio': debt_ratio,
        'MonthlyIncome': monthly_income,
        'NumberOfOpenCreditLines': num_open_credit,
        'NumberOfTimes90DaysLate': num_90_late,
        'NumberRealEstateLoans': num_real_estate,
        'is_default': default
    })
    return df

if __name__ == '__main__':
    lc = generate_lending_club_data(1200)
    lc.to_csv('/working_dir/c_f089660531aacfaf/data/lending_club_sample.csv', index=False)
    print(f"Generated lending_club_sample.csv: {lc.shape}, Default rate: {lc['is_default'].mean():.2%}")
    
    gc = generate_german_credit_data(1000)
    gc.to_csv('/working_dir/c_f089660531aacfaf/data/german_credit_sample.csv', index=False)
    print(f"Generated german_credit_sample.csv: {gc.shape}, Default rate: {gc['is_default'].mean():.2%}")
    
    gmc = generate_give_me_credit_data(1000)
    gmc.to_csv('/working_dir/c_f089660531aacfaf/data/give_me_some_credit_sample.csv', index=False)
    print(f"Generated give_me_some_credit_sample.csv: {gmc.shape}, Default rate: {gmc['is_default'].mean():.2%}")
