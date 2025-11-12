from __future__ import annotations

"""
Venture Capital KPI calculations for Venture-Scope.

This module computes standard VC metrics used to evaluate startup performance:
- Capital Efficiency: Revenue / Total Funding
- Burn Rate & Runway: Cash management metrics
- Traction Index: Composite growth indicator
"""

import pandas as pd
import numpy as np
from typing import Optional


# ==================== STAGE WEIGHTS ====================

STAGE_WEIGHTS = {
    'Pre-Seed': 0.5,
    'Seed': 1.0,
    'Angel': 0.8,
    'Series A': 1.5,
    'Series B': 2.0,
    'Series C': 2.5,
    'Series D+': 3.0,
}


# ==================== KPI FUNCTIONS ====================

def estimate_revenue(
    df: pd.DataFrame,
    method: str = 'funding_based'
) -> pd.Series:
    """
    Estimate annual revenue based on funding and stage.
    
    Since revenue data is rarely public, we use industry benchmarks:
    - Seed: ~0.1x funding as annual revenue
    - Series A: ~0.3x funding as annual revenue
    - Series B: ~0.5x funding as annual revenue
    - Series C+: ~0.8x funding as annual revenue
    """
    revenue_multiples = {
        'Pre-Seed': 0.05,
        'Seed': 0.10,
        'Angel': 0.08,
        'Series A': 0.30,
        'Series B': 0.50,
        'Series C': 0.80,
        'Series D+': 1.00,
    }
    
    stage_multiple = df['stage'].map(revenue_multiples).fillna(0.20)
    estimated_revenue = df['funding_amount'] * stage_multiple
    
    return estimated_revenue


def calculate_capital_efficiency(
    df: pd.DataFrame,
    revenue_col: str = 'estimated_revenue',
    funding_col: str = 'funding_amount'
) -> pd.Series:
    """
    Calculate Capital Efficiency = Revenue / Total Funding Raised
    
    Interpretation:
    - >1.0: Excellent (generates more revenue than funding)
    - 0.5-1.0: Good efficiency
    - 0.2-0.5: Acceptable for growth stage
    - <0.2: Low efficiency (high burn)
    """
    if revenue_col not in df.columns:
        revenue = estimate_revenue(df)
    else:
        revenue = df[revenue_col]
    
    funding = df[funding_col]
    capital_efficiency = revenue / funding.replace(0, np.nan)
    
    return capital_efficiency


def estimate_burn_rate(
    df: pd.DataFrame,
    funding_col: str = 'funding_amount',
    founded_col: str = 'founded_year'
) -> pd.Series:
    """
    Estimate monthly burn rate based on funding and company age.
    
    Assumption: Companies burn through funding over 18-36 months
    depending on stage.
    """
    current_year = 2013  # Crunchbase snapshot year
    company_age = current_year - df[founded_col].fillna(current_year)
    company_age = company_age.clip(lower=1)
    
    burn_period_months = df['stage'].map({
        'Pre-Seed': 12,
        'Seed': 18,
        'Angel': 18,
        'Series A': 24,
        'Series B': 30,
        'Series C': 36,
        'Series D+': 36,
    }).fillna(24)
    
    monthly_burn = df[funding_col] / burn_period_months
    
    return monthly_burn


def calculate_runway(
    df: pd.DataFrame,
    cash_col: Optional[str] = None,
    burn_col: Optional[str] = None
) -> pd.Series:
    """
    Calculate runway in months = Cash / Monthly Burn
    
    If cash not available, estimate as % of total funding remaining.
    """
    if burn_col and burn_col in df.columns:
        monthly_burn = df[burn_col]
    else:
        monthly_burn = estimate_burn_rate(df)
    
    if cash_col and cash_col in df.columns:
        cash = df[cash_col]
    else:
        cash = df['funding_amount'] * 0.50
    
    runway = cash / monthly_burn.replace(0, np.nan)
    
    return runway


def calculate_traction_index(
    df: pd.DataFrame,
    investors_col: str = 'investors_count',
    funding_col: str = 'funding_amount',
    age_col: str = 'founded_year'
) -> pd.Series:
    """
    Calculate Traction Index = composite metric of startup momentum.
    
    Formula: (Funding Growth) × (Investors) × (Stage Weight) / Age
    
    Returns score normalized to 0-100 scale.
    """
    funding_score = np.log10(df[funding_col].clip(lower=1))
    investors_score = df[investors_col].fillna(1).clip(lower=1)
    stage_weight = df['stage'].map(STAGE_WEIGHTS).fillna(1.0)
    
    current_year = 2013
    age = (current_year - df[age_col].fillna(current_year)).clip(lower=1)
    age_factor = 1 / age
    
    traction = funding_score * investors_score * stage_weight * age_factor
    
    traction_normalized = (traction - traction.min()) / (traction.max() - traction.min()) * 100
    
    return traction_normalized


def estimate_rule_of_40(
    df: pd.DataFrame,
    stage_col: str = 'stage',
    use_capital_efficiency: bool = True
) -> pd.Series:
    """
    Estimate Rule of 40 = Revenue Growth (%) + Profit Margin (%)
    
    Since historical revenue data is unavailable, we estimate based on:
    1. Stage-based benchmarks (typical growth/margin profiles)
    2. Capital efficiency adjustment (better efficiency → better Rule of 40)
    
    Args:
        df: DataFrame with stage and capital_efficiency
        stage_col: Column name for funding stage
        use_capital_efficiency: Adjust based on capital efficiency
    
    Returns:
        Estimated Rule of 40 score
    
    Interpretation:
    - ≥40: Excellent balance of growth and profitability
    - 30-40: Good
    - 20-30: Acceptable
    - <20: Concerning
    """
    # Stage benchmarks (industry averages)
    rule_of_40_benchmarks = {
        'Pre-Seed': 80,   # 150% growth, -70% margin
        'Seed': 100,      # 180% growth, -80% margin
        'Angel': 90,      # 160% growth, -70% margin
        'Series A': 100,  # 120% growth, -20% margin
        'Series B': 80,   # 80% growth, 0% margin
        'Series C': 50,   # 50% growth, 0% margin
        'Series D+': 40,  # 30% growth, 10% margin
    }
    
    # Base estimate from stage
    estimated_rule = df[stage_col].map(rule_of_40_benchmarks).fillna(60)
    
    # Adjust based on capital efficiency (if available)
    if use_capital_efficiency and 'capital_efficiency' in df.columns:
        # Companies with efficiency > 0.3 likely have better Rule of 40
        # Companies with efficiency < 0.3 likely have worse Rule of 40
        efficiency_adjustment = (df['capital_efficiency'] - 0.3) * 50
        estimated_rule = estimated_rule + efficiency_adjustment
        
        # Clip to reasonable range
        estimated_rule = estimated_rule.clip(lower=-50, upper=150)
    
    return estimated_rule

def calculate_all_kpis(
    df: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Calculate all KPIs for a startup dataset.
    """
    if verbose:
        print("📊 Calculating Venture Capital KPIs...")
    
    result = df.copy()
    
    if verbose:
        print("  ├─ Estimating annual revenue...")
    result['estimated_revenue'] = estimate_revenue(result)
    
    if verbose:
        print("  ├─ Calculating capital efficiency...")
    result['capital_efficiency'] = calculate_capital_efficiency(result)
    
    if verbose:
        print("  ├─ Estimating burn rate and runway...")
    result['monthly_burn'] = estimate_burn_rate(result)
    result['runway_months'] = calculate_runway(result)
    
    if verbose:
        print("  └─ Calculating traction index...")
    result['traction_index'] = calculate_traction_index(result)

    if verbose:
        print("  ├─ Estimating Rule of 40...")
    result['rule_of_40'] = estimate_rule_of_40(result)
    
    if verbose:
        print("\n✅ KPI calculation complete!")
        print(f"   Added columns: estimated_revenue, capital_efficiency, monthly_burn, runway_months, traction_index")
    
    return result


def kpi_summary(df: pd.DataFrame) -> None:
    """Print summary statistics for calculated KPIs."""
    kpi_cols = [
        'capital_efficiency', 'monthly_burn', 'runway_months', 
        'traction_index', 'estimated_revenue', 'rule_of_40'
    ]
    
    available_kpis = [col for col in kpi_cols if col in df.columns]
    
    print("\n📊 KPI Summary Statistics:")
    print("=" * 80)
    
    for kpi in available_kpis:
        stats = df[kpi].describe()
        print(f"\n{kpi.upper().replace('_', ' ')}:")
        print(f"  Mean:   {stats['mean']:>12,.2f}")
        print(f"  Median: {stats['50%']:>12,.2f}")
        print(f"  Std:    {stats['std']:>12,.2f}")
        print(f"  Min:    {stats['min']:>12,.2f}")
        print(f"  Max:    {stats['max']:>12,.2f}")
    
    print("=" * 80)


# ==================== TESTING ====================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    input_file = Path("data/processed/startups_enriched.csv")
    
    if not input_file.exists():
        print(f"❌ File not found: {input_file}")
        print("   Run the enriched loader first!")
        sys.exit(1)
    
    print(f"📂 Loading data from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"✅ Loaded {len(df):,} companies\n")
    
    df_with_kpis = calculate_all_kpis(df, verbose=True)
    
    print("\n📋 Sample (first 10 rows):")
    print(df_with_kpis[[
        'company', 'stage', 'funding_amount', 'investors_count',
        'capital_efficiency', 'runway_months', 'traction_index'
    ]].head(10).to_string())
    
    kpi_summary(df_with_kpis)
    
    output_file = Path("data/processed/startups_with_kpis.csv")
    df_with_kpis.to_csv(output_file, index=False)
    print(f"\n💾 Saved to: {output_file}")