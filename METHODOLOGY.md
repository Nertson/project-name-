# Methodology & Technical Decisions

*Documentation des décisions techniques et méthodologiques pour Venture-Scope*

---

## 1. Data Sources

**Dataset**: Crunchbase 2013 Snapshot  
**Source**: [Kaggle - justinas/startup-investments](https://www.kaggle.com/datasets/justinas/startup-investments)

### Files Used
- `objects.csv`: Company information (196,553 companies)
- `funding_rounds.csv`: Funding rounds details (52,928 rounds)
- `investments.csv`: Investor relationships (80,902 investments)

---

## 2. Data Filtering Decisions

### 2.1 Entity Type Filter
**Decision**: Keep only `entity_type == 'Company'`

**Rationale**: objects.csv contains multiple entity types (companies, people, investors, products). We filter to companies only.

**Impact**: Removed 266,098 non-company entities (57.5%)

---

### 2.2 Funding Amount Filter
**Decision**: Keep only companies with `funding_amount > $0`

**Rationale**: 
- Most $0 values represent missing data, not true bootstrapped companies
- Our VC-focused KPIs require funding data
- Project scope targets VC-backed startups

**Impact**: Filtered from 196,553 → 27,874 companies (14.2% retention)

**Trade-offs**:
- ✅ Cleaner dataset for VC analysis
- ✅ Enables KPI calculations
- ❌ Selection bias (excludes bootstrapped companies)
- ❌ Survivor bias (VC-backed companies only)

---

## 3. Data Quality Analysis

### 3.1 Investors Count Completeness
**Missing Rate**: 33.5% overall

**Analysis by Stage**:
- Series C: 4.1% missing (excellent)
- Series B: 6.2% missing (excellent)
- Series A: 31.7% missing (acceptable)
- Angel: 45.4% missing (high)

**Analysis by Funding Amount**:
- <$100K: 51.5% missing
- $100K-1M: 53.0% missing
- $1M-10M: 30.4% missing
- $10M+: ~13% missing

**Conclusion**: Missing data concentrated in early-stage and small fundings. This is consistent with Crunchbase's community-sourced nature.

**Mitigation**: Results are most reliable for Series B+ companies with funding ≥ $1M.

---

## 4. KPI Calculations

### 4.1 Capital Efficiency
**Formula**: `Estimated Revenue / Total Funding`

**Interpretation**:
- >1.0: Excellent (generates more revenue than funding raised)
- 0.5-1.0: Good efficiency
- 0.2-0.5: Acceptable for growth stage
- <0.2: Low efficiency (high burn)

**Results**: Median 0.30 (30¢ revenue per $1 raised)

---

### 4.2 Burn Rate & Runway
**Burn Rate Formula**: `Total Funding / Burn Period (months)`

**Burn Period by Stage**:
- Pre-Seed/Seed: 12-18 months
- Series A: 24 months
- Series B: 30 months
- Series C+: 36 months

**Runway Formula**: `Estimated Cash / Monthly Burn`

**Cash Estimate**: 50% of total funding (assumption: companies have deployed half their capital)

**Results**: 
- Median burn rate: $104K/month
- Median runway: 12 months (standard VC expectation)

---

### 4.3 Traction Index
**Formula**: `(log₁₀(Funding) × Investors × Stage Weight) / Age`

**Components**:
- **Funding**: Log scale to handle wide range ($100K - $1B+)
- **Investors**: Count of unique investors (social proof)
- **Stage Weight**: Pre-Seed (0.5), Seed (1.0), Series A (1.5), Series B (2.0), Series C (2.5), Series D+ (3.0)
- **Age**: Years since founding (younger = more impressive)

**Normalization**: Scaled to 0-100 for interpretability

**Interpretation**:
- 0-20: Early traction
- 20-40: Moderate traction
- 40-60: Strong traction
- 60-100: Exceptional traction

---

### 4.4 Rule of 40 (Estimated)

**Standard Formula**: `Revenue Growth Rate (%) + Profit Margin (%)`

**Challenge**: Historical revenue data unavailable in Crunchbase 2013

**Our Approach**: Two-step estimation

#### Step 1: Stage-Based Benchmarks
We apply typical growth/margin profiles by stage:

| Stage | Typical Growth | Typical Margin | Rule of 40 |
|-------|----------------|----------------|------------|
| Seed | 180% | -80% | 100 |
| Angel | 160% | -70% | 90 |
| Series A | 120% | -20% | 100 |
| Series B | 80% | 0% | 80 |
| Series C | 50% | 0% | 50 |
| Series D+ | 30% | 10% | 40 |

#### Step 2: Capital Efficiency Adjustment
Companies with higher capital efficiency likely have better Rule of 40:
```
Adjustment = (Capital Efficiency - 0.30) × 50
Estimated Rule of 40 = Benchmark + Adjustment
```

**Example**:
- Series A startup with capital efficiency 0.50
- Benchmark: 100
- Adjustment: (0.50 - 0.30) × 50 = +10
- Estimated Rule of 40: 110 ✅

**Limitation**: This is an **estimated** metric, not actual Rule of 40. Results should be interpreted as **relative rankings** rather than absolute values.

**Validation**: We will compare estimated Rule of 40 against known outcomes (acquisitions, IPOs) to validate the proxy.

---

### 4.5 Revenue Estimation

**Challenge**: Revenue data is proprietary and rarely disclosed

**Approach**: Stage-based multipliers of total funding

| Stage | Revenue Multiple |
|-------|------------------|
| Pre-Seed | 0.05x funding |
| Seed | 0.10x funding |
| Angel | 0.08x funding |
| Series A | 0.30x funding |
| Series B | 0.50x funding |
| Series C | 0.80x funding |
| Series D+ | 1.00x funding |

**Rationale**: These multiples are based on industry benchmarks where:
- Early stage: Low revenue, high burn
- Growth stage: Revenue scaling
- Late stage: Revenue approaching or exceeding total funding

**Example**: Series A startup with $10M funding → Estimated revenue: $3M/year

---

## 5. Future Improvements

To improve data quality and reduce bias:

1. **Enrich with external data sources** (LinkedIn, company websites)
2. **JOIN with acquisitions.csv** to identify successful exits
3. **Stratified sampling** to ensure representation across sectors/geographies
4. **Sensitivity analysis** on estimation assumptions
5. **Validate estimates** against known public companies

---

## 6. References

- Crunchbase Data Dictionary
- "SaaS Metrics 2.0" - David Skok
- "The Rule of 40" - Brad Feld
- Bessemer Cloud Index benchmarks
- OpenView SaaS Benchmarks

---

*Last Updated: 12.11.2025*  
*Author: Arthur Pillet*