VENTURE-SCOPE a Startup Evaluation & Scoring Engine by Arthur Pillet 

Category: Business & Finance Tools / Data Analysis


MY MOTIVATION AND IDEA 

I’m interested into startups and finance and venture capital combines both subjects. That’s the reason I choose this project. In venture capital, investors constantly must decide which startups are worth their time and money. There’s a correct amount of public data available on Crunchbase, Dealroom, Kaggle datasets, etc… but most of this data is messy and hard to interpret. I have always been impressed by how venture capitalists turn incomplete, noisy data into conviction.

Venture-Scope is a project that tries to reproduce that process with code. My goal is to build a small analytical engine that will take public startup data, cleans it, computes meaningful venture KPIs (growth, burn, efficiency, runway…), and produces a composite score to help prioritize investments.
I will not be trying to predict unicorns but rather create a transparent, reproducible framework for evaluating early-stage startups.


MY APPROACH

I’ll start from a real Kaggle dataset based on Crunchbase (“Startup Investments” by Justinas), which includes hundreds of startups across sectors and funding rounds.

From there:

  1.	Data processing: cleaning inconsistent entries, harmonizing stages (Seed, Series A, etc.), standardizing countries and sectors.

  2.	Feature engineering: computing realistic KPIs using common VC heuristics:
      - Rule of 40 : how fast it’s expanding (growth + margin)
      - Burn Multiple: how fast it spends money (net burn / net new ARR)
      - Capital Efficiency: how well it uses investor’s money (revenue / total funding)
      - Runway : how many month before it runs out of cash(cash / burn rate)
      - Traction Index : how much attention it’s getting (growth × investor count × stage weight)

  3.	Scoring model: normalization with z-scores, adjustable weighting by theme (growth-focused vs profitability-focused investor profiles) to assign each project a 0-100 score 

  4.	Visualization & output: risk–return scatterplots, ranking tables that shows which startup look safer or more promising

The implementation will be done in Python with pandas, numpy, matplotlib, seaborn, plotly, and possibly streamlit for a simple dashboard.


EXPECTED CHALLENGES

The biggest challenge will be data completeness; many startups don’t publish metrics like revenue or burn rate especially in Switzerland.
I’ll handle this by enriching or estimating missing values using stage averages and ratios inspired by VC benchmarks and I will be documenting all assumptions.

The second challenge will be to make the score fair across different sectors and stages. I’ll test both global and within-sector normalization to see which is more meaningful.


SUCCESS CRITERIA

I’ll consider the project successful if:
-	the pipeline runs end-to-end (from raw CSV to clean scores and reports)
-	the visuals clearly show differences between startups
-	the scoring logic is explainable and reproducible

  
STRETCH GOALS

If time allows, I’d like to add:
-	a Monte Carlo simulation of potential portfolio outcomes,
-	and automated PDF exports of the top startups’ profiles.

