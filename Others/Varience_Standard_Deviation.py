import numpy as np

"""
VARIANCE AND STANDARD DEVIATION - MANUAL CALCULATION NOTES
===========================================================

1. VARIANCE (σ² for population, s² for sample)
   - Measures how spread out the data is from the mean
   - Average of squared differences from the mean

2. STANDARD DEVIATION (σ for population, s for sample)
   - Square root of variance
   - Expressed in the same units as the original data
   - More interpretable than variance

MANUAL CALCULATION FORMULAS:
----------------------------

A) POPULATION VARIANCE (σ²):
   σ² = Σ(xi - μ)² / N
   
   Where:
   - xi = each value in the dataset
   - μ = population mean
   - N = total number of values (population size)
   
   Steps:
   1. Calculate the mean (μ) of all values
   2. Subtract the mean from each value: (xi - μ)
   3. Square each difference: (xi - μ)²
   4. Sum all squared differences: Σ(xi - μ)²
   5. Divide by N (total count)

B) SAMPLE VARIANCE (s²):
   s² = Σ(xi - x̄)² / (n - 1)
   
   Where:
   - xi = each value in the dataset
   - x̄ = sample mean
   - n = sample size
   - (n - 1) = degrees of freedom (Bessel's correction)
   
   Steps: Same as population, but divide by (n - 1) instead of n

C) POPULATION STANDARD DEVIATION (σ):
   σ = √(σ²) = √[Σ(xi - μ)² / N]

D) SAMPLE STANDARD DEVIATION (s):
   s = √(s²) = √[Σ(xi - x̄)² / (n - 1)]

KEY DIFFERENCES:
----------------
1. POPULATION vs SAMPLE:
   - Population: Use when you have data for the ENTIRE population
     * Divide by N
     * np.var(data, ddof=0) or np.var(data)  [default]
     * np.std(data, ddof=0) or np.std(data)  [default]
   
   - Sample: Use when you have a SAMPLE from a larger population
     * Divide by (n - 1) for unbiased estimate
     * np.var(data, ddof=1)
     * np.std(data, ddof=1)

2. WHY (n - 1) for Sample? (Bessel's Correction)
   - Using n would underestimate the true population variance
   - (n - 1) provides an unbiased estimator
   - Accounts for the fact that sample mean is used instead of true population mean

EXAMPLE CALCULATION:
--------------------
Data: [1, 2, 200, 2, 1, 3, 2]

Population Variance:
1. Mean (μ) = (1+2+200+2+1+3+2) / 7 = 211/7 = 30.14
2. Differences: (1-30.14), (2-30.14), (200-30.14), (2-30.14), (1-30.14), (3-30.14), (2-30.14)
              = -29.14, -28.14, 169.86, -28.14, -29.14, -27.14, -28.14
3. Squared: 849.14, 791.86, 28852.41, 791.86, 849.14, 736.58, 791.86
4. Sum: 33662.86
5. Variance (σ²) = 33662.86 / 7 = 4808.98

Sample Variance:
- Same steps 1-4
- Variance (s²) = 33662.86 / (7-1) = 33662.86 / 6 = 5610.48

Standard Deviations:
- Population: σ = √4808.98 = 69.35
- Sample: s = √5610.48 = 74.90
"""

# Example with the current data
numbers = [1, 2, 200, 2, 1, 3, 2]

# Calculate mean manually
mean = sum(numbers) / len(numbers)

# Calculate population variance manually
squared_diffs = [(x - mean)**2 for x in numbers]
population_variance_manual = sum(squared_diffs) / len(numbers)
population_std_manual = population_variance_manual ** 0.5

# Calculate sample variance manually
sample_variance_manual = sum(squared_diffs) / (len(numbers) - 1)
sample_std_manual = sample_variance_manual ** 0.5

# Using NumPy
population_variance_numpy = np.var(numbers, ddof=0)  # ddof=0 for population (default)
population_std_numpy = np.std(numbers, ddof=0)

sample_variance_numpy = np.var(numbers, ddof=1)  # ddof=1 for sample
sample_std_numpy = np.std(numbers, ddof=1)

# Display results
print("=" * 70)
print("INPUT DATA")
print("=" * 70)
print(f"Numbers: {numbers}")
print(f"Count: {len(numbers)}")
print(f"Mean: {mean:.2f}")
print()

print("=" * 70)
print("POPULATION STATISTICS (divide by N)")
print("=" * 70)
print(f"Variance (Manual):  {population_variance_manual:.2f}")
print(f"Variance (NumPy):   {population_variance_numpy:.2f}")
print(f"Std Dev (Manual):   {population_std_manual:.2f}")
print(f"Std Dev (NumPy):    {population_std_numpy:.2f}")
print()

print("=" * 70)
print("SAMPLE STATISTICS (divide by n-1)")
print("=" * 70)
print(f"Variance (Manual):  {sample_variance_manual:.2f}")
print(f"Variance (NumPy):   {sample_variance_numpy:.2f}")
print(f"Std Dev (Manual):   {sample_std_manual:.2f}")
print(f"Std Dev (NumPy):    {sample_std_numpy:.2f}")
print()

print("=" * 70)
print("DIFFERENCE BETWEEN POPULATION AND SAMPLE")
print("=" * 70)
print(f"Variance Difference: {sample_variance_numpy - population_variance_numpy:.2f}")
print(f"Std Dev Difference:  {sample_std_numpy - population_std_numpy:.2f}")
print()
print("Note: Sample statistics are ALWAYS larger than population statistics")
print("      because we divide by (n-1) instead of n (Bessel's correction)")
print("=" * 70)
