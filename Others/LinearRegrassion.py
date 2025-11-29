"""
LINEAR REGRESSION - COMPLETE EXPLANATION
=========================================

What is Regression?
-------------------
Regression is a statistical method to model the relationship between:
- A dependent variable (y) - what we want to predict
- One or more independent variables (x) - what we use to make predictions

What is Linear Regression?
---------------------------
Linear regression finds the best-fit straight line through data points.
The equation of the line is: y = mx + b
Where:
- y = predicted value
- m = slope (how steep the line is)
- x = input value
- b = intercept (where line crosses y-axis)

LINEAR REGRESSION OUTPUT PARAMETERS
====================================

1. SLOPE (m)
------------
- Definition: The rate of change of y with respect to x
- Formula: m = Σ[(xi - x̄)(yi - ȳ)] / Σ[(xi - x̄)²]
- Interpretation:
  * Positive slope: y increases as x increases
  * Negative slope: y decreases as x increases
  * Zero slope: no relationship
- Example: slope = 12.28 means for every 1 unit increase in x, y increases by 12.28

2. INTERCEPT (b)
----------------
- Definition: The value of y when x = 0
- Formula: b = ȳ - m*x̄
- Interpretation: Where the regression line crosses the y-axis
- Example: intercept = 92.67 means when x=0, predicted y=92.67

3. R-VALUE (Correlation Coefficient)
-------------------------------------
- Definition: Measures the strength and direction of linear relationship
- Range: -1 to +1
- Formula: r = Σ[(xi - x̄)(yi - ȳ)] / √[Σ(xi - x̄)² * Σ(yi - ȳ)²]

Interpretation:
  * r = +1: Perfect positive correlation (all points on line, upward)
  * r = +0.7 to +1: Strong positive correlation
  * r = +0.3 to +0.7: Moderate positive correlation
  * r = 0 to +0.3: Weak positive correlation
  * r = 0: No correlation
  * r = -0.3 to 0: Weak negative correlation
  * r = -0.7 to -0.3: Moderate negative correlation
  * r = -1 to -0.7: Strong negative correlation
  * r = -1: Perfect negative correlation (all points on line, downward)

Example: r = 0.336 means weak positive correlation
         (x and y have a slight tendency to increase together)

4. R-SQUARED (R² or Coefficient of Determination)
--------------------------------------------------
- Definition: Proportion of variance in y explained by x
- Formula: R² = r²
- Range: 0 to 1 (or 0% to 100%)

Interpretation:
  * R² = 1 (100%): Model explains all variability
  * R² = 0.75 (75%): Model explains 75% of variability
  * R² = 0.5 (50%): Model explains 50% of variability
  * R² = 0 (0%): Model explains none of the variability

Example: R² = 0.336² = 0.113 (11.3%)
         Only 11.3% of variation in y is explained by x
         This is a WEAK model - most variation is due to other factors

5. P-VALUE (Statistical Significance)
--------------------------------------
- Definition: Probability that the observed relationship occurred by chance
- Range: 0 to 1
- Null Hypothesis: There is NO relationship between x and y (slope = 0)

Interpretation:
  * p < 0.001: Extremely strong evidence against null hypothesis (***) 
  * p < 0.01: Strong evidence against null hypothesis (**)
  * p < 0.05: Moderate evidence against null hypothesis (*)
  * p ≥ 0.05: Weak or no evidence against null hypothesis (not significant)

Common Significance Levels:
  * α = 0.05 (5%): Standard threshold
  * α = 0.01 (1%): More stringent
  * α = 0.10 (10%): More lenient

Example: p = 0.343 (34.3%)
         This is > 0.05, so the relationship is NOT statistically significant
         We CANNOT reject the null hypothesis
         The slope might just be due to random chance

Decision Rule:
  * If p < 0.05: Relationship is statistically significant ✓
  * If p ≥ 0.05: Relationship is NOT statistically significant ✗

6. STANDARD ERROR (std_err)
----------------------------
- Definition: Measures the accuracy of the slope estimate
- Interpretation: Average distance that observed values fall from regression line
- Lower is better (more precise estimate)

Formula: SE = √[Σ(yi - ŷi)² / (n-2)] / √[Σ(xi - x̄)²]
Where:
  * yi = actual y values
  * ŷi = predicted y values
  * n = number of data points

Example: std_err = 12.18
         The slope estimate has an uncertainty of ±12.18

Confidence Interval for Slope:
  * 95% CI = slope ± (1.96 × std_err)
  * 95% CI = 12.28 ± (1.96 × 12.18)
  * 95% CI = 12.28 ± 23.87
  * 95% CI = [-11.59, 36.15]
  
  Note: This interval includes 0, which confirms the relationship is not significant!

LEAST SQUARE METHOD
====================
How we find the best-fit line:
1. For each possible line, calculate the residuals (errors)
   Residual = actual y - predicted y
2. Square each residual (to make all positive)
3. Sum all squared residuals (SSE - Sum of Squared Errors)
4. Find the line that MINIMIZES SSE

This gives us the "best-fit" line!

GOODNESS OF FIT - R² METHOD
============================
R² tells us how well the model fits the data:

Total Variation = Explained Variation + Unexplained Variation
SST = SSR + SSE

Where:
- SST (Total Sum of Squares) = Σ(yi - ȳ)²
- SSR (Regression Sum of Squares) = Σ(ŷi - ȳ)²
- SSE (Error Sum of Squares) = Σ(yi - ŷi)²

R² = SSR/SST = 1 - (SSE/SST)

SUMMARY OF YOUR RESULTS
========================
Data: x = [1,2,3,4,5,6,7,8,9,10]
      y = [112,24,256,184,200,50,20,240,360,156]

Regression Equation: y = 12.28x + 92.67

Slope = 12.28
  → For every 1 unit increase in x, y increases by 12.28 units

Intercept = 92.67
  → When x=0, predicted y=92.67

R-value = 0.336
  → Weak positive correlation

R² = 0.113 (11.3%)
  → Model explains only 11.3% of variation
  → This is a POOR fit!

P-value = 0.343
  → NOT statistically significant (p > 0.05)
  → The relationship could be due to random chance
  → We should NOT trust this model for predictions

Standard Error = 12.18
  → High uncertainty in slope estimate

CONCLUSION: This linear model is NOT reliable for making predictions!
The relationship between x and y is too weak and not statistically significant.
"""
"""
লিনিয়ার রিগ্রেশন - সম্পূর্ণ ব্যাখ্যা
=========================================

রিগ্রেশন কি?
-------------------
রিগ্রেশন হলো একটি পরিসংখ্যানগত পদ্ধতি যা দুটি জিনিসের মধ্যে সম্পর্ক খুঁজে বের করে:
- নির্ভরশীল চলক (y) - যা আমরা predict করতে চাই
- স্বাধীন চলক (x) - যা দিয়ে আমরা prediction করি

লিনিয়ার রিগ্রেশন কি?
---------------------------
লিনিয়ার রিগ্রেশন ডেটা পয়েন্টগুলোর মধ্য দিয়ে সবচেয়ে ভালো সরলরেখা খুঁজে বের করে।
সরলরেখার সমীকরণ: y = mx + b
যেখানে:
- y = predicted মান
- m = slope (রেখা কতটা খাড়া)
- x = input মান
- b = intercept (রেখা y-axis কে কোথায় কাটে)

লিনিয়ার রিগ্রেশনের আউটপুট প্যারামিটার
====================================

১. SLOPE (ঢাল) - m
------------
- সংজ্ঞা: x এর সাপেক্ষে y এর পরিবর্তনের হার
- সূত্র: m = Σ[(xi - x̄)(yi - ȳ)] / Σ[(xi - x̄)²]
- ব্যাখ্যা:
  * পজিটিভ slope: x বাড়লে y ও বাড়ে
  * নেগেটিভ slope: x বাড়লে y কমে
  * শূন্য slope: কোনো সম্পর্ক নেই
- উদাহরণ: slope = 12.28 মানে x এ 1 একক বাড়লে y তে 12.28 একক বাড়বে

২. INTERCEPT (ছেদবিন্দু) - b
----------------
- সংজ্ঞা: x = 0 হলে y এর মান কত হবে
- সূত্র: b = ȳ - m*x̄
- ব্যাখ্যা: রিগ্রেশন লাইন y-axis কে যেখানে কাটে
- উদাহরণ: intercept = 92.67 মানে x=0 হলে predicted y=92.67

৩. R-VALUE (সহসম্বন্ধ সহগ)
-------------------------------------
- সংজ্ঞা: রৈখিক সম্পর্কের শক্তি এবং দিক পরিমাপ করে
- পরিসীমা: -1 থেকে +1
- সূত্র: r = Σ[(xi - x̄)(yi - ȳ)] / √[Σ(xi - x̄)² * Σ(yi - ȳ)²]

ব্যাখ্যা:
  * r = +1: নিখুঁত পজিটিভ সহসম্বন্ধ (সব পয়েন্ট লাইনে, উপরের দিকে)
  * r = +0.7 থেকে +1: শক্তিশালী পজিটিভ সহসম্বন্ধ
  * r = +0.3 থেকে +0.7: মাঝারি পজিটিভ সহসম্বন্ধ
  * r = 0 থেকে +0.3: দুর্বল পজিটিভ সহসম্বন্ধ
  * r = 0: কোনো সহসম্বন্ধ নেই
  * r = -0.3 থেকে 0: দুর্বল নেগেটিভ সহসম্বন্ধ
  * r = -0.7 থেকে -0.3: মাঝারি নেগেটিভ সহসম্বন্ধ
  * r = -1 থেকে -0.7: শক্তিশালী নেগেটিভ সহসম্বন্ধ
  * r = -1: নিখুঁত নেগেটিভ সহসম্বন্ধ (সব পয়েন্ট লাইনে, নিচের দিকে)

উদাহরণ: r = 0.336 মানে দুর্বল পজিটিভ সহসম্বন্ধ
         (x এবং y একসাথে বাড়ার সামান্য প্রবণতা আছে)

৪. R-SQUARED (R² বা নির্ধারণ সহগ)
--------------------------------------------------
- সংজ্ঞা: x দ্বারা y এর কতটুকু পরিবর্তন ব্যাখ্যা করা যায়
- সূত্র: R² = r²
- পরিসীমা: 0 থেকে 1 (বা 0% থেকে 100%)

ব্যাখ্যা:
  * R² = 1 (100%): মডেল সব পরিবর্তন ব্যাখ্যা করে
  * R² = 0.75 (75%): মডেল 75% পরিবর্তন ব্যাখ্যা করে
  * R² = 0.5 (50%): মডেল 50% পরিবর্তন ব্যাখ্যা করে
  * R² = 0 (0%): মডেল কোনো পরিবর্তন ব্যাখ্যা করে না

উদাহরণ: R² = 0.336² = 0.113 (11.3%)
         y এর মাত্র 11.3% পরিবর্তন x দ্বারা ব্যাখ্যা করা যায়
         এটি একটি দুর্বল মডেল - বেশিরভাগ পরিবর্তন অন্য কারণে

৫. P-VALUE (পরিসংখ্যানগত তাৎপর্য)
--------------------------------------
- সংজ্ঞা: পর্যবেক্ষিত সম্পর্কটি কাকতালীয়ভাবে ঘটার সম্ভাবনা
- পরিসীমা: 0 থেকে 1
- শূন্য অনুমান: x এবং y এর মধ্যে কোনো সম্পর্ক নেই (slope = 0)

ব্যাখ্যা:
  * p < 0.001: শূন্য অনুমানের বিরুদ্ধে অত্যন্ত শক্তিশালী প্রমাণ (***) 
  * p < 0.01: শক্তিশালী প্রমাণ (**)
  * p < 0.05: মাঝারি প্রমাণ (*)
  * p ≥ 0.05: দুর্বল বা কোনো প্রমাণ নেই (তাৎপর্যপূর্ণ নয়)

সাধারণ তাৎপর্য স্তর:
  * α = 0.05 (5%): স্ট্যান্ডার্ড থ্রেশহোল্ড
  * α = 0.01 (1%): আরো কঠোর
  * α = 0.10 (10%): আরো নমনীয়

উদাহরণ: p = 0.343 (34.3%)
         এটি > 0.05, তাই সম্পর্কটি পরিসংখ্যানগতভাবে তাৎপর্যপূর্ণ নয়
         আমরা শূন্য অনুমান প্রত্যাখ্যান করতে পারি না
         slope টি শুধুমাত্র কাকতালীয় হতে পারে

সিদ্ধান্তের নিয়ম:
  * যদি p < 0.05: সম্পর্ক পরিসংখ্যানগতভাবে তাৎপর্যপূর্ণ ✓
  * যদি p ≥ 0.05: সম্পর্ক পরিসংখ্যানগতভাবে তাৎপর্যপূর্ণ নয় ✗

৬. STANDARD ERROR (মান ত্রুটি)
----------------------------
- সংজ্ঞা: slope অনুমানের নির্ভুলতা পরিমাপ করে
- ব্যাখ্যা: পর্যবেক্ষিত মানগুলো রিগ্রেশন লাইন থেকে গড়ে কতটা দূরে পড়ে
- কম হলে ভালো (আরো সুনির্দিষ্ট অনুমান)

সূত্র: SE = √[Σ(yi - ŷi)² / (n-2)] / √[Σ(xi - x̄)²]
যেখানে:
  * yi = প্রকৃত y মান
  * ŷi = predicted y মান
  * n = ডেটা পয়েন্টের সংখ্যা

উদাহরণ: std_err = 12.18
         slope অনুমানে ±12.18 অনিশ্চয়তা আছে

Slope এর জন্য আত্মবিশ্বাস ব্যবধান:
  * 95% CI = slope ± (1.96 × std_err)
  * 95% CI = 12.28 ± (1.96 × 12.18)
  * 95% CI = 12.28 ± 23.87
  * 95% CI = [-11.59, 36.15]
  
  নোট: এই ব্যবধানে 0 আছে, যা নিশ্চিত করে সম্পর্কটি তাৎপর্যপূর্ণ নয়!

লিস্ট স্কয়ার পদ্ধতি (LEAST SQUARE METHOD)
====================
কিভাবে আমরা সবচেয়ে ভালো লাইন খুঁজি:
১. প্রতিটি সম্ভাব্য লাইনের জন্য, residuals (ত্রুটি) হিসাব করি
   Residual = প্রকৃত y - predicted y
২. প্রতিটি residual বর্গ করি (সব পজিটিভ করার জন্য)
৩. সব বর্গকৃত residuals যোগ করি (SSE - Sum of Squared Errors)
৪. যে লাইন SSE কে ন্যূনতম করে তা খুঁজি

এটি আমাদের "best-fit" লাইন দেয়!

ফিটনেসের গুণমান - R² পদ্ধতি
============================
R² আমাদের বলে মডেল ডেটার সাথে কতটা ভালো মানানসই:

মোট পরিবর্তন = ব্যাখ্যাকৃত পরিবর্তন + অব্যাখ্যাকৃত পরিবর্তন
SST = SSR + SSE

যেখানে:
- SST (Total Sum of Squares) = Σ(yi - ȳ)²
- SSR (Regression Sum of Squares) = Σ(ŷi - ȳ)²
- SSE (Error Sum of Squares) = Σ(yi - ŷi)²

R² = SSR/SST = 1 - (SSE/SST)

আপনার ফলাফলের সারসংক্ষেপ
========================
ডেটা: x = [1,2,3,4,5,6,7,8,9,10]
      y = [112,24,256,184,200,50,20,240,360,156]

রিগ্রেশন সমীকরণ: y = 12.28x + 92.67

Slope = 12.28
  → x এ প্রতি 1 একক বৃদ্ধিতে, y তে 12.28 একক বৃদ্ধি পায়

Intercept = 92.67
  → যখন x=0, predicted y=92.67

R-value = 0.336
  → দুর্বল পজিটিভ সহসম্বন্ধ

R² = 0.113 (11.3%)
  → মডেল মাত্র 11.3% পরিবর্তন ব্যাখ্যা করে
  → এটি একটি খারাপ ফিট!

P-value = 0.343
  → পরিসংখ্যানগতভাবে তাৎপর্যপূর্ণ নয় (p > 0.05)
  → সম্পর্কটি কাকতালীয় হতে পারে
  → আমরা এই মডেলে prediction এর জন্য বিশ্বাস করা উচিত নয়

Standard Error = 12.18
  → slope অনুমানে উচ্চ অনিশ্চয়তা

উপসংহার: এই লিনিয়ার মডেল prediction করার জন্য নির্ভরযোগ্য নয়!
x এবং y এর মধ্যে সম্পর্ক খুবই দুর্বল এবং পরিসংখ্যানগতভাবে তাৎপর্যপূর্ণ নয়।
"""

import matplotlib.pyplot as plt
import scipy as sp
import numpy as np

x = [1,2,3,4,5,6,7,8,9,10]
y = [112,24,256,184,200, 50,20,240,360,156]

# Linear regression    
slope, intercept, r, p, std_err = sp.stats.linregress(x, y)

# Calculate R-squared
r_squared = r**2

print("=" * 70)
print("LINEAR REGRESSION RESULTS")
print("=" * 70)
print(f"Regression Equation: y = {slope:.2f}x + {intercept:.2f}")
print()
print(f"Slope (m):           {slope:.4f}")
print(f"  → For every 1 unit ↑ in x, y changes by {slope:.2f} units")
print()
print(f"Intercept (b):       {intercept:.4f}")
print(f"  → When x=0, predicted y={intercept:.2f}")
print()
print(f"R-value (r):         {r:.4f}")
if abs(r) >= 0.7:
    strength = "Strong"
elif abs(r) >= 0.3:
    strength = "Moderate"
else:
    strength = "Weak"
direction = "positive" if r > 0 else "negative"
print(f"  → {strength} {direction} correlation")
print()
print(f"R-squared (R²):      {r_squared:.4f} ({r_squared*100:.2f}%)")
print(f"  → Model explains {r_squared*100:.2f}% of variation in y")
if r_squared >= 0.7:
    fit = "EXCELLENT"
elif r_squared >= 0.5:
    fit = "GOOD"
elif r_squared >= 0.3:
    fit = "MODERATE"
else:
    fit = "POOR"
print(f"  → Model fit: {fit}")
print()
print(f"P-value (p):         {p:.4f}")
if p < 0.001:
    significance = "EXTREMELY SIGNIFICANT (***)"
elif p < 0.01:
    significance = "VERY SIGNIFICANT (**)"
elif p < 0.05:
    significance = "SIGNIFICANT (*)"
else:
    significance = "NOT SIGNIFICANT"
print(f"  → {significance}")
print(f"  → Relationship is {'RELIABLE' if p < 0.05 else 'NOT RELIABLE'}")
print()
print(f"Standard Error:      {std_err:.4f}")
print(f"  → 95% CI for slope: [{slope - 1.96*std_err:.2f}, {slope + 1.96*std_err:.2f}]")
print("=" * 70)
print()

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', s=100, alpha=0.6, label='Data Points')

# Convert x to numpy array for calculation
x_array = np.array(x)
y_pred = slope*x_array + intercept
plt.plot(x_array, y_pred, color='red', linewidth=2, label=f'Regression Line: y={slope:.2f}x+{intercept:.2f}')

# Add residual lines
for i in range(len(x)):
    plt.plot([x[i], x[i]], [y[i], y_pred[i]], 'g--', alpha=0.5, linewidth=1)

plt.xlabel('X values', fontsize=12)
plt.ylabel('Y values', fontsize=12)
plt.title(f'Linear Regression (R²={r_squared:.3f}, p={p:.3f})', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()