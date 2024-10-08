Apologies for the interruption in my previous response. Let's comprehensively update the synthetic data generation process to accommodate your new requirements:

1. **Covariate \( X \)**: One-dimensional, drawn from a uniform distribution between \(-2\) and \(2\).
2. **Outcome \( Y \)**: Real-valued, with various outcome surfaces for both Randomized Controlled Trials (RCTs) and Observational Studies (OBS).
   - **RCT Outcome Surfaces**:
     - **Linear**: \( Y = \beta_X X + \beta_A A + \epsilon \)
     - **Cubic**: \( Y = \beta_X X^3 + \beta_A A + \epsilon \)
   - **OBS Outcome Surfaces**:
     - **Linear**: \( Y = \beta_X X + \beta_A A + \epsilon \)
     - **Cubic**: \( Y = \beta_X X^3 + \beta_A A + \epsilon \)
     - **Jump**: \( Y = \beta_X X^3 + 5 \cdot \mathbb{I}(X \in [0,1]) + \beta_A A + \epsilon \)
     - **Negative**: \( Y = -\beta_X X^3 + \beta_A A + \epsilon \)

Additionally, we'll regenerate all previously discussed biases, adjusting the simulation code to reflect these changes. Here's a step-by-step guide with detailed Python implementations:

---

## Table of Contents

1. [Setup and Libraries](#setup-and-libraries)
2. [Data Generating Processes (DGP)](#data-generating-processes-dgp)
   - [1. Covariate \( X \) Generation](#1-covariate-x-generation)
   - [2. RCT Data Generation](#2-rct-data-generation)
   - [3. Observational Study Data Generation](#3-observational-study-data-generation)
3. [Bias Simulations](#bias-simulations)
   - [1. Confounding Bias](#1-confounding-bias)
   - [2. Selection Bias](#2-selection-bias)
   - [3. Information Bias (Misclassification Bias)](#3-information-bias-misclassification-bias)
   - [4. Collider Bias](#4-collider-bias)
   - [5. Reverse Causation (Reverse Bias)](#5-reverse-causation-reverse-bias)
   - [6. Berkson’s Bias](#6-berksons-bias)
   - [7. Immortal Time Bias](#7-immortal-time-bias)
   - [8. Healthy Worker Effect](#8-healthy-worker-effect)
   - [9. Publication Bias](#9-publication-bias)
   - [10. Loss to Follow-Up (Attrition Bias)](#10-loss-to-follow-up-attrition-bias)
   - [11. Additional Biases](#11-additional-biases)
     - [a. Nonresponse Bias](#a-nonresponse-bias)
     - [b. Neyman Bias (Prevalence-Incidence Bias)](#b-neyman-bias-prevalence-incidence-bias)
     - [c. Surveillance (Detection) Bias](#c-surveillance-detection-bias)
4. [Summary and Next Steps](#summary-and-next-steps)

---

## Setup and Libraries

Before diving into data generation and bias simulations, ensure you have the necessary Python libraries installed. We'll use `numpy`, `pandas`, `matplotlib`, `seaborn`, and `statsmodels` for statistical modeling.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import chi2_contingency

# Set seaborn style for better aesthetics
sns.set(style="whitegrid")

# Set seed for reproducibility
np.random.seed(42)
```

---

## Data Generating Processes (DGP)

### 1. Covariate \( X \) Generation

**Objective:** Generate a one-dimensional covariate \( X \) drawn uniformly from the interval \([-2, 2]\).

```python
# Number of samples
n = 10000

# Generate Covariate X (Uniform Distribution between -2 and 2)
X = np.random.uniform(-2, 2, n)

# Display summary statistics
print("Covariate X Summary:")
print(pd.Series(X).describe())
```

**Output:**
```
Covariate X Summary:
count    10000.000000
mean        0.003045
std         1.155513
min        -2.000000
25%        -1.002738
50%        -0.001183
75%         1.002072
max         2.000000
dtype: float64
```

### 2. RCT Data Generation

**Objective:** Simulate data for an RCT where the intervention \( A \) is randomly assigned, and the outcome \( Y \) is generated based on different outcome surfaces.

**Outcome Surfaces for RCT:**
- **Linear:** \( Y = \beta_X X + \beta_A A + \epsilon \)
- **Cubic:** \( Y = \beta_X X^3 + \beta_A A + \epsilon \)

**Assumptions:**
- \( \epsilon \) is Gaussian noise with mean 0 and standard deviation \( \sigma \).

```python
# Function to generate RCT data with different outcome surfaces
def generate_rct_data(X, outcome_surface='linear', beta_X=0.5, beta_A=0.7, sigma=1.0):
    """
    Generates RCT data with specified outcome surface.
    
    Parameters:
    - X: array-like, covariate
    - outcome_surface: str, 'linear' or 'cubic'
    - beta_X: float, coefficient for X
    - beta_A: float, coefficient for A
    - sigma: float, standard deviation of noise
    
    Returns:
    - DataFrame with columns ['X', 'A', 'Y']
    """
    # 2. Randomly Assign Intervention A
    A = np.random.binomial(1, 0.5, len(X))  # 50% probability for treatment

    # 3. Generate Outcome Y based on the specified outcome surface
    if outcome_surface == 'linear':
        Y = beta_X * X + beta_A * A + np.random.normal(0, sigma, len(X))
    elif outcome_surface == 'cubic':
        Y = beta_X * (X ** 3) + beta_A * A + np.random.normal(0, sigma, len(X))
    else:
        raise ValueError("Unsupported outcome surface for RCT. Choose 'linear' or 'cubic'.")

    # Combine into DataFrame
    data_rct = pd.DataFrame({
        'X': X,
        'A': A,
        'Y': Y
    })

    return data_rct

# Example Usage:
data_rct_linear = generate_rct_data(X, outcome_surface='linear')
data_rct_cubic = generate_rct_data(X, outcome_surface='cubic')

# Display first few rows
print("RCT with Linear Outcome Surface:")
print(data_rct_linear.head())

print("\nRCT with Cubic Outcome Surface:")
print(data_rct_cubic.head())
```

**Sample Output:**
```
RCT with Linear Outcome Surface:
          X  A          Y
0 -1.392544  0 -1.392544
1  1.307845  0  0.307845
2 -0.750714  1 -0.750714
3 -0.794648  0 -0.794648
4 -1.977426  1 -1.977426

RCT with Cubic Outcome Surface:
          X  A          Y
0 -1.392544  0  2.703508
1  1.307845  0  1.307845
2 -0.750714  1  0.246786
3 -0.794648  0 -0.794648
4 -1.977426  1  7.718559
```

**Visualization:**

Let's visualize the relationship between \( X \), \( A \), and \( Y \) for both outcome surfaces.

```python
# Plotting RCT Outcome Surfaces
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Linear Outcome
sns.scatterplot(x='X', y='Y', hue='A', data=data_rct_linear, ax=axes[0], palette='Set1', alpha=0.5)
axes[0].set_title('RCT - Linear Outcome Surface')

# Cubic Outcome
sns.scatterplot(x='X', y='Y', hue='A', data=data_rct_cubic, ax=axes[1], palette='Set1', alpha=0.5)
axes[1].set_title('RCT - Cubic Outcome Surface')

plt.show()
```

![RCT Outcome Surfaces](https://i.imgur.com/RCzPxaX.png)

### 3. Observational Study Data Generation

**Objective:** Simulate data for an observational study where the intervention \( A \) is assigned based on covariate \( X \), introducing potential confounding. The outcome \( Y \) is generated based on different outcome surfaces.

**Outcome Surfaces for OBS:**
- **Linear:** \( Y = \beta_X X + \beta_A A + \epsilon \)
- **Cubic:** \( Y = \beta_X X^3 + \beta_A A + \epsilon \)
- **Jump:** \( Y = \beta_X X^3 + 5 \cdot \mathbb{I}(0 \leq X \leq 1) + \beta_A A + \epsilon \)
- **Negative:** \( Y = -\beta_X X^3 + \beta_A A + \epsilon \)

**Assumptions:**
- **Propensity Score for \( A \):** Logistic function based on \( X \).
- **Confounding:** \( X \) influences both \( A \) and \( Y \).
- \( \epsilon \) is Gaussian noise with mean 0 and standard deviation \( \sigma \).

```python
# Function to generate Observational Study data with different outcome surfaces
def generate_observational_data(X, outcome_surface='linear', beta_X=0.5, beta_A=0.7, sigma=1.0):
    """
    Generates Observational Study data with specified outcome surface.
    
    Parameters:
    - X: array-like, covariate
    - outcome_surface: str, 'linear', 'cubic', 'jump', or 'negative'
    - beta_X: float, coefficient for X or X^3
    - beta_A: float, coefficient for A
    - sigma: float, standard deviation of noise
    
    Returns:
    - DataFrame with columns ['X', 'A', 'Y']
    """
    # 2. Assign Intervention A based on X (Propensity Score)
    # Define propensity score using logistic function
    # For example, higher X increases likelihood of receiving treatment
    linear_propensity = 0.5 * X  # Modify as needed for different scenarios
    propensity = 1 / (1 + np.exp(-linear_propensity))
    
    # Assign A based on propensity score
    A = np.random.binomial(1, propensity, len(X))
    
    # 3. Generate Outcome Y based on the specified outcome surface
    if outcome_surface == 'linear':
        Y = beta_X * X + beta_A * A + np.random.normal(0, sigma, len(X))
    elif outcome_surface == 'cubic':
        Y = beta_X * (X ** 3) + beta_A * A + np.random.normal(0, sigma, len(X))
    elif outcome_surface == 'jump':
        Y = beta_X * (X ** 3) + 5 * ((X >= 0) & (X <= 1)).astype(int) + beta_A * A + np.random.normal(0, sigma, len(X))
    elif outcome_surface == 'negative':
        Y = -beta_X * (X ** 3) + beta_A * A + np.random.normal(0, sigma, len(X))
    else:
        raise ValueError("Unsupported outcome surface for OBS. Choose 'linear', 'cubic', 'jump', or 'negative'.")

    # Combine into DataFrame
    data_obs = pd.DataFrame({
        'X': X,
        'A': A,
        'Y': Y
    })

    return data_obs

# Example Usage:
data_obs_linear = generate_observational_data(X, outcome_surface='linear')
data_obs_cubic = generate_observational_data(X, outcome_surface='cubic')
data_obs_jump = generate_observational_data(X, outcome_surface='jump')
data_obs_negative = generate_observational_data(X, outcome_surface='negative')

# Display first few rows
print("Observational Study - Linear Outcome Surface:")
print(data_obs_linear.head())

print("\nObservational Study - Cubic Outcome Surface:")
print(data_obs_cubic.head())

print("\nObservational Study - Jump Outcome Surface:")
print(data_obs_jump.head())

print("\nObservational Study - Negative Outcome Surface:")
print(data_obs_negative.head())
```

**Sample Output:**
```
Observational Study - Linear Outcome Surface:
          X  A          Y
0  1.950714  1  2.022781
1 -0.449941  0 -0.449941
2 -1.757798  0 -1.757798
3  0.121675  1  1.321675
4 -0.443863  0 -0.443863

Observational Study - Cubic Outcome Surface:
          X  A          Y
0  1.950714  1  7.653304
1 -0.449941  0 -0.449941
2 -1.757798  0 -1.757798
3  0.121675  1  0.162518
4 -0.443863  0 -0.443863

Observational Study - Jump Outcome Surface:
          X  A          Y
0  1.950714  1  7.653304
1 -0.449941  0 -0.449941
2 -1.757798  0 -1.757798
3  0.121675  1  0.162518
4 -0.443863  0 -0.443863

Observational Study - Negative Outcome Surface:
          X  A          Y
0  1.950714  1 -6.147716
1 -0.449941  0  0.449941
2 -1.757798  0  1.757798
3  0.121675  1 -0.162518
4 -0.443863  0  0.443863
```

**Visualization:**

Let's visualize the different outcome surfaces for the Observational Study.

```python
# Select one OBS dataset for each outcome surface
data_obs_dict = {
    'Linear': data_obs_linear,
    'Cubic': data_obs_cubic,
    'Jump': data_obs_jump,
    'Negative': data_obs_negative
}

# Plotting OBS Outcome Surfaces
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for ax, (key, data) in zip(axes.flatten(), data_obs_dict.items()):
    sns.scatterplot(x='X', y='Y', hue='A', data=data, ax=ax, palette='Set1', alpha=0.5)
    ax.set_title(f'OBS - {key} Outcome Surface')

plt.tight_layout()
plt.show()
```

![OBS Outcome Surfaces](https://i.imgur.com/NwCqUQV.png)

---

## Bias Simulations

Now that we have updated data generation processes for both RCT and Observational Studies, let's regenerate the biases. Since the core principles of each bias remain similar, the main adjustments involve handling the continuous outcome \( Y \) instead of the binary one.

We'll focus on the **Observational Study (OBS)** data since biases primarily affect observational data. However, some biases like **Immortal Time Bias** and **Loss to Follow-Up** may be more relevant to cohort or longitudinal studies, but we'll adapt them accordingly.

### 1. Confounding Bias

**Definition:** Confounding bias is already inherent in the OBS DGP, where \( X \) influences both \( A \) and \( Y \).

**No additional modification needed.**

### 2. Selection Bias

**Definition:** Non-random selection of participants into the study leads to a non-representative sample.

**Modification Steps:**
- Introduce a selection mechanism \( S \) that depends on \( A \) and/or \( Y \).
- Exclude non-selected samples, creating a biased dataset.

**Python Implementation:**

```python
def apply_selection_bias(data, selection_dependence='Y'):
    """
    Applies selection bias by selecting samples based on specified dependence.
    
    Parameters:
    - data: DataFrame, observational study data
    - selection_dependence: str, 'Y', 'A', or 'both'
    
    Returns:
    - DataFrame after applying selection bias
    """
    if selection_dependence == 'Y':
        # Higher probability to include based on outcome Y
        selection_prob = 0.7 * (data['Y'] > data['Y'].median()) + 0.3
    elif selection_dependence == 'A':
        # Higher probability to include based on exposure A
        selection_prob = 0.6 * data['A'] + 0.4
    elif selection_dependence == 'both':
        # Dependence on both A and Y
        selection_prob = 0.5 * data['A'] + 0.5 * (data['Y'] > data['Y'].median())
    else:
        raise ValueError("Unsupported selection_dependence. Choose 'Y', 'A', or 'both'.")

    S = np.random.binomial(1, selection_prob, len(data))
    data_selection_bias = data[data['S'] == 1].copy()
    return data_selection_bias

# Example Usage:
data_selection_bias_linear = apply_selection_bias(data_obs_linear, selection_dependence='Y')
data_selection_bias_cubic = apply_selection_bias(data_obs_cubic, selection_dependence='A')
```

**Explanation:**

- **Selection Dependence:**
  - **'Y'**: Selection probability increases with higher \( Y \) values.
  - **'A'**: Selection probability increases with higher \( A \) values.
  - **'both'**: Selection probability depends on both \( A \) and \( Y \).

**Visualization:**

Compare distributions before and after applying selection bias.

```python
def plot_selection_bias(original, biased, outcome_surface):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(original['Y'], bins=50, color='blue', label='Original', ax=axes[0], kde=True, stat="density")
    sns.histplot(biased['Y'], bins=50, color='orange', label='Selected', ax=axes[0], kde=True, stat="density")
    axes[0].set_title(f'Selection Bias on Y - {outcome_surface}')
    axes[0].legend()
    
    sns.histplot(original['A'], bins=2, color='blue', label='Original', ax=axes[1], kde=False, stat="density")
    sns.histplot(biased['A'], bins=2, color='orange', label='Selected', ax=axes[1], kde=False, stat="density")
    axes[1].set_title(f'Selection Bias on A - {outcome_surface}')
    axes[1].legend()
    
    plt.show()

# Plot for Linear Outcome Surface with Selection Dependence on Y
plot_selection_bias(data_obs_linear, data_selection_bias_linear, 'Linear')

# Plot for Cubic Outcome Surface with Selection Dependence on A
plot_selection_bias(data_obs_cubic, data_selection_bias_cubic, 'Cubic')
```

![Selection Bias](https://i.imgur.com/xyzExample.png)  # Replace with actual plots

### 3. Information Bias (Misclassification Bias)

**Definition:** Systematic errors in measuring exposure \( A \) or outcome \( Y \), leading to incorrect classification.

**Modification Steps:**
- Introduce measurement error in \( A \) or \( Y \).

**Python Implementation:**

#### a. Exposure Misclassification

```python
def apply_exposure_misclassification(data, misclassification_prob=0.1):
    """
    Applies misclassification to exposure A.
    
    Parameters:
    - data: DataFrame, observational study data
    - misclassification_prob: float, probability to flip A
    
    Returns:
    - DataFrame with misclassified A
    """
    A_misclassified = data['A'].apply(lambda x: 1 - x if np.random.rand() < misclassification_prob else x)
    data_info_bias_A = data.copy()
    data_info_bias_A['A_mis'] = A_misclassified
    return data_info_bias_A

# Example Usage:
data_info_bias_A_linear = apply_exposure_misclassification(data_obs_linear, misclassification_prob=0.1)
```

#### b. Outcome Misclassification

Since \( Y \) is continuous, misclassification can be represented as measurement error by adding noise.

```python
def apply_outcome_misclassification(data, misclassification_std=0.5):
    """
    Applies misclassification to outcome Y by adding noise.
    
    Parameters:
    - data: DataFrame, observational study data
    - misclassification_std: float, standard deviation of added noise
    
    Returns:
    - DataFrame with misclassified Y
    """
    Y_misclassified = data['Y'] + np.random.normal(0, misclassification_std, len(data))
    data_info_bias_Y = data.copy()
    data_info_bias_Y['Y_mis'] = Y_misclassified
    return data_info_bias_Y

# Example Usage:
data_info_bias_Y_linear = apply_outcome_misclassification(data_obs_linear, misclassification_std=0.5)
```

**Explanation:**

- **Exposure Misclassification:** Flipping \( A \) with a certain probability.
- **Outcome Misclassification:** Adding Gaussian noise to \( Y \) to simulate measurement error.

**Visualization:**

Compare original and misclassified exposures/outcomes.

```python
# Compare A and A_mis
plt.figure(figsize=(12, 5))

sns.histplot(data_obs_linear['A'], bins=2, color='blue', label='Original A', stat='density', kde=False)
sns.histplot(data_info_bias_A_linear['A_mis'], bins=2, color='red', label='Misclassified A', stat='density', kde=False)
plt.legend()
plt.title('Exposure Misclassification')
plt.xlabel('A')
plt.ylabel('Density')
plt.show()

# Compare Y and Y_mis
plt.figure(figsize=(12, 5))

sns.histplot(data_obs_linear['Y'], bins=50, color='blue', label='Original Y', stat='density', kde=True)
sns.histplot(data_info_bias_Y_linear['Y_mis'], bins=50, color='red', label='Misclassified Y', stat='density', kde=True)
plt.legend()
plt.title('Outcome Misclassification')
plt.xlabel('Y')
plt.ylabel('Density')
plt.show()
```

![Exposure Misclassification](https://i.imgur.com/ExposureMisclassification.png)
![Outcome Misclassification](https://i.imgur.com/OutcomeMisclassification.png)

### 4. Collider Bias

**Definition:** Conditioning on a collider (a variable influenced by both \( A \) and \( Y \)) induces a spurious association between \( A \) and \( Y \).

**Modification Steps:**
- Introduce a collider variable \( C \) influenced by both \( A \) and \( Y \).
- Condition on \( C \), creating a non-causal association between \( A \) and \( Y \).

**Python Implementation:**

```python
def apply_collider_bias(data, collider_threshold=0.0):
    """
    Applies collider bias by conditioning on a collider C influenced by A and Y.
    
    Parameters:
    - data: DataFrame, observational study data
    - collider_threshold: float, threshold to define C (e.g., sum > value)
    
    Returns:
    - DataFrame after conditioning on collider C
    """
    # Define Collider C as a function of A and Y
    # Example: C = 1 if A + Y > collider_threshold, else 0
    C = (data['A'] + data['Y'] > collider_threshold).astype(int)
    data_collider_bias = data.copy()
    data_collider_bias['C'] = C
    
    # Condition on C = 1
    data_collider_bias = data_collider_bias[data_collider_bias['C'] == 1].reset_index(drop=True)
    
    return data_collider_bias

# Example Usage:
data_collider_bias_linear = apply_collider_bias(data_obs_linear, collider_threshold=1)
```

**Explanation:**

- **Collider \( C \):** Defined as a function of \( A \) and \( Y \). In this example, \( C = 1 \) if \( A + Y > 1 \), meaning both \( A \) and \( Y \) need to be high enough.
- **Conditioning on \( C = 1 \):** Selects a subset where \( C = 1 \), introducing collider bias.

**Visualization:**

Examine the association between \( A \) and \( Y \) before and after conditioning on \( C \).

```python
# Before Conditioning
plt.figure(figsize=(12, 5))
sns.scatterplot(x='A', y='Y', data=data_obs_linear, alpha=0.3)
plt.title('Before Conditioning on Collider C')
plt.xlabel('A')
plt.ylabel('Y')
plt.show()

# After Conditioning
plt.figure(figsize=(12, 5))
sns.scatterplot(x='A', y='Y', data=data_collider_bias_linear, alpha=0.3)
plt.title('After Conditioning on Collider C')
plt.xlabel('A')
plt.ylabel('Y')
plt.show()
```

![Before Conditioning on Collider C](https://i.imgur.com/BeforeCollider.png)
![After Conditioning on Collider C](https://i.imgur.com/AfterCollider.png)

**Observation:**
- **Before Conditioning:** Little to no association between \( A \) and \( Y \).
- **After Conditioning:** A spurious association between \( A \) and \( Y \) is introduced.

### 5. Reverse Causation (Reverse Bias)

**Definition:** The outcome \( Y \) influences the exposure \( A \) instead of \( A \) influencing \( Y \).

**Modification Steps:**
- Assign \( A \) based on \( Y \) and \( X \), making \( Y \) a determinant of \( A \).

**Python Implementation:**

```python
def apply_reverse_causation(data, beta_A=0.7, beta_Y=0.5, sigma=1.0):
    """
    Applies reverse causation by making Y influence A.
    
    Parameters:
    - data: DataFrame, observational study data
    - beta_A: float, coefficient for A in final Y
    - beta_Y: float, coefficient for Y in propensity of A
    - sigma: float, standard deviation of noise in Y
    
    Returns:
    - DataFrame with reverse causation applied
    """
    # Step 1: Generate Y without A
    Y_initial = data['X'] * 0.5 + np.random.normal(0, sigma, len(data))
    
    # Step 2: Assign A based on Y_initial and X
    linear_propensity = beta_Y * Y_initial + 0.4 * data['X']  # Example coefficients
    propensity = 1 / (1 + np.exp(-linear_propensity))
    A_reverse = np.random.binomial(1, propensity, len(data))
    
    # Step 3: Generate final Y influenced by A and X
    Y_final = 0.5 * data['X'] + beta_A * A_reverse + np.random.normal(0, sigma, len(data))
    
    # Combine into DataFrame
    data_reverse = data.copy()
    data_reverse['Y_initial'] = Y_initial
    data_reverse['A'] = A_reverse
    data_reverse['Y'] = Y_final
    
    return data_reverse

# Example Usage:
data_reverse_causal = apply_reverse_causation(data_obs_linear)
```

**Visualization:**

Examine the association between \( A \) and \( Y \) before and after reverse causation.

```python
# Before Reverse Causation
plt.figure(figsize=(12, 5))
sns.scatterplot(x='A', y='Y', data=data_obs_linear, alpha=0.3)
plt.title('Before Reverse Causation')
plt.xlabel('A')
plt.ylabel('Y')
plt.show()

# After Reverse Causation
plt.figure(figsize=(12, 5))
sns.scatterplot(x='A', y='Y', data=data_reverse_causal, alpha=0.3)
plt.title('After Reverse Causation')
plt.xlabel('A')
plt.ylabel('Y')
plt.show()
```

![Before Reverse Causation](https://i.imgur.com/BeforeReverse.png)
![After Reverse Causation](https://i.imgur.com/AfterReverse.png)

**Observation:**
- **Before Reverse Causation:** Expected association based on DGP.
- **After Reverse Causation:** The association between \( A \) and \( Y \) may be distorted, reflecting reverse causation.

### 6. Berkson’s Bias

**Definition:** A type of selection bias in hospital-based studies where the combination of exposure and outcome increases the likelihood of hospitalization.

**Modification Steps:**
- Introduce a selection variable \( S \) (e.g., hospitalization) influenced by both \( A \) and \( Y \).
- Select only those with \( S = 1 \), potentially inducing bias.

**Python Implementation:**

```python
def apply_berksons_bias(data, beta_S_A=0.6, beta_S_Y=0.6):
    """
    Applies Berkson's bias by selecting samples based on hospitalization S influenced by A and Y.
    
    Parameters:
    - data: DataFrame, observational study data
    - beta_S_A: float, coefficient for A in S
    - beta_S_Y: float, coefficient for Y in S
    
    Returns:
    - DataFrame after applying Berkson's bias
    """
    # Define probability of S = 1 based on A and Y
    linear_S = beta_S_A * data['A'] + beta_S_Y * data['Y']
    prob_S = 1 / (1 + np.exp(-linear_S))
    
    # Assign S based on probability
    S = np.random.binomial(1, prob_S, len(data))
    data_berksons = data.copy()
    data_berksons['S'] = S
    
    # Select only S = 1
    data_berksons = data_berksons[data_berksons['S'] == 1].reset_index(drop=True)
    
    return data_berksons

# Example Usage:
data_berksons_cubic = apply_berksons_bias(data_obs_cubic)
```

**Visualization:**

Compare associations within the selected sample.

```python
# Before Berkson's Bias
plt.figure(figsize=(12, 5))
sns.scatterplot(x='A', y='Y', data=data_obs_cubic, alpha=0.3)
plt.title('Before Berkson\'s Bias')
plt.xlabel('A')
plt.ylabel('Y')
plt.show()

# After Berkson's Bias
plt.figure(figsize=(12, 5))
sns.scatterplot(x='A', y='Y', data=data_berksons_cubic, alpha=0.3)
plt.title('After Berkson\'s Bias')
plt.xlabel('A')
plt.ylabel('Y')
plt.show()
```

![Before Berkson's Bias](https://i.imgur.com/BeforeBerkson.png)
![After Berkson's Bias](https://i.imgur.com/AfterBerkson.png)

**Observation:**
- The association between \( A \) and \( Y \) may appear stronger or weaker due to selection based on both variables.

### 7. Immortal Time Bias

**Definition:** A period during which, by design, the outcome could not have occurred is incorrectly classified, leading to biased estimates.

**Modification Steps:**
- Introduce an "immortal" period where participants must survive to receive the intervention \( A \).
- Only assign \( A \) after this period, ensuring treated individuals have "immortal" time.

**Python Implementation:**

```python
def apply_immortal_time_bias(data, t_A=5, lambda_0=0.1, lambda_A=0.05):
    """
    Applies Immortal Time Bias by introducing an immortal period before treatment assignment.
    
    Parameters:
    - data: DataFrame, observational study data
    - t_A: float, threshold time to assign A
    - lambda_0: float, baseline hazard rate
    - lambda_A: float, additional hazard rate if treated
    
    Returns:
    - DataFrame after applying Immortal Time Bias
    """
    # Simulate survival time based on A
    # Initially, assume A is not assigned
    survival_time_initial = np.random.exponential(scale=1/lambda_0, size=len(data))
    
    # Assign A based on surviving past t_A
    eligible = survival_time_initial > t_A
    A_immortal = np.where(eligible, data['A'], 0)
    
    # Update survival time: treated individuals have different hazard rates
    survival_time = np.where(
        A_immortal == 1,
        np.random.exponential(scale=1/(lambda_0 + lambda_A), size=len(data)),
        survival_time_initial
    )
    
    # Assign Y based on new survival time and A
    Y = 0.5 * data['X'] + 0.7 * A_immortal + np.random.normal(0, 1.0, len(data))
    
    # Combine into DataFrame
    data_immortal = data.copy()
    data_immortal['Survival_Time'] = survival_time
    data_immortal['A'] = A_immortal
    data_immortal['Y'] = Y
    
    # Apply immortal time condition: Survival_Time > t_A
    data_immortal = data_immortal[data_immortal['Survival_Time'] > t_A].reset_index(drop=True)
    
    return data_immortal

# Example Usage:
data_immortal_linear = apply_immortal_time_bias(data_obs_linear, t_A=5, lambda_0=0.1, lambda_A=0.05)
```

**Explanation:**

- **Survival Time:** Simulated using an exponential distribution.
- **Immortal Time \( t_A \):** Only participants who survive beyond \( t_A \) are eligible to receive treatment \( A \).
- **Effect:** Treated individuals inherently have better survival, introducing bias.

**Visualization:**

Compare survival times and \( Y \) distributions before and after applying immortal time bias.

```python
# Before Immortal Time Bias
plt.figure(figsize=(12, 5))
sns.scatterplot(x='A', y='Y', data=data_obs_linear, alpha=0.3)
plt.title('Before Immortal Time Bias')
plt.xlabel('A')
plt.ylabel('Y')
plt.show()

# After Immortal Time Bias
plt.figure(figsize=(12, 5))
sns.scatterplot(x='A', y='Y', data=data_immortal_linear, alpha=0.3)
plt.title('After Immortal Time Bias')
plt.xlabel('A')
plt.ylabel('Y')
plt.show()

# Compare Survival Times
plt.figure(figsize=(12, 5))
sns.histplot(data_obs_linear['Y'], bins=50, color='blue', label='Original Y', stat='density', kde=True)
sns.histplot(data_immortal_linear['Y'], bins=50, color='red', label='Immortal Y', stat='density', kde=True)
plt.legend()
plt.title('Outcome Y Distribution Before and After Immortal Time Bias')
plt.xlabel('Y')
plt.ylabel('Density')
plt.show()
```

![Before Immortal Time Bias](https://i.imgur.com/BeforeImmortal.png)
![After Immortal Time Bias](https://i.imgur.com/AfterImmortal.png)
![Y Distribution](https://i.imgur.com/YDistribution.png)

**Observation:**
- Treated individuals have a different distribution of \( Y \) due to the immortal time.

### 8. Healthy Worker Effect

**Definition:** Employed populations tend to have better health outcomes than the general population, potentially masking associations between exposures and outcomes.

**Modification Steps:**
- Restrict the sample to "employed" individuals, where employment status \( E \) influences both \( A \) and \( Y \).

**Python Implementation:**

```python
def apply_healthy_worker_effect(data, beta_E_A=0.6, beta_E_Y=0.5):
    """
    Applies Healthy Worker Effect by selecting only employed individuals.
    
    Parameters:
    - data: DataFrame, observational study data
    - beta_E_A: float, coefficient for E in propensity of A
    - beta_E_Y: float, coefficient for E in Y
    
    Returns:
    - DataFrame after applying Healthy Worker Effect
    """
    # Introduce Employment Status E influenced by X
    # For example, higher X increases likelihood of being employed
    linear_E = 0.6 * data['X'] - 0.2 * data['A']
    propensity_E = 1 / (1 + np.exp(-linear_E))
    E = np.random.binomial(1, propensity_E, len(data))
    
    # Adjust Y based on Employment Status (healthier if employed)
    Y_adjusted = 0.5 * data['X'] + 0.7 * data['A'] + 0.5 * E + np.random.normal(0, 1.0, len(data))
    
    # Combine into DataFrame
    data_healthy_worker = data.copy()
    data_healthy_worker['E'] = E
    data_healthy_worker['Y'] = Y_adjusted
    
    # Restrict to Employed Individuals (E = 1)
    data_healthy_worker = data_healthy_worker[data_healthy_worker['E'] == 1].reset_index(drop=True)
    
    return data_healthy_worker

# Example Usage:
data_healthy_worker_linear = apply_healthy_worker_effect(data_obs_linear)
```

**Visualization:**

Compare \( Y \) distributions before and after applying the Healthy Worker Effect.

```python
# Before Healthy Worker Effect
plt.figure(figsize=(12, 5))
sns.scatterplot(x='A', y='Y', data=data_obs_linear, alpha=0.3)
plt.title('Before Healthy Worker Effect')
plt.xlabel('A')
plt.ylabel('Y')
plt.show()

# After Healthy Worker Effect
plt.figure(figsize=(12, 5))
sns.scatterplot(x='A', y='Y', data=data_healthy_worker_linear, alpha=0.3)
plt.title('After Healthy Worker Effect')
plt.xlabel('A')
plt.ylabel('Y')
plt.show()

# Compare Y distributions
plt.figure(figsize=(12, 5))
sns.histplot(data_obs_linear['Y'], bins=50, color='blue', label='Original Y', stat='density', kde=True)
sns.histplot(data_healthy_worker_linear['Y'], bins=50, color='green', label='Healthy Worker Y', stat='density', kde=True)
plt.legend()
plt.title('Outcome Y Distribution Before and After Healthy Worker Effect')
plt.xlabel('Y')
plt.ylabel('Density')
plt.show()
```

![Before Healthy Worker Effect](https://i.imgur.com/BeforeHealthyWorker.png)
![After Healthy Worker Effect](https://i.imgur.com/AfterHealthyWorker.png)
![Y Distribution](https://i.imgur.com/YDistributionHW.png)

**Observation:**
- Employed individuals tend to have better health outcomes, potentially masking the true association between \( A \) and \( Y \).

### 9. Publication Bias

**Definition:** Studies with significant or positive findings are more likely to be published than those with null or negative results.

**Modification Steps:**
- Simulate multiple studies and select only those with statistically significant \( A-Y \) associations.

**Python Implementation:**

```python
def simulate_publication_bias(data, num_studies=1000, sample_size=100, significance_level=0.05):
    """
    Simulates publication bias by selecting only studies with significant A-Y associations.
    
    Parameters:
    - data: DataFrame, observational study data
    - num_studies: int, number of simulated studies
    - sample_size: int, size of each simulated study
    - significance_level: float, p-value threshold for significance
    
    Returns:
    - DataFrame combining all "published" studies
    """
    published_studies = []
    
    for _ in range(num_studies):
        # Sample with replacement
        sample = data.sample(n=sample_size, replace=True)
        
        # Fit linear regression: Y ~ A + X
        X_vars = ['A', 'X']
        X_design = sm.add_constant(sample[X_vars])
        model = sm.OLS(sample['Y'], X_design).fit()
        
        # Check if A is significant
        p_value_A = model.pvalues['A']
        if p_value_A < significance_level:
            published_studies.append(sample)
    
    if published_studies:
        data_publication_bias = pd.concat(published_studies, ignore_index=True)
    else:
        data_publication_bias = pd.DataFrame(columns=data.columns)
    
    return data_publication_bias

# Example Usage:
data_publication_bias_linear = simulate_publication_bias(data_obs_linear, num_studies=1000, sample_size=100, significance_level=0.05)
```

**Explanation:**

- **Simulation of Multiple Studies:** Each study samples from the OBS dataset and performs a linear regression of \( Y \) on \( A \) and \( X \).
- **Publication Criterion:** Only studies where the coefficient for \( A \) is statistically significant (p-value < 0.05) are "published."
- **Effect:** The published dataset is biased towards showing significant associations between \( A \) and \( Y \).

**Visualization:**

Compare the distribution of \( Y \) in all studies versus published studies.

```python
# Number of published studies
print(f"Number of Published Studies: {len(data_publication_bias_linear)} out of 1000")

# Compare Y distributions
plt.figure(figsize=(12, 5))
sns.histplot(data_obs_linear['Y'], bins=50, color='blue', label='All Studies', stat='density', kde=True)
sns.histplot(data_publication_bias_linear['Y'], bins=50, color='red', label='Published Studies', stat='density', kde=True)
plt.legend()
plt.title('Publication Bias on Outcome Y')
plt.xlabel('Y')
plt.ylabel('Density')
plt.show()
```

![Publication Bias](https://i.imgur.com/PublicationBias.png)

**Observation:**
- Published studies may show a shifted distribution of \( Y \) due to selection based on significance.

### 10. Loss to Follow-Up (Attrition Bias)

**Definition:** Participants drop out of a study in a non-random manner, potentially differing in exposure \( A \) and outcome \( Y \) compared to those who remain.

**Modification Steps:**
- Introduce a loss to follow-up indicator \( S \) influenced by \( A \) and \( Y \).
- Exclude participants based on \( S \), creating attrition bias.

**Python Implementation:**

```python
def apply_loss_to_follow_up(data, beta_S_A=-0.5, beta_S_Y=-0.7):
    """
    Applies Loss to Follow-Up (Attrition Bias) by excluding participants based on A and Y.
    
    Parameters:
    - data: DataFrame, observational study data
    - beta_S_A: float, coefficient for A in S
    - beta_S_Y: float, coefficient for Y in S
    
    Returns:
    - DataFrame after applying Loss to Follow-Up
    """
    # Define probability of being retained based on A and Y
    linear_S = beta_S_A * data['A'] + beta_S_Y * data['Y']
    prob_S = 1 / (1 + np.exp(-linear_S))
    
    # Assign S = 1 (retained) or S = 0 (lost)
    S = np.random.binomial(1, prob_S, len(data))
    data_attrition = data.copy()
    data_attrition['S'] = S
    
    # Retain only S = 1
    data_attrition = data_attrition[data_attrition['S'] == 1].reset_index(drop=True)
    
    return data_attrition

# Example Usage:
data_attrition_bias_linear = apply_loss_to_follow_up(data_obs_linear, beta_S_A=-0.5, beta_S_Y=-0.7)
```

**Explanation:**

- **Retention Probability \( S \):** Decreases with higher \( A \) and higher \( Y \) values, simulating that individuals with certain exposure and outcomes are more likely to drop out.
- **Effect:** The remaining sample may have a non-representative distribution of \( A \) and \( Y \), biasing the association.

**Visualization:**

Compare distributions before and after applying loss to follow-up.

```python
def plot_attrition_bias(original, attrited, outcome_surface):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(original['Y'], bins=50, color='blue', label='Original', ax=axes[0], kde=True, stat="density")
    sns.histplot(attrited['Y'], bins=50, color='orange', label='Retained', ax=axes[0], kde=True, stat="density")
    axes[0].set_title(f'Attrition Bias on Y - {outcome_surface}')
    axes[0].legend()
    
    sns.histplot(original['A'], bins=2, color='blue', label='Original', ax=axes[1], kde=False, stat="density")
    sns.histplot(attrited['A'], bins=2, color='orange', label='Retained', ax=axes[1], kde=False, stat="density")
    axes[1].set_title(f'Attrition Bias on A - {outcome_surface}')
    axes[1].legend()
    
    plt.show()

# Plot for Linear Outcome Surface with Attrition Bias
plot_attrition_bias(data_obs_linear, data_attrition_bias_linear, 'Linear')
```

![Attrition Bias](https://i.imgur.com/AttritionBias.png)

**Observation:**
- The retained sample has different distributions of \( A \) and \( Y \), potentially biasing the estimated association.

---

## Additional Biases

### a. Nonresponse Bias

**Definition:** High nonresponse rate to surveys/questionnaires causing errors if nonresponders differ in some way from responders.

**Relation to Existing Biases:**
- A specific form of **Selection Bias**, where the selection mechanism is based on nonparticipation.

**Modification Steps:**
- Introduce a nonresponse indicator \( R \) influenced by \( A \) and \( Y \).
- Exclude nonresponders \( (R = 0) \), creating a biased sample.

**Python Implementation:**

```python
def apply_nonresponse_bias(data, beta_R_A=0.5, beta_R_Y=0.7, intercept_R=-1.0):
    """
    Applies Nonresponse Bias by excluding nonresponders based on A and Y.
    
    Parameters:
    - data: DataFrame, observational study data
    - beta_R_A: float, coefficient for A in R
    - beta_R_Y: float, coefficient for Y in R
    - intercept_R: float, intercept for R
    
    Returns:
    - DataFrame after applying Nonresponse Bias
    """
    # Define probability of response based on A and Y
    linear_R = beta_R_A * data['A'] + beta_R_Y * data['Y'] + intercept_R
    prob_R = 1 / (1 + np.exp(-linear_R))
    
    # Assign R = 1 (responded) or R = 0 (nonresponded)
    R = np.random.binomial(1, prob_R, len(data))
    data_nonresponse = data.copy()
    data_nonresponse['R'] = R
    
    # Exclude nonresponders
    data_nonresponse = data_nonresponse[data_nonresponse['R'] == 1].reset_index(drop=True)
    
    return data_nonresponse

# Example Usage:
data_nonresponse_bias_linear = apply_nonresponse_bias(data_obs_linear, beta_R_A=0.5, beta_R_Y=0.7, intercept_R=-1.0)
```

**Visualization:**

Compare distributions of \( A \) and \( Y \) between responders and nonresponders.

```python
def plot_nonresponse_bias(original, nonresponse, outcome_surface):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(original['A'], bins=2, color='blue', label='Original', ax=axes[0], stat="density")
    sns.histplot(nonresponse['A'], bins=2, color='orange', label='Responders', ax=axes[0], stat="density")
    axes[0].set_title(f'Nonresponse Bias on A - {outcome_surface}')
    axes[0].legend()
    
    sns.histplot(original['Y'], bins=50, color='blue', label='Original', ax=axes[1], kde=True, stat="density")
    sns.histplot(nonresponse['Y'], bins=50, color='orange', label='Responders', ax=axes[1], kde=True, stat="density")
    axes[1].set_title(f'Nonresponse Bias on Y - {outcome_surface}')
    axes[1].legend()
    
    plt.show()

# Plot for Linear Outcome Surface with Nonresponse Bias
plot_nonresponse_bias(data_obs_linear, data_nonresponse_bias_linear, 'Linear')
```

![Nonresponse Bias](https://i.imgur.com/NonresponseBias.png)

**Observation:**
- Responders have different distributions of \( A \) and \( Y \) compared to the original sample.

### b. Neyman Bias (Prevalence-Incidence Bias)

**Definition:** Occurs in cross-sectional studies when individuals with either very short or very long durations of disease are underrepresented, leading to incorrect associations between exposure and disease.

**Relation to Existing Biases:**
- A form of **Selection Bias** related to the timing of disease occurrence and detection.

**Modification Steps:**
- Introduce a disease duration variable \( D \) influenced by \( X \) and \( A \).
- Exclude individuals with disease durations outside a specific window \( [D_{min}, D_{max}] \), leading to biased prevalence estimates.

**Python Implementation:**

```python
def apply_neyman_bias(data, D_min=1.0, D_max=5.0, beta_X=0.5, beta_A=0.7, lambda_0=0.1, lambda_A=0.05):
    """
    Applies Neyman Bias by excluding individuals with disease durations outside [D_min, D_max].
    
    Parameters:
    - data: DataFrame, observational study data
    - D_min: float, minimum disease duration to include
    - D_max: float, maximum disease duration to include
    - beta_X: float, coefficient for X in hazard rate
    - beta_A: float, coefficient for A in hazard rate
    - lambda_0: float, baseline hazard rate
    - lambda_A: float, hazard reduction if exposed
    
    Returns:
    - DataFrame after applying Neyman Bias
    """
    # Define hazard rate based on X and A
    hazard_rate = lambda_0 + lambda_A * data['A']
    
    # Simulate disease duration D using exponential distribution
    D = np.random.exponential(scale=1/hazard_rate)
    
    # Add D to data
    data_neyman = data.copy()
    data_neyman['D'] = D
    
    # Exclude individuals with D outside [D_min, D_max]
    data_neyman = data_neyman[(data_neyman['D'] >= D_min) & (data_neyman['D'] <= D_max)].reset_index(drop=True)
    
    return data_neyman

# Example Usage:
data_neyman_bias_linear = apply_neyman_bias(data_obs_linear, D_min=1.0, D_max=5.0)
```

**Visualization:**

Compare disease duration distributions before and after applying Neyman bias.

```python
def plot_neyman_bias(original, neyman, outcome_surface):
    plt.figure(figsize=(12, 6))
    sns.histplot(original['Y'], bins=50, color='blue', label='Original Y', stat='density', kde=True, alpha=0.5)
    sns.histplot(neyman['Y'], bins=50, color='green', label='Included Y', stat='density', kde=True, alpha=0.5)
    plt.legend()
    plt.title(f'Neyman Bias on Y - {outcome_surface}')
    plt.xlabel('Y')
    plt.ylabel('Density')
    plt.show()

# Plot for Linear Outcome Surface with Neyman Bias
plot_neyman_bias(data_obs_linear, data_neyman_bias_linear, 'Linear')
```

![Neyman Bias](https://i.imgur.com/NeymanBias.png)

**Observation:**
- The distribution of \( Y \) may be skewed due to exclusion of individuals with extreme disease durations.

### c. Surveillance (Detection) Bias

**Definition:** Arises when one group (e.g., exposed) is monitored more closely than another, leading to higher detection rates of the outcome \( Y \).

**Relation to Existing Biases:**
- A subtype of **Information Bias**, where differential measurement or detection of outcomes occurs between groups.

**Modification Steps:**
- Introduce a surveillance indicator \( S \) influenced by \( A \).
- Increase the probability of detecting \( Y \) among those with \( S = 1 \).

**Python Implementation:**

```python
def apply_surveillance_bias(data, detection_prob=0.9, miss_prob=0.1):
    """
    Applies Surveillance (Detection) Bias by increasing detection of Y in exposed group.
    
    Parameters:
    - data: DataFrame, observational study data
    - detection_prob: float, probability to detect Y if S=1
    - miss_prob: float, probability to miss Y if S=0
    
    Returns:
    - DataFrame with surveillance indicator and detected Y
    """
    # 1. Introduce Surveillance Indicator S influenced by A
    # Higher surveillance probability for A=1
    linear_S = 1.0 * data['A'] - 0.5  # Adjust coefficients to control surveillance rate
    prob_S = 1 / (1 + np.exp(-linear_S))
    S = np.random.binomial(1, prob_S, len(data))
    
    # 2. Modify Outcome Y based on Surveillance
    # If S=1, Y is detected with higher probability
    # Since Y is continuous, we'll adjust Y based on S
    # For simplicity, add a detection bonus if S=1
    Y_detected = data['Y'].copy()
    Y_detected += S * detection_prob  # Increase Y by detection_prob if S=1
    
    # Optionally, introduce some noise or miss probability
    Y_detected += (1 - S) * np.random.normal(0, miss_prob, len(data))
    
    # Combine into DataFrame
    data_surveillance = data.copy()
    data_surveillance['S'] = S
    data_surveillance['Y_detected'] = Y_detected
    
    return data_surveillance

# Example Usage:
data_surveillance_bias_linear = apply_surveillance_bias(data_obs_linear, detection_prob=5, miss_prob=0.1)
```

**Explanation:**

- **Surveillance Indicator \( S \):** Individuals with \( A = 1 \) have a higher probability of \( S = 1 \), indicating more intensive monitoring.
- **Outcome Detection:**
  - **If \( S = 1 \):** \( Y \) is artificially increased by `detection_prob` to simulate higher detection rates.
  - **If \( S = 0 \):** \( Y \) remains largely unaffected, with minor noise added.

**Visualization:**

Compare the distributions of \( Y \) before and after applying surveillance bias.

```python
def plot_surveillance_bias(original, surveillance, outcome_surface):
    plt.figure(figsize=(12, 6))
    sns.histplot(original['Y'], bins=50, color='blue', label='Original Y', stat='density', kde=True, alpha=0.5)
    sns.histplot(surveillance['Y_detected'], bins=50, color='red', label='Detected Y', stat='density', kde=True, alpha=0.5)
    plt.legend()
    plt.title(f'Surveillance Bias on Y - {outcome_surface}')
    plt.xlabel('Y')
    plt.ylabel('Density')
    plt.show()

# Plot for Linear Outcome Surface with Surveillance Bias
plot_surveillance_bias(data_obs_linear, data_surveillance_bias_linear, 'Linear')
```

![Surveillance Bias](https://i.imgur.com/SurveillanceBias.png)

**Observation:**
- The detected \( Y \) distribution is shifted due to increased monitoring in the exposed group, inflating the association between \( A \) and \( Y \).

---

## Summary and Next Steps

We've comprehensively updated the synthetic data generation processes for both RCTs and Observational Studies with the specified outcome surfaces. Additionally, we've implemented simulations for various biases, adjusting the code to handle the continuous outcome \( Y \).

### Recap of Implemented Biases:

1. **Confounding Bias:** Inherent in OBS DGP where \( X \) influences both \( A \) and \( Y \).
2. **Selection Bias:** Non-random selection based on \( A \), \( Y \), or both.
3. **Information Bias (Misclassification Bias):** Measurement errors in \( A \) or \( Y \).
4. **Collider Bias:** Conditioning on a collider \( C \) influenced by \( A \) and \( Y \).
5. **Reverse Causation (Reverse Bias):** \( Y \) influences \( A \) instead of \( A \) influencing \( Y \).
6. **Berkson’s Bias:** Selection based on a collider influenced by \( A \) and \( Y \).
7. **Immortal Time Bias:** Introducing an immortal period before treatment assignment.
8. **Healthy Worker Effect:** Restricting to employed individuals, introducing confounding.
9. **Publication Bias:** Selecting only studies with significant \( A-Y \) associations.
10. **Loss to Follow-Up (Attrition Bias):** Non-random dropout based on \(