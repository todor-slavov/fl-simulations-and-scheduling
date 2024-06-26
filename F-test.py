import scipy.stats as stats
import numpy as np

# Example time series data
T_a = [
    [0.9264, 0.9366, 0.9238, 0.93, 0.9291]
]

T_b = [
    [0.8306, 0.8168, 0.8689, 0.8607, 0.8393]
]

alpha = 0.05  # Significance level

for i in range(len(T_a)):
    var_a = np.var(T_a[i], ddof=1)  # Sample variance of T_a_i
    var_b = np.var(T_b[i], ddof=1)  # Sample variance of T_b_i

    print(var_a)
    print(var_b)

    F = var_b / var_a  # F-test statistic
    dfn = len(T_b[i]) - 1  # Degrees of freedom for T_b_i
    dfd = len(T_a[i]) - 1  # Degrees of freedom for T_a_i

    # Critical value for the F-distribution
    F_critical = stats.f.ppf(1 - alpha, dfn, dfd)

    print(f"Time Series Pair {i+1}:")
    print(f"F-test statistic: {F}")
    print(f"Critical value: {F_critical}")
    
    if F < F_critical:
        print("Fail to reject the null hypothesis: Variance of T_a_i is not significantly less than variance of T_b_i.")
    else:
        print("Reject the null hypothesis: Variance of T_a_i is significantly less than variance of T_b_i.")

    print()
