import matplotlib.pyplot as plt
import numpy as np

# Generate data for the first normal distribution
mean1 = 0.843
std_dev1 = 0.019
x1 = np.linspace(-5, 5, 100)
y1 = (1/(std_dev1 * np.sqrt(2 * np.pi))) * np.exp(-(x1 - mean1)**2 / (2 * std_dev1**2))

# Generate data for the second normal distribution
mean2 = 0.929
std_dev2 = 0.004
x2 = np.linspace(-5, 5, 100)
y2 = (1/(std_dev2 * np.sqrt(2 * np.pi))) * np.exp(-(x2 - mean2)**2 / (2 * std_dev2**2))

# Create a plot
plt.figure(figsize=(8, 6))
plt.plot(x1, y1, label=f'N({mean1}, {std_dev1})', color='blue')
plt.plot(x2, y2, label=f'N({mean2}, {std_dev2})', color='orange')
plt.title('Normal Distributions')
plt.xlabel('X')
plt.ylabel('Density')
plt.legend()

# Show the plot
plt.show()
