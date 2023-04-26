import numpy as np
import matplotlib.pyplot as plt

mean, variance, nSamples = 0, 1, 1000

samples = np.random.normal(mean, variance, nSamples)

plt.hist(samples, bins=50, density=True, alpha=0.6, color='b')

x = np.linspace(mean - 4*variance, mean + 4*variance, 100)
pdf = 1/(variance*np.sqrt(2*np.pi))*np.exp(-(x-mean)**2/(2*variance**2))
plt.plot(x, pdf, 'r', linewidth=2)

plt.title('Normal distribution')
plt.xlabel('Value')
plt.ylabel('Probability density')

plt.show()