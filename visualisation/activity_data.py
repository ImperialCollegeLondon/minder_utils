import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

X = np.load('../data/weekly_test/previous/npy/X.npy')
sensors = ['WC1', 'back door', 'bathroom1', 'bedroom1', 'cellar',
           'conservatory', 'dining room', 'fridge door', 'front door',
           'hallway', 'iron', 'kettle', 'kitchen', 'living room',
           'lounge', 'main door', 'microwave', 'multi', 'office']
sns.heatmap(X[0].reshape(24, -1))
plt.ylabel('Time')
plt.xlabel('Sensor')
plt.xticks(np.arange(19), sensors, rotation=90)
plt.tight_layout()
plt.savefig('figures/activity_data.png')
plt.show()
