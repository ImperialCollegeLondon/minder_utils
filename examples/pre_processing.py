import os

os.chdir('..')

from minder_utils.formatting import Formatting
from minder_utils.dataloader import Dataloader

formater = Formatting()
# First time
Dataloader(formater.activity_data, formater.physiological_data,
           formater.environmental_data,
           max_days=7, label_data=True)

# load the saved data
data = Dataloader(None).labelled_data

# Visualise the label
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

label = data['uti_labels'][0]
plt.bar(range(len(label)), label)
plt.ylim(-1, 1)
plt.xlabel('Day')
plt.ylabel('Probability')

ticks = ['Validated Date']
for i in range(0, len(label) - 1, 2):
    ticks.append('%d Day Before' % ((i + 3) // 2))
    ticks.append('%d Day After' % ((i + 3) // 2))
plt.xticks(range(len(label)), ticks, rotation=90)
plt.tight_layout()
plt.savefig('label.png')
