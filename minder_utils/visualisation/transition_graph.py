from minder_utils.formatting import Formatting
from minder_utils.feature_engineering.calculation import build_p_matrix
import os

os.chdir('../../')

from minder_utils.formatting import Formatting
import seaborn as sns
import matplotlib.pyplot as plt

formater = Formatting()
df = formater.activity_data[formater.activity_data.id == '']
matrix, events = build_p_matrix(df.location, True)