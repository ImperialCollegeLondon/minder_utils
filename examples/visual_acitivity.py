from minder_utils.visualisation import Visualisation_Activity
import os

os.chdir('..')

va = Visualisation_Activity()

# get label information
print(va.get_label_info())

# Visualise random patient in random day, without validation
va.raw_data()
va.aggregated_data()
va.normalised_data()

# Visualise random patient in random day, but validated as True
va.reset(valid=True)
va.raw_data()
va.aggregated_data()
va.normalised_data()

# or specify the patient id and the date, Note either could be str or list
va.reset(date=['2021-06-01', '2021-06-02'])
va.raw_data()
va.aggregated_data()
va.normalised_data()