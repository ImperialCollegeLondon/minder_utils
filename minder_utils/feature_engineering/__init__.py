
from minder_utils.feature_engineering.engineer import Feature_engineer


__all__ = ['Feature_engineer']



'''
TODO:
    1. Feature Engineering
    2. Evaluate the feature importance (raw data, extra features)
    3. Evaluate on classifiers
'''
'''
Based on https://www.nhs.uk/conditions/urinary-tract-infections-utis/

NOTE:
    - Every feature should be compared to last n weeks.
    - The comparison / difference measurement should be encapsulated

1. needing to pee more often than usual during the night (nocturia)
    - Compare the frequency of bathroom (Night)

2. needing to pee suddenly or more urgently than usual
    - Compare the time difference between the triggered last sensor and bathroom sensor

3. needing to pee more often than usual
    - Compare the frequency of bathroom (Daytime)

4. a high temperature, or feeling hot and shivery
    - Body temperature (High)

5. a very low temperature below 36C
    - Body temperature (Low)
'''