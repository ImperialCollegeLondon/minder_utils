import os

logs_path = os.path.join(os.path.dirname(__file__), '..', 'logs')

try: 
    os.mkdir(logs_path) 
    print('Creating logs directory')
except OSError as error: 
    print('Logs directory exists')