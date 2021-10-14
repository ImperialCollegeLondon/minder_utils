import requests
import json
import pandas as pd
import io
from pathlib import Path
import importlib.resources as pkg_resources
import sys
from ..util import progress_spinner
from minder_utils.configurations import token_path


class Downloader:
    '''
    This class allows you to download and save the data from minder. Make sure that you 
    have internally saved your token before using this class (see the 
    ```Getting Started.ipynb``` guide).
    
    ``Example``
    
    
    ```
    from minder_utils.download import Downloader
    dl = Downloader()
    category_list = dl.get_category_names('activity')
    dl.export(categories = category_list, since= '2021-10-05', save_path='./data/')

    ```
    This would download all of the activity data from the 5th October 2021, and save it
    as a csv in the directory ```'./data/'```

    '''
    def __init__(self):
        self.url = 'https://research.minder.care/api/'
        self.params = {'Authorization': self.token(), 'Content-Type': 'application/json'}

    def get_info(self):
        '''
        This function returns the available datasets on minder in the form of a
        dictionary

        Returns
        ---------

        - _: dict: 
            This returns a dictionary of the available datasets.
        '''
        try:
            return requests.get(self.url + 'info/datasets', headers=self.params).json()
        except json.decoder.JSONDecodeError:
            print('Get response ', requests.get(self.url + 'info/datasets', headers=self.params))

    def _export_request(self, categories='all', since=None, until=None):
        '''
        This is an internal function that makes the request to download the data.

        Arguments
        ---------

        - categories: list or string: 
            If a list, this is the datasets that will be downloaded. Please use the
            dataset names that can be returned by using the get_category_names function.
            If the string 'all' is supplied, this function will return all of the data. This
            is not good! There should be a good reason to do this.

        - since: valid input to pd.to_datetime(.): 
            This is the date and time from which the data will be loaded.

        '''

        # print('Deleting Existing export request')
        # previously_requests = requests.get(self.url + 'export', headers=self.params).json()
        # for job in previously_requests:
        #     response = requests.delete(self.url + 'export/' + job['id'], headers=self.params)
        #     if response.status_code == 200:
        #         print('Job ID ', job['id'], 'is successfully deleted', response.text)
        #     else:
        #         print('Job ID ', job['id'], 'is NOT deleted. Response code ', response.status_code)
        print('Creating new export request')
        export_keys = {'datasets': {}}
        if since is not None:
            export_keys['since'] = self.convert_to_ISO(since)
        if until is not None:
            export_keys['until'] = self.convert_to_ISO(until)
        info = self.get_info()['Categories']
        for key in info:
            for category in info[key]:
                if category in categories or categories == 'all':
                    export_keys['datasets'][category] = {}
        print('Exporting the ', export_keys['datasets'])
        print('From ', since, 'to', until)
        schedule_job = requests.post(self.url + 'export', data=json.dumps(export_keys), headers=self.params)
        job_id = schedule_job.headers['Content-Location']
        response = requests.get(job_id, headers=self.params).json()
        waiting = True
        while waiting:
            
            if response['status'] == 202:
                response = requests.get(job_id, headers=self.params).json()
                # the following waits for x seconds and runs an animation in the 
                # mean time to make sure the user doesn't think the code is broken
                progress_spinner(30, 'Waiting for the sever to complete the job', new_line_after = False)
                
            elif response['status'] == 500:
                sys.stdout.write('\r')
                sys.stdout.write("Request failed")
                sys.stdout.flush()
                waiting = False
            else:
                sys.stdout.write('\n')
                print("Job is completed, start to download the data")
                waiting = False

    def export(self, since=None, until=None, reload=True, categories='all', save_path='./data/raw_data/'):
        '''
        This is a function that is able to download the data and save it as a csv in save_path.

        Note that ```categories``` refers to the datasets. If you want to get the categories
        for a given set of measurements (ie: activity, care, vital signs, etc) please use
        the method ```.get_category_names('measurement_name')```. Alternatively, if you want to view all of the
        available datasets, please use the method ```.get_category_names('all')```
        

        Arguments
        ---------

        - since: valid input to pd.to_datetime(.): 
            This is the date and time from which the data will be loaded.

        - reload: bool: 
            This value determines whether an export request should be sent. 
            In most cases, this should be true, unless you want to download
            the data from a previously run request.

        - categories: list or string: 
            If a list, this is the datasets that will be downloaded. Please use the
            dataset names that can be returned by using the get_category_names function.
            If the string 'all' is supplied, this function will return all of the data. This
            is not good! There should be a good reason to do this.

        - save_path: string: 
            This is the save path for the data that is downloaded from minder.

        '''
        p = Path(save_path)
        if not p.exists():
            print('Target directory does not exist, creating a new folder')
            p.mkdir()
        if reload:
            self._export_request(categories=categories, since=since, until=until)

        data = requests.get(self.url + 'export', headers=self.params).json()
        export_index = -1
        if not reload:
            if len(data) > 1:
                print('Multiple export requests exist, please choose one to download')
                for idx, job in enumerate(data):
                    print('Job {} '.format(idx).center(50, '='))
                    print('ID: ', job['id'])
                    print('Transaction Time', job['jobRecord']['transactionTime'])
                    print('Export sensors: ', end='')
                    for record in job['jobRecord']['output']:
                        print(record['type'], end=' ')
                    print('')
                export_index = int(input('Enter the index of the job ...'))
                while export_index not in range(len(data)):
                    print('Not a valid input')
                    export_index = int(input('Enter the index of the job ...'))
        print('Start to export job')
        for idx, record in enumerate(data[export_index]['jobRecord']['output']):
            print('Exporting {}/{}'.format(idx + 1, len(data[export_index]['jobRecord']['output'])).ljust(20, ' '),
                  str(record['type']).ljust(20, ' '), end=' ')
            content = requests.get(record['url'], headers=self.params)
            if content.status_code != 200:
                print('Fail, Response code {}'.format(content.status_code))
            else:
                pd.read_csv(io.StringIO(content.text)).to_csv(save_path + record['type'] + '.csv', mode='a',
                                                              header=not Path(save_path + record['type'] + '.csv').exists())
                print('Success')
    
    def get_category_names(self, measurement_name = 'all'):
        '''
        This function allows you to get the category names from a given measurement name.

        Arguments
        ---------

        - measurement_name: str: 
            This is the name of the measurement that you want to get the categories for.
            The default 'all' returns all the possible measurement names.

        Returns
        ---------

        - out: list of strings: 
            This is a list that contains the category names that can be used in the 
            export function.

        '''
        
        if measurement_name == 'all':
            out = self.get_info()['Categories'].values()

        else:
            out = self.get_info()['Categories'][measurement_name].keys()
        
        return list(out)

    def get_measurement_names(self):
        '''
        Please do not use this function. It is a legacy function and will be deleted in the 
        future.
        '''
        return self.get_group_names()

    def get_group_names(self):
        '''
        This function allows you to view the names of the sets of measurements
        that can be downloaded from minder.

        Returns
        ---------

        - out: list of strings: 
            This is a list that contains the names of the sets of measurements.

        '''
        
        out = self.get_info()['Categories'].keys()
        
        return list(out)

    @staticmethod
    def token():
        '''
        This function returns the current user token. This is the token that is saved in the 
        file token_real.json after running the token_save function in settings.

        Returns
        ---------
        
        - token: string: 
            This returns the token in the format that can be used in the api call.

        '''
        token_dir = token_path
        with open(token_dir) as json_file:
            api_keys = json.load(json_file)  
        #with open('./token_real.json', 'r') as f:
            #api_keys = json.loads(f.read())
        return api_keys['token']

    @staticmethod
    def convert_to_ISO(date):
        '''
        Converts the date to ISO.

        Arguments
        ---------
        
        - data: valid input to pd.to_datetime(.):
            This is the date that you want to convert.

        Returns
        ---------

        - out: date:
            This is the date converted to ISO.

        '''
        date = pd.to_datetime(date)
        return date.strftime('%Y-%m-%dT%H:%M:%S.000Z')


if __name__ == '__main__':
    downloader = Downloader()
    downloader.export(reload=True, save_path='../data/raw_data/', categories=['raw_activity_pir'])
