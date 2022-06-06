import requests
import json
import pandas as pd
import io
from pathlib import Path
import sys
import os
from minder_utils.util.util import progress_spinner, reformat_path, save_mkdir, please_dont_fail
from minder_utils.configurations import token_path
import numpy as np
from datetime import date, datetime
import time
import contextlib


import http.client
import logging
from logging import FileHandler

def setup_logging(loglevel):

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y--%H-%M-%S")

    # the file handler receives all messages from level DEBUG on up, regardless
    logging_path = os.path.join(os.path.dirname(__file__), '..', '..', 'logs', '{}_logging.txt'.format(dt_string))
    fileHandler = FileHandler(logging_path)
    fileHandler.setLevel(logging.DEBUG)
    handlers = [fileHandler]

    if loglevel is not None:
        # if a log level is configured, use that for logging to the console
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(loglevel)
        handlers.append(stream_handler)

    if loglevel == logging.DEBUG:
        # when logging at debug level, make http.client extra chatty too
        # http.client *uses `print()` calls*, not logging.
        http.client.HTTPConnection.debuglevel = 1

    # finally, configure the root logger with our choice of handlers
    # the logging level of the root set to DEBUG (defaults to WARNING otherwise).
    logformat = "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
    logging.basicConfig(
        filename=logging_path,
        format=logformat, datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG, #handlers=handlers, 
    )
    logging.getLogger("urllib3").propagate = True


http_client_logger = logging.getLogger("http.client")

def print_to_log(*args):
    http_client_logger.debug(" ".join(args)) 

def do_nothing(*args, **kwargs):
    return

# monkey-patch a `print` global into the http.client module; all calls to
# print() in that module will then use our print_to_log implementation
http.client.print = print_to_log
setup_logging(logging.DEBUG)



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
        print('Sending Request...')
        logging.debug('getting info')

        reponse_func = please_dont_fail(requests.get, tries=3)
        r = reponse_func(self.url + 'info/datasets', headers=self.params)
        if r.status_code in [401, 403]:
            raise TypeError('Authentication failed!'\
                ' Please check your token - it might be out of date. '\
                'You might also not have authorization to complete your request.')
        try:
            return r.json()
        except json.decoder.JSONDecodeError:
            print('Get response ', r)
        logging.debug('info done')

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
        logging.debug('making requests')
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
        reponse_func = please_dont_fail(requests.get, tries=3)
        response = reponse_func(job_id, headers=self.params)
        if response.status_code == 401:
            raise TypeError('Authentication failed!' \
                            ' Please check your token - it might be out of date.')
        response = response.json()
        waiting = True
        while waiting:
            logging.debug('checking status')

            if response['status'] == 202:
                reponse_func = please_dont_fail(requests.get, tries=3)
                response = reponse_func(job_id, headers=self.params).json()
                # the following waits for x seconds and runs an animation in the 
                # mean time to make sure the user doesn't think the code is broken
                progress_spinner(30, 'Waiting for the sever to complete the job', new_line_after=False)

            elif response['status'] == 500:
                sys.stdout.write('\r')
                sys.stdout.write("Request failed")
                sys.stdout.flush()
                waiting = False
            else:
                sys.stdout.write('\n')
                print("Job is completed, start to download the data")
                waiting = False
        logging.debug('no more waiting for the server')

    def _export_request_parallel(self, export_dict):
        '''
        This function allows the user to make parallel export requests. This is useful 
        when the requests have difference since and until dates for the different datasets in 
        the categories.
        
        Arguments
        ---------
        - export_dict: dictionary:
            This dictionary contains the categories to be downloaded as keys, with the since
            and until as values in a tuple.
            For example:
            ```
            { category         : (since                       , until),
             'raw_activity_pir': (pd.to_datetime('2021-10-06'), pd.to_datetime('2021-10-10')),
             'raw_door_sensor' : (pd.to_datetime('2021-10-06'), pd.to_datetime('2021-10-10'))}
            ```

        '''
        categories_list = list(export_dict.keys())

        available_categories_list = self.get_category_names(measurement_name='all')

        for category in categories_list:
            if not category in available_categories_list:
                raise TypeError('Category {} is not available to download. Please check the name.'.format(category))

        print('Creating new parallel export requests')
        logging.debug('creating new parallel export requests')

        # the following creates a list of export keys to be called by the API
        export_key_list = {}
        for category in categories_list:
            since = export_dict[category][0]
            until = export_dict[category][1]
            export_keys = {'datasets': {category: {}}}
            if not since is None:
                export_keys['since'] = self.convert_to_ISO(since)
            if not until is None:
                export_keys['until'] = self.convert_to_ISO(until)

            export_key_list[category] = export_keys

        # scheduling jobs for each of the requests:
        request_url_dict = {}
        schedule_job_dict = {}
        for category in categories_list:
            export_keys = export_key_list[category]
            schedule_job = requests.post(self.url + 'export', data=json.dumps(export_keys), headers=self.params)
            schedule_job_dict[category] = schedule_job
            request_url = schedule_job.headers['Content-Location']
            request_url_dict[category] = request_url

        # checking whether the jobs have been completed:
        waiting = True
        waiting_for = {category: True for category in categories_list}
        job_id_dict = {}
        while waiting:
            logging.debug('checking status')
            for category in categories_list:
                if not waiting_for[category]:
                    continue

                request_url = request_url_dict[category]
                reponse_func = please_dont_fail(requests.get, tries=3)
                response = reponse_func(request_url, headers=self.params)
                if response.status_code in [401, 403]:
                    raise TypeError('Authentication failed!'\
                                ' Please check your token - it might be out of date. '\
                                'You might also not have authorization to complete your request.')
                
                elif response.status_code == 202:
                    waiting_for[category] = True

                elif response.status_code == 500:
                    sys.stdout.write('\r')
                    sys.stdout.write("Request failed for category {}".format(category))
                    sys.stdout.write('\n')
                    sys.stdout.flush()
                    waiting_for[category] = False
                
                else:
                    waiting_for[category] = False

                if not category in job_id_dict:
                    response = response.json()
                    job_id_dict[category] = response['id']


            # if we are no longer waiting for a job to complete, move onto the downloads
            if True in list(waiting_for.values()):
                logging.debug('waiting for server')
                progress_spinner(30, 'Waiting for the sever to complete the job', new_line_after=False)
            else:
                sys.stdout.write('\n')
                sys.stdout.write("The server has finished processing the requests")
                sys.stdout.flush()
                sys.stdout.write('\n')
                waiting = False
        logging.debug('finished waiting for the server')

        return job_id_dict, request_url_dict

    def export(self, since=None, until=None, reload=True,
               categories='all', save_path='./data/raw_data/', append=True, export_index=None,
               save_index=True, return_categories_downloaded=False,
               remove_id=False):
        '''
        This is a function that is able to download the data and save it as a csv in save_path.

        Note that ```categories``` refers to the datasets. If you want to get the categories
        for a given set of measurements (ie: activity, care, vital signs, etc) please use
        the method ```.get_category_names('measurement_name')```. Alternatively, if you want to view all of the
        available datasets, please use the method ```.get_category_names('all')```

        If the data files already exist, the new data will be appended to the end. Be careful, this can cause 
        duplicates! To avoid this, use the ```.refresh()``` function or use ```append = False```     

        Arguments
        ---------

        - since: valid input to pd.to_datetime(.): 
            This is the date and time from which the data will be loaded. If ```None```,
            the earliest possible date is used.
            Default: ```None```

        - until: valid input to pd.to_datetime(.): 
            This is the date and time to which the data will be loaded up until. If ```None```,
            the latest possible date is used.
            Default: ```None```

        - reload: bool: 
            This value determines whether an export request should be sent. 
            In most cases, this should be ```True```, unless you want to download
            the data from a previously run request.
            Default: ```True```

        - categories: list or string: 
            If a list, this is the datasets that will be downloaded. Please use the
            dataset names that can be returned by using the get_category_names function.
            If the string 'all' is supplied, this function will return all of the data. This
            is not good! There should be a good reason to do this.
            Default: ```'all'```

        - save_path: string: 
            This is the save path for the data that is downloaded from minder.
            Default: ```'./data/raw_data/'```

        - append: bool:
            If ```True```, the downloaded data will be appended to the previous data, if it exists.
            If ```False```, the previous data will be overwritten if it exists.

        - export_index: integer:
            You may use this argument to download a previous request. ```-1``` will download
            the most recent request. This argument will over rule the ```reload``` argument.
            Defaults to ```None```.

        - save_index: bool:
            Whether to save the index in the csv file.

        '''
        save_path = reformat_path(save_path)
        p = Path(save_path)
        if not p.exists():
            print('Target directory does not exist, creating a new folder')
            logging.debug('target directory does not exist, creating a new folder')
            save_mkdir(save_path)
        if export_index is None:
            if reload:
                self._export_request(categories=categories, since=since, until=until)
        reponse_func = please_dont_fail(requests.get, tries=3)
        data = reponse_func(self.url + 'export', headers=self.params).json()
        export_index = -1 if export_index is None else export_index
        if export_index is None:
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
        logging.debug('start to export job')
        logging.debug('downloading the data')
        categories_downloaded = []
        for idx, record in enumerate(data[export_index]['jobRecord']['output']):
            print('Exporting {}/{}'.format(idx + 1, len(data[export_index]['jobRecord']['output'])).ljust(20, ' '),
                  str(record['type']).ljust(20, ' '), end=' ')
            reponse_func = please_dont_fail(requests.get, tries=3)
            content = reponse_func(record['url'], headers=self.params)
            if content.status_code != 200:
                print('Fail, Response code {}'.format(content.status_code))
            else:
                if record['type'] in categories_downloaded:
                    mode = 'a'
                    header = False
                else:
                    mode = 'a' if append else 'w'
                    header = not Path(os.path.join(save_path, record['type'] + '.csv')).exists() or mode == 'w'

                csv_content = pd.read_csv(io.StringIO(content.text), index_col=False)
                
                if remove_id:
                    if record['type'] not in ['homes', 'device_types', 'patients']:
                        try:
                            csv_content = csv_content.drop(['id'], axis=1)
                        except KeyError:
                            pass
                
                csv_content.to_csv(os.path.join(save_path, record['type'] + '.csv'),
                                   mode=mode,
                                   header=header, index=save_index)
                categories_downloaded.append(record['type'])
                print('Success')
        logging.debug('done with export')

        if return_categories_downloaded:
            return categories_downloaded
        else:
            return



    def refresh_at_once(self, since, categories=None, until=None,
                        save_path='./data/raw_data/', 
                        export_index=None):
        '''
        This function allows for the user to refresh the data currently saved in the 
        save path. This differs from ```.refresh()``` in that a ```since``` argument
        must be used. This function will download the data between ```until``` and 
        ```since``` and then de-duplicate the files.

        Arguments
        ---------

        - since: valid input to pd.to_datetime(.): 
            This is the date and time from which the data will be loaded. If ```None```,
            the earliest possible date is used.
            Default: ```None```

         - until: valid input to pd.to_datetime(.): 
            This is the date and time to which the data will be loaded up until. If ```None```,
            the latest possible date is used.
            Default: ```None```

        - categories: list or string: 
            If a list, this is the datasets that will be downloaded. Please use the
            dataset names that can be returned by using the get_category_names function.
            If a string is given, only this dataset will be refreshed.

        - save_path: string: 
            This is the save path for the data that is downloaded from minder.
            Default: ```'./data/raw_data/'```
        
        - export_index: integer:
            You may use this argument to download a previous request. ```-1``` will download
            the most recent request. This argument will over rule the ```reload``` argument.
            Defaults to ```None```.

        '''

        save_path = reformat_path(save_path)

        if type(categories) == str:
            if categories == 'all':
                print('You have selected ALL categories... are you sure?')
                progress_spinner(30, 'You have 30 seconds to cancel', new_line_after=True)
                categories = self.get_category_names(categories)
            else:
                categories = [categories]

        print('Checking the index saving method.')
        cols = pd.read_csv(save_path+categories[0]+'.csv', index_col=False, header=0, nrows=0).columns.tolist()
        if np.any(['Unnamed:' in col for col in cols]):
            index_col=0
            save_index=True
        else:
            index_col=False
            save_index=False
        logging.debug('index_col={}, save_index={}'.format(index_col,save_index))
        logging.debug('Getting data for categories {}'.format(categories))
        categories_downloaded = self.export(since=since, until=until, reload=True,
                                                categories=categories, save_path=save_path, 
                                                append=True, export_index=export_index,
                                                return_categories_downloaded=True,
                                                save_index=save_index)

        print('Removing possible duplicates...')
        logging.debug('removing possible duplicates...')

        for cat in categories_downloaded:
            logging.debug('loading {}'.format(cat))
            data = pd.read_csv(save_path + cat + '.csv', index_col=index_col)
            logging.debug('de-duplicating {}'.format(cat))
            data = data.drop_duplicates().reset_index(drop=True)
            logging.debug('saving {}'.format(cat))
            data.to_csv(save_path + cat + '.csv', index=save_index)
        print('Done')

        return





    def refresh(self, until=None, categories=None, save_path='./data/raw_data/'):
        '''
        This function allows for the user to refresh the data currently saved in the 
        save path. It will download the data missing between the saved files and the
        ```until``` argument.

        Arguments
        ---------

         - until: valid input to pd.to_datetime(.): 
            This is the date and time to which the data will be loaded up until. If ```None```,
            the latest possible date is used.
            Default: ```None```

        - categories: list or string: 
            If a list, this is the datasets that will be downloaded. Please use the
            dataset names that can be returned by using the get_category_names function.
            If a string is given, only this dataset will be refreshed.

        - save_path: string: 
            This is the save path for the data that is downloaded from minder.
            Default: ```'./data/raw_data/'```
        

        '''
        if until is None:
            until = datetime.now()
        save_path = reformat_path(save_path)
        if categories is None:
            raise TypeError('Please supply at least one category...')
        if type(categories) == str:
            if categories == 'all':
                categories = self.get_category_names('all')
            else:
                categories = [categories]

        export_dict = {}
        mode_dict = {}
        print('Checking current files...')
        logging.debug('checking dates from current files')
        last_rows = {}
        for category in categories:
            file_path = os.path.join(save_path, category)
            p = Path(file_path + '.csv')
            if not p.exists():
                since = None
            else:
                data = pd.read_csv(file_path + '.csv')
                if 'start_date' in data.columns:
                    # add the following to avoid a duplicate of the last and first row
                    last_rows[category] = data[['start_date']].iloc[-1, :].to_numpy()
                    since = pd.to_datetime(data['start_date'].loc[data['start_date'].last_valid_index()])
                    if self.convert_to_ISO(since) > self.convert_to_ISO(until):
                        # change since to earliest date and overwrite all data for this category
                        since = pd.to_datetime(data[['start_date']].iloc[0, 0])
                        # if the earliest date is after until, then we error
                        if self.convert_to_ISO(since) > self.convert_to_ISO(until):
                            raise TypeError('Please check your inputs. For {} we found that you tried refreshing' \
                                            'to a date earlier than the earliest date in the file.'.format(category))
                        else:
                            mode_dict[category] = 'w'
                    else:
                        mode_dict[category] = 'a'
                
                else:
                    since=None
                    mode_dict[category] = 'w'

            export_dict[category] = (since, until)

        job_id_dict, request_url_dict = self._export_request_parallel(export_dict=export_dict)

        logging.debug('downloading the data')

        reponse_func = please_dont_fail(requests.get, tries=3)
        data = reponse_func(self.url + 'export', headers=self.params).json()
        for category in categories:

            if not category in request_url_dict:
                raise TypeError('Uh-oh! Something seems to have gone wrong.' \
                                'Please check the inputs to the function and try again.' \
                                ' Looks as if category {} caused the problem'.format(category))
            reponse_func = please_dont_fail(requests.get, tries=3)
            content = reponse_func(request_url_dict[category], headers=self.params)
            output = json.load(io.StringIO(content.text))['jobRecord']['output']

            for n_output, data_chunk in enumerate(output):
                reponse_func = please_dont_fail(requests.get, tries=3)
                content = reponse_func(data_chunk['url'], headers=self.params)
                sys.stdout.write('\r')
                sys.stdout.write("For {}, exporting {}/{}".format(category, n_output + 1, len(output)))
                sys.stdout.flush()
                if content.status_code != 200:
                    sys.stdout.write('\n')
                    sys.stdout.write('\r')
                    sys.stdout.write('Fail, Response code {} for category {}'.format(content.status_code, category))
                    sys.stdout.write('\n')
                    sys.stdout.flush()
                else:
                    current_data = pd.read_csv(io.StringIO(content.text))

                    if Path(save_path + category + '.csv').exists():
                        data_to_save = pd.read_csv(save_path + category + '.csv', index_col=0)
                        data_to_save = pd.concat([data_to_save, current_data], ignore_index=True)
                        #data_to_save = data_to_save.append(current_data, ignore_index=True)
                        data_to_save = data_to_save.drop_duplicates(ignore_index=True).reset_index(drop=True)

                    else:
                        data_to_save = current_data

                    '''
                    header = (not Path(save_path + category + '.csv').exists()) or mode_dict[category] == 'w'
                    # checking whether the first line is a duplicate of the end of the previous file
                    if np.all(current_data[['start_date', 'id']].iloc[0, :] == last_rows[category]):
                        current_data.iloc[1:, :].reset_index(drop=True).to_csv(save_path + category + '.csv',
                                                                               mode=mode_dict[category],
                                                                               header=header)
                    else:
                        current_data.to_csv(save_path + category + '.csv', mode=mode_dict[category],
                                            header=header)
                    '''

                    data_to_save.to_csv(save_path + category + '.csv', mode='w',
                                        header=True)

            sys.stdout.write('\n')

        print('Success')

        return

    def get_category_names(self, measurement_name='all'):
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
            out = []
            for value in self.get_info()['Categories'].values():
                out.extend(list(value.keys()))

        else:
            out = list(self.get_info()['Categories'][measurement_name].keys())

        return out

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
            # with open('./token_real.json', 'r') as f:
            # api_keys = json.loads(f.read())
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
        date = pd.to_datetime(date, dayfirst=True)
        return date.strftime('%Y-%m-%dT%H:%M:%S.000Z')


if __name__ == '__main__':
    downloader = Downloader()
    downloader.export(reload=True, save_path='../data/raw_data/', categories=['raw_activity_pir'])
