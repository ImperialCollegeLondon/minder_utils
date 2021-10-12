import requests
import json
import pandas as pd
import io
from pathlib import Path
import time


class Downloader:
    def __init__(self):
        self.url = 'https://research.minder.care/api/'
        self.params = {'Authorization': self.token(), 'Content-Type': 'application/json'}

    def get_info(self):
        try:
            return requests.get(self.url + 'info/datasets', headers=self.params).json()
        except json.decoder.JSONDecodeError:
            print('Get response ', requests.get(self.url + 'info/datasets', headers=self.params))

    def _export_request(self, categories='all', since=None):
        print('Deleting Existing export request')
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
            since = pd.to_datetime(since)
            export_keys['since'] = since.strftime('%Y-%m-%dT%H:%M:%S.000Z')
        info = self.get_info()['Categories']
        for key in info:
            for category in info[key]:
                if category in categories or categories == 'all':
                    export_keys['datasets'][category] = {}
        export_keys['datasets']['device_types'] = {}
        print('Exporting the ', export_keys['datasets'])
        schedule_job = requests.post(self.url + 'export', data=json.dumps(export_keys), headers=self.params)
        job_id = schedule_job.headers['Content-Location']
        response = requests.get(job_id, headers=self.params).json()
        waiting = True
        while waiting:
            if response['status'] == 202:
                print("Waiting the server to complete the job ...")
                response = requests.get(job_id, headers=self.params).json()
                time.sleep(5)
            elif response['status'] == 500:
                print("Request is failed, please try again ...")
            else:
                print("Job is completed, start to download the data")
                waiting = False

    def export(self, since=None, reload=True, categories='all', save_path='./data/raw_data/'):
        p = Path(save_path)
        if not p.exists():
            print('Target directory does not exist, creating a new folder')
            p.mkdir()
        if reload:
            self._export_request(categories, since)

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
                if export_index not in range(len(data)):
                    print('Not a valid input')
                    export_index = int(input('Enter the index of the job ...'))
        print('Start to export job', data[export_index]['id'])
        for idx, record in enumerate(data[export_index]['jobRecord']['output']):
            print('Exporting {}/{}'.format(idx + 1, len(data[export_index]['jobRecord']['output'])).ljust(20, ' '),
                  str(record['type']).ljust(20, ' '), end=' ')
            content = requests.get(record['url'], headers=self.params)
            if content.status_code != 200:
                print('Fail, Response code {}'.format(content.status_code))
            else:
                pd.read_csv(io.StringIO(content.text)).to_csv(save_path + record['type'] + '.csv')
                print('Success')

    @staticmethod
    def token():
        with open('./download/token.json', 'r') as f:
            api_keys = json.loads(f.read())
        return api_keys['token']


if __name__ == '__main__':
    downloader = Downloader()
    downloader.export(reload=True, save_path='../data/raw_data/', categories=['raw_activity_pir'])
