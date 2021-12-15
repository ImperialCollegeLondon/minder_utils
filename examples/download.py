'''
This script introduces how to download the data
'''
import os

os.chdir('..')

from minder_utils.download import Downloader

Downloader().export(since='2021-10-10', until='2021-10-12', reload=True,
                    save_path='./data/activity/', categories=['raw_activity_pir'])

Downloader().refresh(until='2021-12-10',
                    save_path='./data/activity/', categories=['raw_activity_pir'])

