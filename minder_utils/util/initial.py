from minder_utils.dataloader import Dataloader
from minder_utils.formatting import Formatting
from minder_utils.download import Downloader


def first_run(download=False):
    '''
    This function will help you to download ALL the data, then categorise,
    pre-process, and save the data. If you set the refresh as False in config_dri.yaml,
    you will not need to re-process the data again.

    NOTE:
        - The unlabelled data contains only DRI data, if you want to include the TIHM
          data as well, set date in Dataloader as None.
        - It currently takes a while to run, will optimise in future.
    Parameters
    ----------
    download bool, whether to download the data.

    Returns None, the data will be saved and you can load it by Dataloader(None)
    -------

    '''
    if download:
        Downloader().export()

    loader = Formatting()
    dataloader = Dataloader(loader.activity_data, max_days=10, label_data=True)
    # This will automatically process and save the data
    data = dataloader.labelled_data
    unlabelled_data = dataloader.unlabelled_data