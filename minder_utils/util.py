from pathlib import Path
import sys
import time
import os
import shutil


def save_mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def delete_dir(dirpath):
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        print('Deleting existing directory: ', dirpath)
        shutil.rmtree(dirpath)


rocket_base_string = '[---------]'
rocket_progress_indicator_list1 = [rocket_base_string[:n] + '>=>' + rocket_base_string[n:] for n in range(1,11)]
rocket_progress_indicator_list2 = [rocket_base_string[:n] + '<=<' + rocket_base_string[n:] for n in range(1,11)]
rocket_progress_indicator_list = rocket_progress_indicator_list1 + rocket_progress_indicator_list2[::-1]


progress_indicator_dict = {'spinning_wheel': ['\\', '|', '/', '-', '\\', '|', '/', '-'],
                           'rocket': rocket_progress_indicator_list}

def progress_spinner(total_time, statement, new_line_after = True, progress_indicator = 'rocket', time_period = 1):
    '''
    This function prints a spinning wheel for the length of time given in ```total_time```.
    This is useful when it is required to wait for some amount of time but the user wants
    to print something. For example, when waiting for a server to complete its job.

    Arguments
    ---------

    - total_time: int:
        This is the total number of seconds this function will animate for.
    
    - statement: string:
        This is the statement to print before the spinning wheel
    
    - new_line_after: bool:
        This dictates whether to print a new line when the progress bar is finished.
    
    - progress_indicator: string or list:
        If string, make sure it is either ```'spinning_wheel'``` or ```'rocket'```. If it is a 
        list, the values in the list will be the frames of the animation.
    
    - time_period: float:
        This is the time period for the animation.

    '''

    if type(progress_indicator) == str:
        progress_indicator_list = progress_indicator_dict[progress_indicator]

    else: progress_indicator_list = progress_indicator

    sleep_value = time_period/len(progress_indicator_list)
    
    progress_position = 0
    
    time_run = 0
    start = time.time()

    
    while time_run < total_time:
        progress_position = progress_position%len(progress_indicator_list)
        sys.stdout.write('\r')
        sys.stdout.write("{} {}".format(statement, progress_indicator_list[progress_position]))
        sys.stdout.flush()
        
        time.sleep(sleep_value)
        progress_position += 1
        
        time_run = time.time() - start
    
    if new_line_after: sys.stdout.write('\n')
    
    
    return