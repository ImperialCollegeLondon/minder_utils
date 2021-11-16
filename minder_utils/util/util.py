from pathlib import Path
import sys
import time
import os
import shutil
import pickle
import posixpath
import ntpath
import platform


def save_file(obj, file_path, file_name):
    with open(os.path.join(file_path, file_name + '.pickle'), 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_file(file_path, file_name):
    with open(os.path.join(file_path, file_name + '.pickle'), 'rb') as handle:
        data = pickle.load(handle)
    return data


def reformat_path(path):
    if isinstance(path, list):
        path = os.path.join(*path)
    if 'mac' in platform.platform().lower():
        return path
    elif 'windows' in platform.platform().lower():
        return path.replace(os.sep, ntpath.sep)
    elif 'unix' in platform.platform().lower():
        return path.replace(os.sep, posixpath.sep)


def save_mkdir(path):
    Path(reformat_path(path)).mkdir(parents=True, exist_ok=True)


def delete_dir(dirpath):
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        print('Deleting existing directory: ', dirpath)
        shutil.rmtree(dirpath)


rocket_base_string = '[---------]'
rocket_progress_indicator_list1 = [rocket_base_string[:n] + '>=>' + rocket_base_string[n:] for n in range(1, 11)]
rocket_progress_indicator_list2 = [rocket_base_string[:n] + '<=<' + rocket_base_string[n:] for n in range(1, 11)]
rocket_progress_indicator_list = rocket_progress_indicator_list1 + rocket_progress_indicator_list2[::-1]

progress_indicator_dict = {'spinning_wheel': ['\\', '|', '/', '-', '\\', '|', '/', '-'],
                           'rocket': rocket_progress_indicator_list}


def progress_spinner(total_time, statement, new_line_after=True, progress_indicator='rocket', time_period=1):
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

    else:
        progress_indicator_list = progress_indicator

    sleep_value = time_period / len(progress_indicator_list)

    progress_position = 0

    time_run = 0
    start = time.time()

    while time_run < total_time:
        progress_position = progress_position % len(progress_indicator_list)
        sys.stdout.write('\r')
        sys.stdout.write("{} {}".format(statement, progress_indicator_list[progress_position]))
        sys.stdout.flush()

        time.sleep(sleep_value)
        progress_position += 1

        time_run = time.time() - start

    if new_line_after: sys.stdout.write('\n')

    return


class PBar:
    '''
    This is a class for a simple progress bar.
    '''

    def __init__(self, show_length, n_iterations, done_symbol='#', todo_symbol='-'):
        '''
        Arguments
        ---------
            show_length: integer
                This is the length of the progress bar to show.

            n_iterations: integer
                This is the number of iterations the progress bar should expect.

            done_symbol: string
                This is the symbol shown for the finished section of the progress bar.
                By default this is a '#'.
            todo_symbol = string
                This is the symbol shown for the unfinished section of the progress bar.
                By default this is '-'

        '''

        self.show_length = show_length
        self.n_iterations = n_iterations
        self.done_symbol = done_symbol
        self.todo_symbol = todo_symbol
        self.progress = 0

        return

    def update(self, n=1):
        '''
        This is the update function for the progress bar

        Arguments
        ---------
            n: integer
                This is extra progress made

        '''

        self.progress += n

        return

    def give(self):
        '''
        Returns
        ---------
            out: string
                This returns a string of the progress bar.
        '''

        total_bar_length = self.show_length
        current_progress = self.progress if self.progress <= self.n_iterations else self.n_iterations
        n_iterations = self.n_iterations
        hashes_length = int((current_progress) / n_iterations * total_bar_length)
        hashes = self.done_symbol * hashes_length
        dashes = self.todo_symbol * (total_bar_length - hashes_length)

        out = '[{}{}]'.format(hashes, dashes)

        return out

    def show(self):
        '''
        This prints the progress bar.
        '''

        p_bar_to_print = self.give()

        print(p_bar_to_print)

        return
