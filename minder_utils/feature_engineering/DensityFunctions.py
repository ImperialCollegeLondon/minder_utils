import numpy as np
import sys
from minder_utils.util.util import PBar


class BaseDensityCalc:
    '''
    This class allows for the calculations of densities based on previous data.
    For example, it can calculate the probability of a sample having a value,
    given a baseline dataset.
    
    Arguments
    ---------
    
    - save_baseline_array: bool:
        This dictates whether the whole of the baseline array will be saved
        for use in calculating the densities in the future.
        
    - sample: bool:
        This dictates whether a sample of the values should be used to fit the
        model instead of the entire dataset. This is recommended for large datasets.
    
    - sample_size: integer:
        This is the size of the sample to be taken from the array passed in the fit function.
        This will only have an effect if ```sample=True```.
    
    - seed: integer:
        This seed is used within the random operations of this function. Setting the ```seed```
        will make the random operations deterministic.
    
    - verbose: bool:
        This dictates whether the class will print information related to its progress.
    
    '''

    def __init__(self, save_baseline_array=True, sample=False, sample_size=10000, seed=None, verbose=True):

        self.is_fitted = False
        self.save_baseline_array = save_baseline_array
        self.methods_possible = ['reverse_percentiles']
        self.batch_size = -1
        self.sample = sample
        self.sample_size = sample_size
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose

        return

    def fit(self, values):
        '''
        Arguments
        ---------
        
        - values: array:
            This array contains the delta values that will be used in the calculations.
            Please ensure that this is an array of shape (N,L) where N is the number 
            of samples and L is the number of features.
        
        
        '''

        if values.shape[0] < self.sample_size:
            self.sample = False
            if self.verbose:
                print('Sample will not be used, since the sample size given is larger than the dataset.')

        if self.save_baseline_array:

            if len(values.shape) == 1:
                raise TypeError(
                    'Please ensure that values is of shape (N,L) where N is the number of samples and L is the number of features.')

            if self.sample:
                sample_index = self.rng.integers(values.shape[0], size=self.sample_size)
                self._baseline_values = values[sample_index]
            else:
                self._baseline_values = values


        else:
            raise TypeError('save_baseline_array = False is not currently supported.')

        self.is_fitted = True

        if values.shape[0] * values.shape[1] > 10000:
            if self.sample:
                self.batch_size = 10000
            else:
                self.batch_size = 200

        return self

    def _calculate_reverse_percentiles(self, values, kind='rank'):
        '''
        This function calculates the reverse percentiles from an array of values,
        given the baseline_values provided in the ```.fit()``` function.
        

        This function is an edit of the source code of 
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.percentileofscore.html#scipy.stats.percentileofscore
        
        My edits allow for the calculation percentiles on arrays.
        
        Arguments
        ---------
        
        - values: array:
            This is an array containing the values for which to calculate the reverse percentile 
            values for. This array should be of shape (M,L), where L matches the number of features 
            of the array that the class was fitted on (using ```.fit()```) and M is the number of 
            values to calculate the reverse percentile for.
            
        Returns
        --------
        
        - pct: array:
            This is an array containing the reverse percentile values for the values in the input.
        
        '''

        base = np.asarray(self._baseline_values)
        n = base.shape[0]

        if n == 0:
            return 100.0

        if kind == 'rank':
            # the None index forces broadcasting in the dimensions that I need.
            left = np.count_nonzero(base < values[:, None, :], axis=1)
            right = np.count_nonzero(base <= values[:, None, :], axis=1)
            pct = (right + left + (right > left)) * 0.5 / n

        elif kind == 'strict':
            pct = np.count_nonzero(base < values[:, None, :], axis=1) / n

        elif kind == 'weak':
            pct = np.count_nonzero(base <= values[:, None, :], axis=1) / n

        elif kind == 'mean':
            pct = (np.count_nonzero(base < values[:, None, :], axis=1)
                   + np.count_nonzero(base <= values[:, None, :], axis=1)) / n * 0.5

        else:
            raise ValueError("kind can only be 'rank', 'strict', 'weak' or 'mean'")

        return pct

    def _batcher(self, values, batch_size):
        '''
        This function creates list containing the values, each with size of ```batch_size```
        or less.
        '''

        split_index = np.arange(0, values.shape[0], batch_size)[1:]

        out = np.split(values, split_index)
        self.n_batches = split_index.shape[0] + 1

        return out

    def threshold_counter(self, values, threshold, axis=0):
        '''
        This counts the number of times there were breaches in the values against a threshold.
        
        Arguments
        ---------
        
        - values: array:
            This is the array to calculate the breaches on.
        
        - threshold: float:
            This is the threshold to use when calculating the breaches.
        
        - axis: int:
            This is the direction in which the breaches are counted. If 0, the breaches
            are counted over the samples. If 1, over the features.
        
        Returns
        ---------
        
        - out: array:
            This is of shape (1,L) if ```axis = 0``` and (N,1) if ```axis=1```. This means
            that if the number of features is 1 and ```axis=0```, the returned shape will be 
            (1,1).
        
        '''
        breaches = values > threshold

        out = np.sum(breaches, axis=axis, keepdims=True)

        return out

    def transform(self, values, method='reverse_percentiles'):
        '''
        This function calculates the output of the values given a ```method```.
        
        Arguments
        ---------
        
        - values: array:
            This is the arrray for which the transformation is done.
        
        - method: string:
            This is the method that is used to calculate the transformation.
            - ```'reverse_percentiles'```: this setting calculates the reverse percentile.
        
        Returns
        ---------
        
        - out: array:
            This is an array containing the transformed values.
        
        
        '''

        self.methods_possible

        if type(method) == str:
            if not method in self.methods_possible:
                raise TypeError('Please choose from the possible methods: {}'.format(self.methods_possible))
        else:
            raise TypeError(
                'Input must be a string. Please choose from the possible methods:'.format(self.methods_possible))

        if self.batch_size != -1:
            batches = self._batcher(values, self.batch_size)

        else:
            batches = [values]

        out = np.zeros(values.shape[1]).reshape(1, -1)

        bar = PBar(show_length=20, n_iterations=len(batches))
        print_progress = 0
        for nb, batch in enumerate(batches):
            if method == 'reverse_percentiles':
                rp_batch = self._calculate_reverse_percentiles(batch)
                out = np.vstack([out, rp_batch])

            progress = bar.progress / bar.n_iterations
            bar.update(1)
            if self.verbose:
                sys.stdout.write('\r')
                sys.stdout.write('Transforming {} - batch number {} / {}'.format(bar.give(), nb + 1, len(batches)))
                sys.stdout.flush()

        if self.verbose:
            sys.stdout.write('\n')

        out = out[1:]

        return out

    def predict(self, values, method='threshold_counter', method_args={'threshold': 0.7, 'axis': 0},
                transform_method='reverse_percentiles'):
        '''
        This function makes predictions on the ```values```.
        
        Arguments
        ---------
        
        - values: array:
            This is the array of values that we wish to calculate the predictions on.
        
        - method: string:
            This is a string that corresponds to the method that will be used to calculate 
            the predictions. The following methods are available:
            - ```'threshold_counter```: This counts the number of times that the method is above
            a given threshold.
        
        - method_args: dictionary:
            Here, you may pass the arguments which are described under the documentation for each of
            the functions available in the arugment ```method```.
            
        - transform_method: string:
            This is the method used to transform the data before predictions are made. The 
            available options here are described under the documentation for the ```.transform()```
            function.
            
        Returns
        ---------
            
        - out: array
            These are the predictions.
        
        '''

        values_tr = self.transform(values)

        if self.batch_size != -1:
            batches = self._batcher(values_tr, self.batch_size)

        else:
            batches = [values_tr]

        bar = PBar(show_length=20, n_iterations=len(batches))
        print_progress = 0
        out = np.zeros(values.shape[1]).reshape(1, -1)
        for nb, batch in enumerate(batches):

            if method == 'threshold_counter':

                include = ['threshold', 'axis']
                to_include = [argument for argument in include if argument not in method_args]
                if len(to_include) != 0:
                    raise TypeError('Please include the arguments {} in method_args'.format(to_include))

                predict_batch = self.threshold_counter(batch, **method_args)
                out = np.vstack([out, predict_batch])
                out = np.sum(out, axis=method_args['axis'], keepdims=True)

            bar.update(1)
            progress = bar.progress / bar.n_iterations
            if self.verbose:
                sys.stdout.write('\r')
                sys.stdout.write('Predicting   {} - batch number {} / {}'.format(bar.give(), nb + 1, len(batches)))
                sys.stdout.flush()

        if self.verbose:
            sys.stdout.write('\n')

        return out
