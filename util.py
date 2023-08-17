import xarray as xr
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Split data and normalize based on training data
def split_normalize_data(x, y, train_ratio=0.8, method='random', variable_names=None, return_stats=False):

    if method == 'random':
        if return_stats:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(1-train_ratio))
            return x_train, x_test, y_train, y_test, x_train.mean(axis=0), x_train.std(axis=0), y_train.mean(axis=0)
        else:
            return train_test_split(x, y, test_size=(1-train_ratio))
    elif variable_names is None:
        raise ValueError("variable_names must be passed if method is not 'random'.")
    elif method not in variable_names:
        raise ValueError("method must be either 'random' or a variable name.")
    
    index = variable_names.index(method)
    
    x_sorted = x[x[:,index].argsort()]
    y_sorted = y[x[:,index].argsort()]

    x_train = x_sorted[0:int(train_ratio*x_sorted.shape[0]),:]
    x_test = x_sorted[int(train_ratio*x_sorted.shape[0]):,:]
    y_train = y_sorted[0:int(train_ratio*x_sorted.shape[0]),:]
    y_test = y_sorted[int(train_ratio*x_sorted.shape[0]):,:]

    x_train_mean = x_train.mean(axis=0)
    x_train_std  = x_train.std(axis=0)
    y_train_mean = y_train.mean(axis=0)

    x_train = (x_train - x_train_mean) / x_train_std
    x_test = (x_test - x_train_mean) / x_train_std
    y_train = y_train - y_train_mean
    y_test = y_test - y_train_mean

    if return_stats:
        return x_train, x_test, y_train, y_test, x_train_mean, x_train_std, y_train_mean
    else:
        return x_train, x_test, y_train, y_test
    

# Load data
def load_data(variables):
    atms_files = []
    model_files = []

    print('logging atms files:')
    for root, dirs, files in os.walk(".\\data\\atms\\"):
        for name in files:
            file = os.path.join(root, name)
            print(file)
            atms_files.append(file)

    print('logging model files:')
    for root, dirs, files in os.walk(".\\data\\model\\"):
        for name in files:
            file = os.path.join(root, name)
            print(file)
            model_files.append(file)

    print()
    print('-------')
    print()

    x = np.empty((0,len(variables)))
    dT = np.empty((0,22))

    for atms_file, model_file in zip(atms_files, model_files):

        print('loading model dataset',model_file)
        data = xr.load_dataset(model_file)

        print('loading atms dataset',atms_file)
        data_tb = xr.load_dataset(atms_file)
        data_tb = data_tb.tb.squeeze('nScanAng').T.to_dataset().rename_dims({'nProfs': 'obs_id'})

        # Filter data to strictly land with clear-sky conditions for model training
        data_tb = data_tb.where(data.lsm == 1.0, drop=True)
        data_obs = data.where(data.lsm == 1.0, drop=True)
        data_tb = data_tb.where(data_obs.tcc <= 0.1, drop=True)
        data_obs = data_obs.where(data_obs.tcc <= 0.1, drop=True)

        # Create input/output matrices

        x = np.vstack((x,data_obs[variables].to_array().values.T))
        # Add bias term
        #x = np.hstack((x, np.ones((x.shape[0], 1))))

        # Omitting adding date for now: all the same date.
        #x = np.hstack((x, np.full((x.shape[0], 1), date_num)))

        dT = np.vstack((dT,data_obs.obs.values - data_tb.tb.values))
        
    print('finished loading')
    return x, dT

# Returns a list of lists of the elements, up to the specified length with repeats
def combinations(lst, length):
    ret = []
    
    for item in lst:
        ret.append([item])
    
    curr_length = 2
    while curr_length <= length:
        new_lists = []
        for item in lst:
            for old_lst in ret:
                curr = list(old_lst)
                curr.append(item)
                new_lists.append(curr)
                
        curr_length += 1
        ret.extend(new_lists)
    
    return ret