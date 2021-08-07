"""Helper methods for ARcnnV2."""
import copy
import errno
import random
import glob
import os.path
import time
import calendar
import json
import pickle
import netCDF4
import numpy
import pandas

import tensorflow
import tensorflow.keras as keras

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'  # set up tensorflow backend for keras

# from keras import backend as K
import matplotlib.colors
import matplotlib.pyplot as pyplot

# need a single file to set variables
exampfil = "/glade/scratch/wchapman/Reforecast/F036/train/F036_WY_1985_500mb_Clean.nc"
exampfilgefs = "/glade/scratch/wchapman/GEFS/F048/WY_2016_GEFS_F048.nc"

trainst = '11/30/1985'
trainen = '4/02/2019'

# Directories.
DIR_NAME = '.'
NETCDF_TARGET_NAME = 'IVTm'
TARGET_NAME = 'IVTm'
POST_NAME = 'IVT'


NETCDF_IVT_NAME = 'IVT'
NETCDF_Psfc_NAME = 'p_sfc'
NETCDF_U500_NAME = 'u_tr_p'
NETCDF_V500_NAME = 'v_tr_p'
NETCDF_Z500_NAME = 'Z_p'
NETCDF_IWV_NAME = 'IWV'
NETCDF_lat_NAME = 'lat'
NETCDF_lon_NAME = 'lon'
NETCDF_time_NAME= 'time'

#variable names: 
IVTgfs = 'IVT'
psfc = 'p_sfc'
u500 = 'u_tr_p'
v500 = 'v_tr_p'
z500 = 'Z_p'
IWV = 'IWV'
lats = 'lat'
lons = 'lon'

#lat and lon can't come first.... sorry. 
NETCDF_PREDICTOR_NAMES = [
    NETCDF_IVT_NAME, NETCDF_Psfc_NAME, NETCDF_U500_NAME, NETCDF_V500_NAME, NETCDF_Z500_NAME, NETCDF_IWV_NAME,NETCDF_lat_NAME,NETCDF_lon_NAME
]

PREDICTOR_NAMES = [
    IVTgfs, psfc, u500, v500, z500, IWV, lats, lons
]

NUM_VALUES_KEY = 'num_values'
MEAN_VALUE_KEY = 'mean_value'
MEAN_OF_SQUARES_KEY = 'mean_of_squares'

MIN_XENTROPY_DECREASE_FOR_EARLY_STOP = 0.00005
MIN_MSE_DECREASE_FOR_EARLY_STOP = 0.00005
NUM_EPOCHS_FOR_EARLY_STOPPING = 15
NUM_EPOCHS_FOR_REDUCE_LR = 5
FACTOR_REDUCE_LR = 0.7
MIN_LR_REDUCE_TO = 0.00005
MIN_MSE_DECREASE_FOR_REDUCE_LR = 0.0001

PREDICTOR_NAMES_KEY = 'predictor_names'
PREDICTOR_MATRIX_KEY = 'predictor_matrix'
POST_NAME_KEY = IVTgfs      #variable name of the post processed variable 
POST_MATRIX_KEY = 'post_matrix'
TARGET_NAME_KEY = 'target_name'
TARGET_MATRIX_KEY = 'target_matrix'

TRAINING_FILES_KEY = 'training_file_names'
NORMALIZATION_DICT_KEY = 'normalization_dict'
TARGET_DICT_KEY = 'target_dict'
NORMALIZATION_DICT_TARG_KEY = 'normalization_dict_targ'
NUM_EXAMPLES_PER_BATCH_KEY = 'num_examples_per_batch'
NORMALIZATION_DICT_POST_KEY = 'normalization_dict_post'
NUM_TRAINING_BATCHES_KEY = 'num_training_batches_per_epoch'
VALIDATION_FILES_KEY = 'validation_file_names'
NUM_VALIDATION_BATCHES_KEY = 'num_validation_batches_per_epoch'
CNN_FILE_KEY = 'cnn_file_name'
CNN_FEATURE_LAYER_KEY = 'cnn_feature_layer_name'


# Machine-learning constants.
L1_WEIGHT = 0.
L2_WEIGHT = 0.001
NUM_PREDICTORS_TO_FIRST_NUM_FILTERS = 2
NUM_CONV_LAYER_SETS = 2
NUM_CONV_LAYERS_PER_SET = 2
NUM_CONV_FILTER_ROWS = 3
NUM_CONV_FILTER_COLUMNS = 3
CONV_LAYER_DROPOUT_FRACTION = None
USE_BATCH_NORMALIZATION = True
SLOPE_FOR_RELU = 0.2
NUM_POOLING_ROWS = 2
NUM_POOLING_COLUMNS = 2
NUM_DENSE_LAYERS = 3
DENSE_LAYER_DROPOUT_FRACTION = 0.5

NUM_SMOOTHING_FILTER_ROWS = 5
NUM_SMOOTHING_FILTER_COLUMNS = 5

def time_string_to_unix(time_string, time_format):
    """Converts time from string to Unix format.
    Unix format = seconds since 0000 UTC 1 Jan 1970.
    :param time_string: Time string.
    :param time_format: Format of time string (example: "%Y%m%d" or
        "%Y-%m-%d-%H%M%S").
    :return: unix_time_sec: Time in Unix format.
    """

    return calendar.timegm(time.strptime(time_string, time_format))


def time_unix_to_string(unix_time_sec, time_format):
    """Converts time from Unix format to string.
    Unix format = seconds since 0000 UTC 1 Jan 1970.
    :param unix_time_sec: Time in Unix format.
    :param time_format: Desired format of time string (example: "%Y%m%d" or
        "%Y-%m-%d-%H%M%S").
    :return: time_string: Time string.
    """

    return time.strftime(time_format, time.gmtime(unix_time_sec))




def read_image_file(netcdf_file_name,targ_LATinds=None,targ_LONinds=None):
    """Reads AR images from NetCDF file.
    E = number of examples in file
    M = number of rows in each storm-centered grid (lats)
    N = number of columns in each storm-centered grid (lons)
    C = number of channels (predictor variables) 
    targ_inds = list indices of target matrix, you'd like to train on. must be list
    
    :param netcdf_file_name: Path to input file.
    
    :return: image_dict: Dictionary with the following keys.
    image_dict['storm_ids']: length-E list of storm IDs (integers).
    image_dict['storm_steps']: length-E numpy array of storm steps (integers).
    image_dict['predictor_names']: length-C list of predictor names.
    image_dict['predictor_matrix']: E-by-M-by-N-by-C numpy array of predictor
        values.
    image_dict['target_name']: Name of target variable.
    image_dict['target_matrix']: E-by-M-by-N numpy array of target values.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)
    
    predictor_matrix = None

    for this_predictor_name in NETCDF_PREDICTOR_NAMES:
        this_predictor_matrix = numpy.array(
            dataset_object.variables[this_predictor_name][:], dtype=float
        )
        
        if (this_predictor_name == 'lat'):
            this_predictor_matrix_lat= this_predictor_matrix[targ_LATinds]
            LL = None
            
            if isinstance(this_predictor_matrix_lat, float):
                this_predictor_matrix_lat = [this_predictor_matrix_lat]
            
            for pp in range(len(this_predictor_matrix_lat)):
                thisLat = this_predictor_matrix_lat[pp] 
                thisLat = numpy.expand_dims(numpy.tile(thisLat,predictor_matrix.shape[0]),axis=1)
                if LL is None: 
                    LL = thisLat + 0.
                else:
                    LL = numpy.concatenate((LL,thisLat),axis=-1)
            
            
            LL = numpy.squeeze(LL)
            LL = numpy.expand_dims(LL,axis=-1)
            
            predictor_matrix = numpy.concatenate(
                (predictor_matrix, LL), axis=-1)
            
  
        elif (this_predictor_name=='lon'):
            this_predictor_matrix_lon= this_predictor_matrix[targ_LONinds]
            LLo = None
            if isinstance(this_predictor_matrix_lon, float):
                this_predictor_matrix_lon = [this_predictor_matrix_lon]
                
            for pp in range(len(this_predictor_matrix_lon)):
                thisLon = this_predictor_matrix_lon[pp]
                thisLon = numpy.expand_dims(numpy.tile(thisLon,predictor_matrix.shape[0]),axis=1)
                if LLo is None: 
                    LLo = thisLon + 0.
                else: 
                    LLo = numpy.concatenate((LLo,thisLon),axis=-1)
            
            LLo = numpy.squeeze(LLo)
            LLo = numpy.expand_dims(LLo,axis=-1)
            predictor_matrix = numpy.concatenate(
                (predictor_matrix, LLo), axis=-1)
            
        else:
            this_predictor_matrix=this_predictor_matrix[:,targ_LATinds,targ_LONinds]
            this_predictor_matrix = numpy.expand_dims(this_predictor_matrix, axis=-1)

            if predictor_matrix is None:
                predictor_matrix = this_predictor_matrix + 0.
            
            else:
                predictor_matrix = numpy.concatenate(
                    (predictor_matrix, this_predictor_matrix), axis=-1
                
                )
                
            
            
    pred_name_dict = {PREDICTOR_NAMES_KEY: PREDICTOR_NAMES}
    pred_mat_dict = {PREDICTOR_MATRIX_KEY: predictor_matrix}
        
    indPOST = pred_name_dict[PREDICTOR_NAMES_KEY].index(POST_NAME_KEY)
    
    if targ_LATinds is None:
        target_matrix = numpy.array(
            dataset_object.variables[NETCDF_TARGET_NAME][:], dtype=float
        )
        post_matrix = predictor_matrix[indPOST,:,:]
        
    else:
        target_matrix = numpy.array(
            dataset_object.variables[NETCDF_TARGET_NAME][:], dtype=float
        )
        target_matrix = target_matrix[:,targ_LATinds,targ_LONinds]
        target_matrix = numpy.reshape(target_matrix,(target_matrix.shape[0],-1))

    #take logs of all IVT. 
    ##turn off the logging

    indicesPRED = [iii for iii, sss in enumerate(PREDICTOR_NAMES) if 'IVT' in sss]
    indicesTARG = [iii for iii, sss in enumerate([TARGET_NAME]) if 'IVT' in sss]
    indicesPOST = [iii for iii, sss in enumerate([POST_NAME]) if 'IVT' in sss]
    
    
    ##turn off the logging
#     if indicesPRED:
#         tmp = numpy.log(predictor_matrix[...,indicesPRED])
#         tmp[numpy.isinf(tmp)]=0
#         predictor_matrix[...,indicesPRED] = tmp
        
#     if indicesTARG:
#         target_matrix = numpy.log(target_matrix)
    
    predictor_matrix = numpy.squeeze(predictor_matrix)
    
    dicter = {
        PREDICTOR_NAMES_KEY: PREDICTOR_NAMES,
        PREDICTOR_MATRIX_KEY: predictor_matrix,
        TARGET_NAME_KEY: TARGET_NAME,
        TARGET_MATRIX_KEY: target_matrix,
    } 
    return dicter



def _update_normalization_params(intermediate_normalization_dict, new_values):
    """Updates normalization params for one predictor.
    :param intermediate_normalization_dict: Dictionary with the following keys.
    intermediate_normalization_dict['num_values']: Number of values on which
        current estimates are based.
    intermediate_normalization_dict['mean_value']: Current estimate for mean.
    intermediate_normalization_dict['mean_of_squares']: Current mean of squared
        values.
    :param new_values: numpy array of new values (will be used to update
        `intermediate_normalization_dict`).
    :return: intermediate_normalization_dict: Same as input but with updated
        values.
    """
    if MEAN_VALUE_KEY not in intermediate_normalization_dict:
        intermediate_normalization_dict = {
            NUM_VALUES_KEY: 0,
            MEAN_VALUE_KEY: 0.,
            MEAN_OF_SQUARES_KEY: 0.
        }

    these_means = numpy.array([
        intermediate_normalization_dict[MEAN_VALUE_KEY], numpy.mean(new_values)
    ])
    these_weights = numpy.array([
        intermediate_normalization_dict[NUM_VALUES_KEY], new_values.size
    ])

    intermediate_normalization_dict[MEAN_VALUE_KEY] = numpy.average(
        these_means, weights=these_weights)

    these_means = numpy.array([
        intermediate_normalization_dict[MEAN_OF_SQUARES_KEY],
        numpy.mean(new_values ** 2)
    ])

    intermediate_normalization_dict[MEAN_OF_SQUARES_KEY] = numpy.average(
        these_means, weights=these_weights)

    intermediate_normalization_dict[NUM_VALUES_KEY] += new_values.size
    return intermediate_normalization_dict




def _get_standard_deviation(intermediate_normalization_dict):
    """Computes stdev from intermediate normalization params.
    :param intermediate_normalization_dict: See doc for
        `_update_normalization_params`.
    :return: standard_deviation: Standard deviation.
    """

    num_values = float(intermediate_normalization_dict[NUM_VALUES_KEY])
    multiplier = num_values / (num_values - 1)

    return numpy.sqrt(multiplier * (
        intermediate_normalization_dict[MEAN_OF_SQUARES_KEY] -
        intermediate_normalization_dict[MEAN_VALUE_KEY] ** 2
    ))


def get_image_normalization_params(netcdf_file_names,targ_LATinds=None,targ_LONinds=None):
    """Computes normalization params (mean and stdev) for each predictor.
    :param netcdf_file_names: 1-D list of paths to input files.
    :return: normalization_dict: See input doc for `normalize_images`.
    """

    predictor_names = None
    norm_dict_by_predictor = None

    for this_file_name in netcdf_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        if (targ_LATinds is None) & (targ_LONinds is None):
            this_image_dict = read_image_file(this_file_name)
        else:
            this_image_dict = read_image_file(this_file_name,targ_LATinds,targ_LONinds)
        
        if predictor_names is None:
            predictor_names = this_image_dict[PREDICTOR_NAMES_KEY]
            norm_dict_by_predictor = [{}] * len(predictor_names)

        for m in range(len(predictor_names)):
            norm_dict_by_predictor[m] = _update_normalization_params(
                intermediate_normalization_dict=norm_dict_by_predictor[m],
                new_values=this_image_dict[PREDICTOR_MATRIX_KEY][..., m]
            )

    print('\n')
    normalization_dict = {}
    for m in range(len(predictor_names)):
        this_mean = norm_dict_by_predictor[m][MEAN_VALUE_KEY]
        this_stdev = _get_standard_deviation(norm_dict_by_predictor[m])

        normalization_dict[predictor_names[m]] = numpy.array(
            [this_mean, this_stdev]
        )

        message_string = (
            'Mean and standard deviation for "{0:s}" = {1:.4f}, {2:.4f}'
        ).format(predictor_names[m], this_mean, this_stdev)
        print(message_string)

    return normalization_dict



def get_image_normalization_params_targ(netcdf_file_names,targ_LATinds=None,targ_LONinds=None):
    """Computes normalization params (mean and stdev) for the target variable.
    :param netcdf_file_names: 1-D list of paths to input files.
    :param targ*lons: desired target lat lon indices
    :return: normalization_dict: See input doc for `normalize_images`.
    """

    targ_names = None
    norm_dict_by_targ = None

    for this_file_name in netcdf_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        
        if (targ_LATinds is None) & (targ_LONinds is None):
            this_image_dict = read_image_file(this_file_name)
        else:
            this_image_dict = read_image_file(this_file_name,targ_LATinds,targ_LONinds)    
        
        if targ_names is None:
            targ_names = this_image_dict[TARGET_NAME_KEY]
            norm_dict_by_targ = [{}] * len(targ_names[0])

        for m in range(len(targ_names[0])):
            norm_dict_by_targ[m] = _update_normalization_params(
                intermediate_normalization_dict=norm_dict_by_targ[m],
                new_values=numpy.ndarray.flatten(this_image_dict[TARGET_MATRIX_KEY])
            )

    print('\n')
    normalization_dict = {}


    for m in range(len(targ_names[0])):
        this_mean = norm_dict_by_targ[m][MEAN_VALUE_KEY]
        this_stdev = _get_standard_deviation(norm_dict_by_targ[m])

        normalization_dict[targ_names] = numpy.array(
            [this_mean, this_stdev]
        )
        message_string = (
            'Mean and standard deviation for "{0:s}" = {1:.4f}, {2:.4f}'
        ).format(targ_names, this_mean, this_stdev)
        print(message_string)

    return normalization_dict


def get_image_normalization_params_post(netcdf_file_names,targ_LATinds=None,targ_LONinds=None):
    """Computes normalization params (mean and stdev) for each predictor.
    :param netcdf_file_names: 1-D list of paths to input files.
    :param targ*lons: desired target lat lon indices
    :return: normalization_dict: See input doc for `normalize_images`.
    """

    post_names = None
    norm_dict_by_post = None

    for this_file_name in netcdf_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        
        if (targ_LATinds is None) & (targ_LONinds is None):
            this_image_dict = read_image_file(this_file_name)
        else:
            this_image_dict = read_image_file(this_file_name,targ_LATinds,targ_LONinds)    
        
        if post_names is None:
            post_names = this_image_dict[POST_NAME_KEY]
            norm_dict_by_post = [{}] * len(post_names[0])
        
        
        for m in range(len(post_names[0])):
            norm_dict_by_post[m] = _update_normalization_params(
                intermediate_normalization_dict=norm_dict_by_post[m],
                new_values=this_image_dict[POST_MATRIX_KEY][..., m]
            )

    print('\n')
    normalization_dict = {}

    for m in range(len(post_names[0])):
        this_mean = norm_dict_by_post[m][MEAN_VALUE_KEY]
        this_stdev = _get_standard_deviation(norm_dict_by_post[m])

        normalization_dict[post_names] = numpy.array(
            [this_mean, this_stdev]
        )
        message_string = (
            'Mean and standard deviation for "{0:s}" = {1:.4f}, {2:.4f}'
        ).format(post_names, this_mean, this_stdev)
        print(message_string)

    return normalization_dict



def count_samps(netcdf_file_names):
    """determines number of samples in list
    """
    targ_names = None
    norm_dict_by_targ = None
    num_samps = 0
    for this_file_name in netcdf_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_image_dict = read_image_file(this_file_name)
        num_samps = num_samps+this_image_dict[TARGET_MATRIX_KEY].shape[0]

    return num_samps





def normalize_images(
        predictor_matrix, predictor_names, normalization_dict=None):
    """Normalizes images to z-scores.
    E = number of examples in file
    M = number of rows in each grid (lats)
    N = number of columns in each grid (lons)
    C = number of channels (predictor variables)
    
    :param predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :param predictor_names: length-C list of predictor names.
    :param normalization_dict: Dictionary.  Each key is the name of a predictor
        value, and the corresponding value is a length-2 numpy array with
        [mean, standard deviation].  If `normalization_dict is None`, mean and
        standard deviation will be computed for each predictor.
    :return: predictor_matrix: Normalized version of input.
    :return: normalization_dict: See doc for input variable.  If input was None,
        this will be a newly created dictionary.  Otherwise, this will be the
        same dictionary passed as input.
        
    If you want a multiple stream model with a post-process matrix... poststream=True. 
    """
    num_predictors = len(predictor_names)
    if normalization_dict is None:
        normalization_dict = {}

        for m in range(num_predictors):
            this_mean = numpy.mean(predictor_matrix[..., m])
            this_stdev = numpy.std(predictor_matrix[..., m], ddof=1)
            
            normalization_dict[predictor_names[m]] = numpy.array(
                [this_mean, this_stdev]
            )

    for m in range(num_predictors):
        this_mean = normalization_dict[predictor_names[m]][0]
        this_stdev = normalization_dict[predictor_names[m]][1]
        
        if this_stdev ==0:
            predictor_matrix[..., m] =0
        else:
            predictor_matrix[..., m] = (
                (predictor_matrix[..., m] - this_mean) / float(this_stdev))
    
    return predictor_matrix, normalization_dict


def normalize_images_targ(
        targ_matrix, targ_names, normalization_dict):
    """Normalizes images to z-scores.
    E = number of examples in file
    M = number of rows in each grid (lats)
    N = number of columns in each grid (lons)
    C = number of channels (predictor variables)
    
    :param predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :param predictor_names: length-C list of predictor names.
    :param normalization_dict: Dictionary.  Each key is the name of a predictor
        value, and the corresponding value is a length-2 numpy array with
        [mean, standard deviation].  If `normalization_dict is None`, mean and
        standard deviation will be computed for each predictor.
    :return: predictor_matrix: Normalized version of input.
    :return: normalization_dict: See doc for input variable.  If input was None,
        this will be a newly created dictionary.  Otherwise, this will be the
        same dictionary passed as input.
    """
    num_targs = len(targ_names.split())
    for m in range(num_targs):
        this_mean = normalization_dict[targ_names.split()[m]][0]
        this_stdev = normalization_dict[targ_names.split()[m]][1]
       
        targ_matrix = ((targ_matrix - this_mean) / float(this_stdev)
            )

    return targ_matrix, normalization_dict

def normalize_images_post(
        post_matrix, post_names, normalization_dict):
    """Normalizes images to z-scores.
    E = number of examples in file
    M = number of rows in each grid (lats)
    N = number of columns in each grid (lons)
    C = number of channels (predictor variables)
    
    :param predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :param predictor_names: length-C list of predictor names.
    :param normalization_dict: Dictionary.  Each key is the name of a predictor
        value, and the corresponding value is a length-2 numpy array with
        [mean, standard deviation].  If `normalization_dict is None`, mean and
        standard deviation will be computed for each predictor.
    :return: predictor_matrix: Normalized version of input.
    :return: normalization_dict: See doc for input variable.  If input was None,
        this will be a newly created dictionary.  Otherwise, this will be the
        same dictionary passed as input.
    """
    num_post = len(post_names.split())
    for m in range(num_post):
        this_mean = normalization_dict[post_names.split()[m]][0]
        this_stdev = normalization_dict[post_names.split()[m]][1]
       
        post_matrix = ((post_matrix - this_mean) / float(this_stdev)
            )

    return post_matrix, normalization_dict

def denormalize_images(predictor_matrix, predictor_names, normalization_dict):
    """Denormalizes images from z-scores back to original scales.
    :param predictor_matrix: See doc for `normalize_images`.
    :param predictor_names: Same.
    :param normalization_dict: Same.
    :return: predictor_matrix: Denormalized version of input.
    """

    num_predictors = len(predictor_names)

    for m in range(num_predictors):
        this_mean = normalization_dict[predictor_names[m]][0]
        this_stdev = normalization_dict[predictor_names[m]][1]

        predictor_matrix[..., m] = (
            this_mean + this_stdev * predictor_matrix[..., m]
        )

    return predictor_matrix


def denormalize_images_targ(targ_matrix, targ_names, normalization_dict):
    """Denormalizes images from z-scores back to original scales.
    :param predictor_matrix: See doc for `normalize_images`.
    :param predictor_names: Same.
    :param normalization_dict: Same.
    :return: predictor_matrix: Denormalized version of input.
    """   
    
    num_targs = len(targ_names.split())
    
    for m in range(num_targs):
        this_mean = normalization_dict[targ_names.split()[0]][0]
        this_stdev = normalization_dict[targ_names.split()[0]][1]
    
        targ_matrix = (
                this_mean + this_stdev * targ_matrix
            )

    return targ_matrix


def read_keras_model(hdf5_file_name):
    """Reads Keras model from HDF5 file.
    :param hdf5_file_name: Path to input file.
    :return: model_object: Instance of `keras.models.Model`.
    """

    return keras.models.load_model(
        hdf5_file_name, custom_objects=METRIC_FUNCTION_DICT)


def find_model_metafile(model_file_name, raise_error_if_missing=False):
    """Finds metafile for machine-learning model.
    :param model_file_name: Path to file with trained model.
    :param raise_error_if_missing: Boolean flag.  If True and metafile is not
        found, this method will error out.
    :return: model_metafile_name: Path to file with metadata.  If file is not
        found and `raise_error_if_missing = False`, this will be the expected
        path.
    :raises: ValueError: if metafile is not found and
        `raise_error_if_missing = True`.
    """

    model_directory_name, pathless_model_file_name = os.path.split(
        model_file_name)
    model_metafile_name = '{0:s}/{1:s}_metadata.json'.format(
        model_directory_name, os.path.splitext(pathless_model_file_name)[0]
    )

    if not os.path.isfile(model_metafile_name) and raise_error_if_missing:
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            model_metafile_name)
        raise ValueError(error_string)

    return model_metafile_name


def _metadata_numpy_to_list(model_metadata_dict):
    """Converts numpy arrays in model metadata to lists.
    This is needed so that the metadata can be written to a JSON file (JSON does
    not handle numpy arrays).
    This method does not overwrite the original dictionary.
    :param model_metadata_dict: Dictionary created by `train_cnn` or
        `train_ucn`.
    :return: new_metadata_dict: Same but with lists instead of numpy arrays.
    """

    new_metadata_dict = copy.deepcopy(model_metadata_dict)

    if NORMALIZATION_DICT_KEY in new_metadata_dict.keys():
        this_norm_dict = new_metadata_dict[NORMALIZATION_DICT_KEY]

        for this_key in this_norm_dict.keys():
            print(this_key)
            if isinstance(this_norm_dict[this_key], numpy.ndarray):
                this_norm_dict[this_key] = this_norm_dict[this_key].tolist() 
    
    if NORMALIZATION_DICT_TARG_KEY in model_metadata_dict.keys():
        this_norm_dict_targ = new_metadata_dict[NORMALIZATION_DICT_TARG_KEY]
        for this_key in  this_norm_dict_targ.keys():
            print(this_key)
            this_norm_dict_targ[this_key] = this_norm_dict_targ[this_key].tolist()
            
    if NORMALIZATION_DICT_POST_KEY in model_metadata_dict.keys():
        this_norm_dict_post = new_metadata_dict[NORMALIZATION_DICT_POST_KEY]
        for this_key in  this_norm_dict_post.keys():
            print(this_key)
            this_norm_dict_post[this_key] = this_norm_dict_post[this_key].tolist()

    return new_metadata_dict


def _metadata_list_to_numpy(model_metadata_dict):
    """Converts lists in model metadata to numpy arrays.
    This method is the inverse of `_metadata_numpy_to_list`.
    This method overwrites the original dictionary.
    :param model_metadata_dict: Dictionary created by `train_cnn` or
        `train_ucn`.
    :return: model_metadata_dict: Same but numpy arrays instead of lists.
    """

    if NORMALIZATION_DICT_KEY in model_metadata_dict.keys():
        this_norm_dict = model_metadata_dict[NORMALIZATION_DICT_KEY]

        for this_key in this_norm_dict.keys():
            this_norm_dict[this_key] = numpy.array(this_norm_dict[this_key])
            
    if NORMALIZATION_DICT_TARG_KEY in model_metadata_dict.keys():
        this_norm_dict_targ = model_metadata_dict[NORMALIZATION_DICT_TARG_KEY]

        for this_key in  this_norm_dict_targ.keys():
            this_norm_dict_targ[this_key] = numpy.array(this_norm_dict_targ[this_key])

    return model_metadata_dict


def write_model_metadata(model_metadata_dict, json_file_name):
    """Writes metadata for machine-learning model to JSON file.
    :param model_metadata_dict: Dictionary created by `train_cnn` or
        `train_ucn`.
    :param json_file_name: Path to output file.
    """

    _create_directory(file_name=json_file_name)

    new_metadata_dict = _metadata_numpy_to_list(model_metadata_dict)
    
    print(new_metadata_dict)
    with open(json_file_name, 'w') as this_file:
        json.dump(new_metadata_dict, this_file)


def read_model_metadata(json_file_name):
    """Reads metadata for machine-learning model from JSON file.
    :param json_file_name: Path to output file.
    :return: model_metadata_dict: Dictionary with keys listed in doc for
        `train_cnn` or `train_ucn`.
    """

    with open(json_file_name) as this_file:
        model_metadata_dict = json.load(this_file)
        return _metadata_list_to_numpy(model_metadata_dict)

    
# def apply_cnn(cnn_model_object, predictor_matrix, verbose=True,
#               output_layer_name=None):
#     """Applies trained CNN (convolutional neural net) to new data.
#     E = number of examples in file
#     M = number of rows in each grid (lats)
#     N = number of columns in each grid (lons)
#     C = number of channels (predictor variables)
#     :param cnn_model_object: Trained instance of `keras.models.Model`.
#     :param predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
#     :param verbose: Boolean flag.  If True, progress messages will be printed.
#     :param output_layer_name: Name of output layer.  If
#         `output_layer_name is None`, this method will use the actual output
#         layer, so will return predictions.  If `output_layer_name is not None`,
#         will return "features" (outputs from the given layer).
#     If `output_layer_name is None`...
#     :return: forecast_probabilities: length-E numpy array with forecast
#         probabilities of positive class (label = 1).
#     If `output_layer_name is not None`...
#     :return: feature_matrix: numpy array of features (outputs from the given
#         layer).  There is no guarantee on the shape of this array, except that
#         the first axis has length E.
#     """

#     num_examples = predictor_matrix.shape[0]
#     num_examples_per_batch = 1000

#     if output_layer_name is None:
#         model_object_to_use = cnn_model_object
#     else:
#         model_object_to_use = keras.models.Model(
#             inputs=cnn_model_object.input,
#             outputs=cnn_model_object.get_layer(name=output_layer_name).output
#         )

#     output_array = None

#     for i in range(0, num_examples, num_examples_per_batch):
#         this_first_index = i
#         this_last_index = min(
#             [i + num_examples_per_batch - 1, num_examples - 1]
#         )

#         if verbose:
#             print('Applying model to examples {0:d}-{1:d} of {2:d}...'.format(
#                 this_first_index, this_last_index, num_examples
#             ))

#         these_indices = numpy.linspace(
#             this_first_index, this_last_index,
#             num=this_last_index - this_first_index + 1, dtype=int)

#         this_output_array = model_object_to_use.predict(
#             predictor_matrix[these_indices, ...],
#             batch_size=num_examples_per_batch)

#         if output_layer_name is None:
#             this_output_array = this_output_array[:, -1]

#         if output_array is None:
#             output_array = this_output_array + 0.
#         else:
#             output_array = numpy.concatenate(
#                 (output_array, this_output_array), axis=0
#             )

#     return output_array



# def deep_learning_generator(netcdf_file_names, num_examples_per_batch,
#                             normalization_dict,normalization_dict_targ,targ_LATinds=None,
#                            targ_LONinds=None):
#     """Generates training examples for deep-learning model on the fly.
#     E = number of examples 
#     M = number of rows in each grid (lats)
#     N = number of columns in each grid (lons)
#     C = number of channels (predictor variables)
#     :param netcdf_file_names: 1-D list of paths to input (NetCDF) files.
#     :param num_examples_per_batch: Number of examples per training batch.
#     :param normalization_dict: See doc for `normalize_images`.  You cannot leave
#         this as None.
    
#     :return: predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
#     :return: target_values: length-E numpy array of target values (integers in
#         0...1).
#     :raises: TypeError: if `normalization_dict is None`.
#     """

#     if normalization_dict is None:
#         error_string = 'normalization_dict cannot be None.  Must be specified.'
#         raise TypeError(error_string)
        
#     if normalization_dict_targ is None:
#         error_string = 'normalization_dict_targ cannot be None.  Must be specified.'
#         raise TypeError(error_string)

#     random.shuffle(netcdf_file_names)
#     num_files = len(netcdf_file_names)
#     file_index = 0

#     num_examples_in_memory = 0
#     full_predictor_matrix = None
#     full_target_matrix = None
#     predictor_names = None

#     while True:
#         while num_examples_in_memory < num_examples_per_batch:
#             print('Reading data from: "{0:s}"...'.format(
#                 netcdf_file_names[file_index]
#             ))
            
#             if (targ_LATinds is None) & (targ_LONinds is None):
#                 this_image_dict = read_image_file(netcdf_file_names[file_index])
#             else:
#                 this_image_dict = read_image_file(netcdf_file_names[file_index],targ_LATinds,targ_LONinds)
            
#             predictor_names = this_image_dict[PREDICTOR_NAMES_KEY]
#             targ_names = this_image_dict[TARGET_NAME_KEY]

#             file_index += 1
#             if file_index >= num_files:
#                 file_index = 0

#             if full_target_matrix is None or full_target_matrix.size == 0:
#                 full_predictor_matrix = (
#                     this_image_dict[PREDICTOR_MATRIX_KEY] + 0.
#                 )
#                 full_target_matrix = this_image_dict[TARGET_MATRIX_KEY] + 0.

#             else:
#                 full_predictor_matrix = numpy.concatenate(
#                     (full_predictor_matrix,
#                      this_image_dict[PREDICTOR_MATRIX_KEY]),
#                     axis=0
#                 )

#                 full_target_matrix = numpy.concatenate(
#                     (full_target_matrix, this_image_dict[TARGET_MATRIX_KEY]),
#                     axis=0
#                 )

#             num_examples_in_memory = full_target_matrix.shape[0]

#         batch_indices = numpy.linspace(
#             0, num_examples_in_memory - 1, num=num_examples_in_memory,
#             dtype=int)
#         batch_indices = numpy.random.choice(
#             batch_indices, size=num_examples_per_batch, replace=False)

        
#         predictor_matrix, _ = normalize_images(
#             predictor_matrix=full_predictor_matrix[batch_indices, ...],
#             predictor_names=predictor_names,
#             normalization_dict=normalization_dict)
#         predictor_matrix = predictor_matrix.astype('float32')
        
        
#         target_values, _ = normalize_images_targ(
#             targ_matrix=full_target_matrix[batch_indices, ...],
#             targ_names = targ_names, 
#             normalization_dict = normalization_dict_targ)
#         target_values = target_values.astype('float32')


#         num_examples_in_memory = 0
#         full_predictor_matrix = None
#         full_target_matrix = None

#         yield (predictor_matrix, target_values)
        

        

# def train_cnn(
#         cnn_model_object, training_file_names, normalization_dict,
#         normalization_dict_targ,num_examples_per_batch, num_epochs,
#         num_training_batches_per_epoch, output_model_file_name,
#         validation_file_names=None, num_validation_batches_per_epoch=None,
#     targ_LATinds=None, targ_LONinds=None):
    
#     """Trains CNN (convolutional neural net).
#     :param cnn_model_object: Untrained instance of `keras.models.Model` (may be
#         created by `setup_cnn`).
#     :param training_file_names: 1-D list of paths to training files (must be
#         readable by `read_image_file`).
#     :param normalization_dict: See doc for `deep_learning_generator`.
#     param normalization_dict_targ: See doc for `deep_learning_generator`.
#     :param num_examples_per_batch: Same.
#     :param num_epochs: Number of epochs.
#     :param num_training_batches_per_epoch: Number of training batches furnished
#         to model in each epoch.
#     :param output_model_file_name: Path to output file.  The model will be saved
#         as an HDF5 file (extension should be ".h5", but this is not enforced).
#     :param validation_file_names: 1-D list of paths to training files (must be
#         readable by `read_image_file`).  If `validation_file_names is None`,
#         will omit on-the-fly validation.
#     :param num_validation_batches_per_epoch:
#         [used only if `validation_file_names is not None`]
#         Number of validation batches furnished to model in each epoch.
#     :return: cnn_metadata_dict: Dictionary with the following keys.
#     cnn_metadata_dict['training_file_names']: See input doc.
#     cnn_metadata_dict['normalization_dict']: Same.
#     cnn_metadata_dict['normalization_dict_targ']: Same.
#     cnn_metadata_dict['num_examples_per_batch']: Same.
#     cnn_metadata_dict['num_training_batches_per_epoch']: Same.
#     cnn_metadata_dict['validation_file_names']: Same.
#     cnn_metadata_dict['num_validation_batches_per_epoch']: Same.
#     """
    
#     #configure GPU: 
#     config = tensorflow.ConfigProto(allow_soft_placement=False, log_device_placement=False)
#     config.gpu_options.allow_growth = True
#     sess = tensorflow.Session(config=config)
#     K.set_session(sess)


#     _create_directory(file_name=output_model_file_name)

#     if validation_file_names is None:
#         checkpoint_object = keras.callbacks.ModelCheckpoint(
#             filepath=output_model_file_name, monitor='loss', verbose=1,
#             save_best_only=False, save_weights_only=False, mode='min',
#             period=1)
#     else:
#         checkpoint_object = keras.callbacks.ModelCheckpoint(
#             filepath=output_model_file_name, monitor='val_loss', verbose=1,
#             save_best_only=True, save_weights_only=False, mode='min',
#             period=1)

#     list_of_callback_objects = [checkpoint_object]
    
#     print('Normalization dict targ:', normalization_dict_targ)
#     cnn_metadata_dict = {
#         TRAINING_FILES_KEY: training_file_names,
#         NORMALIZATION_DICT_KEY: normalization_dict,
#         NORMALIZATION_DICT_TARG_KEY: normalization_dict_targ,
#         NUM_EXAMPLES_PER_BATCH_KEY: num_examples_per_batch,
#         NUM_TRAINING_BATCHES_KEY: num_training_batches_per_epoch,
#         VALIDATION_FILES_KEY: validation_file_names,
#         NUM_VALIDATION_BATCHES_KEY: num_validation_batches_per_epoch
#     }
    
#     if (targ_LATinds is None) & (targ_LONinds is None):
#         training_generator = deep_learning_generator(
#             netcdf_file_names=training_file_names,
#             num_examples_per_batch=num_examples_per_batch,
#             normalization_dict=normalization_dict,
#             normalization_dict_targ=normalization_dict_targ)
#     else:
#         training_generator = deep_learning_generator(
#             netcdf_file_names=training_file_names,
#             num_examples_per_batch=num_examples_per_batch,
#             normalization_dict=normalization_dict,
#             normalization_dict_targ=normalization_dict_targ,
#             targ_LATinds=targ_LATinds, targ_LONinds=targ_LONinds)

#     if validation_file_names is None:
#         cnn_model_object.fit_generator(
#             generator=training_generator,
#             steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
#             verbose=1, callbacks=list_of_callback_objects, workers=0)

#         return cnn_metadata_dict

#     early_stopping_object = keras.callbacks.EarlyStopping(
#         monitor='val_loss', min_delta=MIN_MSE_DECREASE_FOR_EARLY_STOP,
#         patience=NUM_EPOCHS_FOR_EARLY_STOPPING, verbose=1, mode='min')

#     list_of_callback_objects.append(early_stopping_object)
    
#     reduce_on_plateau_object = keras.callbacks.ReduceLROnPlateau(
#         monitor='val_loss', min_delta=MIN_MSE_DECREASE_FOR_REDUCE_LR,factor = FACTOR_REDUCE_LR,
#         patience=NUM_EPOCHS_FOR_REDUCE_LR, verbose=1, mode='min',min_lr=MIN_LR_REDUCE_TO)
    
#     list_of_callback_objects.append(reduce_on_plateau_object)

#     if (targ_LATinds is None) & (targ_LONinds is None):
#         validation_generator = deep_learning_generator(
#             netcdf_file_names=validation_file_names,
#             num_examples_per_batch=num_examples_per_batch,
#             normalization_dict=normalization_dict,
#             normalization_dict_targ=normalization_dict_targ)
#     else:
#         validation_generator = deep_learning_generator(
#             netcdf_file_names=validation_file_names,
#             num_examples_per_batch=num_examples_per_batch,
#             normalization_dict=normalization_dict,
#             normalization_dict_targ=normalization_dict_targ,
#             targ_LATinds=targ_LATinds, targ_LONinds=targ_LONinds)

#     with tensorflow.device("/device:GPU:0"):
#         K.get_session().run(tensorflow.global_variables_initializer())
#         cnn_model_object.fit_generator(
#             generator=training_generator,
#             steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
#             verbose=1, callbacks=list_of_callback_objects, workers=0,
#             validation_data=validation_generator,
#             validation_steps=num_validation_batches_per_epoch)

#     return cnn_metadata_dict




def deep_learning_generator_ms(netcdf_file_names, num_examples_per_batch,
                            normalization_dict,normalization_dict_targ,normalization_dict_post,
                            targ_LATinds=None,targ_LONinds=None):
    """Generates training examples for deep-learning model on the fly.
    ... mult stream version of the generator. 
    E = number of examples 
    M = number of rows in each grid (lats)
    N = number of columns in each grid (lons)
    C = number of channels (predictor variables)
    :param netcdf_file_names: 1-D list of paths to input (NetCDF) files.
    :param num_examples_per_batch: Number of examples per training batch.
    :param normalization_dict: See doc for `normalize_images`.  You cannot leave
        this as None.
    :param normalization_targ: See doc for `normalize_images_targ`.  You cannot leave
        this as None.
    :param normalization_post: See doc for `normalize_images_post`.  You cannot leave
        this as None.
    
    :return: predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :return: post_matrix: length-E numpy array of post values
    :return: target_values: length-E numpy array of target values (integers in
        0...1).
    :raises: TypeError: if `normalization_dict is None`.
    """

    if normalization_dict is None:
        error_string = 'normalization_dict cannot be None.  Must be specified.'
        raise TypeError(error_string)
        
    if normalization_dict_targ is None:
        error_string = 'normalization_dict_targ cannot be None.  Must be specified.'
        raise TypeError(error_string)
        
    if normalization_dict_post is None:
        error_string = 'normalization_dict_post cannot be None.  Must be specified.'
        raise TypeError(error_string)

    random.shuffle(netcdf_file_names)
    num_files = len(netcdf_file_names)
    file_index = 0

    num_examples_in_memory = 0
    full_predictor_matrix = None
    full_target_matrix = None
    full_post_matrix = None
    predictor_names = None
    targ_names = None
    post_names = None

    while True:
        while num_examples_in_memory < num_examples_per_batch:
            print('Reading data from: "{0:s}"...'.format(
                netcdf_file_names[file_index]
            ))
            
            if (targ_LATinds is None) & (targ_LONinds is None):
                this_image_dict = read_image_file(netcdf_file_names[file_index])
            else:
                this_image_dict = read_image_file(netcdf_file_names[file_index],targ_LATinds,targ_LONinds)
            
            predictor_names = this_image_dict[PREDICTOR_NAMES_KEY]
            targ_names = this_image_dict[TARGET_NAME_KEY]
            post_names = this_image_dict[POST_NAME_KEY]

            file_index += 1
            if file_index >= num_files:
                file_index = 0

            if full_target_matrix is None or full_target_matrix.size == 0:
                full_predictor_matrix = (
                    this_image_dict[PREDICTOR_MATRIX_KEY] + 0.
                )
                full_target_matrix = this_image_dict[TARGET_MATRIX_KEY] + 0.
                full_post_matrix = this_image_dict[POST_MATRIX_KEY] + 0.

            else:
                full_predictor_matrix = numpy.concatenate(
                    (full_predictor_matrix,
                     this_image_dict[PREDICTOR_MATRIX_KEY]),
                    axis=0
                )

                full_target_matrix = numpy.concatenate(
                    (full_target_matrix, this_image_dict[TARGET_MATRIX_KEY]),
                    axis=0
                )
                
                full_post_matrix = numpy.concatenate(
                    (full_post_matrix, this_image_dict[POST_MATRIX_KEY]),
                    axis=0
                )
                
            num_examples_in_memory = full_target_matrix.shape[0]

        batch_indices = numpy.linspace(
            0, num_examples_in_memory - 1, num=num_examples_in_memory,
            dtype=int)
        batch_indices = numpy.random.choice(
            batch_indices, size=num_examples_per_batch, replace=False)

        
        predictor_matrix, _ = normalize_images(
            predictor_matrix=full_predictor_matrix[batch_indices, ...],
            predictor_names=predictor_names,
            normalization_dict=normalization_dict)   
        predictor_matrix = predictor_matrix.astype('float32')
        
        target_values, _ = normalize_images_targ(
            targ_matrix=full_target_matrix[batch_indices, ...],
            targ_names = targ_names, 
            normalization_dict = normalization_dict_targ)
        target_values = target_values.astype('float32')
        
        post_matrix, _ = normalize_images_post(
            post_matrix=full_post_matrix[batch_indices, ...],
            post_names = post_names, 
            normalization_dict = normalization_dict_post)
        post_matrix = post_matrix.astype('float32')

        num_examples_in_memory = 0
        full_predictor_matrix = None
        full_target_matrix = None
        full_post_matrix = None

        yield [predictor_matrix, post_matrix], target_values
        

        

def train_cnn_ms(
        cnn_model_object, training_file_names, normalization_dict,
        normalization_dict_targ,normalization_dict_post,num_examples_per_batch, num_epochs,
        num_training_batches_per_epoch, output_model_file_name,
        validation_file_names=None, num_validation_batches_per_epoch=None,
    targ_LATinds=None, targ_LONinds=None):
    
    """Trains CNN (convolutional neural net).
    ... multi stream training with post process variable. 
    :param cnn_model_object: Untrained instance of `keras.models.Model` (may be
        created by `setup_cnn`).
    :param training_file_names: 1-D list of paths to training files (must be
        readable by `read_image_file`).
    :param normalization_dict: See doc for `deep_learning_generator_ms`.
    :param normalization_dict_targ: See doc for `deep_learning_generator_ms`.
    :param normalization_dict_post: See doc for `deep_learning_generator_ms`.
    :param num_examples_per_batch: Same.
    :param num_epochs: Number of epochs.
    :param num_training_batches_per_epoch: Number of training batches furnished
        to model in each epoch.
    :param output_model_file_name: Path to output file.  The model will be saved
        as an HDF5 file (extension should be ".h5", but this is not enforced).
    :param validation_file_names: 1-D list of paths to training files (must be
        readable by `read_image_file`).  If `validation_file_names is None`,
        will omit on-the-fly validation.
    :param num_validation_batches_per_epoch:
        [used only if `validation_file_names is not None`]
        Number of validation batches furnished to model in each epoch.
    :return: cnn_metadata_dict: Dictionary with the following keys.
    cnn_metadata_dict['training_file_names']: See input doc.
    cnn_metadata_dict['normalization_dict']: Same.
    cnn_metadata_dict['normalization_dict_targ']: Same.
    cnn_metadata_dict['num_examples_per_batch']: Same.
    cnn_metadata_dict['num_training_batches_per_epoch']: Same.
    cnn_metadata_dict['validation_file_names']: Same.
    cnn_metadata_dict['num_validation_batches_per_epoch']: Same.
    """
    
    #configure GPU: 
    config = tensorflow.compat.v1.ConfigProto(allow_soft_placement=False, log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tensorflow.compat.v1.Session(config=config)
    tensorflow.compat.v1.keras.backend.set_session(sess)
#     keras.backend.set_session(sess)


    _create_directory(file_name=output_model_file_name)

    if validation_file_names is None:
#         checkpoint_object = keras.callbacks.ModelCheckpoint(
#             filepath=output_model_file_name, monitor='loss', verbose=1,
#             save_best_only=False, save_weights_only=False, mode='min',
#             period=1)
        
        checkpoint_object = keras.callbacks.ModelCheckpoint(filepath=output_model_file_name, monitor='loss', verbose=1, save_best_only=False, save_weights_only=True, mode='min', period=1)

    else:
        
        checkpoint_object = keras.callbacks.ModelCheckpoint(filepath=output_model_file_name, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min', period=1)
        
#         checkpoint_object = keras.callbacks.ModelCheckpoint(
#             filepath=output_model_file_name, monitor='val_loss', verbose=1,
#             save_best_only=True, save_weights_only=False, mode='min',
#             period=1)

    list_of_callback_objects = [checkpoint_object]
    
    print('Normalization dict targ:', normalization_dict_targ)
    cnn_metadata_dict = {
        TRAINING_FILES_KEY: training_file_names,
        NORMALIZATION_DICT_KEY: normalization_dict,
        NORMALIZATION_DICT_TARG_KEY: normalization_dict_targ,
        NORMALIZATION_DICT_POST_KEY: normalization_dict_post,
        NUM_EXAMPLES_PER_BATCH_KEY: num_examples_per_batch,
        NUM_TRAINING_BATCHES_KEY: num_training_batches_per_epoch,
        VALIDATION_FILES_KEY: validation_file_names,
        NUM_VALIDATION_BATCHES_KEY: num_validation_batches_per_epoch
    }
    
    if (targ_LATinds is None) & (targ_LONinds is None):
        training_generator = deep_learning_generator_ms(
            netcdf_file_names=training_file_names,
            num_examples_per_batch=num_examples_per_batch,
            normalization_dict=normalization_dict,
            normalization_dict_targ=normalization_dict_targ,
            normalization_dict_post=normalization_dict_post)
    else:
        training_generator = deep_learning_generator_ms(
            netcdf_file_names=training_file_names,
            num_examples_per_batch=num_examples_per_batch,
            normalization_dict=normalization_dict,
            normalization_dict_targ=normalization_dict_targ,
            normalization_dict_post=normalization_dict_post,
            targ_LATinds=targ_LATinds, targ_LONinds=targ_LONinds)

    if validation_file_names is None:
        cnn_model_object.fit_generator(
            generator=training_generator,
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, callbacks=list_of_callback_objects, workers=0)

        return cnn_metadata_dict

    early_stopping_object = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=MIN_MSE_DECREASE_FOR_EARLY_STOP,
        patience=NUM_EPOCHS_FOR_EARLY_STOPPING, verbose=1, mode='min')

    list_of_callback_objects.append(early_stopping_object)
    
    reduce_on_plateau_object = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', min_delta=MIN_MSE_DECREASE_FOR_REDUCE_LR,factor = FACTOR_REDUCE_LR,
        patience=NUM_EPOCHS_FOR_REDUCE_LR, verbose=1, mode='min',min_lr=MIN_LR_REDUCE_TO)
    
    list_of_callback_objects.append(reduce_on_plateau_object)

    if (targ_LATinds is None) & (targ_LONinds is None):
        validation_generator = deep_learning_generator_ms(
            netcdf_file_names=validation_file_names,
            num_examples_per_batch=num_examples_per_batch,
            normalization_dict=normalization_dict,
            normalization_dict_targ=normalization_dict_targ,
            normalization_dict_post=normalization_dict_post)
    else:
        validation_generator = deep_learning_generator_ms(
            netcdf_file_names=validation_file_names,
            num_examples_per_batch=num_examples_per_batch,
            normalization_dict=normalization_dict,
            normalization_dict_targ=normalization_dict_targ,
            normalization_dict_post=normalization_dict_post,
            targ_LATinds=targ_LATinds, targ_LONinds=targ_LONinds)

    with tensorflow.device("/device:GPU:0"):
#         keras.backend.get_session().run(tensorflow.global_variables_initializer())
        tensorflow.compat.v1.global_variables_initializer()
        print('YO!:',num_validation_batches_per_epoch)
        print('YO!:',num_training_batches_per_epoch)
        print('YO!:',num_epochs)
        print('YO!:',training_generator)
        
        cnn_model_object.fit_generator(
            generator=training_generator,
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, callbacks=list_of_callback_objects, workers=0,
            validation_data=validation_generator,
            validation_steps=num_validation_batches_per_epoch)

    return cnn_metadata_dict



def _create_directory(directory_name=None, file_name=None):
    """Creates directory (along with parents if necessary).
    This method creates directories only when necessary, so you don't have to
    worry about it overwriting anything.
    :param directory_name: Name of desired directory.
    :param file_name: [used only if `directory_name is None`]
        Path to desired file.  All directories in path will be created.
    """

    if directory_name is None:
        directory_name = os.path.split(file_name)[0]

    try:
        os.makedirs(directory_name)
    except OSError as this_error:
        if this_error.errno == errno.EEXIST and os.path.isdir(directory_name):
            pass
        else:
            raise
            
            
            
            
# def apply_cnn(cnn_model_object, predictor_matrix, verbose=True,
#               output_layer_name=None):
#     """Applies trained CNN (convolutional neural net) to new data.
#     E = number of examples in file
#     M = number of rows in each grid
#     N = number of columns in each grid
#     C = number of channels (predictor variables)
#     :param cnn_model_object: Trained instance of `keras.models.Model`.
#     :param predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
#     :param verbose: Boolean flag.  If True, progress messages will be printed.
#     :param output_layer_name: Name of output layer.  If
#         `output_layer_name is None`, this method will use the actual output
#         layer, so will return predictions.  If `output_layer_name is not None`,
#         will return "features" (outputs from the given layer).
#     If `output_layer_name is None`...
#     :return: forecast_probabilities: length-E numpy array with forecast
#         probabilities of positive class (label = 1).
#     If `output_layer_name is not None`...
#     :return: feature_matrix: numpy array of features (outputs from the given
#         layer).  There is no guarantee on the shape of this array, except that
#         the first axis has length E.
#     """

#     num_examples = predictor_matrix.shape[0]
#     num_examples_per_batch = 1000

#     if output_layer_name is None:
#         model_object_to_use = cnn_model_object
#     else:
#         model_object_to_use = keras.models.Model(
#             inputs=cnn_model_object.input,
#             outputs=cnn_model_object.get_layer(name=output_layer_name).output
#         )

#     output_array = None

#     for i in range(0, num_examples, num_examples_per_batch):
#         this_first_index = i
#         this_last_index = min(
#             [i + num_examples_per_batch - 1, num_examples - 1]
#         )

#         if verbose:
#             print('Applying model to examples {0:d}-{1:d} of {2:d}...'.format(
#                 this_first_index, this_last_index, num_examples
#             ))

#         these_indices = numpy.linspace(
#             this_first_index, this_last_index,
#             num=this_last_index - this_first_index + 1, dtype=int)

#         this_output_array = model_object_to_use.predict(
#             predictor_matrix[these_indices, ...],
#             batch_size=num_examples_per_batch)

#         if output_layer_name is None:
#             this_output_array = this_output_array[:, -1]

#         if output_array is None:
#             output_array = this_output_array + 0.
#         else:
#             output_array = numpy.concatenate(
#                 (output_array, this_output_array), axis=0
#             )

#     return output_array


def get_latlon_ind(latlonfolder):
    ##gets the lat lon indices from the desired folder. 
    ##good to compare with the AnEn method. 

    latfind=numpy.array([])
    lonfind=numpy.array([])
    latind=numpy.array([])
    lonind=numpy.array([])
    latlonfolder=latlonfolder+'/lat*lon*'
    allfolds = glob.glob(latlonfolder)
    allfolds = sorted(allfolds)
    for ff in allfolds:
        ff2 = ff.split('lat')[1]
        ff3 = (ff2.split('lon'))
        
        latfind = numpy.append(latfind,float(ff3[0]))
        lonfind = numpy.append(lonfind,float(ff3[1])-360)

    latm = netCDF4.Dataset(exampfil).variables['lat'][:]
    lonm = netCDF4.Dataset(exampfil).variables['lon'][:]

    for ii in range(latfind.shape[0]):
        lage = latfind[ii]
        loge = lonfind[ii]
        
        if not numpy.where(loge==lonm)[0]:
            continue
        
        if not numpy.where(lage==latm)[0]:
            continue
        
        latind = numpy.append(latind,(numpy.where(lage==latm)[0]))
        lonind = numpy.append(lonind,(numpy.where(loge==lonm)[0]))

    latind = list(latind.astype(int))
    lonind = list(lonind.astype(int))

    return latfind, lonfind, latind, lonind


def get_latlon_ind_gefs(latlonfolder):
    ##gets the lat lon indices from the desired folder. 
    ##good to compare with the AnEn method. 

    latfind=numpy.array([])
    lonfind=numpy.array([])
    latind=numpy.array([])
    lonind=numpy.array([])
    latlonfolder=latlonfolder+'/lat*lon*'
    allfolds = glob.glob(latlonfolder)
    allfolds = sorted(allfolds)
    for ff in allfolds:
        ff2 = ff.split('lat')[1]
        ff3 = (ff2.split('lon'))
        
        latfind = numpy.append(latfind,float(ff3[0]))
        lonfind = numpy.append(lonfind,float(ff3[1])-360)

    latm = netCDF4.Dataset(exampfilgefs).variables['lat'][:].data
    lonm = netCDF4.Dataset(exampfilgefs).variables['lon'][:].data
    
    lonm = lonm-360
#     print('latfind shape',latfind.shape[0])
#     print('lonner:',lonm)
#     print('lon',lonm)
#     print('lat',latm)
#     print(latfind.shape[0])
    for ii in range(latfind.shape[0]):
        lage = latfind[ii]
        loge = lonfind[ii]
#         print(lage)
        
        if len(numpy.where(loge==lonm)[0])==0:
            print(numpy.where(loge==lonm)[0])
            print('length:',len(numpy.where(loge==lonm)[0] ))
            print('skipped',loge)
            print('skipped',lage)
            continue
        
        if len(numpy.where(lage==latm)[0])==0:
            continue
        
        latind = numpy.append(latind,(numpy.where(lage==latm)[0]))
        lonind = numpy.append(lonind,(numpy.where(loge==lonm)[0]))

    latind = list(latind.astype(int))
    lonind = list(lonind.astype(int))

    return latfind, lonfind, latind, lonind




def deep_learning_generator_ss(netcdf_file_names, num_examples_per_batch,
                            normalization_dict,normalization_dict_targ,normalization_dict_post,
                            targ_LATinds=None,targ_LONinds=None):
    """Generates training examples for deep-learning model on the fly.
    ... mult stream version of the generator. 
    E = number of examples 
    M = number of rows in each grid (lats)
    N = number of columns in each grid (lons)
    C = number of channels (predictor variables)
    :param netcdf_file_names: 1-D list of paths to input (NetCDF) files.
    :param num_examples_per_batch: Number of examples per training batch.
    :param normalization_dict: See doc for `normalize_images`.  You cannot leave
        this as None.
    :param normalization_targ: See doc for `normalize_images_targ`.  You cannot leave
        this as None.
    :param normalization_post: See doc for `normalize_images_post`.  You cannot leave
        this as None.
    
    :return: predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :return: post_matrix: length-E numpy array of post values
    :return: target_values: length-E numpy array of target values (integers in
        0...1).
    :raises: TypeError: if `normalization_dict is None`.
    """

    if normalization_dict is None:
        error_string = 'normalization_dict cannot be None.  Must be specified.'
        raise TypeError(error_string)
        
    if normalization_dict_targ is None:
        error_string = 'normalization_dict_targ cannot be None.  Must be specified.'
        raise TypeError(error_string)
        
    if normalization_dict_post is None:
        error_string = 'normalization_dict_post cannot be None.  Must be specified.'
        raise TypeError(error_string)

    random.shuffle(netcdf_file_names)
    num_files = len(netcdf_file_names)
    file_index = 0

    num_examples_in_memory = 0
    full_predictor_matrix = None
    full_target_matrix = None
    full_post_matrix = None
    predictor_names = None
    targ_names = None
    post_names = None

    while True:
        while num_examples_in_memory < num_examples_per_batch:
            print('Reading data from: "{0:s}"...'.format(
                netcdf_file_names[file_index]
            ))
            
            if (targ_LATinds is None) & (targ_LONinds is None):
                this_image_dict = read_image_file(netcdf_file_names[file_index])
            else:
                this_image_dict = read_image_file(netcdf_file_names[file_index],targ_LATinds,targ_LONinds)
            
            predictor_names = this_image_dict[PREDICTOR_NAMES_KEY]
            targ_names = this_image_dict[TARGET_NAME_KEY]
            post_names = this_image_dict[POST_NAME_KEY]

            file_index += 1
            if file_index >= num_files:
                file_index = 0

            if full_target_matrix is None or full_target_matrix.size == 0:
                full_predictor_matrix = (
                    this_image_dict[PREDICTOR_MATRIX_KEY] + 0.
                )
                full_target_matrix = this_image_dict[TARGET_MATRIX_KEY] + 0.
                full_post_matrix = this_image_dict[POST_MATRIX_KEY] + 0.

            else:
                full_predictor_matrix = numpy.concatenate(
                    (full_predictor_matrix,
                     this_image_dict[PREDICTOR_MATRIX_KEY]),
                    axis=0
                )

                full_target_matrix = numpy.concatenate(
                    (full_target_matrix, this_image_dict[TARGET_MATRIX_KEY]),
                    axis=0
                )
                
                full_post_matrix = numpy.concatenate(
                    (full_post_matrix, this_image_dict[POST_MATRIX_KEY]),
                    axis=0
                )
                
            num_examples_in_memory = full_target_matrix.shape[0]

        batch_indices = numpy.linspace(
            0, num_examples_in_memory - 1, num=num_examples_in_memory,
            dtype=int)
        batch_indices = numpy.random.choice(
            batch_indices, size=num_examples_per_batch, replace=False)

        
        predictor_matrix, _ = normalize_images(
            predictor_matrix=full_predictor_matrix[batch_indices, ...],
            predictor_names=predictor_names,
            normalization_dict=normalization_dict)   
        predictor_matrix = predictor_matrix.astype('float32')
        
        target_values, _ = normalize_images_targ(
            targ_matrix=full_target_matrix[batch_indices, ...],
            targ_names = targ_names, 
            normalization_dict = normalization_dict_targ)
        target_values = target_values.astype('float32')
        
        post_matrix, _ = normalize_images_post(
            post_matrix=full_post_matrix[batch_indices, ...],
            post_names = post_names, 
            normalization_dict = normalization_dict_post)
        post_matrix = post_matrix.astype('float32')

        num_examples_in_memory = 0
        full_predictor_matrix = None
        full_target_matrix = None
        full_post_matrix = None

        yield post_matrix, target_values
        
        
        
def train_cnn_ss(
        cnn_model_object, training_file_names, normalization_dict,
        normalization_dict_targ,normalization_dict_post,num_examples_per_batch, num_epochs,
        num_training_batches_per_epoch, output_model_file_name,
        validation_file_names=None, num_validation_batches_per_epoch=None,
    targ_LATinds=None, targ_LONinds=None):
    
    """Trains CNN (convolutional neural net).
    ... multi stream training with post process variable. 
    :param cnn_model_object: Untrained instance of `keras.models.Model` (may be
        created by `setup_cnn`).
    :param training_file_names: 1-D list of paths to training files (must be
        readable by `read_image_file`).
    :param normalization_dict: See doc for `deep_learning_generator_ms`.
    :param normalization_dict_targ: See doc for `deep_learning_generator_ms`.
    :param normalization_dict_post: See doc for `deep_learning_generator_ms`.
    :param num_examples_per_batch: Same.
    :param num_epochs: Number of epochs.
    :param num_training_batches_per_epoch: Number of training batches furnished
        to model in each epoch.
    :param output_model_file_name: Path to output file.  The model will be saved
        as an HDF5 file (extension should be ".h5", but this is not enforced).
    :param validation_file_names: 1-D list of paths to training files (must be
        readable by `read_image_file`).  If `validation_file_names is None`,
        will omit on-the-fly validation.
    :param num_validation_batches_per_epoch:
        [used only if `validation_file_names is not None`]
        Number of validation batches furnished to model in each epoch.
    :return: cnn_metadata_dict: Dictionary with the following keys.
    cnn_metadata_dict['training_file_names']: See input doc.
    cnn_metadata_dict['normalization_dict']: Same.
    cnn_metadata_dict['normalization_dict_targ']: Same.
    cnn_metadata_dict['num_examples_per_batch']: Same.
    cnn_metadata_dict['num_training_batches_per_epoch']: Same.
    cnn_metadata_dict['validation_file_names']: Same.
    cnn_metadata_dict['num_validation_batches_per_epoch']: Same.
    """
    
    #configure GPU: 
    config = tensorflow.compat.v1.ConfigProto(allow_soft_placement=False, log_device_placement=False)
    config.gpu_options.allow_growth = True
    sess = tensorflow.compat.v1.Session(config=config)
    tensorflow.compat.v1.keras.backend.set_session(sess)
#     keras.backend.set_session(sess)


    _create_directory(file_name=output_model_file_name)

    if validation_file_names is None:
#         checkpoint_object = keras.callbacks.ModelCheckpoint(
#             filepath=output_model_file_name, monitor='loss', verbose=1,
#             save_best_only=False, save_weights_only=False, mode='min',
#             period=1)
        
        checkpoint_object = keras.callbacks.ModelCheckpoint(filepath=output_model_file_name, monitor='loss', verbose=1, save_best_only=False, save_weights_only=True, mode='min', period=1)

    else:
        
        checkpoint_object = keras.callbacks.ModelCheckpoint(filepath=output_model_file_name, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min', period=1)
        
#         checkpoint_object = keras.callbacks.ModelCheckpoint(
#             filepath=output_model_file_name, monitor='val_loss', verbose=1,
#             save_best_only=True, save_weights_only=False, mode='min',
#             period=1)

    list_of_callback_objects = [checkpoint_object]
    
    print('Normalization dict targ:', normalization_dict_targ)
    cnn_metadata_dict = {
        TRAINING_FILES_KEY: training_file_names,
        NORMALIZATION_DICT_KEY: normalization_dict,
        NORMALIZATION_DICT_TARG_KEY: normalization_dict_targ,
        NORMALIZATION_DICT_POST_KEY: normalization_dict_post,
        NUM_EXAMPLES_PER_BATCH_KEY: num_examples_per_batch,
        NUM_TRAINING_BATCHES_KEY: num_training_batches_per_epoch,
        VALIDATION_FILES_KEY: validation_file_names,
        NUM_VALIDATION_BATCHES_KEY: num_validation_batches_per_epoch
    }
    
    if (targ_LATinds is None) & (targ_LONinds is None):
        training_generator = deep_learning_generator_ss(
            netcdf_file_names=training_file_names,
            num_examples_per_batch=num_examples_per_batch,
            normalization_dict=normalization_dict,
            normalization_dict_targ=normalization_dict_targ,
            normalization_dict_post=normalization_dict_post)
    else:
        training_generator = deep_learning_generator_ss(
            netcdf_file_names=training_file_names,
            num_examples_per_batch=num_examples_per_batch,
            normalization_dict=normalization_dict,
            normalization_dict_targ=normalization_dict_targ,
            normalization_dict_post=normalization_dict_post,
            targ_LATinds=targ_LATinds, targ_LONinds=targ_LONinds)

    if validation_file_names is None:
        cnn_model_object.fit_generator(
            generator=training_generator,
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, callbacks=list_of_callback_objects, workers=0)

        return cnn_metadata_dict

    early_stopping_object = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=MIN_MSE_DECREASE_FOR_EARLY_STOP,
        patience=NUM_EPOCHS_FOR_EARLY_STOPPING, verbose=1, mode='min')

    list_of_callback_objects.append(early_stopping_object)
    
    reduce_on_plateau_object = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', min_delta=MIN_MSE_DECREASE_FOR_REDUCE_LR,factor = FACTOR_REDUCE_LR,
        patience=NUM_EPOCHS_FOR_REDUCE_LR, verbose=1, mode='min',min_lr=MIN_LR_REDUCE_TO)
    
    list_of_callback_objects.append(reduce_on_plateau_object)

    if (targ_LATinds is None) & (targ_LONinds is None):
        validation_generator = deep_learning_generator_ss(
            netcdf_file_names=validation_file_names,
            num_examples_per_batch=num_examples_per_batch,
            normalization_dict=normalization_dict,
            normalization_dict_targ=normalization_dict_targ,
            normalization_dict_post=normalization_dict_post)
    else:
        validation_generator = deep_learning_generator_ss(
            netcdf_file_names=validation_file_names,
            num_examples_per_batch=num_examples_per_batch,
            normalization_dict=normalization_dict,
            normalization_dict_targ=normalization_dict_targ,
            normalization_dict_post=normalization_dict_post,
            targ_LATinds=targ_LATinds, targ_LONinds=targ_LONinds)

    with tensorflow.device("/device:GPU:0"):
#         keras.backend.get_session().run(tensorflow.global_variables_initializer())
        tensorflow.compat.v1.global_variables_initializer()
        print('YO!:',num_validation_batches_per_epoch)
        print('YO!:',num_training_batches_per_epoch)
        print('YO!:',num_epochs)
        print('YO!:',training_generator)
        
        cnn_model_object.fit_generator(
            generator=training_generator,
            steps_per_epoch=num_training_batches_per_epoch, epochs=num_epochs,
            verbose=1, callbacks=list_of_callback_objects, workers=0,
            validation_data=validation_generator,
            validation_steps=num_validation_batches_per_epoch)

    return cnn_metadata_dict



def deep_learning_generator_ss_mv(netcdf_file_names, num_examples_per_batch,
                            normalization_dict,normalization_dict_targ,
                            targ_LATinds=None,targ_LONinds=None):
    """Generates training examples for deep-learning model on the fly.
    ... mult stream version of the generator. 
    E = number of examples 
    M = number of rows in each grid (lats)
    N = number of columns in each grid (lons)
    C = number of channels (predictor variables)
    :param netcdf_file_names: 1-D list of paths to input (NetCDF) files.
    :param num_examples_per_batch: Number of examples per training batch.
    :param normalization_dict: See doc for `normalize_images`.  You cannot leave
        this as None.
    :param normalization_targ: See doc for `normalize_images_targ`.  You cannot leave
        this as None.
    :param normalization_post: See doc for `normalize_images_post`.  You cannot leave
        this as None.
    
    :return: predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :return: post_matrix: length-E numpy array of post values
    :return: target_values: length-E numpy array of target values (integers in
        0...1).
    :raises: TypeError: if `normalization_dict is None`.
    """

    if normalization_dict is None:
        error_string = 'normalization_dict cannot be None.  Must be specified.'
        raise TypeError(error_string)
        
    if normalization_dict_targ is None:
        error_string = 'normalization_dict_targ cannot be None.  Must be specified.'
        raise TypeError(error_string)

    random.shuffle(netcdf_file_names)
    num_files = len(netcdf_file_names)
    file_index = 0

    num_examples_in_memory = 0
    full_predictor_matrix = None
    full_target_matrix = None
    full_post_matrix = None
    predictor_names = None
    targ_names = None
    post_names = None
    
    
    if (targ_LATinds is None) & (targ_LONinds is None):
        num_examples_per_batch = num_examples_per_batch
    else:
        num_examples_per_batch = int(num_examples_per_batch/targ_LONinds.shape[0])
            
    
    while True:
        while num_examples_in_memory < num_examples_per_batch:
            print('Reading data from: "{0:s}"...'.format(
                netcdf_file_names[file_index]
            ))
            
            if (targ_LATinds is None) & (targ_LONinds is None):
                this_image_dict = read_image_file(netcdf_file_names[file_index])
            else:
                this_image_dict = read_image_file(netcdf_file_names[file_index],targ_LATinds,targ_LONinds)
            
            predictor_names = this_image_dict[PREDICTOR_NAMES_KEY]
            targ_names = this_image_dict[TARGET_NAME_KEY]

            file_index += 1
            if file_index >= num_files:
                file_index = 0

            if full_target_matrix is None or full_target_matrix.size == 0:
                full_predictor_matrix = (
                    this_image_dict[PREDICTOR_MATRIX_KEY] + 0.
                )
                full_target_matrix = this_image_dict[TARGET_MATRIX_KEY] + 0.

            else:
                full_predictor_matrix = numpy.concatenate(
                    (full_predictor_matrix,
                     this_image_dict[PREDICTOR_MATRIX_KEY]),
                    axis=0
                )

                full_target_matrix = numpy.concatenate(
                    (full_target_matrix, this_image_dict[TARGET_MATRIX_KEY]),
                    axis=0
                )
                
                
            num_examples_in_memory = full_target_matrix.shape[0]

        batch_indices = numpy.linspace(
            0, num_examples_in_memory - 1, num=num_examples_in_memory,
            dtype=int)
        batch_indices = numpy.random.choice(
            batch_indices, size=num_examples_per_batch, replace=False)

        
        predictor_matrix, _ = normalize_images(
            predictor_matrix=full_predictor_matrix[batch_indices, ...],
            predictor_names=predictor_names,
            normalization_dict=normalization_dict)   
        predictor_matrix = predictor_matrix.astype('float32')
        
        target_values, _ = normalize_images_targ(
            targ_matrix=full_target_matrix[batch_indices, ...],
            targ_names = targ_names, 
            normalization_dict = normalization_dict_targ)
        target_values = target_values.astype('float32')

        num_examples_in_memory = 0
        full_predictor_matrix = None
        full_target_matrix = None
        full_post_matrix = None
        
        if len(predictor_matrix.shape)>2:
            predictor_matrix = numpy.reshape(predictor_matrix,(predictor_matrix.shape[0]*predictor_matrix.shape[1],predictor_matrix.shape[-1]))
            target_values = numpy.reshape(target_values,(target_values.shape[0]*target_values.shape[1],1))

        yield predictor_matrix,target_values
        
        
        
def read_image_file_ss(netcdf_file_name,targ_LATinds=None,targ_LONinds=None):
    """Reads AR images from NetCDF file.
    E = number of examples in file
    M = number of rows in each storm-centered grid (lats)
    N = number of columns in each storm-centered grid (lons)
    C = number of channels (predictor variables) 
    targ_inds = list indices of target matrix, you'd like to train on. must be list
    
    :param netcdf_file_name: Path to input file.
    
    :return: image_dict: Dictionary with the following keys.
    image_dict['storm_ids']: length-E list of storm IDs (integers).
    image_dict['storm_steps']: length-E numpy array of storm steps (integers).
    image_dict['predictor_names']: length-C list of predictor names.
    image_dict['predictor_matrix']: E-by-M-by-N-by-C numpy array of predictor
        values.
    image_dict['target_name']: Name of target variable.
    image_dict['target_matrix']: E-by-M-by-N numpy array of target values.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)
    
    predictor_matrix = None

    for this_predictor_name in NETCDF_PREDICTOR_NAMES:
        this_predictor_matrix = numpy.array(
            dataset_object.variables[this_predictor_name][:,targ_LATinds,targ_LONinds], dtype=float
        )
        this_predictor_matrix = numpy.expand_dims(
            this_predictor_matrix, axis=-1)

        if predictor_matrix is None:
            predictor_matrix = this_predictor_matrix + 0.
        else:
            predictor_matrix = numpy.concatenate(
                (predictor_matrix, this_predictor_matrix), axis=-1
            )
        
    pred_name_dict = {PREDICTOR_NAMES_KEY: PREDICTOR_NAMES}
    pred_mat_dict = {PREDICTOR_MATRIX_KEY: predictor_matrix}
        
    indPOST = pred_name_dict[PREDICTOR_NAMES_KEY].index(POST_NAME_KEY)
    
    if targ_LATinds is None:
        target_matrix = numpy.array(
            dataset_object.variables[NETCDF_TARGET_NAME][:], dtype=float
        )
        post_matrix = predictor_matrix[indPOST,:,:]
        
    else:
        target_matrix = numpy.array(
            dataset_object.variables[NETCDF_TARGET_NAME][:], dtype=float
        )
        target_matrix = target_matrix[:,targ_LATinds,targ_LONinds]
        target_matrix = numpy.reshape(target_matrix,(target_matrix.shape[0],-1))

    #take logs of all IVT. 
    indicesPRED = [iii for iii, sss in enumerate(PREDICTOR_NAMES) if 'IVT' in sss]
    indicesTARG = [iii for iii, sss in enumerate([TARGET_NAME]) if 'IVT' in sss]
    indicesPOST = [iii for iii, sss in enumerate([POST_NAME]) if 'IVT' in sss]
    
#     if indicesPRED:
#         predictor_matrix[:,:,:,indicesPRED] = numpy.log(predictor_matrix[:,:,:,indicesPRED])
        
#     if indicesTARG:
#         target_matrix = numpy.log(target_matrix)
    
    predictor_matrix = numpy.squeeze(predictor_matrix)
    return {
        PREDICTOR_NAMES_KEY: PREDICTOR_NAMES,
        PREDICTOR_MATRIX_KEY: predictor_matrix,
        TARGET_NAME_KEY: TARGET_NAME,
        TARGET_MATRIX_KEY: target_matrix,
    }





def read_image_file_time(netcdf_file_name,targ_LATinds=None,targ_LONinds=None):
    """Reads AR images from NetCDF file.
    E = number of examples in file
    M = number of rows in each storm-centered grid (lats)
    N = number of columns in each storm-centered grid (lons)
    C = number of channels (predictor variables) 
    targ_inds = list indices of target matrix, you'd like to train on. must be list
    
    :param netcdf_file_name: Path to input file.
    
    :return: image_dict: Dictionary with the following keys.
    image_dict['storm_ids']: length-E list of storm IDs (integers).
    image_dict['storm_steps']: length-E numpy array of storm steps (integers).
    image_dict['predictor_names']: length-C list of predictor names.
    image_dict['predictor_matrix']: E-by-M-by-N-by-C numpy array of predictor
        values.
    image_dict['target_name']: Name of target variable.
    image_dict['target_matrix']: E-by-M-by-N numpy array of target values.
    """

    dataset_object = netCDF4.Dataset(netcdf_file_name)
    daters =netCDF4.Dataset(netcdf_file_name).variables[NETCDF_time_NAME]
    daters = netCDF4.num2date(daters[:],units=daters.units,calendar=daters.calendar)
    
    datdat = []
    for dd in daters:
        datdat.append(time.strftime(str(dd))[0:10])
    
    predictor_matrix = None

    for this_predictor_name in NETCDF_PREDICTOR_NAMES:
        this_predictor_matrix = numpy.array(
            dataset_object.variables[this_predictor_name][:], dtype=float
        )
        
        if (this_predictor_name == 'lat'):
            this_predictor_matrix_lat= this_predictor_matrix[targ_LATinds]
            LL = None
            
            if isinstance(this_predictor_matrix_lat, float):
                this_predictor_matrix_lat = [this_predictor_matrix_lat]
            
            for pp in range(len(this_predictor_matrix_lat)):
                thisLat = this_predictor_matrix_lat[pp] 
                thisLat = numpy.expand_dims(numpy.tile(thisLat,predictor_matrix.shape[0]),axis=1)

                if LL is None: 
                    LL = thisLat + 0.
                else:
                    LL = numpy.concatenate((LL,thisLat),axis=-1)
            
            
            LL = numpy.squeeze(LL)
            LL = numpy.expand_dims(LL,axis=-1)
            
            predictor_matrix = numpy.concatenate(
                (predictor_matrix, LL), axis=-1)
            
  
        elif (this_predictor_name=='lon'):
            this_predictor_matrix_lon= this_predictor_matrix[targ_LONinds]
            LLo = None
            if isinstance(this_predictor_matrix_lon, float):
                this_predictor_matrix_lon = [this_predictor_matrix_lon]
                
            for pp in range(len(this_predictor_matrix_lon)):
                thisLon = this_predictor_matrix_lon[pp]
                thisLon = numpy.expand_dims(numpy.tile(thisLon,predictor_matrix.shape[0]),axis=1)
                if LLo is None: 
                    LLo = thisLon + 0.
                else: 
                    LLo = numpy.concatenate((LLo,thisLon),axis=-1)
            
            LLo = numpy.squeeze(LLo)
            LLo = numpy.expand_dims(LLo,axis=-1)
            predictor_matrix = numpy.concatenate(
                (predictor_matrix, LLo), axis=-1)
            
        else:
            this_predictor_matrix=this_predictor_matrix[:,targ_LATinds,targ_LONinds]
            this_predictor_matrix = numpy.expand_dims(this_predictor_matrix, axis=-1)

            if predictor_matrix is None:
                predictor_matrix = this_predictor_matrix + 0.
            
            else:
                predictor_matrix = numpy.concatenate(
                    (predictor_matrix, this_predictor_matrix), axis=-1
                
                )
                
            
            
    pred_name_dict = {PREDICTOR_NAMES_KEY: PREDICTOR_NAMES}
    pred_mat_dict = {PREDICTOR_MATRIX_KEY: predictor_matrix}
        
    indPOST = pred_name_dict[PREDICTOR_NAMES_KEY].index(POST_NAME_KEY)
    
    if targ_LATinds is None:
        target_matrix = numpy.array(
            dataset_object.variables[NETCDF_TARGET_NAME][:], dtype=float
        )
        post_matrix = predictor_matrix[indPOST,:,:]
        
    else:
        target_matrix = numpy.array(
            dataset_object.variables[NETCDF_TARGET_NAME][:], dtype=float
        )
        target_matrix = target_matrix[:,targ_LATinds,targ_LONinds]
        target_matrix = numpy.reshape(target_matrix,(target_matrix.shape[0],-1))
    
    
    #take logs of all IVT. 
    ##turn off the logging

    indicesPRED = [iii for iii, sss in enumerate(PREDICTOR_NAMES) if 'IVT' in sss]
    indicesTARG = [iii for iii, sss in enumerate([TARGET_NAME]) if 'IVT' in sss]    
    
#     ##turn off the logging
#     if indicesPRED:
#         tmp = numpy.log(predictor_matrix[...,indicesPRED])
#         tmp[numpy.isinf(tmp)]=0
#         predictor_matrix[...,indicesPRED] = tmp
        
#     if indicesTARG:
#         target_matrix = numpy.log(target_matrix)
#     predictor_matrix = numpy.squeeze(predictor_matrix)
    
    
    dicter = {
        PREDICTOR_NAMES_KEY: PREDICTOR_NAMES,
        PREDICTOR_MATRIX_KEY: predictor_matrix,
        TARGET_NAME_KEY: TARGET_NAME,
        TARGET_MATRIX_KEY: target_matrix,
    } 
    return dicter,datdat




def deep_learning_generator_ss_mv_time(netcdf_file_names, num_examples_per_batch,
                            normalization_dict,normalization_dict_targ,
                            targ_LATinds=None,targ_LONinds=None):
    """Generates training examples for deep-learning model on the fly.
    ... mult stream version of the generator. 
    E = number of examples 
    M = number of rows in each grid (lats)
    N = number of columns in each grid (lons)
    C = number of channels (predictor variables)
    :param netcdf_file_names: 1-D list of paths to input (NetCDF) files.
    :param num_examples_per_batch: Number of examples per training batch.
    :param normalization_dict: See doc for `normalize_images`.  You cannot leave
        this as None.
    :param normalization_targ: See doc for `normalize_images_targ`.  You cannot leave
        this as None.
    :param normalization_post: See doc for `normalize_images_post`.  You cannot leave
        this as None.
    
    :return: predictor_matrix: E-by-M-by-N-by-C numpy array of predictor values.
    :return: post_matrix: length-E numpy array of post values
    :return: target_values: length-E numpy array of target values (integers in
        0...1).
    :raises: TypeError: if `normalization_dict is None`.
    """

    if normalization_dict is None:
        error_string = 'normalization_dict cannot be None.  Must be specified.'
        raise TypeError(error_string)
        
    if normalization_dict_targ is None:
        error_string = 'normalization_dict_targ cannot be None.  Must be specified.'
        raise TypeError(error_string)

    random.shuffle(netcdf_file_names)
    num_files = len(netcdf_file_names)
    file_index = 0

    num_examples_in_memory = 0
    full_predictor_matrix = None
    full_target_matrix = None
    full_post_matrix = None
    predictor_names = None
    targ_names = None
    post_names = None
    
    times_pd = pandas.date_range(start=trainst, end=trainen)
    times_pd =times_pd.strftime('%Y-%m-%d')
    tmvec = numpy.linspace(-2, 2,len(times_pd))

    
    if (targ_LATinds is None) & (targ_LONinds is None):
        num_examples_per_batch = num_examples_per_batch
    else:
        num_examples_per_batch = int(num_examples_per_batch/targ_LONinds.shape[0])
            
    
    while True:
        while num_examples_in_memory < num_examples_per_batch:
            print('Reading data from: "{0:s}"...'.format(
                netcdf_file_names[file_index]
            ))
            
            if (targ_LATinds is None) & (targ_LONinds is None):
                [this_image_dict,timy] = read_image_file_time(netcdf_file_names[file_index])
            else:
                [this_image_dict,timy] = read_image_file_time(netcdf_file_names[file_index],targ_LATinds,targ_LONinds)
            
            predictor_names = this_image_dict[PREDICTOR_NAMES_KEY]
            targ_names = this_image_dict[TARGET_NAME_KEY]

            file_index += 1
            if file_index >= num_files:
                file_index = 0

            if full_target_matrix is None or full_target_matrix.size == 0:
                full_predictor_matrix = (
                    this_image_dict[PREDICTOR_MATRIX_KEY] + 0.
                )
                full_target_matrix = this_image_dict[TARGET_MATRIX_KEY] + 0.
                
                this_timy = timy
                
                en_tim = numpy.zeros([len(timy),1])
                for cc,sd in enumerate(timy): 
                    inds = sd==times_pd
                    en_tim[cc,:] = tmvec[inds]
                this_timy_ind = en_tim

            else:
                full_predictor_matrix = numpy.concatenate(
                    (full_predictor_matrix,
                     this_image_dict[PREDICTOR_MATRIX_KEY]),
                    axis=0
                )

                full_target_matrix = numpy.concatenate(
                    (full_target_matrix, this_image_dict[TARGET_MATRIX_KEY]),
                    axis=0
                )
                
                this_timy = this_timy+timy
                
                en_tim = numpy.zeros([len(timy),1])
                for cc,sd in enumerate(timy): 
                    inds = sd==times_pd
                    en_tim[cc,:] = tmvec[inds]
                
                this_timy_ind = numpy.concatenate((this_timy_ind,en_tim),axis=0)
                
                
            num_examples_in_memory = full_target_matrix.shape[0]

        batch_indices = numpy.linspace(
            0, num_examples_in_memory - 1, num=num_examples_in_memory,
            dtype=int)
        batch_indices = numpy.random.choice(
            batch_indices, size=num_examples_per_batch, replace=False)

        
        this_timy = [this_timy[i] for i in batch_indices.astype(int)]
        
        beens = numpy.empty([len(this_timy),targ_LONinds.shape[0]], dtype="U15")
        for hh, dats in enumerate(this_timy):
            sets = numpy.tile(dats,[1,targ_LONinds.shape[0]])
            beens[hh,:]=sets
            
            
        beens_ind = numpy.empty([len(this_timy_ind),targ_LONinds.shape[0]], dtype='float32')
        for hh, dats in enumerate(this_timy_ind):
            sets = numpy.tile(dats,[1,targ_LONinds.shape[0]])
            beens_ind[hh,:]=sets
        
        predictor_matrix, _ = normalize_images(
            predictor_matrix=full_predictor_matrix[batch_indices, ...],
            predictor_names=predictor_names,
            normalization_dict=normalization_dict)   
        predictor_matrix = predictor_matrix.astype('float32')
        
        
        target_values, _ = normalize_images_targ(
            targ_matrix=full_target_matrix[batch_indices, ...],
            targ_names = targ_names, 
            normalization_dict = normalization_dict_targ)
        target_values = target_values.astype('float32')

        num_examples_in_memory = 0
        full_predictor_matrix = None
        full_target_matrix = None
        full_post_matrix = None
        
        if len(predictor_matrix.shape)>2:
            predictor_matrix = numpy.reshape(predictor_matrix,(predictor_matrix.shape[0]*predictor_matrix.shape[1],predictor_matrix.shape[-1]))
            target_values = numpy.reshape(target_values,(target_values.shape[0]*target_values.shape[1],1))
            beens = numpy.reshape(beens,(beens.shape[0]*beens.shape[1],1))
            beens_ind = numpy.reshape(beens_ind,(beens_ind.shape[0]*beens_ind.shape[1],1))

        yield predictor_matrix,target_values,beens,beens_ind