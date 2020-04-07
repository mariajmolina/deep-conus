import keras
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
from keras.models import Model, save_model, load_model, Sequential
from keras.layers import Dense, Activation, Conv2D, Input, AveragePooling2D, MaxPooling2D
from keras.layers import SpatialDropout2D, Flatten, LeakyReLU, Dropout, BatchNormalization, LeakyReLU
from keras.optimizers import SGD, Adam
from keras.regularizers import l2

import xarray as xr
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


class EvaluateDLModel:
    
    """Class instantiation of EvaluateDLModel:
    
    Here we load the variable data for evaluation of the trained deep convolutional neural network.
    
    Attributes:
        climate (str): Whether to interpolate variable in the ``current`` or ``future`` climate simulation.
        method (str): Method for parsing data. Options include ``random``, ``month``, ``season``, ``year``.
        variables (str): Numpy array of variable name strings. Options include ``EU``, ``EV``, ``TK``, ``QVAPOR``, ``WMAX``, 
                         ``W_vert``,``PRESS``,``DBZ``,``CTT``,``UH25``, and``UH03``.
        var_directory (str): Directory where the test subset variable data is saved.
        model_directory (str): Directory where the deep learning model is saved.
        model_num (str): The number of the model as it was saved.
        eval_directory (str): Directory where evaluation files will be saved.
        mask (boolean): Whether to train using the masked data or the non-masked data. Defaults to False.
        random_choice (int): The integer the respective ``random`` method file was saved as. Defaults to ``None``.
        month_choice (int): Month for analysis. Defaults to ``None``.
        season_choice (str): Three-month season string, if ``method==season`` (e.g., 'DJF'). Defaults to ``None``.
        year_choice (int): Year for analysis. Defaults to ``None``.
        obs_threshold (float): Decimal value that denotes whether model output is a ``1`` or ``0``. Defaults to ``0.5``.
        
    Todo:
        * Add loading and handling of test subsets that were created using the ``month``, ``season``, and ``year`` methods.
        
    """
    
    
    def __init__(self, climate, method, variables, var_directory, model_directory, model_num, eval_directory, mask=False, 
                 random_choice=None, month_choice=None, season_choice=None, year_choice=None, obs_threshold=0.5):
        
        if climate!='current' and climate!='future':
            raise Exception("Please enter current or future as string for climate period selection.")
        if climate=='current' or climate=='future':
            self.climate=climate
        
        if method!='random' and method!='month' and method!='season' and method!='year':
            raise Exception("Please enter ``random``, ``month``, ``season``, or ``year`` as method.")
        if method=='random' or method=='month' or method=='season' or method=='year':
            self.method=method
            
        self.variables=variables
        self.var_directory=var_directory
        self.model_directory=model_directory
        self.eval_directory=eval_directory
        
        self.mask=mask
        if not self.mask:
            self.mask_str='nomask'
        if self.mask:
            self.mask_str='mask'
    
        self.random_choice=random_choice 
        self.month_choice=month_choice 
        self.season_choice=season_choice 
        self.year_choice=year_choice
        self.obs_threshold=obs_threshold
        
    
    def month_translate(self):
        
        """Convert integer month to string month.
        
        Returns:
            out (str): Input month as string.
            
        Raises:
            ValueError: If the month is not within the study's range (Dec-May).
            
        """
        var={12:'dec',
             1:'jan',
             2:'feb',
             3:'mar',
             4:'apr',
             5:'may'}
        try:
            out=var[self.month_choice]
            return out
        except:
            raise ValueError("Please enter month integer from Dec-May.")
    
    
    def open_test_files(self):

        """Open the subset test data files.
        
        Returns:
            the_data: Dictionary of opened Xarray data arrays containing selected variable training data.
            
        """
        the_data={}
        if self.method=='random':
            for var in self.variables:
                the_data[var]=xr.open_dataset(
                    f'/{self.var_directory}/{self.climate}_{self.variable_translate(var).lower()}_{self.mask_str}_{self.method}_test{self.random_choice}.nc')
        if self.method=='month':
            for var in self.variables:
                var   ##TODO 
        if self.method=='season':
            for var in self.variables:
                var   ##TODO 
        if self.method=='year':
            for var in self.variables:
                var   ##TODO 
        return the_data


    def assemble_and_concat(self, **kwargs):
        
        """Eagerly load the testing labels and data.
        
        Args:
            **kwargs: Dictionary containing opened variable testing data, which was opened with ``self.open_test_files()``.
            
        Returns:
            X_test, label: Eagerly loaded test data and labels as a numpy array.
        
        """
        thedata={}
        for key, value in kwargs.items():
            thedata[key]=value.X_test.values
            label=value.X_test_label.values
        if len(kwargs) > 1:
            X_test=np.concatenate(list(thedata.values()), axis=3)
        if len(kwargs)==1:
            X_test=np.squeeze(np.asarray(list(thedata.values())))
        kwargs=None
        return X_test, label
    
    
    def remove_nans(self, X_test, label):
        
        """Assemble the variables and remove any ``nan`` values from the data. 
        Assigns new attributes to class, including ``test_data`` and ``test_labels``, which are the test data and labels 
        for evaluating deep learning model skill.
        
        Args:
            X_test (numpy array): Test data.
            label (numpy array): Label data.
        
        """
        data = xr.Dataset({
                    'X_test':(['b','x','y','features'], X_test),
                    'X_test_label':(['b'], label),
                    },
                    ).dropna(dim='b')
        X_test=None
        label=None
        self.test_data=data.X_test.values
        self.test_labels=data.X_test_label.values
    
    
    def load_and_predict(self):
        
        """Load the DL model and generate predictions with the input variable data.
        Make sure to input variables (features) in the same shape as were input to the model during training.
        Assigns new class attributes, ``model_probability_forecasts`` and ``model_binary_forecasts``, which are 
        the probabilistic and dichotomous predictions from the deep learning model.
        
        """
        model=load_model(f'{model_directory}/model_{self.model_num}_{self.climate}.h5')
        self.model_probability_forecasts=model.predict(self.test_data)
        self.model_binary_forecasts=np.round(self.model_probability_forecasts.reshape(len(self.model_probability_forecasts)),0)
        
    
    def create_contingency_matrix(self):
        
        """Create the contingency 2x2 matrix using deep learning model binary predictions and test labels.
        Assigns new class attribute, ``cont_matrix``.
        
        """
        self.cont_matrix=contingency_matrix(labels_true=self.test_labels, labels_pred=self.model_binary_forecasts)
        
    
    def threat_score(self):

        """Threat score (Gilbert 1884) or critical success index. The worst score is zero, while the best score is one.
        From Wilks book: "Proportion correct for the quantity being forecast after removing correct no forecasts from consideration".
            
        Returns:
            Threat score (float).
            
        """    
        return np.divide(self.cont_matrix[0][0], self.cont_matrix[0][0] + self.cont_matrix[0][1] + self.cont_matrix[1][0])


    def proportion_correct(self):

        """Returns: The proportion correct (Finley 1884).
            
        """
        return np.divide(self.cont_matrix[0][0] + self.cont_matrix[1][1], 
                         self.cont_matrix[0][0] + self.cont_matrix[0][1] + self.cont_matrix[1][0] + self.cont_matrix[1][1])


    def bias(self):

        """Returns: The bias ratio. Unbiased = 1, bias > 1 means overforecasting, Bias < 1 means underforecasting.
            
        """
        return np.divide(self.cont_matrix[0][0] + self.cont_matrix[0][1], self.cont_matrix[0][0] + self.cont_matrix[1][0])


    def false_alarm_ratio(self):

        """False alarm ratio measures the fraction of ``positive`` forecasts that turned out to be wrong (Doswell et al. 1990).
        Best FAR is zero, the worst score is 1.
        
        Returns:
            False alarm ratio (float).
            
        """
        return np.divide(self.cont_matrix[0][1], self.cont_matrix[0][0] + self.cont_matrix[0][1])


    def hit_rate(self):

        """Also called the probability of detection (POD; Doswell et al. 1990).
        This metric measures the ratio of correct forecasts to the number of times the event occurred.
            
        Returns:
            Hit rate (float).

        """
        return np.divide(self.cont_matrix[0][0], self.cont_matrix[0][0] + self.cont_matrix[1][0])


    def false_alarm_rate(self):

        """Also known as the probability of false detection (POFD).
        This metric is the ratio of false alarms to the total number of non-occurrences of the event.
        
        Returns:
            False alarm rate (float).
        
        """
        return np.divide(self.cont_matrix[0][1], self.cont_matrix[0][1] + self.cont_matrix[1][1])

    
    def assign_thresholds(self):
        
        """Assign an array of probability thresholds (values between zero and one) to use in subsequent ROC curve and probability skill metrics.
        Assigns new class attribute ``self.thresholds`` with the array of thresholds.
        
        """
        _, _, self.thresholds = roc_curve(self.test_labels, self.model_probability_forecasts)
    
            
    def nonscalar_metrics_and_save(self):
        
        """Evaluate the DL model using varying thresholds and a series of error metrics and save results as a csv file.
        
        """
        self.assign_thresholds()
        self.contingency_tables=pd.DataFrame(np.zeros((self.thresholds.size, 8), dtype=int),
                                               columns=["TP", "FP", "FN", "TN", "Threshold", "POD", "POFD", "FAR"])
        for t, threshold in enumerate(self.thresholds):
            tp = np.count_nonzero(np.logical_and((self.model_probability_forecasts >= threshold).reshape(-1), (self.test_labels >= self.obs_threshold)))
            fp = np.count_nonzero(np.logical_and((self.model_probability_forecasts >= threshold).reshape(-1), (self.test_labels < self.obs_threshold)))
            fn = np.count_nonzero(np.logical_and((self.model_probability_forecasts < threshold).reshape(-1), (self.test_labels >= self.obs_threshold)))
            tn = np.count_nonzero(np.logical_and((self.model_probability_forecasts < threshold).reshape(-1), (self.test_labels < self.obs_threshold)))
            try:
                pod = float(tp) / (float(tp) + float(fn))
            except ZeroDivisionError:
                pod = 0.
            try:
                pofd = float(fp) / (float(fp) + float(tn))
            except ZeroDivisionError:
                pofd = 0.
            try:
                far = float(fp) / (float(fp) + float(tp))
            except ZeroDivisionError:
                far = 0.
            self.contingency_tables.iloc[t] += [tp, fp, fn, tn, threshold, pod, pofd, far]
            
        if self.method='random':
            self.contingency_tables.to_csv(f'{self.eval_directory}/probability_results_{self.mask_str}_{self.method}_test{self.model_num}.csv')
                

    def scalar_metrics_and_save(self):
        
        """Evaluate the DL model using a scalar threshold and a series of error metrics and save results as a csv file.
        
        """
        self.contingency_tables = pd.DataFrame(columns=["TP", "FP", "FN", "TN", "Threshold", "POD", "POFD", "FAR", 
                                                        "CSI", "ProportionCorrect", "Bias", "HitRate", "FalseAlarmRate"])
        
        tp=np.count_nonzero(np.logical_and((self.model_binary_forecasts >= self.obs_threshold).reshape(-1), (self.test_labels >= self.obs_threshold)))
        fp=np.count_nonzero(np.logical_and((self.model_binary_forecasts >= self.obs_threshold).reshape(-1), (self.test_labels < self.obs_threshold)))
        fn=np.count_nonzero(np.logical_and((self.model_binary_forecasts < self.obs_threshold).reshape(-1), (self.test_labels >= self.obs_threshold)))
        tn=np.count_nonzero(np.logical_and((self.model_binary_forecasts < self.obs_threshold).reshape(-1), (self.test_labels < self.obs_threshold))) 
        csi=self.threat_score()
        pc=self.proportion_correct()
        bs=self.bias()
        hr=self.hit_rate()
        farate=self.false_alarm_rate()
        
        self.contingency_tables.iloc[0] = [tp, fp, fn, tn, threshold, pod, pofd, far, csi, pc, bs, hr, farate]
        
        if self.method='random':
            self.contingency_tables.to_csv(f'{self.eval_directory}/scalar_results_{self.mask_str}_{self.method}_test{self.model_num}.csv')
        
        
