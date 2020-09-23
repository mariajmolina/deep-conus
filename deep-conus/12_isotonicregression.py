import numpy as np
import xarray as xr
import pandas as pd
from keras.models import load_model
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
import pickle


class IsotonicRegress:
    
    """Class instantiation of IsotonicRegress. 
    Here, an isotonic regression is fit to the DL Convnet model predictions.
    The goal is to reduce DL model overconfident output and improve forecast reliability.
    
    Attributes:
        climate (str): Whether to interpolate variable in the ``current`` or ``future`` climate simulation.
        method (str): Method for parsing data. Options include ``random``, ``month``, ``season``, ``year``.
        variables (str): Numpy array of variable name strings. Options include ``EU``, ``EV``, ``TK``, ``QVAPOR``, ``WMAX``, 
                         ``W_vert``,``PRESS``,``DBZ``,``CTT``,``UH25``, ``MASK``, and``UH03``.
        var_directory (str): Directory where the test subset variable data is saved.
        model_directory (str): Directory where the deep learning model is saved.
        model_num (int): Number assignment for the model.
        eval_directory (str): Directory where evaluation files will be saved.
        mask (boolean): Whether to train using the masked data or the non-masked data. Defaults to ``False``.
        mask_train (boolean): Whether to train using masked state variable data. Defaults to ``False``. Will override ``mask`` to ``True``.
        unbalanced (boolean): Whether training data will be artificially balanced (``False``) or left unbalanced (``True``). Defaults to ``False``. 
        validation (boolean): Whether to extract a validation set from the original unbalanced dataset. Defaults to ``False``.
        random_choice (int): The integer the respective ``random`` method file was saved as. Defaults to ``None``.
        month_choice (int): Month for analysis. Defaults to ``None``.
        season_choice (str): Three-month season string, if ``method==season`` (e.g., 'DJF'). Defaults to ``None``.
        year_choice (int): Year for analysis. Defaults to ``None``.
        obs_threshold (float): Decimal value that denotes whether model output is a ``1`` or ``0``. Defaults to ``0.5``.
        print_sequential (boolean): Whether the sequential function calls to save files will output status statements. Defaults to ``True``.
    
    """
    
    def __init__(self, climate, method, variables, var_directory, model_directory, model_num, eval_directory, 
                 mask=False, mask_train=False, unbalanced=True, validation=True,
                 random_choice=None, month_choice=None, season_choice=None, year_choice=None, 
                 obs_threshold=0.5, print_sequential=True):
        
        if climate!='current' and climate!='future':
            raise Exception("Please enter ``current`` or ``future`` as string for climate period selection.")
        else:
            self.climate=climate
        if method!='random' and method!='month' and method!='season' and method!='year':
            raise Exception("Please enter ``random``, ``month``, ``season``, or ``year`` as method.")
        else:
            self.method=method
        self.variables=variables
        self.var_directory=var_directory
        self.model_directory=model_directory
        self.model_num=model_num
        self.eval_directory=eval_directory
        self.mask_train=mask_train
        self.unbalanced=unbalanced
        self.validation=validation
        if not self.validation:
            raise Exception("Unbalanced validation set is needed for this analysis.")
        if not mask_train:
            self.mask=mask
            if not self.mask:
                self.mask_str='nomask'
            if self.mask:
                self.mask_str='mask'
        if mask_train:
            self.mask=True
            self.mask_str='mask'
        self.random_choice=random_choice 
        self.month_choice=month_choice 
        self.season_choice=season_choice 
        self.year_choice=year_choice
        self.obs_threshold=obs_threshold
        self.print_sequential=print_sequential

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

    def variable_translate(self, variable):
        
        """Variable name for the respective filenames.
           
        Args:
            variable (str): The variable to feed into the dictionary.
            
        Returns:
            variable (str): The variable name to use for opening saved files.
            
        Raises:
            ValueError: If provided variable is not available.
            
        """
        var={
               'EU':'EU',
               'EV':'EV',
               'TK':'TK',
               'QVAPOR':'QVAPOR',
               'WMAX':'MAXW',
               'W_vert':'W',
               'PRESS':'P',
               'DBZ':'DBZ',
               'CTT':'CTT',
               'UH25':'UH25',
               'UH03':'UH03',
               'MASK':'MASK',
              }
        try:
            out=var[variable]
            return out
        except:
            raise ValueError("Please enter ``TK``, ``EV``, ``EU``, ``QVAPOR``, ``PRESS``, ``W_vert``, ``UH25``, ``MASK``, ``UH03``, ``MAXW``, ``CTT``, or ``DBZ`` as variable.")

    def open_test_files(self):

        """Open the subset test data files.
        
        Returns:
            the_data: Dictionary of opened Xarray data arrays containing selected variable training data.
            
        """
        the_data={}
        if self.method=='random':
            for var in self.variables:
                the_data[var]=xr.open_dataset(
                    f'/{self.var_directory}/{self.climate}_{self.variable_translate(var).lower()}_{self.mask_str}_{self.method}_valid1_unbalanced_valid.nc')                        
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
            testdata, label: Eagerly loaded test data and labels as a numpy array.
        
        """
        thedata={}
        for key, value in kwargs.items():
            if not self.mask_train:
                thedata[key]=value.X_valid.values
            if self.mask_train:
                thedata[key]=np.where(np.repeat(kwargs['MASK'].X_valid.values, value.X_valid.shape[-1],3)==0, 0, value.X_valid.values)
            label=value.X_valid_label.values
        if len(kwargs) > 1:
            testdata=np.concatenate(list(thedata.values()), axis=3)
        if len(kwargs)==1:
            testdata=np.squeeze(np.asarray(list(thedata.values())))
        thedata=None
        return testdata, label

    def remove_nans(self, testdata, label):
        
        """Assemble the variables and remove any ``nan`` values from the data. 
        Assigns new attributes to class, including ``test_data`` and ``test_labels``, which are the test data and labels 
        for evaluating deep learning model skill.
        
        Args:
            testdata (numpy array): Test data.
            label (numpy array): Label data.
        
        """
        data=xr.Dataset({
                    'X_valid':(['c','x','y','features'], testdata),
                    'X_valid_label':(['c'], label),
                    },
                    ).dropna(dim='c')
        return data

    def load_and_predict(self, valid_data):
        
        """Load the DL model and generate predictions with the input variable data.
        Make sure to input variables (features) in the same shape as were input to the model during training.
        Assigns new class attributes, ``model_probability_forecasts`` and ``model_binary_forecasts``, which are 
        the probabilistic and dichotomous predictions from the deep learning model.
        
        """
        model=load_model(f'{self.model_directory}/model_{self.model_num}_current.h5')
        self.model_probability_forecasts=model.predict(valid_data)
        
        
    def plot_raw(self, data):
        
        """Plot predictions and labels.
        
        """
        fig = plt.figure()
        plt.plot(self.model_probability_forecasts.reshape(-1)[:], data, 'b.', markersize=5)
        return plt.show()

    def train_isotonic(self, data1, data2):
        
        """Train the Isotonic Regression.
        
        """
        ir = IsotonicRegression()
        ir.fit(data1, data2)
        return ir

    def generate_predictions(self, model, data):
        
        """Generate the Isotonic model predictions.
        
        """
        return model.predict(data)

    def save_model_pickle(self, model):
        
        """Save the model.
        
        """
        pkl_filename = f'{self.model_directory}/isotonic_model{self.model_num}.pkl'
        with open(pkl_filename, 'wb') as file:
            pickle.dump(model, file)

    def visualize_predictions(self, model, data1, data2):
        fig = plt.figure()
        plt.plot(data1, data2, 'b.', markersize=5)
        return plt.show()

    def sequence_isotonic(self):
        
        """Sequence the function calls for the Isotonic Model training.
        
        """
        print("Opening files...")
        data=self.open_test_files()
        print("Assemble and concat files...")
        valid, label=self.assemble_and_concat(**data)
        print("Removing nans...")
        newdata=self.remove_nans(valid, label)
        valid=None; label=None; data=None
        print("DL model predictions...")
        self.load_and_predict(newdata['X_valid'].values)
        print("Training isotonic model...")
        ir=file.train_isotonic(data1=self.model_probability_forecasts.reshape(-1), 
                               data2=newdata['X_valid_label'].values.astype(np.float32))
        print("Saving isotonic model...")
        self.save_model_pickle(ir)
