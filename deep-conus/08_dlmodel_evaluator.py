from keras.models import load_model
import xarray as xr
import numpy as np
import pandas as pd
from sklearn.metrics.cluster import contingency_matrix
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
        mask (boolean): Whether to train using the masked data or the non-masked data. Defaults to ``False``.
        random_choice (int): The integer the respective ``random`` method file was saved as. Defaults to ``None``.
        month_choice (int): Month for analysis. Defaults to ``None``.
        season_choice (str): Three-month season string, if ``method==season`` (e.g., 'DJF'). Defaults to ``None``.
        year_choice (int): Year for analysis. Defaults to ``None``.
        obs_threshold (float): Decimal value that denotes whether model output is a ``1`` or ``0``. Defaults to ``0.5``.
        print_sequential (boolean): Whether the sequential function calls to save files will output status statements.
                                    Defaults to ``True``.
        
    Raises:
        Exceptions: Checks whether correct values were input for ``climate`` and ``method``.
        
    Todo:
        * Add loading and handling of test subsets that were created using the ``month``, ``season``, and ``year`` methods.
        * Troubleshoot bug with ``nans`` showing up in UH03 data.
        
    """
    
    def __init__(self, climate, method, variables, var_directory, model_directory, model_num, eval_directory, mask=False, 
                 random_choice=None, month_choice=None, season_choice=None, year_choice=None, obs_threshold=0.5, print_sequential=True):
        
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
              }
        try:
            out=var[variable]
            return out
        except:
            raise ValueError("Please enter ``TK``, ``EV``, ``EU``, ``QVAPOR``, ``PRESS``, ``W_vert``, ``UH25``, ``UH03``, ``MAXW``, ``CTT``, or ``DBZ`` as variable.")
            
    
    def add_dbz(self):
        
        """Function that adds ``DBZ`` variable to last dim of test data array.
        
        """
        self.variables=np.append(self.variables, 'DBZ')
        
        
    def add_uh25(self):
        
        """Function that adds ``UH25`` variable to last dim of test data array.
        
        """
        self.variables=np.append(self.variables, 'UH25')
        
        
    def add_uh03(self):
        
        """Function that adds ``UH03`` variable to last dim of test data array.
        
        """
        self.variables=np.append(self.variables, 'UH03')
            
            
    def open_test_files(self):

        """Open the subset test data files.
        
        Returns:
            the_data: Dictionary of opened Xarray data arrays containing selected variable training data.
            
        """
        self.add_dbz()
        self.add_uh25()
        #self.add_uh03()    #add once UH03 data is fixed
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
            testdata, label: Eagerly loaded test data and labels as a numpy array.
        
        """
        thedata={}
        for key, value in kwargs.items():
            thedata[key]=value.X_test.values
            label=value.X_test_label.values
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
                    'X_test':(['b','x','y','features'], testdata),
                    'X_test_label':(['b'], label),
                    },
                    ).dropna(dim='b')
        self.test_data=data.X_test.values
        self.test_labels=data.X_test_label.values
        testdata=None
        label=None
    
    
    def load_and_predict(self):
        
        """Load the DL model and generate predictions with the input variable data.
        Make sure to input variables (features) in the same shape as were input to the model during training.
        Assigns new class attributes, ``model_probability_forecasts`` and ``model_binary_forecasts``, which are 
        the probabilistic and dichotomous predictions from the deep learning model.
        
        """
        model=load_model(f'{self.model_directory}/model_{self.model_num}_{self.climate}.h5')
        self.model_probability_forecasts=model.predict(self.test_data[...,:-2])     #change to -3 once UH03 data is fixed
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
        _, _, self.thresholds=roc_curve(self.test_labels, self.model_probability_forecasts)
    
            
    def nonscalar_metrics_and_save(self):
        
        """Evaluate the DL model using varying thresholds and a series of error metrics and save results as a csv file.
        
        """
        self.assign_thresholds()
        self.contingency_nonscalar_table=pd.DataFrame(np.zeros((self.thresholds.size, 8), dtype=int),
                                                      columns=["TP", "FP", "FN", "TN", "Threshold", "POD", "POFD", "FAR"])
        for t, threshold in enumerate(self.thresholds):
            tp=np.count_nonzero(np.logical_and((self.model_probability_forecasts >= threshold).reshape(-1), (self.test_labels >= self.obs_threshold)))
            fp=np.count_nonzero(np.logical_and((self.model_probability_forecasts >= threshold).reshape(-1), (self.test_labels < self.obs_threshold)))
            fn=np.count_nonzero(np.logical_and((self.model_probability_forecasts < threshold).reshape(-1), (self.test_labels >= self.obs_threshold)))
            tn=np.count_nonzero(np.logical_and((self.model_probability_forecasts < threshold).reshape(-1), (self.test_labels < self.obs_threshold)))
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
            self.contingency_nonscalar_table.iloc[t] += [tp, fp, fn, tn, threshold, pod, pofd, far]
        if self.method=='random':
            self.contingency_nonscalar_table.to_csv(f'{self.eval_directory}/probability_results_{self.mask_str}_model{self.model_num}_{self.method}{self.random_choice}.csv')
                

    def scalar_metrics_and_save(self):
        
        """Evaluate the DL model using a scalar threshold and a series of error metrics and save results as a csv file.
        
        """
        self.contingency_scalar_table = pd.DataFrame(np.zeros((1, 13), dtype=int),
                                                     columns=["TP", "FP", "FN", "TN", "Threshold", "POD", "POFD", "FAR", 
                                                              "CSI", "ProportionCorrect", "Bias", "HitRate", "FalseAlarmRate"])
        tp=np.count_nonzero(np.logical_and((self.model_binary_forecasts >= self.obs_threshold).reshape(-1), (self.test_labels >= self.obs_threshold)))
        fp=np.count_nonzero(np.logical_and((self.model_binary_forecasts >= self.obs_threshold).reshape(-1), (self.test_labels < self.obs_threshold)))
        fn=np.count_nonzero(np.logical_and((self.model_binary_forecasts < self.obs_threshold).reshape(-1), (self.test_labels >= self.obs_threshold)))
        tn=np.count_nonzero(np.logical_and((self.model_binary_forecasts < self.obs_threshold).reshape(-1), (self.test_labels < self.obs_threshold))) 
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
        self.create_contingency_matrix()
        csi=self.threat_score()
        pc=self.proportion_correct()
        bs=self.bias()
        hr=self.hit_rate()
        farate=self.false_alarm_rate()
        self.contingency_scalar_table.iloc[0] += [tp, fp, fn, tn, self.obs_threshold, pod, pofd, far, csi, pc, bs, hr, farate]
        if self.method=='random':
            self.contingency_scalar_table.to_csv(f'{self.eval_directory}/scalar_results_{self.mask_str}_model{self.model_num}_{self.method}{self.random_choice}.csv')
        
        
    def grab_verification_indices(self):
        
        """Extract the indices of various test cases for follow-up interpretation compositing.
        
        """
        #true - positives
        self.tp_indx=np.asarray(np.where(np.logical_and((self.model_binary_forecasts >= self.obs_threshold).reshape(-1), (self.test_labels >= self.obs_threshold)))).squeeze()
        #true - positives, prediction probability exceeding 95% confidence (very correct, severe)
        self.tp_99_indx=np.asarray(np.where(np.logical_and((self.model_probability_forecasts >= 0.99).reshape(-1), (self.test_labels >= self.obs_threshold)))).squeeze()
        #false - positives
        self.fp_indx=np.asarray(np.where(np.logical_and((self.model_binary_forecasts >= self.obs_threshold).reshape(-1), (self.test_labels < self.obs_threshold)))).squeeze()
        #false - positives, prediction probability exceeding 95% confidence (very incorrect, severe)
        self.fp_99_indx=np.asarray(np.where(np.logical_and((self.model_binary_forecasts >= 0.99).reshape(-1), (self.test_labels < self.obs_threshold)))).squeeze()
        #false - negatives
        self.fn_indx=np.asarray(np.where(np.logical_and((self.model_binary_forecasts < self.obs_threshold).reshape(-1), (self.test_labels >= self.obs_threshold)))).squeeze()
        #false - negatives; prediction probability below 5% (very incorrect, nonsevere)
        self.fn_01_indx=np.asarray(np.where(np.logical_and((self.model_binary_forecasts < 0.01).reshape(-1), (self.test_labels >= self.obs_threshold)))).squeeze()
        #true negative
        self.tn_indx=np.asarray(np.where(np.logical_and((self.model_binary_forecasts < self.obs_threshold).reshape(-1), (self.test_labels < self.obs_threshold)))).squeeze()
        #true negative, prediction probability below 5% (very correct, nonsevere)
        self.tn_01_indx=np.asarray(np.where(np.logical_and((self.model_binary_forecasts < 0.01).reshape(-1), (self.test_labels < self.obs_threshold)))).squeeze()
        
        
    def add_heights_to_variables(self, variable):
        
        """Variable name for the respective filenames.
           
        Args:
            variable (str): The variable to feed into the dictionary.
            
        Returns:
            variable (str): The variable name with respective height AGL.
            
        Raises:
            ValueError: If provided variable is not available.
            
        """
        var={
               'EU':['EU1','EU3','EU5','EU7'],
               'EV':['EV1','EV3','EV5','EV7'],
               'TK':['TK1','TK3','TK5','TK7'],
               'QVAPOR':['QVAPOR1','QVAPOR3','QVAPOR5','QVAPOR7'],
               'WMAX':'MAXW',
               'W_vert':['W1','W3','W5','W7'],
               'PRESS':['P1','P3','P5','P7'],
               'DBZ':'DBZ',
               'CTT':'CTT',
               'UH25':'UH25',
               'UH03':'UH03',
              }
        try:
            out=var[variable]
            return out
        except:
            raise ValueError("Please enter ``TK``, ``EV``, ``EU``, ``QVAPOR``, ``PRESS``, ``W_vert``, ``UH25``, ``UH03``, ``MAXW``, ``CTT``, or ``DBZ`` as variable.")
        
        
    def apply_variable_dictionary(self):
        
        """Create the list of features to feed as Xarray dataset attribute in ``save_indx_variables``.
        
        """
        var_list=[self.add_heights_to_variables(var) for var in self.variables[:-2]]   #change to -3 when UH03 is fixed
        var_list2=[m for me in var_list for m in me]
        var_list2=np.append(var_list2, 'DBZ')
        var_list2=np.append(var_list2, 'UH25')
        #var_list2=np.append(var_list2, 'UH03')      #UNCOMMENT when UH03 is fixed
        return var_list2
        
        
    def save_indx_variables(self):
        
        """Extract the respective test data cases using indices from ``grab_verification_indices``.
        
        """
        self.grab_verification_indices()
        data=xr.Dataset({
            'tp':(['a','x','y','features'], self.test_data[self.tp_indx,:,:,:].squeeze()),
            'tp_99':(['b','x','y','features'], self.test_data[self.tp_99_indx,:,:,:].squeeze()),
            'fp':(['c','x','y','features'], self.test_data[self.fp_indx,:,:,:].squeeze()),
            'fp_99':(['d','x','y','features'], self.test_data[self.fp_99_indx,:,:,:].squeeze()),
            'fn':(['e','x','y','features'], self.test_data[self.fn_indx,:,:,:].squeeze()),
            'fn_01':(['f','x','y','features'], self.test_data[self.fn_01_indx,:,:,:].squeeze()),
            'tn':(['g','x','y','features'], self.test_data[self.tn_indx,:,:,:].squeeze()),
            'tn_01':(['h','x','y','features'], self.test_data[self.tn_01_indx,:,:,:].squeeze()),
            },
            coords=
            {'features':(['features'], self.apply_variable_dictionary()),
            })
        if self.method=='random':
            data.to_netcdf(f'{self.eval_directory}/composite_results_{self.mask_str}_model{self.model_num}_{self.method}{self.random_choice}.nc')
        
        
    def sequence_the_evaluation(self):
        
        """Automation of the sequence of functions to produce deep learning model evaluation files.
        
        """
        if self.print_sequential:
            print("Opening and preparing the test files...")
        data=self.open_test_files()
        testdata, labels=self.assemble_and_concat(**data)
        self.remove_nans(testdata, labels)
        if self.print_sequential:
            print("Generating DL predictions...")
        self.load_and_predict()
        if self.print_sequential:
            print("Generating probabilistic and nonprobabilistic skill scores...")
        self.nonscalar_metrics_and_save()
        self.scalar_metrics_and_save()
        if self.print_sequential:
            print("Saving the indexed variables...")
        self.save_indx_variables()
        if self.print_sequential:
            print("Evaluation is complete.")
        
