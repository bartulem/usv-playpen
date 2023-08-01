"""
@author: bartulem
Code to regress audio to video sync data.
"""

import numpy as np
from sklearn import linear_model


class LinRegression:

    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def split_train_test_and_regress(self, quantum_seed=None):
        """
        Description
        ----------
        This method splits the data into train/test sets, obtains a simple linear
        model, predicts test data based on the model and estimates prediction
        errors.
        ----------

        Parameters
        ----------
        Contains the following set of parameters
            x_data (np.ndarray)
                Data used for prediction, should be LED on start samples in audio file.
            y_data (np.ndarray)
                Data to be predicted, should be LED on start frames in video file(s).
        ----------

        Returns
        ----------
        prediction_error_array (np.ndarray)
            The discrepancy between predicted and true frames in the test sample.
        ----------
        """

        # set random seed
        np.random.seed(quantum_seed)

        # chose random indices for training
        train_indices = np.sort(np.random.choice(a=range(self.x_data.shape[0]), size=int(round(self.x_data.shape[0]*.5)), replace=False))
        test_indices = [ti for ti in range(self.x_data.shape[0]) if ti not in train_indices]

        split_data = {'x_train': np.take(a=self.x_data, indices=train_indices, axis=0), 'x_test': np.take(a=self.x_data, indices=test_indices, axis=0),
                      'y_train': np.take(a=self.y_data, indices=train_indices, axis=0), 'y_test': np.take(a=self.y_data, indices=test_indices, axis=0)}

        regress_data = {key: val.reshape((val.shape[0], 1)) for key, val in split_data.items()}

        # train model
        lm = linear_model.LinearRegression()
        lm.fit(regress_data['x_train'], regress_data['y_train'])

        # predict test data
        y_predictions = np.round(np.floor(np.ravel(lm.predict(regress_data['x_test']))))
        prediction_error_array = y_predictions - np.ravel(regress_data['y_test'])

        return prediction_error_array
