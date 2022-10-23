import dill as pickle
import os
from model_maker import ModelMaker


def save_model(model_key, model):
    """
    Saves the model to the directory ../models

    Parameters
    ----------
        model_key : int
            an integer key of the model
        model : sklearn.base.BaseEstimator
            the model to save
    """

    with open('../models/' + str(model_key), 'wb') as file:
        pickle.dump(model, file)


def load_model(model_key):
    """
    Loads the model from the directory ../models

    Parameters
    ----------
        model_key : int
            an integer key of the model
    """

    with open('../models/' + str(model_key), 'rb') as file:
        model = pickle.load(file)
    return model


def add_model(model_key, model_type):
    """
    Adds new model and saves it.

    Parameters:
        model_key : int
             an integer key of the model
        model_type : str
            type of the model ('RandomForestClassifier', 'LogisticRegression')
    """

    added_model = ModelMaker(model_key, model_type)
    save_model(model_key, added_model)


def fit_model(model_key, model_params=None):
    """
    Fits the chosen model.

    Parameters
    ----------
        model_key : int
                 an integer key of the model
        model_params : dict
                    parameters for the model
    """

    model = load_model(model_key)
    for param in model_params:
        if param not in model.get_params().keys():
            return 'Unrecognized model parameters', 400
    else:
        model.fit(model_params=model_params)
        save_model(model_key, model)


def model_predict(model_key):
    """
    Returns predict of the model on test data.

    Parameters:
        model_key : int
                 an integer key of the model
    """

    model = load_model(model_key)
    y_pred = model.predict()
    return y_pred


def delete_model(model_key):
    """
    Deletes the chosen model.

    Parameters
    ----------
        model_key : int
                 an integer key of the model
    """

    os.remove('../models/' + str(model_key))
