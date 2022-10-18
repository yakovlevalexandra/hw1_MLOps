import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def prep_data():
    """
    Preprocesses data for a model.
    """

    full_data = pd.read_csv('./data/hotel_bookings_small.csv')
    num_features = ["lead_time", "arrival_date_week_number", "arrival_date_day_of_month",
                    "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
                    "babies", "is_repeated_guest", "previous_cancellations",
                    "previous_bookings_not_canceled", "agent", "company",
                    "required_car_parking_spaces", "total_of_special_requests", "adr"]

    cat_features = ["hotel", "arrival_date_month", "meal", "market_segment",
                    "distribution_channel", "reserved_room_type", "deposit_type", "customer_type"]

    features = num_features + cat_features

    x = full_data.drop(["is_canceled"], axis=1)[features]
    y = full_data["is_canceled"]

    num_transformer = SimpleImputer(strategy="constant")

    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_features),
                                                   ("cat", cat_transformer, cat_features)])

    x = preprocessor.fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=42)
    return x_train, x_test, y_train, y_test


class ModelMaker:
    """
    A class to make a model.

    ...

    Attributes
    ----------
    model_key : int
        an integer key for the model
    model_type : str
        type of the model ('RandomForestClassifier', 'LogisticRegression')

    Methods
    -------
    fit(model_params):
        Fits the model with the chosen parameters.
    predict():
        Returns predict of the model on test data.
    get_params():
        Gets all available parameters for the model.
    """

    def __init__(self, model_key, model_type):
        """
        Makes all necessary attributes for the model object.

        Parameters
        ----------
            model_key : int
                 an integer key for the model
            model_type : str
                 type of the model ('RandomForestClassifier', 'LogisticRegression')
        """

        models_dict = {'RandomForestClassifier': RandomForestClassifier(), 'LogisticRegression': LogisticRegression()}
        self.model_key = model_key
        self.model_type = model_type
        self.model = models_dict[self.model_type]
        self.x_train, self.x_test, self.y_train, self.y_test = prep_data()

    def fit(self, model_params):
        """
        Fits the model with the chosen parameters.

        Parameters
        ----------
            model_params : dict
                parameters for the model
        """

        self.model.set_params(**model_params)
        self.model.fit(self.x_train, self.y_train)

    def predict(self):
        """
        Returns predict of the model on test data.
        """

        y_pred = self.model.predict(self.x_test)
        return y_pred

    def get_params(self):
        """
        Gets all available parameters for the model.
        """

        return self.model.get_params()
