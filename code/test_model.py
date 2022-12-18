from model_maker import ModelMaker
import pandas as pd


def test_model_maker_predict_random_forest():
    data = pd.read_csv('./data/hotel_bookings_small.csv')
    model = ModelMaker(1, 'RandomForestClassifier', data)
    model.fit({'random_state': 42})
    assert len(model.predict())!=0


def test_model_maker_predict_logreg():
    data = pd.read_csv('./data/hotel_bookings_small.csv')
    model = ModelMaker(1, 'LogisticRegression', data)
    model.fit({'random_state': 42})
    assert len(model.predict())!=0