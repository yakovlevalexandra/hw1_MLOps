import flask
import json
import dill as pickle
import os
from ModelMaker import ModelMaker
from flask import Response
from flask_restx import Api, Resource, fields


def save_model(model_key, model):
    with open('./models/' + str(model_key), 'wb') as file:
        pickle.dump(model, file)


def load_model(model_key):
    with open('./models/' + str(model_key), 'rb') as file:
        model = pickle.load(file)
    return model


def add_model(model_key, model_type):
    added_model = ModelMaker(model_key, model_type)
    save_model(model_key, added_model)


def fit_model(model_key, model_params=None):
    model = load_model(model_key)
    for param in model_params:
        if param not in model.get_params().keys():
            return 'Unrecognized model parameters', 400
    else:
        model.fit(model_params=model_params)
        save_model(model_key, model)


def model_predict(model_key):
    model = load_model(model_key)
    y_pred = model.predict()
    return y_pred


def delete_model(model_key):
    os.remove('./models/' + str(model_key))


app = flask.Flask(__name__)
api = Api(app)

available_models = {}

add_model_params = api.model('Params for adding a model',
                             {'model_key': fields.Integer(description='Model key', example=1),
                              'model_type': fields.String(description='Model type', example='LogisticRegression')})

fit_model_params = api.model('Params for fitting a model',
                             {'model_key': fields.Integer(description='Model key', example=1),
                              'model_params': fields.Arbitrary(description='Model parameters', example=json.dumps({'random_state': 42}))})

predict_model_params = api.model('Params for model prediction',
                                 {'model_key': fields.Integer(description='Model key', example=1)})

delete_model_params = api.model('Params for model deletion',
                                {'model_key': fields.Integer(description='Model key', example=1)})


@api.route('/available_models')
class AvailableModels(Resource):
    @api.response(200, 'Success')
    @api.response(500, 'Internal Server Error')
    def get(self):
        return available_models


@api.route('/model/add')
class AddModel(Resource):
    @api.expect(add_model_params)
    @api.response(200, 'Success')
    @api.response(400, 'Bad Request')
    @api.response(500, 'Internal Server Error')
    def post(self):
        model_key = api.payload['model_key']
        model_type = api.payload['model_type']
        if model_type not in ['RandomForestClassifier', 'LogisticRegression']:
            return 'Model type not available, try: RandomForestClassifier, LogisticRegression', 400
        else:
            available_models[model_key] = model_type
            add_model(model_key, model_type)
            return f'Successfully added model {model_key}', 200


@api.route('/model/fit')
class FitModel(Resource):
    @api.expect(fit_model_params)
    @api.response(200, 'Success')
    @api.response(400, 'Bad Request')
    @api.response(500, 'Internal Server Error')
    def post(self):
        model_key = api.payload['model_key']
        model_params = api.payload['model_params']
        if model_key not in available_models:
            return 'Model key not available', 400
        else:
            fit_model(model_key, model_params)
            return f'Successfully fitted model {model_key}', 200


@api.route('/model/predict')
class ModelPredict(Resource):
    @api.expect(predict_model_params)
    @api.response(200, 'Success')
    @api.response(400, 'Bad Request')
    @api.response(500, 'Internal Server Error')
    def get(self):
        model_key = api.payload['model_key']
        if model_key not in available_models:
            return 'Model key not available', 400
        else:
            pred = model_predict(model_key)
            return json.dumps({f'Model {model_key} prediction': pred.astype(int).tolist()})


@api.route('/model/delete')
class DeleteModel(Resource):
    @api.expect(delete_model_params)
    @api.response(200, 'Success')
    @api.response(400, 'Bad Request')
    @api.response(500, 'Internal Server Error')
    def delete(self):
        model_key = api.payload['model_key']
        if model_key not in available_models:
            return 'Model key not available', 400
        else:
            delete_model(model_key)
            del available_models[model_key]
            return f'Successfully deleted model {model_key}', 200


if __name__ == '__main__':
    app.run(debug=True)
