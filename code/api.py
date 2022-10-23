import flask
import json
import pandas as pd
from model_store import save_model, load_model, add_model, fit_model, model_predict, delete_model
from flask_restx import Api, Resource, fields


app = flask.Flask(__name__)
api = Api(app)

available_models = {}

add_model_params = api.model('Params for adding a model',
                             {'model_key': fields.Integer(description='Model key', example=1),
                              'model_type': fields.String(description='Model type', example='LogisticRegression'),
                              'data': fields.Arbitrary(description='Data for model')})

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
        data = pd.DataFrame(json.loads(api.payload['data']))
        if model_type not in ['RandomForestClassifier', 'LogisticRegression']:
            return 'Model type not available, try: RandomForestClassifier, LogisticRegression', 400
        else:
            available_models[model_key] = model_type
            add_model(model_key, model_type, data)
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
    app.run()
