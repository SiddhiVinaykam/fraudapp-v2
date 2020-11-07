
import logging
import joblib
import json
import click
import os
import uuid
import time
from collections import OrderedDict
# from . import pipeline
from io import BytesIO
from typing import Dict, List
from flask import request

import physarum
from physarum import phyml
import kfserving
from physarum.features import _download_uri


from sklearnPipeline import execute
from physarum.phyml import sklearn_node, sklearn_pipeline

from sklearnPipeline import (
	ContinuousColumnSelector,
	SimpleImputer,
	SelectKBest,
	SelectFromModel,
	CategoricalColumnSelector,
	DataEncoder,
	LogisticRegression,
)

def main():
	if 'KF_DOCKER_IMAGE' in os.environ and 'CI_COMMIT_SHA' in os.environ:
		image_url = os.environ['KF_DOCKER_IMAGE']
	else:
		image_url = 'gcr.io/physarum-dev-platform/physarum/physarum-apparelapp-dev:0.1'

	app = phyml.PhymlApp(
		name='customer-app',
		platform='GCP',
		image_url='image',
		package_entrypoint='sdasdas',
		inference_port=8000,
		k8s_namespace='kubeflow',
	)

	app.with_envs(
		{
			'GCP_PROJECT': 'physarum-streaming',
			'GCP_ZONE': 'us-central1-b',
			'K8S_CLUSTER': 'physarum-serving',
			'K8S_NAMESPACE': 'kubeflow',
			'LAKE_BUCKET': 'physarum-dev',
			'LAKE_PATH': 'physarum/applications/customer_app',
			'WAREHOUSE_DATASET': 'customer_app',
			'WAREHOUSE_LOCATION': 'US',
			'KFP_ARTIFACT_PATH': './artifacets'
		}
	)

	@sklearn_pipeline(ml_app=app.mlpipelines, name="Fraud_Detection_Pipeline_", description="sklearn training", task="Classification", parameter_tuning=None,     parameter_tuning_params=None, save_output="output", cache_dirpath='/tmp/tmp-cache')
	def Fraud_Detection_Pipeline_(train_path=None, test_path=None, output_path=None, sep="	"):

		ContinuousColumnSelector_1_1 = sklearn_node(transformer = ContinuousColumnSelector, config= {'continuous_columns': None, 'unique_counts': 20}).after(SimpleImputer_2_1)
		SimpleImputer_2_1 = sklearn_node(transformer = SimpleImputer, config= {'strategy': 'mean'}).after(SelectKBest_2_1)
		SelectKBest_2_1 = sklearn_node(transformer = SelectKBest, config= {'score_func': 'f_regression', 'k': 'all'}).after(SelectFromModel_1_1)
		SelectFromModel_1_1 = sklearn_node(transformer = SelectFromModel, config= {'estimator': 'LogisticRegression', 'estimator_params': OrderedDict(), 'threshold': None}).after(LogisticRegression_1_1)
		CategoricalColumnSelector_1_1 = sklearn_node(transformer = CategoricalColumnSelector, config= {'unique_counts': 20}).after(SimpleImputer_1_1)
		SimpleImputer_1_1 = sklearn_node(transformer = SimpleImputer, config= {'strategy': 'most_frequent'}).after(DataEncoder_1_1)
		DataEncoder_1_1 = sklearn_node(transformer = DataEncoder, config= {'columns': None}).after(SelectKBest_1_1)
		SelectKBest_1_1 = sklearn_node(transformer = SelectKBest, config= {'score_func': 'f_regression', 'k': 'all'}).after(SelectFromModel_1_1)
		LogisticRegression_1_1 = sklearn_node(transformer = LogisticRegression, config= {'C': [1, 3], 'tol': [0.0001, 0.001], 'penalty': ['l1', 'l2']})

	@phyml.pipeline(app.pipelines, experiment="Perform Fraud detection.")
	def Fraud_Detection_pipeline(message: str = "Your face app is Ready!"):

		select_into_1 = pipeline.select_into()
		export_csv_1 = pipeline.export_csv()
		analyze_categorical_features_1 = pipeline.analyze_categorical_features()
		build_matrix_1 = pipeline.build_matrix()
		custom_model_1 = pipeline.custom_model()
		random_forest_op_1 = pipeline.random_forest_op()

    @phyml.kf_predictor(app.inference)
    class SklearnCustomModel(kfserving.KFModel):
        def __init__(self,
                    name='sklearn-serving',
                    model_path='gs://physarum-phyml-serving/physarum/applications/customer_app/wfid-local/artifacts/v2/mypipeline2/encoder.joblib',
                    encoder_path='gs://physarum-phyml-serving/physarum/applications/customer_app/wfid-local/artifacts/v2/mypipeline2/model.joblib',):

            super().__init__(name)
            self.name = name
            self.model_path = _download_uri(model_path, '/tmp/model.joblib')
            self.encoder_path = _download_uri(encoder_path, '/tmp/encoder.joblib')
            self.ready = False

        def load(self):
            self.pipeline = joblib.load(self.model_path)
            self.encoder = joblib.load(self.encoder_path)
            # logging.info('Model loaded !!')
            self.ready = True

        def predict(self, request: Dict) -> Dict:
            inputs = request['instances']
            df = pd.DataFrame(inputs)
            predictions = self.pipeline.predict(df)
            predictions = self.encoder.inverse_transform(predictions)
            probability = self.pipeline.predict_proba(df)
            results = []
            for pred,prob in zip(predictions,probability.values):
                results.append({'prediction':str(pred), 'probability':prob[0]})
            # time.sleep(3)
            return {'predictions': results}
	@phyml.deploy_kf_inference(app.inference)
	def deploy_inference(deployment: phyml.PhymlInferenceDeployment):
		(
		deployment
		.inference_graph()
			.add_model(name='sklearn_test3')
			.image_pull_choice('IfNotPresent')
			.with_resources(limit_cpu='1milicpu', limit_memory='1Gi', request_cpu='0.5milicpu', request_memory='1Gi')

		.canary_graph(canary_traffic_percent=0)
			.add_model(name='sklearn_test3')
			.image_pull_choice('IfNotPresent')
			.with_resources(limit_cpu='1milicpu', limit_memory='1Gi', request_cpu='0.5milicpu', request_memory='1Gi')

		.build_service(namespace='default', autoscaling_target=0)
		)
	app.start()


if __name__ == "__main__":
	main()
