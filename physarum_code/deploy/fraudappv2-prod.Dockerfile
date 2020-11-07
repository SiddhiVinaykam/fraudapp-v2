#FROM gcr.io/streaming-project-277811/physarum/physarum-sklearn-dev:physarum-sklearn-0.2
FROM gcr.io/streaming-project-277811/physarum/physarum-serving-test:0.5
# manual installation
RUN pip install xgboost

# add the application level code
ADD ./src /pkg_src/fraudappv2
WORKDIR /pkg_src/fraudappv2
RUN pip install -e .
WORKDIR /fraudappv2
