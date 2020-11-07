#ARG GCP_PROJECT=random

#FROM gcr.io/$GCP_PROJECT/physarum/physarum-phyml-dev
FROM gcr.io/physarum-streaming/physarum/physarum-phyml-dev

# Install gcloud
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && apt-get update -y && apt-get install google-cloud-sdk -y

# Install kubectl
RUN mkdir /kubectl
WORKDIR /kubectl
RUN curl -LO https://storage.googleapis.com/kubernetes-release/release/`curl -s https://storage.googleapis.com/kubernetes-release/release/stable.txt`/bin/linux/amd64/kubectl
RUN chmod +x ./kubectl
ENV PATH="${PATH}:/kubectl"

# add the core library
ADD ./src/phy-sarum/ /pkg_src/phy-sarum
RUN pip uninstall -y physarum
WORKDIR /pkg_src/phy-sarum
RUN pip install -e .

# will be moved to the base image - only temporarily here
# RUN apt-get update && apt-get install -y gcc unixodbc-dev
# add the sklearn library
ADD ./src/sklearn/src/sklearn/ /pkg_src/sklearn
RUN pip uninstall -y sklearn
WORKDIR /pkg_src/sklearn
RUN pip install -e .

# manual installation
RUN pip install xgboost

ADD ./src/physerving/ /pkg_src/physerving
RUN pip uninstall -y physerving
WORKDIR /pkg_src/physerving
RUN pip install -e .
