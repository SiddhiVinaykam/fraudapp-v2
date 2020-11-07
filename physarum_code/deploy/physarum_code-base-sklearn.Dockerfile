FROM gcr.io/streaming-project-277811/physarum/physarum-phyml-dev:physarum-0.1

# add the core library
ADD ./src/phy-sarum/ /pkg_src/phy-sarum
RUN pip uninstall physarum
WORKDIR /pkg_src/phy-sarum
RUN pip install -e .


# add the sklearn library
ADD ./src/sklearn /pkg_src/sklearn
RUN pip uninstall sklearn
WORKDIR /pkg_src/sklearn
RUN pip install -e .

# manual installation
RUN pip install xgboost