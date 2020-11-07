FROM gcr.io/physarum-streaming/physarum/physarum-tekton-base

# Install application
ADD ./src /pkg_src/fraudappv2
WORKDIR /pkg_src/fraudappv2
RUN pip install -e .
WORKDIR /fraudappv2