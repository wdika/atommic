ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:23.08-py3

# build an image that includes only the atommic dependencies, ensures that dependencies
# are included first for optimal caching, and useful for building a development
# image (by specifying build target as `atommic-deps`)
FROM ${BASE_IMAGE} as atommic-deps

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update &&  \
    apt-get upgrade -y &&  \
    apt-get install -y --no-install-recommends \
    libsndfile1 sox \
    libfreetype6 \
    swig && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/

# install atommic dependencies
WORKDIR /tmp/

COPY requirements .
RUN for f in requirements*.txt; do pip3 install --disable-pip-version-check --no-cache-dir -r $f; done

# copy atommic source into a scratch image
FROM scratch as atommic-src
COPY . .

# start building the final container
FROM atommic-deps as atommic
ARG ATOMMIC_VERSION=1.0.1

# Check that atommic_VERSION is set. Build will fail without this. Expose atommic and base container
# version information as runtime environment variable for introspection purposes
RUN /usr/bin/test -n "$ATOMMIC_VERSION" && \
  /bin/echo "export ATOMMIC_VERSION=${ATOMMIC_VERSION}" >> /root/.bashrc && \
  /bin/echo "export BASE_IMAGE=${BASE_IMAGE}" >> /root/.bashrc

# Install ATOMMIC
RUN --mount=from=atommic-src,target=/tmp/atommic cd /tmp/atommic && pip install --no-cache-dir ".[all]" || bash

RUN --mount=from=atommic-src,target=/tmp/atommic cd /tmp/atommic && pip install atommic || bash

# Check that the module can be imported
RUN python -c "import atommic"

# Check install
RUN python -c "import atommic.collections.multitask.rs as atommic_mrs" &&  \
    python -c "import atommic.collections.quantitative as atommic_qmri" && \
    python -c "import atommic.collections.reconstruction as atommic_rec" && \
    python -c "import atommic.collections.segmentation as atommic_seg"

# copy projects/tools/tests into container for end user
WORKDIR /workspace/atommic
COPY projects /workspace/atommic/projects
COPY tests /workspace/atommic/tests
COPY tools /workspace/atommic/tools
COPY tutorials /workspace/atommic/tutorials

RUN printf "#!/bin/bash\njupyter lab --no-browser --allow-root --ip=0.0.0.0" >> start-jupyter.sh && \
  chmod +x start-jupyter.sh \
