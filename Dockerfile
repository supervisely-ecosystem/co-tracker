FROM supervisely/cotracker:1.0.1

RUN pip install git+https://github.com/supervisely/supervisely.git@bbox_tracking_debug

COPY . /repo

WORKDIR /