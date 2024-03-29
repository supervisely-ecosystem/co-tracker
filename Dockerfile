FROM supervisely/cotracker:1.0.1

RUN pip install git+https://github.com/supervisely/supervisely.git@bbox_tracking_debug

WORKDIR /app
COPY supervisely_integration/serve /app

EXPOSE 80

ENV APP_MODE=production ENV=production

ENTRYPOINT ["python3", "-u", "-m", "uvicorn", "src.main:model.app"]
CMD ["--host", "0.0.0.0", "--port", "80"]
