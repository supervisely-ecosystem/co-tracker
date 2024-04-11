FROM supervisely/cotracker:1.0.1

RUN pip install git+https://github.com/supervisely/supervisely.git@inference-improvements

WORKDIR /app
COPY . /app

EXPOSE 80

ENV PYTHONPATH "${PYTHONPATH}:/app/supervisely_integration/serve/src"
ENV APP_MODE=production ENV=production

ENTRYPOINT ["python3", "-u", "-m", "uvicorn", "supervisely_integration.serve.src.main:model.app"]
CMD ["--host", "0.0.0.0", "--port", "80"]
