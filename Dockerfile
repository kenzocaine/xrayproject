FROM python:3.8.6-buster

COPY api /api
COPY xrayproject /xrayproject
COPY model_architecture.json /model_architecture.json
COPY model_weights.h5 /model_weights.h5
COPY requirements.txt /requirements.txt

RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT
