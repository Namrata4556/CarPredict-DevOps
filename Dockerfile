FROM python:3.12-slim

WORKDIR /app

COPY app.py .
COPY used_car_prediction.py .
COPY used_car_price_dataset.csv .

RUN pip install --no-cache-dir streamlit pandas numpy scikit-learn matplotlib seaborn

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
