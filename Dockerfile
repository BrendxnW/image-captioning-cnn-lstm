FROM python:3.13.7

WORKDIR /src

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "src.scripts.generate_caption"]