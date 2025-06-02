FROM python:3

WORKDIR D:\ShapeSynth

COPY env.yml .

RUN conda env create -f env.yml

SHELL ["conda", "run". "-n", "ShapeSynth", "/bin/bash", "-c"]

COPY requirements.txt .

RUN pip -install -r requirements. txt

COPY . .

CMD [ "conda", "run", "-n", "ShapeSynth", "python"]