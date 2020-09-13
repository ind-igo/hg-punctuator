FROM continuumio/miniconda3

WORKDIR /app

COPY . .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "punctuator_service", "/bin/bash", "-c"]
#RUN python -c "import nltk; nltk.download('punkt')"
RUN python -m nltk.downloader punkt

# The code to run when container is started:
EXPOSE 8080
ENTRYPOINT ["conda", "run", "-n", "punctuator_service", "python", "server.py"]