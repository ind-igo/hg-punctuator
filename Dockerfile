FROM continuumio/miniconda3

WORKDIR /app

ADD https://hieroglyph.s3.amazonaws.com/Demo-Europarl-EN.pcl .
COPY . .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "punctuator_service", "/bin/bash", "-c"]
#RUN python -c "import nltk; nltk.download('punkt')"
RUN python -m nltk.downloader punkt

# The code to run when container is started:
EXPOSE 8080
ENTRYPOINT ["conda", "run", "-n", "punctuator_service", "python", "server.py"]