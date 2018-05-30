# Instructions copied from - https://hub.docker.com/_/python/
FROM python:3-onbuild

# Meta-data
LABEL maintainer="Chun-Yi Sung <juneyi1@gmail.com>" \
      description="This image contains a pickled version of the predictive model\
      for predcting the ICD9 code based on medeical condition input by a user.\
      The model can be accessed by a Web App (created in Flask).\
      Fitting and pickling was done in Model.py"

# tell the port number the container should expose
EXPOSE 5000

# fetch app specific deps
#RUN python -m nltk.downloader punkt

# run the command
CMD ["python", "./web_app.py"]
