FROM python:3.9.12-slim

# install pipenv
RUN  python -m pip install pipenv

# set a directory for the app
WORKDIR /app

# copy dependencies into a working directory
COPY ["Pipfile", "Pipfile.lock", "./"]

# install dependencies
RUN pipenv install --deploy --system

# copy model and flask applicantion to working directory
COPY ["model.py", "model.bin", "./"]

# define the port number the container should expose
EXPOSE 9696

# commad to run app on docker
ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:9696", "model:app"]