FROM python:3.9.12-slim

# install pipenv
RUN python -m pip install pipenv

# set a directory for the app
WORKDIR  /app

# copy dependecies into a working directory
COPY ["Pipfile", "Pipfile.lock", "./"]

# install dependencies
RUN pipenv install --deploy --system

# copy model and flask application with template folder to working directory
COPY ["app.py", "model.bin", ".", "./"]

# define the port number the container should expose
EXPOSE $PORT

# run command to run app on heroku local server
CMD ["waitress-serve", "--listen=0.0.0.0:$PORT", "app:app"]
 