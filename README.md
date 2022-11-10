# ML Zoomcamp Midterm Project

This repository contains code and document of the mid-term project **Home Loan Approval Prediction** for the course ML Zoomcamp 2022 conducted by Alexey Grigorev.

> * Project Link: https://home-loan-approval.herokuapp.com/
> 
> * Course Link: [mlbookcamp-code/course-zoomcamp at master · alexeygrigorev/mlbookcamp-code · GitHub](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp)

## 1. About the project

A loan is a sum of money that one or more individuals or companies borrow from banks or other financial institutions; to financially manage events. In doing so, the borrower incurs a debt, which he has to pay back with interest within a given period. 

Loans can be education loans or personal loans. The most common type of loan is a **Home Loan** or **Mortgage**. A mortgage is a type of secured loan which is taken for the purchase of a property.

 A typical process for home loans starts with an application from the customer; after that company will validate the eligibility of the customer for a loan, based on the result application can be accepted or rejected. The loan approval is the most crucial stage of a loan process and takes a long time to complete. We can automate this process of eligibility of customers for a home loan by using a Machine Learning algorithm.

**Problem Statement**

The project objective is to build a web application using machine learning model to predict whether a particular customer is eligible for a home loan based on the customer details provided while filling out the online application form.

**The dataset**

The dataset is taken from the analytics vidya hackathon platform. You can download the data from [here](https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/#ProblemStatement). There are three data files available to download while I have used only the `train.csv` file to train and validate a machine learning model.

This dataset consist of 13 attributes and  614 records of customer details such as loan id number, gender, marital status, income of applicant and coapplicant loan amount etc.

This is binary classification problem, where we will predict the `Loan_Status` which describes `Y` for application that are approved by company or bank and `N` for the rejected applications.

The description of data attributes is as follows:

| Attributes        | Description                                                        |
| ----------------- | ------------------------------------------------------------------ |
| Loan_ID           | unique loan id                                                     |
| Gender            | gender male/female                                                 |
| Married           | Whether applicant is married or not (Y/N)                          |
| Dependents        | Number of dependents on applicant                                  |
| Education         | Education level of applicant(graduate/undergraduate)               |
| Self_Employed     | self employed (y/n)                                                |
| ApplicantIncome   | Applicant income                                                   |
| CoapplicantIncome | Coapplicant income                                                 |
| LoanAmount        | Loan amount in thousands                                           |
| Loan_Amount_Term  | Amount of time the lender gives you to repay your laon.(in months) |
| Credit_History    | if Credit history meets guidlines 1/ 0                             |
| Property_Area     | urban/semiurban / rural                                            |
| Loan_Status       | loan approved(y/n)                                                 |


## 2. Folder Structure

The following tree graph shows the folder structure of this repository.

```gitignore
|-- README.md
|-- data.csv
|-- notebook.ipynb
|-- train.py
|-- model.bin
|-- predict.py
|-- request.py
|-- Pipenv
|-- Pipenv.lock
|-- \local-deploy
|   |-- model.py
|   |-- Dockerfile
|   |-- model.bin
|   |-- Pipfile
|   |-- Pipfile.lock
|   |-- request.py
|-- \cloud-deploy
|   |-- templates
|   |   |-- index.html
|   |-- app.py
|   |-- Dockerfile
|   |-- model.bin
|   |-- Pipfile
|   |-- Pipfile.lock
|   |-- Procfile
|--\images
```

* **README.md**
  
  > This is markdown file contains description document for the project code and its deployment on local machine and on cloud platform called `Heroku`.

* **data.csv**
  
  > `data.csv` is dataset used for model development and training.

* **notebook.ipynb**
  
  > This  jupyter notebook contains data preparation, explotatory data analysis, data transformation, various models training and parameter tunning on datafile.

* **train.py**
  
  > Python script to develop a best selected model from jupyter notebook model training and to save a model and preprocessor in a pickle file

* **model.bin**
  
  > Saved pickle file from train.py contains classifier and preprocessor instances. Classifier is machine learning classifier with tunned parameter and Preprocessor is a pipeline of column tansformer to transform the data.

* **predict.py**
  
  > Python script to load `model.bin` pickle file and to create web service application using flask framework.

* **request.py**
  
  > Python script to test flask web service application on local machine.

* **Pipenv,  Pipenv.lock**
  
  > Contains all dependencies required for project.

* **Dockerfile**
  
  > Dockerfile contains a list of commands that Docker client calls while creating an image for this project.

* **Local deploy**
  
  > This directory contains all required files to deploy web application on local machine. 
  > 
  > * `Dockerfile` contains docker image for the project. 
  > 
  > * `pipenv` and `pipenv.lock` for dependencies.
  > 
  > * `model.bin` pickle file for machine learning model. 
  > 
  > * `model.py` is flask application script for web service.
  > 
  > * `request.py` to test application.

* **Cloud deploy**
  
  >  This directory contains all required files to deploy web application on cloud plotform `Heroku`.
  > 
  > * `Dockerfile` contains docker image for the project.
  > 
  > * `pipenv` and `pipenv.lock` for dependencies.
  > - `model.bin` pickle file for machine learning model.
  > 
  > - `app.py` is flask application script file.
  > 
  > - `templates` directory contains `index.html` html file for home page.
  > 
  > - `Procfile` is necessary for heroku deployment which contains command for execution of an app on startup.

* **Images**
  > It contains the screenshot images of web applicantion.

## 3. Approach

### 3.1 Exploratory Data Analysis

After a data preparation, Correlation data analysis and data visualization performed to find relation between feature variables and response variable. Further, to confirm the findings from visualization, `chi-square` hypothesis test perfromed on categorical variables while `ANOVA` test is performed on numerical variables.

#### 3.1.1 Chi-2 hypothesis test for categorical variables

To find the relation between categorical features and response variable.

* H0(null hypothesis): feature and response variable(`loan_status`) are dependent.

* H1(alternate hypothesis): feature and response variable are not dependent.

The result shows that, status of loan approval does not depends on feature variables `gender`, `dependents`, and  `self-employed`  while it shows strong correlation with feature `credit_history` and `property_area`.

#### 3.1.2 ANOVA hypothesis test for numerical variables

To find if there is statistically significant difference in distribution of means of numerical features for the groups of categories of response variable.

- H0(null hypothesis): The group mean of each class of response variable(`Y/N`) is same.

- H1(alternate hypothesis): There is difference between these two group means.

The result shows that there is no statistical significant difference in group mean of each category of response variable for both `applicant_income` and `coapplicant_income` features.

### 3.2 Data Transformation

* Log transformation performed on  `applicant_income`, `coapplicant_income` and `loan_amount` to make long tail distribution of these variables into a normal distribution.

* `Preprocessor` pipeline is created to perform data transformation using `ColumnTransformer` . A `StandardScaler` is used on numerical data for standardization after filling missing values using `KNNImputer` with `n_neighbors=3` and `OneHotEncoder` on categorical data to transform data after filling missing data with `mode` value using `SimpleImputer`.

* For further evaluation, the data is split into training, validation, and test datasets in a ratio of 70:20:10.

### 3.3 Model Evaluation & Optimization

* ***F1-Score*** : The f1-score is selected for model evaluation since the class distribution of the response variable is imbalanced.

* First, base model `LogisiticRegression` algorithm is used to test overall performance. Next, 10-fold cross validation is perfromed and evaluated using f1-score on several machine learning classification algorithms. Among these `RandomForestClassifier`, `LogisticRegression`, `GradientBoostingClassifier` and `SupportVectorMachine` performs well. Therefore, selected for further evaluation and parameter optimization.

* Various parameters tune using cross-validation on training data. The value that gives a higher f1-score is selected.

* **GradientBoostingClassifier**
  
  > * n_estimators = 250
  > 
  > * learning_rate = 0.03
  > 
  > * max_depth = 3
  > 
  > * min_samples_split = 200
  > 
  > * min_samples_leaf = 50
  > 
  > * max_features = 11
  > 
  > * warm_start = True
  > 
  > * random_state = 42

* **RandomForestClassifier**
  
  > - n_estimators = 110
  > 
  > - max_depth = 15
  > 
  > - min_samples_split = 3
  > 
  > - min_samples_leaf = 1
  > 
  > - max_features = 'sqrt'
  > 
  > - n_jobs = -1
  > 
  > - random_state = 42

* **SupportVectorMachine**
  
  > * C = 3
  > 
  > * gamma = 0.1
  > 
  > * random_state = 42

* **LogisticRegression**
  
  > * C = 0.1
  > 
  > * random_state = 42

* `GradientBoostingClassifier` performs well on both validation and test data with an f1-score of 0.7789 and 0.7801 respectively. Hence it is selected as a final model.

## 4. Dependency and Environment Management

To run all the scripts of this project without any error locally, you must install all libraries and dependencies used during model training and deployment. Python `Pipenv` is used to manage, all virutal environment and package dependecies.

The steps to install `pipenv` and package dependencies on the windows operating system are as follows.

1. To Install pipenv using pip on Windows OS, open a command prompt and run the following command.
   
   ```
   python -m pip install pipenv
   ```

2. Download the project from the GitHub repository, or you can clone this repository.
   
   ```
   git clone enter-link-here
   ```

3. Open the folder and go to the directory where Pipenv and Pipenv.lock file is present. Create a virtual environment and install all dependencies from  `Pipenv.lock` file.
   
   ```
   pipenv install -d
   ```

4. Activate the pipenv virtual environment using the following command.
   
   ```
   pipenv shell
   ```
   
   Now, you can train a model in this virtual environment by running the train.py script.
   
   ```
   python train.py
   ```
   
   The train.py will train a `GradientBoostingClassifier` model and save a model and preprocessor as a `model.bin` file

### 4.1 Model deployment as a web service on local machine

The `predict.py` is a script for the flask application to deploy the model locally and to test the model deployment. We have to run the  `request.py` python script.

Open a command prompt, activate the virtual environment, and run the following command to run the `predict.py` web service application.

```
waitress-serve --listen=0.0.0.0:9696 predict:app
```

Open another command prompt in the same directory as above and run `request.py` t o test / make request to this web service. 

```
python request.py
```

### 4.2 Local deployment as Docker Container

The `local_deploy` folder contains all code and dependencies to deploy a model in docker. Here, `model.py` is the same prediction model web application `predict.py` 

Steps required to deploy the model as web service inside a docker container on a local machine.

To containerize a web application in docker on a local machine, you must have installed Docker on your computer.  You can download Docker Desktop [here](https://docs.docker.com/desktop/install/windows-install/). Read the manual to install docker on the system properly.

After installation, open the Docker Desktop application window which takes some time to start.

After running the docker, Open the command prompt in the model file directory and run the following command to build a docker image named `loan_approval`

```
docker build -t loan_approval .
```

Create a docker container from the image and run an application using following command.

```
docker run -it -p 9696:9696 loan_approval:latest
```

Again, you can test or request web service running on docker container by running python scriptAgain, you can test or request web service running on the docker container by running python script `request.py`  on another command prompt from the same directory.

```
python request.py
```

### 4.3 Cloud deployment as a web service on Heroku

The `cloud_deploy` folder contains all code and dependencies to deploy a model on the cloud platform Heroku using the docker container. Here, `app.py` is a prediction model web application with a front end.

The following screen shot shows the deployed the model as a web service on the cloud and its response from the web service.

***Web application Homepage***
![index](https://github.com/madhuri-15/home-loan-approval-prediction/blob/main/images/homepage.png)


***Request***
![index](https://github.com/madhuri-15/home-loan-approval-prediction/blob/main/images/input.png)


***Response***
![index](https://github.com/madhuri-15/home-loan-approval-prediction/blob/main/images/output.png)
