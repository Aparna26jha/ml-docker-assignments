# RentApp MLOps and Jenkins Assignment

This repository contains a simple house rent prediction machine learning application and Jenkins practice files.

## Applications

### Rent Prediction Application

The `app.py` program:

- Loads rent data from a CSV file
- Trains a Linear Regression model
- Evaluates the model
- Predicts the monthly rent of an example property

### Jenkins Python Application

The `hello_jenkins.py` program prints a Hello World message and is executed through a Jenkins Freestyle job.

## Technologies Used

- Python
- pandas
- scikit-learn
- Git and GitHub
- Docker
- Jenkins
- Ubuntu

## Run the RentApp

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py