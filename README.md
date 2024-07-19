# Disaster Response Pipeline Project

### Summary
This project's goal is to provide a way for messages sent during/after a disaster to be classified into categories.
These classifications can help first responders and other personnel to triage and provide aid to those most in need.

### Files in this repository
```
- app
  - templates
  - requirements.txt
  - run.py
  - starting_verb_extrator.py
- data
  - disaster_categories.csv
  - disaster_messages.csv
  - DisasterResponse.db
  - process_data.py
- models
  - starting_verb_extrator.py
  - train_classifier.py
```

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/
