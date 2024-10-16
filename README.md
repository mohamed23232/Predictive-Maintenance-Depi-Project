# Predictive Maintenance System

This project focuses on applying machine learning to a predictive maintenance use case. In industrial settings, sensors are deployed to monitor equipment, network devices, and other field assets. These sensors help predict when components need to be replaced or repaired before failure occurs, avoiding costly downtime and poor customer experiences. Predictive maintenance models can also augment regular schedule-based maintenance by preventing unnecessary replacements of parts that are still in good condition, reducing operational costs.

## Project Structure

- `data/`: Contains the dataset used for the project, sourced from the UCI AI4I 2020 Predictive Maintenance Dataset.
- `notebooks/`: Includes Jupyter notebooks for data exploration, feature selection, and model evaluation.
- `src/`: The main source code for data preprocessing, model training, and evaluation.
- `app/`: Flask web application for deploying the machine learning model.
- `models/`: Saved machine learning models, including the Voting Classifier used for prediction.
- `README.md`: Documentation of the project.

## Dataset

The **AI4I 2020 Predictive Maintenance Dataset** is synthetic and contains features such as:

- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear
- Machine failure indicator (target label)

Failures are categorized into five modes: **Tool Wear Failure (TWF), Heat Dissipation Failure (HDF), Power Failure (PWF), Overstrain Failure (OSF),** and **Random Failure (RNF)**.

**Dataset link**
https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020/code

### Objective

The primary goal of this project is to develop a machine learning model that can correctly predict machine failures, so maintenance teams can take preventive action before a failure causes downtime.

## Key Features

- **Handling Class Imbalance**: We addressed class imbalance by up-sampling the minority class (machine failures) to improve model performance.
- **Feature Selection**: Core features like torque, rotational speed, and tool wear were selected based on correlation with machine failures.
- **Modeling Approach**: We employed a **Voting Classifier**, combining multiple models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost) to enhance predictive accuracy and reduce overfitting.

## Results

The final **Voting Classifier** model achieved an accuracy of **99.4%**, offering robust and generalized predictions by combining the strengths of individual models.

## Web Application

We deployed the model using a Flask web application. Users can input machine parameters to receive predictions on whether a machine will fail. The app scales input data using the pre-trained MinMaxScaler and makes predictions using the Voting Classifier.

### Running the Application

1. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

2. Run the Flask application:
    ```bash
    python app.py
    ```

3. Access the web interface at `http://localhost:5000`.

## Contributors

- Mohamed Ahmed Rabea
- Muhammad Usama Aowad
- Useif Muhammed Saad Ebrahim
- Ahmed Abdelsalam Abdelmaqsoud
