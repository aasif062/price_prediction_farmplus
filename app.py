import pandas as pd
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

# Initialize Flask app
app = Flask(__name__)

# Global variables for model and encoders
model = None
label_encoders = {}

# Train and save the model if not already saved
def train_and_save_model():
    global model, label_encoders

    # Path to dataset
    dataset_path = 'dataset.csv'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset not found. Please place the dataset.csv file in the same directory.")

    # Load dataset
    df = pd.read_csv(dataset_path)

    # Handle missing values
    df.fillna(method='ffill', inplace=True)

    # Encode categorical columns
    categorical_columns = ['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade']
    for column in categorical_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Define features and target variable
    X = df[['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade']]
    y = df['Modal Price']

    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Random Forest Regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model and encoders to disk
    joblib.dump(model, 'crop_price_model.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')

    # Evaluate model (Optional, for logging)
    y_pred = model.predict(X_test)
    print("Model Evaluation:")
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R-squared:", r2_score(y_test, y_pred))
    print("Model and label encoders saved successfully!")

# Load the trained model and label encoders
def load_model_and_encoders():
    global model, label_encoders
    if os.path.exists('crop_price_model.pkl') and os.path.exists('label_encoders.pkl'):
        model = joblib.load('crop_price_model.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
    else:
        train_and_save_model()

# Define route for the main page
@app.route('/')
def home():
    return render_template('crop_pred.html')

# Define route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Retrieve form inputs
        State = request.form['State']
        District = request.form['District']
        Market = request.form['Market']
        Commodity = request.form['Commodity']
        Variety = request.form['Variety']
        Grade = request.form['Grade']

        # Prepare the input data
        input_data = pd.DataFrame([[State, District, Market, Commodity, Variety, Grade]],
                                  columns=['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade'])

        # Encode categorical features
        try:
            for column in ['State', 'District', 'Market', 'Commodity', 'Variety', 'Grade']:
                input_data[column] = label_encoders[column].transform(input_data[column])
        except KeyError:
            return render_template('crop_pred.html', prediction="Invalid input: Some values are not recognized.")

        # Make a prediction
        predicted_price = model.predict(input_data)[0]
        predicted_price_int = int(predicted_price)

        # Return prediction to the HTML page
        return render_template('crop_pred.html', prediction=f"{predicted_price_int} Rupees Per Quintal")

# if __name__ == '__main__':
#     # Load or train model and encoders
#     load_model_and_encoders()

#     # Run the Flask app
#     app.run(debug=True)
