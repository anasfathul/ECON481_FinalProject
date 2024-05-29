import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def randomForest_function(df1):
    """
    Function that generate features importance from random forest model with 100 tree
    """
    # Filter if needed
    #car_crash_data = car_crash_data[(car_crash_data["County"] == "King") & (car_crash_data["City"] == "Seattle")]
    car_crash_data = df1
    
    # Convert the "Collision Date" column to datetime
    car_crash_data['Collision Date'] = pd.to_datetime(car_crash_data['Collision Date'], errors='coerce')
    
    # Dropping specified columns
    columns_to_drop = [
        'Primary Road Number', 'Secondary Trafficway', 
        'Secondary Road Number', 'Block Number', 'Mile Post',
        'Object Struck', 'Associated State Road Number'
    ]
    car_crash_data_cleaned = car_crash_data.drop(columns=columns_to_drop)
    
    # Encoding categorical variables
    label_encoder = LabelEncoder()
    for column in ['Weather Condition', 'Lighting Condition', 'Jurisdiction',
                   'Agency', 'Primary Trafficway', 'City', 'County']:
        car_crash_data_cleaned[column] = label_encoder.fit_transform(car_crash_data_cleaned[column])
    
    # Features and target variable
    X = car_crash_data_cleaned.drop(columns=['Collision Report Number', 'Collision Date', 'Injury Severity'])
    y = car_crash_data_cleaned['Injury Severity']
    
    # Encode target variable
    y = label_encoder.fit_transform(y)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build a Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_classifier.predict(X_test)
    
    # Evaluation
    classification_rep = classification_report(y_test, y_pred)
    confusion_mat = confusion_matrix(y_test, y_pred)
    
    # Get feature importances
    feature_importances = rf_classifier.feature_importances_
    feature_names = X.columns
    
    # Create a DataFrame for the feature importances
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)
    
    # Print evaluation metrics
    print("Classification Report:\n", classification_rep)
    print("Confusion Matrix:\n", confusion_mat)
    print("Top 10 Feature Importances:\n", importance_df)
    
    # Visualizing feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

def weather_condition(df2):
    # Grouping data by Lighting Condition and calculating summary statistics
    weather_summary = df2.groupby('Weather Condition').agg({
        'Collision Report Number': 'count',
        'Motor Vehicles Involved': 'sum',
        'Passengers Involved': 'sum',
        'Commercial Carrier Involved': 'sum',
        'School Bus Involved': 'sum',
        'Pedestrians Involved': 'sum',
        'Pedalcyclists Involved': 'sum',
        'AADT': 'mean'
    }).reset_index()
    
    # Renaming columns for clarity
    weather_summary.rename(columns={
        'Collision Report Number': 'Total Collisions',
        'Motor Vehicles Involved': 'Total Vehicles Involved',
        'Passengers Involved': 'Total Passengers Involved',
        'Commercial Carrier Involved': 'Total Commercial Carriers Involved',
        'School Bus Involved': 'Total School Buses Involved',
        'Pedestrians Involved': 'Total Pedestrians Involved',
        'Pedalcyclists Involved': 'Total Pedalcyclists Involved',
        'AADT': 'Average AADT'
    }, inplace=True)
    
    # Bar plot for total collisions by lighting condition
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Weather Condition', y='Total Collisions', data=weather_summary, palette='viridis')
    plt.title('Total Collisions by Weather Condition')
    plt.xlabel('Weather Condition')
    plt.ylabel('Total Collisions')
    plt.xticks(rotation=45)
    plt.show()

def lighting_condition(df3):
    # Grouping data by Lighting Condition and calculating summary statistics
    lighting_summary = df3.groupby('Lighting Condition').agg({
        'Collision Report Number': 'count',
        'Motor Vehicles Involved': 'sum',
        'Passengers Involved': 'sum',
        'Commercial Carrier Involved': 'sum',
        'School Bus Involved': 'sum',
        'Pedestrians Involved': 'sum',
        'Pedalcyclists Involved': 'sum',
        'AADT': 'mean'
    }).reset_index()
    
    # Renaming columns for clarity
    lighting_summary.rename(columns={
        'Collision Report Number': 'Total Collisions',
        'Motor Vehicles Involved': 'Total Vehicles Involved',
        'Passengers Involved': 'Total Passengers Involved',
        'Commercial Carrier Involved': 'Total Commercial Carriers Involved',
        'School Bus Involved': 'Total School Buses Involved',
        'Pedestrians Involved': 'Total Pedestrians Involved',
        'Pedalcyclists Involved': 'Total Pedalcyclists Involved',
        'AADT': 'Average AADT'
    }, inplace=True)
    
    # Bar plot for total collisions by lighting condition
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Lighting Condition', y='Total Collisions', data=lighting_summary, palette='viridis')
    plt.title('Total Collisions by Lighting Condition')
    plt.xlabel('Lighting Condition')
    plt.ylabel('Total Collisions')
    plt.xticks(rotation=45)
    plt.show()

def stratified_sample_RF(df4):

    # filter out state road > 100
    filtered_df = df4.groupby('Associated State Road Number').filter(lambda x: len(x) > 100)
    # sample 100 from each state road
    sampled_df = filtered_df.groupby('Associated State Road Number').apply(lambda x: x.sample(100)).reset_index(drop=True)
    # call rf function
    randomForest_function(sampled_df)

if __name__ == '__main__':
    # Load the dataset
    car_crash_data = pd.read_csv('Car_Crash_Cleaned_AADT.csv')
    randomForest_function(car_crash_data)
    weather_condition(car_crash_data)
    lighting_condition(car_crash_data)
    





