from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import *
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats as st
from statsmodels.stats.diagnostic import het_breuschpagan

def PredictionReport(X_values: pd.DataFrame, Y_values: pd.DataFrame, classifier: str = "Gaussian") -> None:
    Observation = X_values.to_numpy()
    Results = Y_values.to_numpy().ravel()
    X_train, X_test, y_train, y_test = train_test_split(Observation, Results, test_size=0.2, random_state=0)
    
    gnb = GaussianNB()
    if classifier == "Multinomial":
        gnb = MultinomialNB()
    elif classifier == "Complement":
        gnb = ComplementNB()
    elif classifier == "Categorical":
        gnb = CategoricalNB()
        
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    print("Number of mislabeled points out of a total %d points : %d"
          % (X_test.shape[0], (y_test != y_pred).sum()))
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Generate classification report
    class_report = classification_report(y_test, y_pred)
    
    # Print classification report
    print("\nClassification Report:")
    print(class_report)
    values_index = Y_values['Injury Severity'].unique().tolist()
    values_index.sort()
    cm_df = pd.DataFrame(conf_matrix,
                     index = values_index, 
                     columns = values_index)
    #Plotting the confusion matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_df, annot=True, fmt='g')
    plt.title('Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    plt.show()

    unique_values, counts = np.unique(y_pred, return_counts=True)
    print("Predicted values:")
    for value, count in zip(unique_values, counts):
        print(f"{value} occurs {count} times")
    unique_values, counts = np.unique(y_test, return_counts=True)
    
    print("\nActual values:")
    for value, count in zip(unique_values, counts):
        print(f"{value} occurs {count} times")

    return None

def Prepare_Data_for_Prediction(df) -> list:
    LightingCond = {'Dark-Street Lights On':3, 'Dark-No Street Lights':5, 'Daylight':1,
       'Unknown':1, 'Dark-Street Lights Off':4, 'Dawn':0,
       'Dark - Unknown Lighting':5, 'Dusk':2, 'Other':1}

    WeatherCond = {'Clear':0,
              'Partly Cloudy': 1,
              'Overcast': 2,
              'Raining': 5,
              'Snowing': 6,
              'Sleet or Hail or Freezing Rain': 7,
              'Other':0, 'Severe Crosswind':4,
       'Fog or Smog or Smoke':3, 'Blowing Sand or Dirt or Snow':8}
    
    Known_Injury = df[(df["Injury Severity"] != "Unknown Injury Collision")]
    X = Known_Injury[['Associated State Road Number','Mile Post','Intersection Related', 'Weather Condition',
       'Lighting Condition', 'Motor Vehicles Involved',
       'Passengers Involved', 'Commercial Carrier Involved',
       'School Bus Involved', 'Pedestrians Involved', 'Pedalcyclists Involved',
       'AADT']]
    values = {"Lighting Condition": 'Unknown', "Weather Condition": 'Other', 'Passengers Involved':0.0, 'Commercial Carrier Involved':0.0,
       'School Bus Involved':0.0, 'Pedestrians Involved':0.0, 'Pedalcyclists Involved':0.0}
    X=X.fillna(value=values)
    X=X.dropna()
    Conditions = ['Weather Condition','Lighting Condition']
    for Condition in Conditions:
        use_list = []
        states = []
        if Condition == 'Weather Condition':
            use_dict = WeatherCond
            states = X['Weather Condition'].unique()
        else:
            use_dict = LightingCond
            states = X['Lighting Condition'].unique()
        for i in np.arange(X[Condition].unique().shape[0]):
            X.loc[X[Condition] == states[i], Condition] = use_dict[states[i]]
    Y = Data[['Injury Severity']].iloc[X.index.to_list()]

    return {"X": X, "Y":Y}

def AADT_Distribution(df)-> None:
    print(df['AADT'].value_counts())
    plt.hist(df['AADT'].to_numpy(), bins = 100)
    plt.show
    print(st.mode(df['AADT'].to_numpy()))

def LSData(df: pd.DataFrame, AADT: int = 0):

    RData = df[df['County'].notna() & df['Weather Condition'].notna() & df['Lighting Condition'].notna()]
    RData = RData.join(pd.get_dummies(RData[['County','Weather Condition','Lighting Condition']], dtype=float))
    RData = RData.drop(['County','Weather Condition','Lighting Condition',
        'Collision Report Number', 'Collision Date', 'City', 'Jurisdiction',
       'Agency', 'Primary Trafficway', 'Primary Road Number',
       'Secondary Trafficway', 'Secondary Road Number', 'Block Number',
                   'Object Struck', 'Mile Post', 'Associated State Road Number'], axis=1)

    if AADT in df['AADT'].unique():
        RData = RData[RData['AADT'] == AADT]
        RData = RData.drop(['AADT'], axis = 1)
    
    RData = RData.replace({'Injury Severity': {"No Injury Collision": 0, 
                                   'Minor Injury Collision': 0,
                                                 "Unknown Injury Collision":0,
                                  "Serious Injury Collision": 1, 
                                   'Fatal Collision': 1}})
    RData = RData.dropna()
    RY = RData['Injury Severity']
    RX = RData.drop(['Injury Severity'], axis=1)
    X_model = sm.add_constant(RX)
    model = sm.OLS(RY, X_model)
    results = model.fit(cov_type = "HC1")
    print(results.summary())
    return None

if __name__ == '__main__':
    # Load the dataset
    car_crash_data = pd.read_csv('Car_Crash_Cleaned_AADT.csv')
    X, Y = Prepare_Data_for_Prediction(car_crash_data)[0], Prepare_Data_for_Prediction(car_crash_data)[1]
    PredictionReport(X, Y, "Categorical")
    PredictionReport(X, Y, "Complement")
    AADT_Dist(car_crash_data)
    LSData(car_crash_data)