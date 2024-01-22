import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
import shap
import time

def load_model():
    start_time = time.time()  # Recording start time
    
    # Loading data
    print("Loading data...")
    data = pd.read_excel('./etat_securite.xlsx')
    print("Data loaded successfully.")
    
    df = data.copy()
    
    # Data preprocessing
    features = df.drop('Situation_Surpoids', axis=1)
    target = df['Situation_Surpoids']
    
    # Encoding
    code = {'Acceptable(Normale)': 1, 'Précaire': 2, 'Alarmante(Alerte)': 3, 'Critique(Urgence)': 4}  
    for col in df.select_dtypes('object').columns:
        df.loc[:, col] = df[col].replace(code)
    
    # Imputing missing values for all columns with 'most_frequent' strategy
    imputer = SimpleImputer(strategy='most_frequent')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    
    # Splitting data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df_imputed, target, test_size=0.2, random_state=42)
    
    print(f"Debug - Data Shape: {df.shape}")
    print(f"Debug - X_train Shape: {X_train.shape}, y_train Shape: {y_train.shape}")

    # Creating and training the model
    print("Training the model...")
    food_security_model = SVC(kernel='linear', C=1)
    food_security_model.fit(X_train, y_train)
    print("Model trained successfully.")

    # Initializing SHAP with the trained model
    print("Initializing SHAP...")
    explainer = shap.Explainer(food_security_model, X_train)
    print("SHAP initialized successfully.")

    # Recording end time and calculating duration
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Model loading time: {elapsed_time:.2f} seconds")
    
    return food_security_model, explainer, X_train, y_train

def predict(model, features):
    food_security_model = model
    prediction = food_security_model.predict(features)[0]
    return prediction

def calculate_shap(model, features):
    _, explainer = model
    features = features.apply(pd.to_numeric, errors='coerce')
    
    # Calculating SHAP values
    shap_values = explainer.shap_values(features)
    
    # print(shap_values)

    return shap_values

# Function to map region names to numeric values
def map_region_to_numeric(region_name):
    region_mapping = {
        "madagascar": 0,
        "diana": 1,
        "sava": 2,
        "itasy": 3,
        "analamanga": 4,
        "vakinankaratra": 5,
        "bongolava": 6,
        "sofia": 7,
        "boeny": 8,
        "betsiboka": 9,
        "melaky": 10,
        "alaotra-mangoro": 11,
        "antsinana": 12,
        "analanjirofo": 13,
        "ambatosoa": 14,
        "amoron'i mania": 15,
        "vatovavy": 16,
        "fitovinany": 17,
        "haute matsiatra": 18,
        "atsimo-atsinanana": 19,
        "ihorombe": 20,
        "menabe": 21,
        "atsimo-andrefana": 22,
        "androy": 23
    }

    # Convert region to lowercase before looking in the dictionary
    region_name_lower = region_name.lower()

    # Return the corresponding numeric value if it exists, otherwise return -1 or a default value
    return region_mapping.get(region_name_lower, -1)

def interpret_shap(feature_names, shap_values, region_name):
    interpretations = []

    if shap_values.ndim == 1:
        for i, val in enumerate(shap_values):
            feature_name = feature_names[i]
            interpretations.append(
                f"For the region {region_name}, the value of the feature '{feature_name}' has an impact of {val:.4f} on the prediction."
            )
    else:
        interpretations.append(
            f"For the region {region_name}, the value of the feature has multiple components and cannot be interpreted simply. The components are: {shap_value.tolist()}"
        )

    return interpretations

def main():
    st.title("Prediction of Phenomena")

    # Load the model
    model, explainer, X_train, y_train = load_model()

    # User interface to input features
    region_name = st.text_input("Enter the region:")
    region_numeric = map_region_to_numeric(region_name)

    # Check if the region is valid
    if region_numeric == -1:
        st.warning("The entered region is not valid. Please enter a valid region.")
        return

    date = st.date_input("Enter the date:")

    # Extract features from date
    year = date.year
    month = date.month
    day = date.day

    # Button to make the prediction
    if st.button("Predict"):
        start_prediction_time = time.time()  # Record prediction start time

          # Preprocess features (excluding "Overweight_Situation")
        features1 = pd.DataFrame({
            "DATE": [year],
            "REGION": [region_numeric],
            "Situation_Surpoids": [3],
            "Situation_MC": [1], 
            "Situation_MA": [2],
            "Proportion_population_insuffisance_calorique": [5],
            "EPIDEMOLOGIE": [60],
            "Taux_couverture_vaccinale_complete": [28],
            "Taille_moyenne_menages_Individus": [4],
            "Pourcentage_femmes_couvertes_assurance_maladie": [2],
            "Pourcentage_membres_menages_toilettes_ameliorees": [20],
            "Pourcentage_membres_menages_lieu_lavage_mains_eau_savon_detergent": [26]
        })
        features2 = pd.DataFrame({
            "DATE": [year],
            "REGION": [region_numeric],
            "Situation_Surpoids": [2],
            "Situation_MC": [1], 
            "Situation_MA": [4],
            "Proportion_population_insuffisance_calorique": [70],
            "EPIDEMOLOGIE": [70],
            "Taux_couverture_vaccinale_complete": [18],
            "Taille_moyenne_menages_Individus": [5],
            "Pourcentage_femmes_couvertes_assurance_maladie": [0.5],
            "Pourcentage_membres_menages_toilettes_ameliorees": [8],
            "Pourcentage_membres_menages_lieu_lavage_mains_eau_savon_detergent": [2]
        })
        features3 = pd.DataFrame({
            "DATE": [year],
            "REGION": [region_numeric],
            "Situation_Surpoids": [4],
            "Situation_MC": [1], 
            "Situation_MA": [3],
            "Proportion_population_insuffisance_calorique": [40],
            "EPIDEMOLOGIE": [20],
            "Taux_couverture_vaccinale_complete": [52],
            "Taille_moyenne_menages_Individus": [4],
            "Pourcentage_femmes_couvertes_assurance_maladie": [3],
            "Pourcentage_membres_menages_toilettes_ameliorees": [6],
            "Pourcentage_membres_menages_lieu_lavage_mains_eau_savon_detergent": [26]
        })
        
        # Use X_train column names to ensure consistency
        features1.columns = X_train.columns

        # Make predictions
        prediction1 = predict(model, features1)
        prediction2 = predict(model, features2)
        prediction3 = predict(model, features3)

        # Display predictions
        st.write(f"Prediction of Underweight: {prediction1}")
        st.write(f"Prediction of Chronic Malnutrition: {prediction2}")
        st.write(f"Prediction of Acute Malnutrition: {prediction3}")

        # Calculate and display SHAP values
        shap_values = calculate_shap((model, explainer), features1)
        
         # Record end time of prediction and calculate duration
        end_prediction_time = time.time()
        elapsed_prediction_time = end_prediction_time - start_prediction_time
        print(f"Prediction time: {elapsed_prediction_time:.4f} seconds")

        # Get descending order of feature indices by impact
        feature_order = list(reversed(np.argsort(shap_values[0])))

        print(f"Debug - Feature Order: {feature_order}")
        
        #SHAP Values
        st.write("SHAP Values:")
        st.write(f"{shap_values}")

        # Display SHAP results with comments suitable for nutritionists

        st.write("Interpretation of SHAP Values:")

        feature_names = features1.columns 

        for feature_index in feature_order[0]:
            if 0 <= feature_index < len(shap_values[0]):
                feature_name = feature_names[feature_index]  
                shap_value = shap_values[0][feature_index]

                print(f"Debug - Feature Index: {feature_index}, Feature Name: {feature_name}, SHAP Value Shape: {shap_value.shape}")

                interpretation = interpret_shap(feature_names, shap_value, region_name)

                st.write(interpretation)

                print(f"Debug - All SHAP Values for {feature_name}:\n{shap_value}")

if __name__ == "__main__":
    main()
