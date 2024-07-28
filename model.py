import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.impute import SimpleImputer
import shap
import time
import matplotlib.pyplot as plt
import io

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
    
    # Encoding categorical features
    code = {'Acceptable(Normale)': 1, 'Précaire': 2, 'Alarmante(Alerte)': 3, 'Critique(Urgence)': 4}  
    for col in df.select_dtypes('object').columns:
        df[col] = df[col].replace(code)
    
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
    
    return food_security_model, explainer, X_train

def predict(model, features):
    return model.predict(features)

def calculate_shap(model_explainer, features):
    model, explainer = model_explainer
    features = features.apply(pd.to_numeric, errors='coerce')
    
    # Calculating SHAP values
    shap_values = explainer.shap_values(features)
    
    # Return SHAP values
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
    region_name_lower = region_name.lower()
    return region_mapping.get(region_name_lower, -1)

def interpret_shap(feature_names, shap_values, region_name):
    interpretations = []
    if len(shap_values.shape) == 2:  # If shap_values is a 2D array (multiple features)
        for i in range(shap_values.shape[1]):
            feature_name = feature_names[i]
            shap_value = shap_values[0, i]  # Use the first instance for explanation
            interpretations.append(
                f"For the region {region_name}, the value of the feature '{feature_name}' has an impact of {shap_value:.4f} on the prediction."
            )
    else:  # If shap_values is a 1D array (single feature)
        for i, val in enumerate(shap_values):
            feature_name = feature_names[i]
            interpretations.append(
                f"For the region {region_name}, the value of the feature '{feature_name}' has an impact of {val:.4f} on the prediction."
            )
    return interpretations

def main():
    st.title("Prediction of Phenomena")

    # Load the model
    model, explainer, X_train = load_model()

    # User interface to input features
    region_name = st.text_input("Enter the region:")
    region_numeric = map_region_to_numeric(region_name)

    if region_numeric == -1:
        st.warning("The entered region is not valid. Please enter a valid region.")
        return

    date = st.date_input("Enter the date:")
    year = date.year
    month = date.month
    day = date.day

    # Button to make the prediction
    if st.button("Predict"):
        start_prediction_time = time.time()

        # Preprocess features
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
        features1 = features1.reindex(columns=X_train.columns)
        features2 = features2.reindex(columns=X_train.columns)
        features3 = features3.reindex(columns=X_train.columns)

        # Make predictions
        prediction1 = predict(model, features1)
        prediction2 = predict(model, features2)
        prediction3 = predict(model, features3)

        # Display predictions
        st.write(f"Prediction of Underweight: {prediction1}")
        st.write(f"Prediction of Chronic Malnutrition: {prediction2}")
        st.write(f"Prediction of Acute Malnutrition: {prediction3}")

        prediction_time = time.time() - start_prediction_time
        st.write(f"Prediction time: {prediction_time:.4f} seconds")

        # Calculate SHAP values
        shap_values = calculate_shap((model, explainer), features1)

        # Display SHAP values
        st.write("SHAP Values:")
        st.write(shap_values)

        # Interpret SHAP values
        st.write("Interpretation of SHAP Values:")
        feature_names = X_train.columns

        for feature_index in range(len(feature_names)):
            if feature_index < len(shap_values[0]):
                feature_name = feature_names[feature_index]
                shap_value = shap_values[0][feature_index]

                interpretation = interpret_shap(feature_names, shap_value, region_name)
                st.write(interpretation)

     # Plot SHAP heatmap
        st.write("SHAP Heatmap:")
        shap_plot_io = io.BytesIO()

        # Flatten the SHAP values if needed
        shap_values_flat = np.array(shap_values).reshape(-1, len(features1.columns))

        # Create the heatmap plot
        plt.figure(figsize=(12, 8))  # Adjust the figure size as needed
        shap_values_df = pd.DataFrame(shap_values_flat, columns=features1.columns)
        sns.heatmap(shap_values_df, annot=True, cmap="coolwarm", cbar=True)
        plt.title("SHAP Heatmap showing the impact of each feature on the model's predictions.", fontsize=12)
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels if necessary

        # Adjust layout to ensure everything fits
        plt.tight_layout()

        plt.savefig(shap_plot_io, format='png')
        plt.close()

        shap_plot_io.seek(0)
        st.image(shap_plot_io, caption="SHAP Heatmap")

if __name__ == "__main__":
    main()
