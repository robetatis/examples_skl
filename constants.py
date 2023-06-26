path_to_data = 'data/churn.csv'

primary_key = 'customerID'

features_to_encode = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
    'PhoneService', 'MultipleLines',  'InternetService', 'PaymentMethod', 
    'PaymentMethod', 'PaperlessBilling','Contract', 'StreamingMovies',
    'StreamingTV', 'TechSupport', 'DeviceProtection', 'OnlineBackup',
    'OnlineSecurity', 'Dependents', 'Partner'
    ]

numeric_features = [
    'tenure', 'MonthlyCharges', 'TotalCharges',
]

y_name = 'Churn'

X_names = [
    'gender_Female', 'gender_Male', 'SeniorCitizen_0', 'SeniorCitizen_1',
    'Partner_No', 'Partner_Yes', 'Dependents_No', 'Dependents_Yes',
    'PhoneService_No', 'PhoneService_Yes', 'MultipleLines_No',
    'MultipleLines_No_phone_service', 'MultipleLines_Yes',
    'InternetService_DSL', 'InternetService_Fiber_optic', 'InternetService_No',
    'PaymentMethod_Bank_transfer_automatic',
    'PaymentMethod_Credit_card_automatic', 'PaymentMethod_Electronic_check',
    'PaymentMethod_Mailed_check', 'PaymentMethod_Bank_transfer_automatic',
    'PaymentMethod_Credit_card_automatic', 'PaymentMethod_Electronic_check',
    'PaymentMethod_Mailed_check', 'PaperlessBilling_No', 'PaperlessBilling_Yes',
    'Contract_Month-to-month', 'Contract_One_year', 'Contract_Two_year',
    'StreamingMovies_No', 'StreamingMovies_No_internet_service',
    'StreamingMovies_Yes', 'StreamingTV_No', 'StreamingTV_No_internet_service',
    'StreamingTV_Yes', 'TechSupport_No', 'TechSupport_No_internet_service',
    'TechSupport_Yes', 'DeviceProtection_No',
    'DeviceProtection_No_internet_service', 'DeviceProtection_Yes',
    'OnlineBackup_No', 'OnlineBackup_No_internet_service', 'OnlineBackup_Yes',
    'OnlineSecurity_No', 'OnlineSecurity_No_internet_service',
    'OnlineSecurity_Yes', 'Dependents_No', 'Dependents_Yes', 'Partner_No',
    'Partner_Yes', 'tenure', 'MonthlyCharges', 'TotalCharges',
]

