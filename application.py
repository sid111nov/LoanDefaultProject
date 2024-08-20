import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer

model = tf.keras.models.load_model('./artifacts/model.h5')

with open('artifacts/imputers.pickle','rb') as f:
    imputers = pickle.load(f)

with open('artifacts/encoders.pickle','rb') as f:
    encoders = pickle.load(f)    

cat_imputer = imputers['SimpleImputer']
num_imputer = imputers['KNNImputer']

   
scaler = encoders['scaler']
ohencoders = encoders['ohencoders']

st.title('Loan Default Predictor')

loan_year = st.selectbox('Year of Loan',range(2000,2024))
loan_limit = st.selectbox('loan limit',ohencoders['loan_limit'].categories_[0])
gender = st.selectbox('gender',ohencoders['Gender'].categories_[0])
approv_in_adv = st.selectbox('approval in advance',ohencoders['approv_in_adv'].categories_[0])
loan_type = st.selectbox('approval in advance',ohencoders['loan_type'].categories_[0])
loan_purpose = st.selectbox('purpose of loan',ohencoders['loan_purpose'].categories_[0])
Credit_Worthiness = st.selectbox('creditworthiness of loanee',ohencoders['Credit_Worthiness'].categories_[0])
open_credit = st.selectbox('open credit accounts',ohencoders['open_credit'].categories_[0])
business_or_commercial = st.selectbox('business or commercial account',ohencoders['business_or_commercial'].categories_[0])
loan_amount = st.number_input('Loan Amount')

rate_of_interest = st.number_input('interest rate charged on the loan')
interest_rate_spread = st.number_input('difference between the interest rate on the loan and a benchmark interest rate')
upfront_charges	= st.number_input('initial charges associated with securing the loan')
term = st.number_input('duration of the loan in months')

neg_ammortization = st.selectbox('does loan allows for negative ammortization',ohencoders['Neg_ammortization'].categories_[0])	 
interest_only = st.selectbox('loan has an interest-only payment option',ohencoders['interest_only'].categories_[0])
lump_sum_payment = st.selectbox('if a lump sum payment is required at the end of the loan term',ohencoders['lump_sum_payment'].categories_[0])	
property_value = st.number_input('value of the property being financed')
construction_type = st.selectbox('type of construction',ohencoders['construction_type'].categories_[0])

occupancy_type = st.selectbox('occupancy type',ohencoders['occupancy_type'].categories_[0])	
Secured_by	= st.selectbox('secured by',ohencoders['Secured_by'].categories_[0]) 
total_units	= st.selectbox( 'number of units in the property being financed', ohencoders['total_units'].categories_[0])

income = st.number_input('applicant\'s annual income')
credit_type	 =  st.selectbox('applicant\'s type of credit',ohencoders['credit_type'].categories_[0])  
Credit_Score = 	st.number_input('applicant\'s credit score')
co_applicant_credit_type = st.selectbox('co-applicant\'s type of credit',ohencoders['co-applicant_credit_type'].categories_[0])

age = st.selectbox('age of applicant',ohencoders['age'].categories_[0])	

submission_of_application = st.selectbox('how the application was submitted',ohencoders['submission_of_application'].categories_[0]) 
ltv = st.number_input('loan-to-value ratio')	
region = st.selectbox('geographic region where the property is located', ohencoders['Region'].categories_[0] )
security_Type = st.selectbox('type of security or collateral backing the loan',ohencoders['Security_Type'].categories_[0])	 

dtir1 = st.number_input('debt-to-income ratio')	

cat_col_frame = [ohencoders['loan_limit'].transform([[loan_limit]])[0], ohencoders['Gender'].transform([[gender]])[0] , 
           ohencoders['approv_in_adv'].transform([[approv_in_adv]])[0],
           ohencoders['loan_type'].transform([[loan_type]])[0],
           ohencoders['loan_purpose'].transform([[loan_purpose]])[0],
        ohencoders['Credit_Worthiness'].transform([[Credit_Worthiness]])[0], 
        ohencoders['open_credit'].transform([[open_credit]])[0] ,
          ohencoders['business_or_commercial'].transform([[business_or_commercial]])[0] ,
        ohencoders['Neg_ammortization'].transform([[neg_ammortization]])[0] ,
        ohencoders['interest_only'].transform([[interest_only]])[0] , 
       ohencoders['lump_sum_payment'].transform([[lump_sum_payment]])[0], 
       ohencoders['construction_type'].transform([[construction_type]])[0] , 
       ohencoders['occupancy_type'].transform([[occupancy_type]])[0] ,
       ohencoders['Secured_by'].transform([[Secured_by]])[0] , 
       ohencoders['total_units'].transform([[total_units]])[0] ,
        ohencoders['credit_type'].transform([[credit_type]])[0] , 
        ohencoders['co-applicant_credit_type'].transform([[co_applicant_credit_type]])[0] ,
         ohencoders['age'].transform([[age]])[0] ,
        ohencoders['submission_of_application'].transform([[submission_of_application]])[0] , 
        ohencoders['Region'].transform([[region]])[0] , 
        ohencoders['Security_Type'].transform([[security_Type]])[0] ]

cat_col = [loan_limit,gender ,            approv_in_adv,  loan_type,
           loan_purpose,        Credit_Worthiness,         open_credit ,
          business_or_commercial ,        neg_ammortization ,        interest_only , 
       lump_sum_payment,        construction_type ,        occupancy_type ,
       Secured_by ,        total_units ,        credit_type , 
        co_applicant_credit_type ,        age ,
        submission_of_application ,         region ,        security_Type ]

num_col = [loan_year, loan_amount, rate_of_interest, interest_rate_spread,
        upfront_charges, term, property_value, income, Credit_Score,
        ltv, dtir1]

cat_col_names = ['loan_limit', 'Gender', 'approv_in_adv', 'loan_type', 'loan_purpose',
        'Credit_Worthiness', 'open_credit', 'business_or_commercial',
        'Neg_ammortization', 'interest_only', 'lump_sum_payment',
        'construction_type', 'occupancy_type', 'Secured_by', 'total_units',
        'credit_type', 'co-applicant_credit_type', 'age',
        'submission_of_application', 'Region', 'Security_Type']

num_col_names = ['year', 'loan_amount', 'rate_of_interest', 'Interest_rate_spread',
        'Upfront_charges', 'term', 'property_value', 'income', 'Credit_Score',
        'LTV', 'dtir1']

df_catagorical = pd.DataFrame({col: pd.Series(dtype='object') for col in cat_col_names})

df_catagorical.loc[0] = cat_col

df_numerical = pd.DataFrame({col: pd.Series(dtype='float64') for col in num_col_names})

df_numerical.loc[0] = num_col


data_check = pd.concat([df_numerical, df_catagorical], axis=1)



data_check[num_col_names] = num_imputer.transform(data_check[num_col_names])

data_check[cat_col_names] = cat_imputer.transform(data_check[cat_col_names])

data_check[num_col_names] = scaler.transform(data_check[num_col_names])





loan_limit_enc = ohencoders['loan_limit'].transform([[loan_limit]]) 
gender_enc = ohencoders['Gender'].transform([[gender]]) 
approv_in_adv_enc = ohencoders['approv_in_adv'].transform([[approv_in_adv]])
loan_type_enc = ohencoders['loan_type'].transform([[loan_type]])
loan_purpose_enc =  ohencoders['loan_purpose'].transform([[loan_purpose]])
Credit_Worthiness_enc =  ohencoders['Credit_Worthiness'].transform([[Credit_Worthiness]])
open_credit_enc =    ohencoders['open_credit'].transform([[open_credit]])
business_or_commercial_enc  =   ohencoders['business_or_commercial'].transform([[business_or_commercial]])
Neg_ammortization_enc =    ohencoders['Neg_ammortization'].transform([[neg_ammortization]])
interest_only_enc =        ohencoders['interest_only'].transform([[interest_only]])
lump_sum_payment_enc =        ohencoders['lump_sum_payment'].transform([[lump_sum_payment]])
construction_type_enc =        ohencoders['construction_type'].transform([[construction_type]])
occupancy_type_enc =        ohencoders['occupancy_type'].transform([[occupancy_type]])
Secured_by_enc =        ohencoders['Secured_by'].transform([[Secured_by]])
total_units_enc =        ohencoders['total_units'].transform([[total_units]])
credit_type_enc =         ohencoders['credit_type'].transform([[credit_type]]) 
co_applicant_credit_type_enc =         ohencoders['co-applicant_credit_type'].transform([[co_applicant_credit_type]])
age_enc =          ohencoders['age'].transform([[age]])
submission_of_application_enc =         ohencoders['submission_of_application'].transform([[submission_of_application]]) 
region_enc =         ohencoders['Region'].transform([[region]])
security_Type_enc =         ohencoders['Security_Type'].transform([[security_Type]])


loan_limit_encoded_df= pd.DataFrame(loan_limit_enc,columns=ohencoders['loan_limit'].get_feature_names_out())
gender_encoded_df= pd.DataFrame(gender_enc,columns=ohencoders['Gender'].get_feature_names_out())
approv_in_adv_df= pd.DataFrame(approv_in_adv_enc,columns=ohencoders['approv_in_adv'].get_feature_names_out())
loan_type_encoded_df= pd.DataFrame(loan_type_enc,columns=ohencoders['loan_type'].get_feature_names_out())
loan_purpose_enc_encoded_df= pd.DataFrame(loan_purpose_enc,columns=ohencoders['loan_purpose'].get_feature_names_out())
Credit_Worthiness_encoded_df= pd.DataFrame(Credit_Worthiness_enc,columns=ohencoders['Credit_Worthiness'].get_feature_names_out())
open_credit_encoded_df= pd.DataFrame(open_credit_enc,columns=ohencoders['open_credit'].get_feature_names_out())
boc_encoded_df= pd.DataFrame(business_or_commercial_enc,columns=ohencoders['business_or_commercial'].get_feature_names_out())
Neg_ammortization_encoded_df= pd.DataFrame(Neg_ammortization_enc,columns=ohencoders['Neg_ammortization'].get_feature_names_out())
interest_only_encoded_df= pd.DataFrame(interest_only_enc,columns=ohencoders['interest_only'].get_feature_names_out())
lump_sum_payment_encoded_df= pd.DataFrame(lump_sum_payment_enc,columns=ohencoders['lump_sum_payment'].get_feature_names_out())
construction_type_encoded_df= pd.DataFrame(construction_type_enc,columns=ohencoders['construction_type'].get_feature_names_out())

occupancy_type_encoded_df= pd.DataFrame(occupancy_type_enc,columns=ohencoders['occupancy_type'].get_feature_names_out())
Secured_by_encoded_df= pd.DataFrame(Secured_by_enc,columns=ohencoders['Secured_by'].get_feature_names_out())
total_units_encoded_df= pd.DataFrame(total_units_enc,columns=ohencoders['total_units'].get_feature_names_out())
credit_type_encoded_df= pd.DataFrame(credit_type_enc,columns=ohencoders['credit_type'].get_feature_names_out())
cact_encoded_df= pd.DataFrame(co_applicant_credit_type_enc,columns=ohencoders['co-applicant_credit_type'].get_feature_names_out())


age_encoded_df= pd.DataFrame(age_enc,columns=ohencoders['age'].get_feature_names_out())
soa_encoded_df= pd.DataFrame(submission_of_application_enc,columns=ohencoders['submission_of_application'].get_feature_names_out())
region_encoded_df= pd.DataFrame(region_enc,columns=ohencoders['Region'].get_feature_names_out())
security_Type_encoded_df= pd.DataFrame(security_Type_enc,columns=ohencoders['Security_Type'].get_feature_names_out())


df_catagorical_combined = pd.concat([loan_limit_encoded_df,gender_encoded_df,approv_in_adv_df,loan_type_encoded_df,
loan_purpose_enc_encoded_df,Credit_Worthiness_encoded_df,open_credit_encoded_df,
boc_encoded_df,Neg_ammortization_encoded_df,interest_only_encoded_df,lump_sum_payment_encoded_df,construction_type_encoded_df,
occupancy_type_encoded_df,Secured_by_encoded_df,total_units_encoded_df,credit_type_encoded_df,
cact_encoded_df,age_encoded_df,soa_encoded_df,region_encoded_df,security_Type_encoded_df], axis=1)

input_data = pd.concat([df_numerical, df_catagorical_combined], axis=1)



prediction = model.predict(input_data)

if(prediction ==1 ):
    st.write("The person will default")

else:
    st.title("The person will not default") 