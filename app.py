import pickle
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

df = pd.read_csv("zindi_checkpoint/Expresso_churn_dataset.csv", low_memory=False)
df = df.dropna().reset_index()
df_head = df.head()

#the classifier and scaler
with open('zindi_checkpoint/Expresso_churn_model.pkl', "rb") as model_file:
    classifier = pickle.load(model_file)
###########################################################################
with open('zindi_checkpoint/Expresso_scaler.pkl', "rb") as scaler_file:
    scaler = pickle.load(scaler_file)
###########################################################################

#python function to predict churn
def predict_churn(features):
    scaled_features = scaler.transform([features])
    prediction = classifier.predict(scaled_features)[0]
    return prediction

#columns to use for predictions
columns = ['REGION', 'TENURE', 'MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 
           'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO', 'ZONE1', 'ZONE2', 
           'REGULARITY', 'TOP_PACK', 'FREQ_TOP_PACK']


# partie streamlit
st.image("https://saaiassociation.co.za/wp-content/uploads/2023/05/maxresdefault.jpg")

st.title("ZINDI Predictions App")
st.header("Expresso Telecommunications")
st.dataframe(df_head)

#
st.sidebar.title("ZINDI Sidebar")
fig = px.histogram(df, x="CHURN", y="REGION")
st.sidebar.plotly_chart(fig, use_container_width=True)

data_volume_by_region = df.groupby('REGION')['DATA_VOLUME'].mean()
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(data_volume_by_region.index, data_volume_by_region.values, color='skyblue')
ax.set_title('Average Data Volume by Region')
ax.set_ylabel('Average Data Volume')
ax.set_xlabel('Region')
plt.xticks(rotation=45, ha='right') 
st.pyplot(fig)

st.markdown("""
    <style>   /* balise style qui represente le code CSS */
    .main {
        background-color: grey;
    }
    .title {
        font-size: 36px;
        font-weight: bold;
        color: #53f805;
        font-family: cursive;
    }
    .subheader {
        font-size: 20px;
        color: #53f805;
        font-family: cursive;    
    }
    .input-field {
        margin: 10px 0;
    }
    .predict-button {
        background-color: #000000;
        color: white;
        font-size: 20px;
        padding: 10px;
        margin: 20px 0;
    }
    .st-bc {
    background-color: #ff94871f;
}
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title"> Expresso Churn Prediction </div>', unsafe_allow_html=True)
st.markdown('<div class="subheader"> Please enter the customer details below: </div>', unsafe_allow_html=True)

features = []
form = st.form("Expresso Churn Prediction")
#adding the features to the form
columns = ['REGION', 'TENURE', 'MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT', 
           'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO', 'ZONE1', 'ZONE2', 
           'REGULARITY', 'TOP_PACK', 'FREQ_TOP_PACK']
for column in columns:
    feature = form.number_input(f'{column}', key=f'{column}')
    features.append(feature)

predict_button = form.form_submit_button("Predict")
if predict_button:
    container = st.container(border=True)
    prediction = predict_churn(features)
    if prediction == 1:
        container.write("Customer is likely to leave soon!")
    else:
        container.write("Customer is likely to stay.")

