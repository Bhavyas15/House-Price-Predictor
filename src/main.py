import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import xgboost
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

df=pd.read_csv('data/final_df (1).csv')
with open(r'data/HousePricePredictionModel (1).pkl', 'rb') as file:
    model = pickle.load(file)
with open("data/data_description.txt", "r") as file:
    content = file.read()

option = st.sidebar.radio('Select an option:', ['Price Prediction', 'Exploratory Data Analysis', 'Information'])
st.title('House Price Prediction 🏡')
numcols=[]
catcols=[]
for feature in df.columns:
    if df[feature].dtype=='O':
        catcols.append(feature)
    else:
        numcols.append(feature)
discrete_cols=[feature for feature in numcols if len(df[feature].unique())<25]
cont_cols=[feature for feature in numcols if len(df[feature].unique())>=25]

if option=='Price Prediction':
    st.header('Enter Preferences and Predict')
    #Handling discrete valued columns
    OverallQual=st.number_input("Overall material and finish quality", 1,10,step=1,value=int(df['OverallQual'].mode()[0]))
    BsmtFullBath=st.number_input('Basement full bathrooms',0,5,step=1,value=int(df['BsmtFullBath'].mode()[0]))
    FullBath=st.number_input('Full bathrooms',1,8,step=1,value=int(df['FullBath'].mode()[0]))
    HalfBath=st.number_input('Half bathrooms',0,5,step=1,value=int(df['HalfBath'].mode()[0]))
    BedroomAbvGr=st.number_input('Bedrooms', 1,15,step=1,value=int(df['BedroomAbvGr'].mode()[0]))
    Fireplaces=st.number_input('Fireplaces', 0,5,step=1,value=int(df['Fireplaces'].mode()[0]))

    # Continuous Variables
    LotFrontage=st.slider('Lot Frotage', 20,350,value=int(df['LotFrontage'].mode()[0]))
    LotArea=st.slider('Lot Area', 1000,22000,value=int(df['LotArea'].mode()[0]))
    YearBuilt=st.slider('Original Construction Year',1850,2020,value=int(df['YearBuilt'].mean()))
    YearRemodAdd=st.slider('Remodel Year',YearBuilt,2025,value=int(df['YearRemodAdd'].mean()))
    MasVnrArea=st.slider('Masonry veneer area in square feet',0,2000,value=int(df['MasVnrArea'].mean()))
    BsmtFinSF1=st.slider('Basement Finished Area',0,6000,value=int(df['BsmtFinSF1'].mean()))
    BsmtUnfSF=st.slider('Basement Unfinished Area',0,2500,value=int(df['BsmtUnfSF'].mean()))
    TotalBsmtSF=BsmtFinSF1+BsmtUnfSF
    st.write(f'Total Basement Surface Area: {TotalBsmtSF}')
    FFlrSF=st.slider('First Floor Surface Area',300,6000,value=int(df['1stFlrSF'].mode()[0]))
    SFlrSF=st.slider('Second Floor Surface Area',0,FFlrSF+100,value=int(df['2ndFlrSF'].mode()[0]))
    GrLivArea=FFlrSF+SFlrSF
    st.write(f'Above Ground Living Area: {GrLivArea}')
    GarageArea=st.slider('Garage Area',0,1500,value=int(df['GarageArea'].mean()))
    WoodDeckSF=st.slider('Wood Deck Area',0,1500,value=int(df['WoodDeckSF'].mean()))
    OpenPorchSF=st.slider('Open Porch Area ',0,800,value=int(df['OpenPorchSF'].mean()))


    col1,col2,col3=st.columns([1,1,1])
    columns=[col1,col2,col3]
    # Convert categorical inputs to DataFrame and apply one-hot encoding
    cat_inp=[]
    i=0
    for feature in catcols:
        sel_input=columns[i%3].radio(f'Select {feature}',df[feature].unique())
        cat_inp.append(sel_input)
        i+=1

    def category_onehot_multicols(multicols):
        df_final=df
        i=0
        for fields in multicols:
            df1=pd.get_dummies(df[fields],drop_first=True)

            final=df.drop([fields],axis=1,inplace=True)
            if i==0:
                df_final=df1.copy()
            else:
                df_final=pd.concat([df_final,df1], axis=1)
            i+=1

        df_final=pd.concat([df,df_final], axis=1)
        return df_final

    final_df=category_onehot_multicols(catcols)
    final_df=final_df.loc[:,~final_df.columns.duplicated()]
    final_df= final_df.replace({False: 0, True: 1})
    ffinal_df=final_df.drop(['SalePrice'],axis=1)

    b=pd.DataFrame(columns=ffinal_df.columns[20:])
    new_row = {col: 1 if col in cat_inp else 0 for col in b.columns}
    b = pd.concat([b, pd.DataFrame([new_row])], ignore_index=True)

    # Covert numerical features
    inputs_num=pd.DataFrame([[LotFrontage,LotArea,OverallQual,YearBuilt,YearRemodAdd,MasVnrArea,
            BsmtFinSF1,BsmtUnfSF,TotalBsmtSF,FFlrSF,SFlrSF,GrLivArea,BsmtFullBath,
            FullBath,HalfBath,BedroomAbvGr,Fireplaces,GarageArea,WoodDeckSF,OpenPorchSF]], 
            columns=df.drop(columns=['SalePrice']).select_dtypes(exclude=['object']).columns)
    inputs=pd.concat([inputs_num,b], axis=1)

    #Normalization
    scaler = MinMaxScaler()
    ffinal_df=scaler.fit_transform(ffinal_df)
    # ffinal_df
    inputs_scaled=scaler.transform(inputs)
    # inputs_scaled
    exp_output=model.predict(inputs_scaled)

    st.header(f'Predicted Price : $ {np.round(exp_output[0])}')

elif option=='Information':
    st.header('Information of Parameters \n\n')
    st.write(content)

else:
    c1,c1,c3=st.columns([1,5,1])
    st.write('\n')
    st.subheader('Relationship with Discrete Features')
    for feature in discrete_cols:
        fig, ax = plt.subplots(figsize=(8, 5))  # Create a figure
        df.groupby(feature)['SalePrice'].median().plot()
        ax.set_xlabel(feature)
        ax.set_ylabel('Sales Price')
        ax.set_title(f'{feature} vs Sale Price')
        
        st.pyplot(fig)  # Display plot in Streamlit
    st.subheader('Relationship with Continuous Features')
    for feature in cont_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        # Plot histogram
        ax.hist(df[feature], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Histogram of {feature}', fontsize=14)
        plt.grid(True)
        st.pyplot(fig)
    st.subheader('Relationship with Categorical Features')
    for feature in catcols:
        # Create a figure for Streamlit
        fig, ax = plt.subplots(figsize=(8, 5))

        # Create bar plot of median SalePrice for each category
        df.groupby(feature)['SalePrice'].median().plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
        # Customize plot
        ax.set_title(f'{feature} vs Median SalePrice', fontsize=14)
        plt.xticks()  # Rotate x-axis labels for better readability

        # Display plot in Streamlit
        st.pyplot(fig)
