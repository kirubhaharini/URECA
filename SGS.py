import streamlit as st
import pandas as pd
import datetime
import plotly.express as px
from windrose import WindroseAxes
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math
from st_btn_select import st_btn_select
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import numpy as np
import matplotlib.pyplot as plt
import pickle

'''
Dashboard for Serangoon Secondary School
Data from 2020 to 2022
'''

def SGS(state):
    df = pd.read_excel('SGS after outlier removal.xlsx')

    #for presenting stats
    st.markdown(f"""<h3 style='text-align: left; color: darkgoldenrod;'>{'Summary'}</h3>""", unsafe_allow_html=True)
    col1,col2,col3 = st.columns([2,2,2])
    col4,col5,col6 = st.columns([2,2,2])
    with col1:
        st.write('Avg Rainfall (cm)')
    with col2:
        st.write('Avg Temp (°C)')
    with col3:
        st.write('Avg Humidity (%)')


    #expander for data
    state.expander = st.expander(label='Show data')
    with state.expander:
        placeholder2 = st.empty()
    
    #toggle switch 
    a1, b1, c = st.columns([5,4,3])
    with a1,b1:
        st.write(' ')
    choice = ''
    with c:
        st.write('Select View:')
        choice = st_btn_select(('Actual Data', 'Predictions'))#,  nav=True)


    #for climograph
    a, climo_col, b = st.columns([3,6,3])

    with climo_col:
        placeholder = st.empty()
    with a,b:
        st.write(' ')  #just for formatting


    temp_col,wind_col = st.columns(2)
    air_col,light_col = st.columns(2)
    #for temp graph
    with temp_col:
        placeholder1 = st.empty() #for graph - hourly temp

    #windrose
    with wind_col:
        placeholder_windrose = st.empty()
    
    #airpressure
    with air_col:
        placeholder_air = st.empty()


    #date
    state.date_slider = st.sidebar.date_input("Date(s)",[min(df['Date']),max(df['Date'])],min_value=min(df['Date']),max_value=max(df['Date']))

    #devices
    devices = df['Site'].unique().tolist()
    state.devices_filter = st.sidebar.multiselect('Site(s)',devices,key='devices')

    #temperature
    min_temp = min(df['new temp'])
    max_temp = max(df['new temp'])
    state.temperature = st.sidebar.slider('Temperature (°C)',min_temp,max_temp,(min_temp,max_temp),step=0.1)

    #humidity
    min_hum = min(df['new humidity'])
    max_hum = max(df['new humidity'])
    state.humidity = st.sidebar.slider('Humidity (%)',max_hum,min_hum,(min_hum,max_hum),step=0.1)

    #air pressure
    min_air = min(df['new air pressure'])
    max_air = max(df['new air pressure'])
    state.air = st.sidebar.slider('Air Pressure (hPa)',max_air,min_air,(min_air,max_air),step=0.1)

     #rainfall
    min_rainfall = min(df['new rainfall'])
    max_rainfall = max(df['new rainfall'])
    state.rainfall = st.sidebar.slider('Rainfall (cm)',max_rainfall,min_rainfall,(min_rainfall,max_rainfall),step=0.1)

    
    #apply filter

    if state.date_slider:
        min_date = state.date_slider[0]
        if state.date_slider[1]:  
            max_date = state.date_slider[1]
        else: max_date = min_date
        # st.write(min_date,max_date)
    else:  #if not filter - default values
        min_date = min(df['Date'])
        max_date = max(df['Date'])

    if state.devices_filter:
        pass
    else:
        state.devices_filter = devices 

    if state.temperature:
        min_temp = state.temperature[0]
        max_temp = state.temperature[1]

    if state.humidity:
        min_hum = state.humidity[0]
        max_hum = state.humidity[1]
    
    if state.air:
        min_air= state.air[0]
        max_air = state.air[1]

    if state.rainfall:
        min_rainfall = state.rainfall[0]
        max_rainfall = state.rainfall[1]



    filtered_df = pd.DataFrame()
    for row in range(len(df)):
        if min_date <= df.loc[row,'Date'] <= max_date: #date
            if df.loc[row,'Site'] in state.devices_filter: #device
                if min_temp<=df.loc[row,'new temp']<=max_temp:#temperature
                    if min_hum<=df.loc[row,'new humidity']<=max_hum:#humidity
                        if min_air<=df.loc[row,'new air pressure']<=max_air:#light
                            if min_rainfall<=df.loc[row,'new rainfall']<=max_rainfall:#rainfall
                                filtered_df = filtered_df.append(df.iloc[row,:])
  

    
    placeholder2.dataframe(filtered_df)

    
    if len(filtered_df)!=0:
        ##NOTE: MIGHT HAVE ERRORS if eg rainfall in filtered df is '-' or NaN etc
        with col4: 
            st.write(round(filtered_df['new rainfall'].mean(),2))
        with col5:
            st.write(round(filtered_df['new temp'].mean(),2))
        with col6:
            st.write(round(filtered_df['new humidity'].mean(),2))

        if choice == 'Actual Data':
            ####################### hourly temp graph #######################
            tempp = filtered_df[['hour','new temp']]
            tempp.set_index('hour',inplace=True)
            temp = tempp.groupby('hour').mean()
            temp_fig = px.line(temp, y='new temp',labels={
                     "new temp": "Temperature (℃) "})
            temp_fig.update_layout(title="Hourly Temperature",
                            template="plotly_white",title_x=0.5,legend=dict(orientation='h'))
            placeholder1.plotly_chart(temp_fig)


            ####################### climograph #######################
            avg_df = filtered_df.groupby('Date',as_index=False).mean()
            climograph = make_subplots(specs=[[{"secondary_y": True}]])#this a one cell subplot
            climograph.update_layout(title="Climograph",
                            template="plotly_white",title_x=0.5,legend=dict(orientation='h'))

            trace1 = go.Bar(x=avg_df['Date'], y=avg_df['new rainfall'], opacity=0.5,name='Rainfall (cm)',marker_color ='#1f77b4')

            trace2p = go.Scatter(x=avg_df['Date'], y=avg_df['new temp'],name='Temperature ((°C))',mode='lines+markers',line=dict(color='#e377c2', width=2))

            climograph.add_trace(trace1, secondary_y=False)

            climograph.add_trace(trace2p, secondary_y=True)

            climograph.update_yaxes(#left yaxis
                            title= 'cm',showgrid= False, secondary_y=False)
            climograph.update_yaxes(#right yaxis
                            showgrid= True, 
                            title= '°C',
                            secondary_y=True)
            placeholder.plotly_chart(climograph)

            ####################### windrose #######################
            wind_df = filtered_df[filtered_df['Wind Direction']!=np.nan]
            wind_df = filtered_df[filtered_df['Wind Direction']!=0]
            freq = wind_df[['Wind Direction','Wind Speed (km/h)']].value_counts()
            freq = freq.reset_index() 
            freq = freq.rename({0: "frequency"},axis=1)

            windrose = px.bar_polar(freq, r='frequency', 
                                theta="Wind Direction",
                                color="Wind Speed (km/h)",
                            color_discrete_sequence= px.colors.sequential.Plasma_r)
            windrose.update_layout(title="Windrose",
                            template="plotly_white",title_x=0.45,legend=dict(orientation='h'))

            placeholder_windrose.plotly_chart(windrose)

            

            ####################### air pressure #######################
            air = px.line(avg_df,x=avg_df['Date'], y=avg_df['new air pressure'],title='Average Air Pressure per Month',labels=
            {"new air pressure": "Air Pressure (hPa)"})
            air.update_layout(title_x=0.5)
            placeholder_air.plotly_chart(air)