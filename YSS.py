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
import pmdarima as pm
from numpy import asarray
import xgboost
from xgboost import XGBRegressor
import statsmodels

'''
Dashboard for Yishun Secondary School
Data from 2021
'''

def YSS(state):
    df = pd.read_excel('FINAL_PROCESSED_YSS.xlsx')

    
#####################################################################

    #for presenting stats
    st.markdown(f"""<h3 style='text-align: left; color: darkgoldenrod;'>{'Summary'}</h3>""", unsafe_allow_html=True)
    col1,col2,col3 = st.columns([2,2,2])
    col4,col5,col6 = st.columns([2,2,2])
    with col1:
        st.write('Avg CO2 (ppm)')
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


    climo_col,temp_col = st.columns(2)
    light_col,co2_col = st.columns(2)
    noise_col, dust_col = st.columns(2) 
    
    #for climograph
    with climo_col:
        placeholder = st.empty()

    #for temp graph
    with temp_col:
        placeholder1 = st.empty() #for graph - hourly temp
    
    #co2
    with co2_col:
        placeholder_co2 = st.empty()

    #light
    with light_col:
        placeholder_light = st.empty()

    #noise
    with noise_col:
        placeholder_noise = st.empty()
    
    #dust
    with dust_col:
        placeholder_dust = st.empty()

    #date
    state.date_slider = st.sidebar.date_input("Date(s)",[min(df['Date']),max(df['Date'])],min_value=min(df['Date']),max_value=max(df['Date']))

    #devices
    devices = df['Device ID'].unique().tolist()
    state.devices_filter = st.sidebar.multiselect('Device(s)',devices,key='devices')

    #temperature
    min_temp = min(df['Temperature (°C)'])
    max_temp = max(df['Temperature (°C)'])
    state.temperature = st.sidebar.slider('Temperature (°C)',min_temp,max_temp,(min_temp,max_temp),step=0.1)

    #humidity
    min_hum = min(df['Humidity (%)'])
    max_hum = max(df['Humidity (%)'])
    state.humidity = st.sidebar.slider('Humidity (%)',max_hum,min_hum,(min_hum,max_hum),step=0.1)

     #light
    min_light = float(min(df['Visible Light (lm)']))
    max_light = float(max(df['Visible Light (lm)']))
    state.light = st.sidebar.slider('Visible Light (lm)',max_light,min_light,(min_light,max_light),step=0.1)

     #co2
    min_co2 = min(df['CO2 (ppm)'].dropna())
    max_co2 = max(df['CO2 (ppm)'].dropna())
    state.co2 = st.sidebar.slider('CO2 (ppm)',max_co2,min_co2,(min_co2,max_co2),step=0.1)

     #noise
    min_noise = min(df['Noise (0-1023)'].dropna())
    max_noise = max(df['Noise (0-1023)'].dropna())
    state.noise = st.sidebar.slider('Noise (0-1023)',max_noise,min_noise,(min_noise,max_noise),step=1.0)

    #dust
    min_dust = min(df['Dust (µg/m3)'])
    max_dust = max(df['Dust (µg/m3)'])
    state.dust = st.sidebar.slider('Dust (µg/m3)',max_dust,min_dust,(min_dust,max_dust),step=1)

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
    
    if state.light:
        min_light= state.light[0]
        max_light = state.light[1]

    if state.co2:
        min_co2 = state.co2[0]
        max_co2 = state.co2[1]

    if state.noise:
        min_noise = state.noise[0]
        max_noise = state.noise[1]

    if state.dust:
        min_dust = state.dust[0]
        max_dust = state.dust[1]


    filtered_df = pd.DataFrame()
    for row in range(len(df)):
        if min_date <= df.loc[row,'Date'] <= max_date: #date
            if df.loc[row,'Device ID'] in state.devices_filter: #device
                if min_temp<=df.loc[row,'Temperature (°C)']<=max_temp:#temperature
                    if min_hum<=df.loc[row,'Humidity (%)']<=max_hum:#humidity
                        if min_light<=df.loc[row,'Visible Light (lm)']<=max_light:#light
                            if min_co2<=df.loc[row,'CO2 (ppm)']<=max_co2:#co2
                                if min_noise<=df.loc[row,'Noise (0-1023)']<=max_noise:#noise
                                    if min_dust<=df.loc[row,'Dust (µg/m3)']<=max_dust:#dust
                                        filtered_df = filtered_df.append(df.iloc[row,:])
  

    
    placeholder2.dataframe(filtered_df)


    if len(filtered_df)!=0:
        ##NOTE: MIGHT HAVE ERRORS if eg rainfall in filtered df is '-' or NaN etc
        with col4: 
            st.write(round(filtered_df['CO2 (ppm)'].mean(),2))
        with col5:
            st.write(round(filtered_df['Temperature (°C)'].mean(),2))
        with col6:
            st.write(round(filtered_df['Humidity (%)'].mean(),2))

        if choice == 'Actual Data':
            ####################### hourly temp graph #######################
            tempp = filtered_df[['hour','Temperature (°C)']]
            tempp.set_index('hour',inplace=True)
            temp = tempp.groupby('hour').mean()
            temp_fig = px.line(temp, y='Temperature (°C)')
            temp_fig.update_layout(title="Hourly Temperature",
                            template="plotly_white",title_x=0.5,legend=dict(orientation='h'))
            placeholder1.plotly_chart(temp_fig)


            ####################### climograph #######################
            avg_df = filtered_df.groupby('Date',as_index=False).mean()
            for row in range(len(avg_df)):
                avg_df.loc[row,'Day'] = avg_df.loc[row,'Date'].strftime('%d %B %Y')
                avg_df.loc[row,'Month'] = avg_df.loc[row,'Date'].strftime('%B')
            
            climograph = make_subplots(specs=[[{"secondary_y": True}]])#this a one cell subplot
            climograph.update_layout(title="Temperature Data (°C)",
                            template="plotly_white",title_x=0.5,legend=dict(orientation='h'))

            trace1 = go.Scatter(x=avg_df['Date'], y=avg_df['Temperature (°C)'],name='Temperature ((°C))',mode='lines+markers',line=dict(color='#e377c2', width=2))

            climograph.add_trace(trace1, secondary_y=True)

            # climograph.update_yaxes(#left yaxis
            #                 title= 'mm',showgrid= False, secondary_y=False)
            climograph.update_yaxes(#right yaxis
                            showgrid= True, 
                            title= '°C',
                            secondary_y=True)
            placeholder.plotly_chart(climograph)


            ####################### co2 levels #######################        
            co2 = px.line(avg_df,x=avg_df['Date'], y=avg_df['CO2 (ppm)'],title='Average CO2 Level per Day')
            co2.update_layout(title_x=0.5)
            placeholder_co2.plotly_chart(co2)

            ####################### visible light and IR #######################
            

            light_df = avg_df[['Date','Visible Light (lm)']]
            light_df['Visible Light (lm)'] = light_df['Visible Light (lm)'].fillna(light_df['Visible Light (lm)'].mean())
            IR_df = avg_df[['Date','IR (lm)']]
            IR_df['IR (lm)'] = IR_df['IR (lm)'].fillna(IR_df['IR (lm)'].mean())
            light_df['category'] = 'Visible Light (lm)'
            IR_df['category'] = 'IR (lm)'
            final_df = pd.concat([light_df,IR_df],ignore_index=True)
            final_df['Light (lm)'] = final_df['Visible Light (lm)'].fillna(final_df['IR (lm)'])
            light_fig = px.area(final_df, x="Date", y="Light (lm)", color="category",title = 'Visible Light and IR Levels')
            placeholder_light.plotly_chart(light_fig)


            ####################### noise #######################
            sites = df['Device ID'].unique()
            df2 = df.groupby(['hour','Device ID'],as_index=False).mean()
            bubble_size = []
            hover_text = []

            for index, row in df2.iterrows():
                hover_text.append(( 'Noise Level: {noise}<br>'
                                ).format(noise=row['Noise (0-1023)']
                                                        ))
                if row['Noise (0-1023)']>0:
                    bubble_size.append(int(math.sqrt(row['Noise (0-1023)'])))
                else:
                    bubble_size.append(0)

            df2['text'] = hover_text
            
            # Create figure
            noise_fig = go.Figure()

            for device in sites:
                new_df = df2[df2['Device ID']==device]
                avg_df = new_df.groupby(['hour','Device ID'],as_index=False).mean()
                noise_fig.add_trace(go.Scatter(
                    x=new_df['hour'], y=new_df['Noise (0-1023)'],
                    name=device,text=(new_df['text']),
                    ))

            # Tune marker appearance and layout
            noise_fig.update_traces(mode='markers', marker=dict(sizemode='area',line_width=2))

            noise_fig.update_layout(title="Average Noise (0-1023) by Hour of Day",
                            template="plotly_white",title_x=0.5)
            noise_fig.update_layout(
                xaxis=dict(
                    title='Hour of day',
                    gridwidth=2,
                ),
                yaxis=dict(
                    title='Noise Level',
                    gridwidth=2,
                ),
            )

            placeholder_noise.plotly_chart(noise_fig)

            ####################### dust #######################
            bubble_size = []
            hover_text = []

            for index, row in df2.iterrows():
                hover_text.append(( 'Dust Level: {dust}<br>'#+
                                #'Noise Level: {noise}<br>'
                                ).format(dust=row['Dust (µg/m3)']#,noise = row['Noise (0-1023)']
                                                        ))
            #     print(row)
                if row['Dust (µg/m3)']>0:
                    bubble_size.append(int(math.sqrt(row['Dust (µg/m3)'])))
                else:
                    bubble_size.append(0)

            df2['text'] = hover_text


            # Create figure
            dust_fig = go.Figure()

            for device in sites:
                new_df = df2[df2['Device ID']==device]
                avg_df = new_df.groupby(['hour','Device ID'],as_index=False).mean()
                dust_fig.add_trace(go.Scatter(
                    x=new_df['hour'], y=new_df['Dust (µg/m3)'],
                    name=device,text=(new_df['text']),
                    ))

            # Tune marker appearance and layout
            dust_fig.update_traces(mode='markers', marker=dict(sizemode='area', line_width=2))

            dust_fig.update_layout(title="Average Dust (µg/m3) by Hour of Day",
                            template="plotly_white",title_x=0.5)
            dust_fig.update_layout(
                xaxis=dict(
                    title='Hour of day',
                    gridwidth=2,
                ),
                yaxis=dict(
                    title='Dust',
                    gridwidth=2,
                ),
            )

            placeholder_dust.plotly_chart(dust_fig)


        elif choice == 'Predictions':
                    ##### using resampled data from July
                    resampled_df = pd.read_excel('resampled YSS.xlsx')            
                    df2 = resampled_df.set_index('Time')

                    pred1, pred2 = st.columns(2)
                    with pred1:
                        ###################  temp predictions  ################
                        new_temp_column = resampled_df[['Time', 'Temperature (°C)']]  #df1
                        new_temp_column.dropna(inplace=True)
                        new_temp_column.reset_index(inplace = True)
                        new_temp_column.drop('index',axis=1,inplace=True)
                        new_temp_column.columns = ['ds', 'y'] 
                       
                        #load model
                        with open('Prophet_Temp_YSS.pkl','rb') as f:
                            temp_model = pickle.load(f)

                        temp_future = temp_model.make_future_dataframe(periods=500,freq='H')
                        temp_forecast=temp_model.predict(temp_future)

                        temp_pred = go.Figure()
                        temp_pred.add_trace(go.Scatter(
                            x=temp_forecast.ds,
                            y=temp_forecast.yhat,
                            name = '<b>Forecast</b>', # Style name/legend entry with html tags
                        ))
                        temp_pred.add_trace(go.Scatter(
                            x=new_temp_column.ds,
                            y=new_temp_column.y,
                            name='Actual',
                        ))
                        temp_pred.update_layout(title='Actual vs Predicted Temperature (°C)',title_x = 0.5)
                        
                        st.plotly_chart(temp_pred)
                        

                    with pred2:
                        ###################  humidity predictions  ################
                        new_hum_column = resampled_df[['Time', 'Humidity (%)']]  
                        new_hum_column.dropna(inplace=True)
                        new_hum_column.reset_index(inplace = True)
                        new_hum_column.drop('index',axis=1,inplace=True)
                        new_hum_column.columns = ['ds', 'y'] 
                       
                        #load model
                        with open('Prophet_Hum_YSS.pkl','rb') as f:
                            hum_model = pickle.load(f)

                        hum_future= hum_model.make_future_dataframe(periods=500,freq='H')
                        hum_forecast=hum_model.predict(hum_future)
                        hum_pred = go.Figure()
                        hum_pred.add_trace(go.Scatter(
                            x=hum_forecast.ds,
                            y=hum_forecast.yhat,
                            name = '<b>Forecast</b>', 
                        ))
                        hum_pred.add_trace(go.Scatter(
                            x=new_hum_column.ds,
                            y=new_hum_column.y,
                            name='Actual'
                        ))
                        hum_pred.update_layout(title='Actual vs Predicted Humidity (%)',title_x = 0.5)
                        
                        st.plotly_chart(hum_pred)


                    ###################  noise predictions  ################
                    noise_data = resampled_df[['Noise (0-1023)','Time']]
                    date_time_obj = datetime.datetime.strptime('2021-05-07', '%Y-%m-%d').date()  #cuz a lot of gaps after this date
                    noise_data = noise_data[noise_data.Time.dt.date <= date_time_obj]
                    noise_data = noise_data.set_index('Time')
                    noise_data = noise_data.fillna(noise_data.mean())
                    actual_noise = noise_data

                    with open('XGBoost_Noise_YSS.pkl','rb') as f:
                        noise_model = pickle.load(f)

                    n_periods = 500

                    for i in range(n_periods):
                        
                        # construct an input for a new preduction
                        row = noise_data[-2:].values.flatten()
                        # make a one-step prediction
                        yhat = noise_model.predict(asarray([row]))
                        temp_df = noise_data.tail(1)
                        temp_df['time'] = temp_df.index
                        temp_df = temp_df.reset_index()
                        next_time = temp_df.loc[0,'time'] + datetime.timedelta(hours = 1)

                        extrapolated_value = pd.DataFrame([yhat],[next_time])
                        extrapolated_value.columns = ['Noise (0-1023)']
                        noise_data = pd.concat([noise_data,extrapolated_value])
                    
                    noise_forecast = noise_data[len(actual_noise):]
                    
                    noise_fig = go.Figure()

                    noise_fig.add_trace(go.Scatter(
                        x=noise_forecast.index,
                        y=noise_forecast['Noise (0-1023)'],
                        name = '<b>Forecast</b>', # Style name/legend entry with html tags
                    ))
                    noise_fig.add_trace(go.Scatter(
                        x=actual_noise.index,
                        y=actual_noise['Noise (0-1023)'],
                        name='Actual',
                    ))

                    noise_fig.update_layout(title='Actual vs Predicted Noise (0-1023)',title_x = 0.5)
                    st.plotly_chart(noise_fig)
                    

                    #choose date and predict
                    state.pred_date_slider = st.date_input("Select date to predict",max(resampled_df['Date'])+datetime.timedelta(days = 1),min_value=max(resampled_df['Date']))
                    p_date_df = pd.DataFrame({'ds':[state.pred_date_slider]})
                    
                    date = state.pred_date_slider
                    ###noise:
                    last_date = actual_noise.tail(1)
                    last_date['time'] = last_date.index
                    last_date = last_date.reset_index()
                    last_date = last_date.loc[0,'time']
                    periods = date - last_date.date()
                    n_periods = periods.days * 24 
                    date_pred_noise = actual_noise
                    for i in range(n_periods):
                        
                        # construct an input for a new preduction
                        row = date_pred_noise[-2:].values.flatten()
                        # make a one-step prediction
                        yhat = noise_model.predict(asarray([row]))
                        temp_df = date_pred_noise.tail(1)
                        temp_df['time'] = temp_df.index
                        temp_df = temp_df.reset_index()
                        next_time = temp_df.loc[0,'time'] + datetime.timedelta(hours = 1)
                        extrapolated_value = pd.DataFrame([yhat],[next_time])
                        extrapolated_value.columns = ['Noise (0-1023)']
                        date_pred_noise = pd.concat([date_pred_noise,extrapolated_value])

                    predicted = date_pred_noise.tail(1).reset_index()
                    pred_noise = round(predicted.loc[0,'Noise (0-1023)'],0)

                    ###temp:
                    predicted=temp_model.predict(p_date_df)
                    pred_temp = round(predicted.yhat[0],2)

                    ###hum:
                    predicted=hum_model.predict(p_date_df)
                    pred_hum = round(predicted.yhat[0],2)

                    st.write('Weather Forecast for '+str(state.pred_date_slider)+' :')
                    st.write('Temperature: '+str(pred_temp)+' °C')
                    st.write('Humidity: '+str(pred_hum)+' %')
                    st.write('Noise Level: '+str(pred_noise))

                    st.write('\n[Note: Prediction is only based on data from March to May 2021]')
                    

    else:       
        with col4:
            st.write('NA')
        with col5:
            st.write('NA')
        with col6:
            st.write('NA')