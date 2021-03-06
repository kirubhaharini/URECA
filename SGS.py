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
Dashboard for Serangoon Secondary School
Data from 2020 to 2022
'''

def SGS(state):
    df = pd.read_excel('FINAL_PROCESSED_SGS.xlsx')

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
    min_temp = min(df['Temperature (℃)'])
    max_temp = max(df['Temperature (℃)'])
    state.temperature = st.sidebar.slider('Temperature (℃)',min_temp,max_temp,(min_temp,max_temp),step=0.1)

    #humidity
    min_hum = min(df['Relative Humidity (%)'])
    max_hum = max(df['Relative Humidity (%)'])
    state.humidity = st.sidebar.slider('Humidity (%)',max_hum,min_hum,(min_hum,max_hum),step=0.1)

    #air pressure
    min_air = min(df['Air Pressure (hPa)'])
    max_air = max(df['Air Pressure (hPa)'])
    state.air = st.sidebar.slider('Air Pressure (hPa)',max_air,min_air,(min_air,max_air),step=0.1)

     #rainfall
    min_rainfall = min(df['Rainfall (cm)'])
    max_rainfall = max(df['Rainfall (cm)'])
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
                if min_temp<=df.loc[row,'Temperature (℃)']<=max_temp:#temperature
                    if min_hum<=df.loc[row,'Relative Humidity (%)']<=max_hum:#humidity
                        if min_air<=df.loc[row,'Air Pressure (hPa)']<=max_air:#light
                            if min_rainfall<=df.loc[row,'Rainfall (cm)']<=max_rainfall:#rainfall
                                filtered_df = filtered_df.append(df.iloc[row,:])
  


    placeholder2.dataframe(filtered_df.astype(str))

    
    if len(filtered_df)!=0:
        ##NOTE: MIGHT HAVE ERRORS if eg rainfall in filtered df is '-' or NaN etc
        with col4: 
            st.write(round(filtered_df['Rainfall (cm)'].mean(),2))
        with col5:
            st.write(round(filtered_df['Temperature (℃)'].mean(),2))
        with col6:
            st.write(round(filtered_df['Relative Humidity (%)'].mean(),2))

        if choice == 'Actual Data':
            ####################### hourly temp graph #######################
            tempp = filtered_df[['hour','Temperature (℃)']]
            tempp.set_index('hour',inplace=True)
            temp = tempp.groupby('hour').mean()
            temp_fig = px.line(temp, y='Temperature (℃)')
            temp_fig.update_layout(title="Hourly Temperature",
                            template="plotly_white",title_x=0.5,legend=dict(orientation='h'))
            placeholder1.plotly_chart(temp_fig)


            ####################### climograph #######################
            avg_df = filtered_df.groupby('Date',as_index=False).mean()
            climograph = make_subplots(specs=[[{"secondary_y": True}]])#this a one cell subplot
            climograph.update_layout(title="Climograph",
                            template="plotly_white",title_x=0.5,legend=dict(orientation='h'))

            trace1 = go.Bar(x=avg_df['Date'], y=avg_df['Rainfall (cm)'], opacity=0.5,name='Rainfall (cm)',marker_color ='#1f77b4')

            trace2p = go.Scatter(x=avg_df['Date'], y=avg_df['Temperature (℃)'],name='Temperature ((℃))',mode='lines+markers',line=dict(color='#e377c2', width=2))

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
            air = px.line(avg_df,x=avg_df['Date'], y=avg_df['Air Pressure (hPa)'],title='Average Air Pressure per Month')
            air.update_layout(title_x=0.5)
            placeholder_air.plotly_chart(air)

        
        elif choice == 'Predictions':
            ##### using resampled data from July
            resampled_df = pd.read_excel('resampled SGS.xlsx')            
            df2 = resampled_df.set_index('timestamp')

            pred1, pred2 = st.columns(2)
            with pred1:
                ###################  temp predictions  ################
                temp_data = df2[['Temperature (℃)']]

                with open('XGBoost_Temp_SGS.pkl','rb') as f:
                    temp_model = pickle.load(f)
                
                n_periods = 500
                for i in range(n_periods):
                    
                    # construct an input for a new preduction
                    row = temp_data[-2:].values.flatten()
                    # make a one-step prediction
                    yhat = temp_model.predict(asarray([row]))
                    temp_df = temp_data.tail(1)
                    temp_df['time'] = temp_df.index
                    temp_df = temp_df.reset_index()
                    next_time = temp_df.loc[0,'time'] + datetime.timedelta(hours = 1)

                    extrapolated_value = pd.DataFrame([yhat],[next_time])
                    extrapolated_value.columns = ['Temperature (℃)']
                    temp_data = pd.concat([temp_data,extrapolated_value])

                actual_temp = df2[['Temperature (℃)']]
                temp_forecast = pd.concat([actual_temp,temp_data]).drop_duplicates(keep=False)
                
                temp_fig = go.Figure()

                temp_fig.add_trace(go.Scatter(
                    x=temp_forecast.index,
                    y=temp_forecast['Temperature (℃)'],
                    name = '<b>Forecast</b>', # Style name/legend entry with html tags
                ))
                temp_fig.add_trace(go.Scatter(
                    x=actual_temp.index,
                    y=actual_temp['Temperature (℃)'],
                    name='Actual',
                ))

                temp_fig.update_layout(title='Actual vs Predicted Temperature (℃)',title_x = 0.5)

                st.plotly_chart(temp_fig)


            with pred2:
                ###################  humidity predictions  ################
                hum_data = df2[['Relative Humidity (%)']]

                with open('XGBoost_Hum_SGS.pkl','rb') as f:
                    hum_model = pickle.load(f)

                n_periods = 500

                for i in range(n_periods):
                    
                    # construct an input for a new preduction
                    row = hum_data[-2:].values.flatten()
                    # make a one-step prediction
                    yhat = hum_model.predict(asarray([row]))
                    temp_df = hum_data.tail(1)
                    temp_df['time'] = temp_df.index
                    temp_df = temp_df.reset_index()
                    next_time = temp_df.loc[0,'time'] + datetime.timedelta(hours = 1)
                    

                    extrapolated_value = pd.DataFrame([yhat],[next_time])
                    extrapolated_value.columns = ['Relative Humidity (%)']
                    hum_data = pd.concat([hum_data,extrapolated_value])

                actual_hum = df2[['Relative Humidity (%)']]
                hum_forecast = hum_data[len(actual_hum):]
                
                hum_fig = go.Figure()

                hum_fig.add_trace(go.Scatter(
                    x=hum_forecast.index,
                    y=hum_forecast['Relative Humidity (%)'],
                    name = '<b>Forecast</b>', # Style name/legend entry with html tags
                ))
                hum_fig.add_trace(go.Scatter(
                    x=actual_hum.index,
                    y=actual_hum['Relative Humidity (%)'],
                    name='Actual',
                ))

                hum_fig.update_layout(title='Actual vs Predicted Relative Humidity (%)',title_x = 0.5)
                st.plotly_chart(hum_fig)

            ###################  rainfall predictions  ################
            #df2_reset = df2.reset_index()
            new_rainfall_df = resampled_df[['timestamp', 'Rainfall (cm)']] 
            new_rainfall_df.set_index('timestamp',inplace=True)
            # Create Training and Test - 80:20
            train = new_rainfall_df['Rainfall (cm)'][:430]
            test = new_rainfall_df['Rainfall (cm)'][430:] 

            # #load model
            # with open('ARIMA_Rainfall_SGS.pkl','rb') as f:
            #     rainfall_model = joblib.load('test arima.pkl')#pickle.load(f)
            
            #autoarima
            rainfall_model = pm.auto_arima(train, start_p=1, start_q=1,
                    test='adf',       # use adftest to find optimal 'd'
                    max_p=3, max_q=3, # maximum p and q
                    m=1,              # frequency of series
                    d=None,           # let model determine 'd'
                    seasonal=False,   # No Seasonality
                    start_P=0, 
                    D=0, 
                    trace=True,
                    error_action='ignore',  
                    suppress_warnings=True, 
                    stepwise=True)


            n_periods = 500
            date_to_forecast_from = (train.index[-1] + datetime.timedelta(days=1)).date()
            m  = pd.date_range(date_to_forecast_from,periods=n_periods,freq='H')

            fc = rainfall_model.predict(n_periods=n_periods)
            index_of_fc = m
            # make series for plotting purpose
            fc_series = pd.Series(fc, index=index_of_fc)

            rainfall_pred = go.Figure()

            rainfall_pred.add_trace(go.Histogram(
                x=fc_series.index,
                y=fc_series,
                name = '<b>Forecast</b>', # Style name/legend entry with html tags
            ))
            rainfall_pred.add_trace(go.Histogram(
                x=new_rainfall_df.index,
                y=new_rainfall_df['Rainfall (cm)'],
                name='Actual',
            ))

            rainfall_pred.update_layout(title='Actual vs Predicted Rainfall (cm)',title_x = 0.5)

            st.plotly_chart(rainfall_pred)
            

            #choose date and predict
            state.pred_date_slider = st.date_input("Select date to predict",max(resampled_df['Date'])+datetime.timedelta(days = 1),min_value=max(resampled_df['Date']))
            p_date_df = pd.DataFrame({'ds':[state.pred_date_slider]})
            
            date = state.pred_date_slider
            ###temp:
            last_date = actual_temp.tail(1)
            last_date['time'] = last_date.index
            last_date = last_date.reset_index()
            last_date = last_date.loc[0,'time']
            periods = date - last_date.date()
            n_periods = periods.days * 24 
            date_pred_temp = actual_temp
            for i in range(n_periods):
                
                # construct an input for a new preduction
                row = date_pred_temp[-2:].values.flatten()
                # make a one-step prediction
                yhat = temp_model.predict(asarray([row]))
                temp_df = date_pred_temp.tail(1)
                temp_df['time'] = temp_df.index
                temp_df = temp_df.reset_index()
                next_time = temp_df.loc[0,'time'] + datetime.timedelta(hours = 1)
                extrapolated_value = pd.DataFrame([yhat],[next_time])
                extrapolated_value.columns = ['Temperature (℃)']
                date_pred_temp = pd.concat([date_pred_temp,extrapolated_value])

            predicted = date_pred_temp.tail(1).reset_index()
            pred_temp = round(predicted.loc[0,'Temperature (℃)'],2)

            ###hum:
            last_date = actual_hum.tail(1)
            last_date['time'] = last_date.index
            last_date = last_date.reset_index()
            last_date = last_date.loc[0,'time']
            periods = date - last_date.date()
            n_periods = periods.days * 24 
            date_pred_hum = actual_hum
            for i in range(n_periods):
                
                # construct an input for a new preduction
                row = date_pred_hum[-2:].values.flatten()
                # make a one-step prediction
                yhat = temp_model.predict(asarray([row]))
                temp_df = date_pred_hum.tail(1)
                temp_df['time'] = temp_df.index
                temp_df = temp_df.reset_index()
                next_time = temp_df.loc[0,'time'] + datetime.timedelta(hours = 1)
                extrapolated_value = pd.DataFrame([yhat],[next_time])
                extrapolated_value.columns = ['Relative Humidity (%)']
                date_pred_hum = pd.concat([date_pred_hum,extrapolated_value])

            predicted = date_pred_hum.tail(1).reset_index()
            pred_hum = round(predicted.loc[0,'Relative Humidity (%)'],2)

            ###rainfall:
            n = (state.pred_date_slider - train.index[-1].date()).days
            fc = rainfall_model.predict(n_periods=n)
            predicted=fc[-1]
            pred_rainfall = round(predicted,2)


            st.write('Weather Forecast for '+str(state.pred_date_slider)+' :')
            st.write('Temperature: '+str(pred_temp)+' °C')
            st.write('Humidity: '+str(pred_hum)+' %')
            st.write('Rainfall: '+str(pred_rainfall)+' cm')

            st.write('\n[Note: Prediction is only based on April 2022 data]')
            

    else:       
        with col4:
            st.write('NA')
        with col5:
            st.write('NA')
        with col6:
            st.write('NA')