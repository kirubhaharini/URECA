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

# import hydralit_components as hc
# import time

'''
Dashboard for Nan Chiau High School
Data from June to Aug 2021
'''

def NCHS(state):
    
    #use final merged dataset instead
    df = pd.read_excel('FINAL_PROCESSED_NCHS.xlsx')

#####################################################################

    #for presenting stats
    st.markdown(f"""<h3 style='text-align: left; color: darkgoldenrod;'>{'Summary'}</h3>""", unsafe_allow_html=True)
    col1,col2,col3 = st.columns([2,2,2])
    col4,col5,col6 = st.columns([2,2,2])
    with col1:
        st.write('Avg Rainfall (mm)')
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
    co2_col,light_col = st.columns(2)
    #for temp graph
    with temp_col:
        placeholder1 = st.empty() #for graph - hourly temp

    #windrose
    with wind_col:
        placeholder_windrose = st.empty()
    
    #co2
    with co2_col:
        placeholder_co2 = st.empty()

    #light
    with light_col:
        placeholder_light = st.empty()

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
    min_co2 = min(df['CO2 (ppm)'])
    max_co2 = max(df['CO2 (ppm)'])
    state.co2 = st.sidebar.slider('CO2 (ppm)',max_co2,min_co2,(min_co2,max_co2),step=0.1)

     #rainfall
    min_rainfall = min(df['Rainfall (mm)'])
    max_rainfall = max(df['Rainfall (mm)'])
    state.rainfall = st.sidebar.slider('Rainfall (mm)',max_rainfall,min_rainfall,(min_rainfall,max_rainfall),step=0.1)


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

    if state.rainfall:
        min_rainfall = state.rainfall[0]
        max_rainfall = state.rainfall[1]



    filtered_df = pd.DataFrame()
    for row in range(len(df)):
        if min_date <= df.loc[row,'Date'] <= max_date: #date
            if df.loc[row,'Device ID'] in state.devices_filter: #device
                if min_temp<=df.loc[row,'Temperature (°C)']<=max_temp:#temperature
                    if min_hum<=df.loc[row,'Humidity (%)']<=max_hum:#humidity
                        if min_light<=df.loc[row,'Visible Light (lm)']<=max_light:#light
                            if min_co2<=df.loc[row,'CO2 (ppm)']<=max_co2:#co2
                                if min_rainfall<=df.loc[row,'Rainfall (mm)']<=max_rainfall:#rainfall
                                    filtered_df = filtered_df.append(df.iloc[row,:])
  

    
    placeholder2.dataframe(filtered_df)


    if len(filtered_df)!=0:
        ##NOTE: MIGHT HAVE ERRORS if eg rainfall in filtered df is '-' or NaN etc
        with col4: 
            st.write(round(filtered_df['Rainfall (mm)'].mean(),2))
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
            climograph.update_layout(title="Climograph",
                            template="plotly_white",title_x=0.5,legend=dict(orientation='h'))

            trace1 = go.Bar(x=avg_df['Date'], y=avg_df['Rainfall (mm)'], opacity=0.5,name='Rainfall (mm)',marker_color ='#1f77b4')

            trace2p = go.Scatter(x=avg_df['Date'], y=avg_df['Temperature (°C)'],name='Temperature ((°C))',mode='lines+markers',line=dict(color='#e377c2', width=2))

            #The first trace is referenced to the default xaxis, yaxis (ie. xaxis='x1', yaxis='y1')
            climograph.add_trace(trace1, secondary_y=False)

            #The second trace is referenced to xaxis='x1'(i.e. 'x1' is common for the two traces) 
            #and yaxis='y2' (the right side yaxis)

            climograph.add_trace(trace2p, secondary_y=True)

            climograph.update_yaxes(#left yaxis
                            title= 'mm',showgrid= False, secondary_y=False)
            climograph.update_yaxes(#right yaxis
                            showgrid= True, 
                            title= '°C',
                            secondary_y=True)
            placeholder.plotly_chart(climograph)

            ####################### windrose #######################
            wind_df = filtered_df[filtered_df['direction of wind']!='NA']
            freq = wind_df[['direction of wind','Wind Speed (m/s)']].value_counts()
            freq = freq.reset_index() 
            freq = freq.rename({0: "frequency"},axis=1)
            
            windrose = px.bar_polar(freq, r='frequency', 
                                theta="direction of wind",
                                color="Wind Speed (m/s)",
                            color_discrete_sequence= px.colors.sequential.Plasma_r)
            windrose.update_layout(title="Windrose",
                            template="plotly_white",title_x=0.45,legend=dict(orientation='h'))

            placeholder_windrose.plotly_chart(windrose)


            ####################### co2 levels #######################        
            co2 = px.line(avg_df,x=avg_df['Date'], y=avg_df['CO2 (ppm)'],title='Average CO2 Level per Day')
            co2.update_layout(title_x=0.5)
            placeholder_co2.plotly_chart(co2)

            ####################### visible light and UV #######################
            sites = filtered_df['Device ID'].unique()
            temp_df = filtered_df.groupby(['hour','Device ID'],as_index=False).mean()
            bubble_size = []
            hover_text = []

            for index, row in temp_df.iterrows():
                hover_text.append(( #'Date: {date}<br>'+
                                'Visible Light: {light}<br>'+
                                'UV Index: {UV}<br>').format(#date=row['Date'],
                                                        light=row['Visible Light (lm)'],
                                                        UV=row['UV (UV Index)'] ))
            #     print(row)
                if row['Visible Light (lm)']>0:
                    bubble_size.append(math.sqrt(row['Visible Light (lm)']))
                else:
                    bubble_size.append(0)

            temp_df['text'] = hover_text
            temp_df['size'] = bubble_size
            sizeref = 2.*max(temp_df['size'])/(100**2)


            # Create figure
            fig = go.Figure()

            for device in sites:
                new_df = temp_df[temp_df['Device ID']==device]
                # avg_df = new_df.groupby(['hour','Device ID'],as_index=False).mean()
                fig.add_trace(go.Scatter(
                    x=new_df['hour'], y=new_df['Visible Light (lm)'],
                    name=device,text=new_df['text'],
                    marker_size=new_df['size'],
                    ))

            # Tune marker appearance and layout
            fig.update_traces(mode='markers', marker=dict(sizemode='area',
                                                        sizeref=sizeref, line_width=2))

            fig.update_layout(title="Average Visible Light level by Hour of Day",
                            template="plotly_white",title_x=0.5)
            fig.update_layout(
                xaxis=dict(
                    title='Hour of day',
            #         gridcolor='white',
                    gridwidth=2,
                ),
                yaxis=dict(
                    title='Visible Light (lm)',
            #         gridcolor='white',
                    gridwidth=2,
                ),
            #     paper_bgcolor='rgb(243, 243, 243)',
            #     plot_bgcolor='rgb(243, 243, 243)',
            )
            placeholder_light.plotly_chart(fig)
            ################################################################

            ####################### calmap #######################
            # import calmap
            # temp = df.copy().set_index(pd.DatetimeIndex(df['Date']))
            # #temp.set_index('date', inplace=True)
            # # fig, ax = calmap.calendarplot(temp['Rainfall (mm)'], fig_kws={"figsize":(15,4)})
            # # plt.title("Hours raining")
            # fig, ax = calmap.calendarplot(temp['Rainfall (mm)'],how=u'sum', fig_kws={"figsize":(15,4)})
            # st.image(fig)
            # plt.title("Total Rainfall Daily")
            ###################################################################

        elif choice == 'Predictions':
            
            july_df = df[df['Time'].dt.strftime('%m')=='07'] 
            # ^ no max since can predict any date after max date in july data:)
            
            pred1, pred2 = st.columns(2)
            with pred1:
                ###################  temp predictions  ################

                new_temp_column = july_df[['Time', 'new temp']] 
                new_temp_column.dropna(inplace=True)
                new_temp_column.reset_index(inplace = True)
                new_temp_column.drop('index',axis=1,inplace=True)
                # new_column.columns
                new_temp_column.columns = ['ds', 'y'] 

                #load model
                with open('prophet_temp_model_NCHS.pkl','rb') as f:
                    temp_model = pickle.load(f)
                
                #prediction
                temp_future = temp_model.make_future_dataframe(periods=1000,freq='33.4min')
                temp_forecast=temp_model.predict(temp_future)
                # import plotly.graph_objects as go

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
                new_hum_column = july_df[['Time', 'new humidity']] 
                new_hum_column.dropna(inplace=True)
                new_hum_column.reset_index(inplace = True)
                new_hum_column.drop('index',axis=1,inplace=True)
                new_hum_column.columns = ['ds', 'y'] 
                with open('prophet_humidity_model_NCHS.pkl','rb') as f:
                    hum_model = pickle.load(f)

                hum_future= hum_model.make_future_dataframe(periods=1000,freq='33.4min')
                hum_forecast=hum_model.predict(hum_future)
                hum_pred = go.Figure()
                hum_pred.add_trace(go.Scatter(
                    x=hum_forecast.ds,
                    y=hum_forecast.yhat,
                    name = '<b>Forecast</b>', # Style name/legend entry with html tags
                ))
                hum_pred.add_trace(go.Scatter(
                    x=new_hum_column.ds,
                    y=new_hum_column.y,
                    name='Actual'
                ))
                hum_pred.update_layout(title='Actual vs Predicted Humidity (%)',title_x = 0.5)
                
                st.plotly_chart(hum_pred)
            
            ###################  rainfall predictions  ################
            new_rainfall_column = july_df[['Time', 'Rainfall (mm)']] 
            new_rainfall_column.dropna(inplace=True)
            new_rainfall_column.reset_index(inplace = True)
            new_rainfall_column.drop('index',axis=1,inplace=True)
            new_rainfall_column.columns = ['ds', 'y'] 

            #load model
            with open('prophet_rainfall_model_NCHS.pkl','rb') as f:
                rainfall_model = pickle.load(f)

            rainfall_future= rainfall_model.make_future_dataframe(periods=1000,freq='33.4min')
            rainfall_forecast=rainfall_model.predict(rainfall_future)
            rainfall_pred = go.Figure()

            rainfall_pred.add_trace(go.Histogram(
                x=rainfall_forecast.ds,
                y=rainfall_forecast.yhat,
                name = '<b>Forecast</b>', # Style name/legend entry with html tags
            ))
            rainfall_pred.add_trace(go.Histogram(
                x=new_rainfall_column.ds,
                y=new_rainfall_column.y,
                name='Actual',
            ))

            rainfall_pred.update_layout(title='Actual vs Predicted Rainfall (cm)',title_x = 0.5)

            st.plotly_chart(rainfall_pred)


            #choose date and predict
            state.pred_date_slider = st.date_input("Select date to predict",max(july_df['Date']),min_value=max(july_df['Date']))

            p_date_df = pd.DataFrame({'ds':[state.pred_date_slider]})
            
            predicted=temp_model.predict(p_date_df)
            pred_temp = round(predicted.yhat[0],2)

            predicted=hum_model.predict(p_date_df)
            pred_hum = round(predicted.yhat[0],2)

            predicted=rainfall_model.predict(p_date_df)
            pred_rainfall = round(predicted.yhat[0],2)


            st.write('Weather Forecast on '+str(state.pred_date_slider)+' :')
            st.write('Temperature: '+str(pred_temp)+' °C')
            st.write('Humidity: '+str(pred_hum)+' %')
            st.write('Rainfall: '+str(pred_rainfall)+' cm')

            st.write('\n[Note: Prediction is only based on July data]')
            
    else:
        with col4:
            st.write('NA')
        with col5:
            st.write('NA')
        with col6:
            st.write('NA')