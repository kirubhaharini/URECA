import pandas as pd
import streamlit as st
import datetime
import numpy as np
import plotly.express as px
from windrose import WindroseAxes
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math

'''
Dashboard for Dunearn Secondary School
Data from Nov 2019 to Apr 2021
'''

def DSS(state):

    ############## preprocessing ##############
    df = pd.read_excel('Dunearn_Weather Readings.xlsx')

    df['date'] = [datetime.datetime.date(d) for d in df['timestamp']]
    df.drop(['Unnamed: 10','Unnamed: 11','id'], inplace=True, axis=1)
    cols = ['Wind Speed (km/h)','Wind Direction','Rainfall (cm)']
    for col in cols:
        df[col] = df[col].replace('-',0)
        df[col] = df[col].replace('???',0)

    #checking and removing negative values 
    df = df[df['Temperature (℃)']>=0]
    df = df[df['Relative Humidity (%)']>=0]
    df = df[df['Rainfall (cm)']>=0]
    df = df.reset_index()

    df['hour'] = ''
    for row in range(len(df)):

        time = (df.loc[row,'timestamp'])
        df.loc[row,'hour'] = time.hour
    
    df1 = df.groupby(['moteID','date'],as_index=False).mean()
    
    df1['Temperature (℃)'] = df1['Temperature (℃)'].replace(0,df1['Temperature (℃)'].median())
    df1 = df1.reset_index()
    for row in range(len(df1)):
        df1.loc[row,'year'] = int((df1.loc[row,'date']).year)
    
    
    #setting up page

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
        choice = st.radio("View:",('Actual data','Predictions'))
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

    # st.write(choice)


    #for climograph
    a, climo_col, b = st.columns([3,6,3])

    with climo_col:
        placeholder = st.empty()
    with a,b:
        st.write(' ')  #just for formatting


    temp_col,wind_col = st.columns(2)
    light_col,co2_col = st.columns(2)
    #for temp graph
    with temp_col:
        placeholder1 = st.empty() #for graph - hourly temp

    #windrose
    with wind_col:
        placeholder_windrose = st.empty()
    
    #co2
    with co2_col:
       st.write('')

    #light
    with light_col:
        placeholder_light = st.empty()

    #date
    state.date_slider = st.sidebar.date_input("Date(s)",[min(df['date']),max(df['date'])],min_value=min(df['date']),max_value=max(df['date']))

    #devices
    devices = df['moteName'].unique().tolist()
    state.devices_filter = st.sidebar.multiselect('Device(s)',devices,key='devices')

    #temperature
    min_temp = min(df['Temperature (℃)'])
    max_temp = max(df['Temperature (℃)'])
    state.temperature = st.sidebar.slider('Temperature (°C)',min_temp,max_temp,(min_temp,max_temp),step=0.1)

    #humidity
    min_hum = min(df['Relative Humidity (%)'])
    max_hum = max(df['Relative Humidity (%)'])
    state.humidity = st.sidebar.slider('Humidity (%)',max_hum,min_hum,(min_hum,max_hum),step=1)

     #light
    min_light = min(df['Light'])
    max_light = max(df['Light'])
    state.light = st.sidebar.slider('Light',max_light,min_light,(min_light,max_light),step=1)

   
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
        min_date = min(df['date'])
        max_date = max(df['date'])

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

    if state.rainfall:
        min_rainfall = state.rainfall[0]
        max_rainfall = state.rainfall[1]



    filtered_df = pd.DataFrame()
    for row in range(len(df)):
        if min_date <= df.loc[row,'date'] <= max_date: #date
            if df.loc[row,'moteName'] in state.devices_filter: #device
                if min_temp<=df.loc[row,'Temperature (℃)']<=max_temp:#temperature
                    if min_hum<=df.loc[row,'Relative Humidity (%)']<=max_hum:#humidity
                        if min_light<=df.loc[row,'Light']<=max_light:#light
                            if min_rainfall<=df.loc[row,'Rainfall (cm)']<=max_rainfall:#rainfall
                                filtered_df = filtered_df.append(df.iloc[row,:])
  
    filtered_df = filtered_df.drop('Wind Direction',axis=1)  #need to check this
    # st.write(filtered_df)
    placeholder2.dataframe(filtered_df)


    if len(filtered_df)!=0:
        ##NOTE: MIGHT HAVE ERRORS if eg rainfall in filtered df is '-' or NaN etc
        with col4: 
            st.write(round(filtered_df['Rainfall (cm)'].mean(),2))
        with col5:
            st.write(round(filtered_df['Temperature (℃)'].mean(),2))
        with col6:
            st.write(round(filtered_df['Relative Humidity (%)'].mean(),2))

        if choice == 'Actual data':
            ########### climograph ##############
            avg_df = df1.groupby('date',as_index=False).mean()
            for row in range(len(avg_df)):
                avg_df.loc[row,'Day'] = avg_df.loc[row,'date'].strftime('%d %B %Y')
                avg_df.loc[row,'Month'] = avg_df.loc[row,'date'].strftime('%B')

            climograph = make_subplots(specs=[[{"secondary_y": True}]])#this a one cell subplot
            climograph.update_layout(title="Climograph",#+str(int(year)),
                            template="plotly_white",title_x=0.5,legend=dict(orientation='h'))

            trace1 = go.Bar(x=avg_df['date'], y=avg_df['Rainfall (cm)'], opacity=0.5,name='Rainfall (cm)',marker_color ='#1f77b4')

            trace2p = go.Scatter(x=avg_df['date'], y=avg_df['Temperature (℃)'],name='Temperature ((°C))',mode='lines+markers',line=dict(color='#e377c2', width=2))

            #The first trace is referenced to the default xaxis, yaxis (ie. xaxis='x1', yaxis='y1')
            climograph.add_trace(trace1, secondary_y=False)

            #The second trace is referenced to xaxis='x1'(i.e. 'x1' is common for the two traces) 
            #and yaxis='y2' (the right side yaxis)

            climograph.add_trace(trace2p, secondary_y=True)

            climograph.update_yaxes(#left yaxis
                            title= 'cm',showgrid= False, secondary_y=False)
            climograph.update_yaxes(#right yaxis
                            showgrid= True, 
                            title= '°C',
                            secondary_y=True)
            placeholder.plotly_chart(climograph)
            
            ############ hourly temperature ############

            tempp = df[['hour','Temperature (℃)']]
            tempp.set_index('hour',inplace=True)
            temp = tempp.groupby(['hour']).mean()
            temp_fig = px.line(temp, y='Temperature (℃)')
            temp_fig.update_layout(title="Hourly Temperature",
                            template="plotly_white",title_x=0.5,legend=dict(orientation='h'))

            placeholder1.plotly_chart(temp_fig)
            ############ windrose ###############
            wind_df = df[df['Wind Direction']!=0]
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

            ############# Light ##############
            sites = df['moteName'].unique()
            df = df.groupby(['hour','moteName'],as_index=False).mean()
            bubble_size = []
            hover_text = []

            for index, row in df.iterrows():
                hover_text.append(( #'Date: {date}<br>'+
                                'Visible Light: {light}<br>'
                                ).format(light=row['Light']
                                                        ))
            #     print(row)
                if row['Light']>0:
                    bubble_size.append(int(math.sqrt(row['Light'])))
                else:
                    bubble_size.append(0)

            df['text'] = hover_text
            df['size'] = (bubble_size)
            sizeref = 2.*max(df['size'])/(100**2)


            # Create figure
            fig = go.Figure()

            for device in sites:
                new_df = df[df['moteName']==device]
                avg_df = new_df.groupby(['hour','moteName'],as_index=False).mean()
                fig.add_trace(go.Scatter(
                    x=new_df['hour'], y=new_df['Light'],
                    name=device,text=(new_df['text']),
                    marker_size=(new_df['size']),
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
                    title='Light',
            #         gridcolor='white',
                    gridwidth=2,
                ),
            #     paper_bgcolor='rgb(243, 243, 243)',
            #     plot_bgcolor='rgb(243, 243, 243)',
            )
                            
            placeholder_light.plotly_chart(fig) 
        
        
        
        elif choice == 'Predictions':
            st.write('hi')



   
    else:
        with col4:
            st.write('NA')
        with col5:
            st.write('NA')
        with col6:
            st.write('NA')