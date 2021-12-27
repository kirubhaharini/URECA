import streamlit as st
import pandas as pd
import datetime
import plotly.express as px
from windrose import WindroseAxes
from plotly.subplots import make_subplots
import plotly.graph_objects as go

'''
Dashboard for Nan Chiau High School
Data from June to Aug 2021
'''

def NCHS(state):

    #merge dfs
    df1 = pd.read_excel("NCHS Weather Sensor Data.xlsx",'2021-06')
    df2 = pd.read_excel("NCHS Weather Sensor Data.xlsx",'2021-07')
    df3 = pd.read_excel("NCHS Weather Sensor Data_New.xlsx")
    df =  pd.concat([df1,df2,df3])
    df = df.reset_index(drop=True)


    df['hour'] = ''
    for row in range(len(df)):
        #changing nchsmote1 to mote1
        if 'nchs' in df.loc[row,'Device ID']:
            df.loc[row,'Device ID'] = df.loc[row,'Device ID'].replace('nchs','')

        time = (df.loc[row,'Time'])
        df.loc[row,'hour'] = time.hour
    df['Date'] = [datetime.datetime.date(d) for d in df['Time']]

    # st.write(df)

    #########average per day - for climograph################
    # #AVG by day
    # avg_df = df.groupby('Date',as_index=False).mean()
    # for row in range(len(avg_df)):
    #     avg_df.loc[row,'Day'] = avg_df.loc[row,'Date'].strftime('%d %B %Y')
    #     avg_df.loc[row,'Month'] = avg_df.loc[row,'Date'].strftime('%B')
    
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
    #for climograph
    placeholder = st.empty()
    #for temp graph
    placeholder1 = st.empty() #for graph - hourly temp
    #windrose
    st.write('windrose is not for this data - need to change later')
    temp_wind_df = px.data.wind()
    st.write(temp_wind_df)
    fig = px.bar_polar(temp_wind_df, r="frequency", theta="direction",
                            color="strength",
                        color_discrete_sequence= px.colors.sequential.Plasma_r)
    st.plotly_chart(fig)

    #date
    state.date_slider = st.sidebar.date_input("Date(s)",[min(df['Date']),max(df['Date'])],min_value=min(df['Date']),max_value=max(df['Date']))

    #devices
    devices = df['Device ID'].unique().tolist()
    state.devices_filter = st.sidebar.multiselect('Device(s)',devices,key='devices')

    #temperature
    min_temp = min(df['Temperature (°C)'])
    max_temp = max(df['Temperature (°C)'])
    state.temperature = st.sidebar.slider('Temperature',min_temp,max_temp,(min_temp,max_temp),step=0.1)

    #humidity
    min_hum = min(df['Humidity (%)'])
    max_hum = max(df['Humidity (%)'])
    state.humidity = st.sidebar.slider('Humidity',max_hum,min_hum,(min_hum,max_hum),step=0.1)




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



    filtered_df = pd.DataFrame()
    for row in range(len(df)):
        if min_date <= df.loc[row,'Date'] <= max_date: #date
            if df.loc[row,'Device ID'] in state.devices_filter: #device
                if min_temp<=df.loc[row,'Temperature (°C)']<=max_temp:#temperature
                    if min_hum<=df.loc[row,'Humidity (%)']<=max_hum:#humidity
                        filtered_df = filtered_df.append(df.iloc[row,:])
  

    
    placeholder2.table(filtered_df)


    if len(filtered_df)!=0:

        tempp = filtered_df[['hour','Temperature (°C)']]
        tempp.set_index('hour',inplace=True)
        temp = tempp.groupby('hour').mean()
        fig = px.line(temp, y='Temperature (°C)')
        fig.update_layout(title="Hourly Temperature",
                        template="plotly_white",title_x=0.5,legend=dict(orientation='h'))
        placeholder1.plotly_chart(fig)


        ########climograph#######################
        avg_df = filtered_df.groupby('Date',as_index=False).mean()
        for row in range(len(avg_df)):
            avg_df.loc[row,'Day'] = avg_df.loc[row,'Date'].strftime('%d %B %Y')
            avg_df.loc[row,'Month'] = avg_df.loc[row,'Date'].strftime('%B')
        
        fig_combi = make_subplots(specs=[[{"secondary_y": True}]])#this a one cell subplot
        fig_combi.update_layout(title="Climograph",
                        template="plotly_white",title_x=0.5,legend=dict(orientation='h'))

        trace1 = go.Bar(x=avg_df['Date'], y=avg_df['Rainfall (mm)'], opacity=0.5,name='Rainfall (mm)',marker_color ='#1f77b4')

        trace2p = go.Scatter(x=avg_df['Date'], y=avg_df['Temperature (°C)'],name='Temperature ((°C))',mode='lines+markers',line=dict(color='#e377c2', width=2))

        #The first trace is referenced to the default xaxis, yaxis (ie. xaxis='x1', yaxis='y1')
        fig_combi.add_trace(trace1, secondary_y=False)

        #The second trace is referenced to xaxis='x1'(i.e. 'x1' is common for the two traces) 
        #and yaxis='y2' (the right side yaxis)

        fig_combi.add_trace(trace2p, secondary_y=True)

        fig_combi.update_yaxes(#left yaxis
                        title= 'mm',showgrid= False, secondary_y=False)
        fig_combi.update_yaxes(#right yaxis
                        showgrid= True, 
                        title= '°C',
                        secondary_y=True)
        ######################################################################
        placeholder.plotly_chart(fig_combi)



 ##NOTE: MIGHT HAVE ERRORS if eg rainfall in filtered df is '-' or NaN etc
        with col4: 
            st.write(round(filtered_df['Rainfall (mm)'].mean(),2))
        with col5:
            st.write(round(filtered_df['Temperature (°C)'].mean(),2))
        with col6:
            st.write(round(filtered_df['Humidity (%)'].mean(),2))
   
    else:
        with col4:
            st.write('NA')
        with col5:
            st.write('NA')
        with col6:
            st.write('NA')