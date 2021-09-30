import streamlit as st
import pandas as pd
import datetime
import school_selection
import plotly.express as px
from streamlit.report_thread import get_report_ctx
from streamlit.hashing import _CodeHasher
from streamlit.server.server import Server
import sessionstate



def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)
    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    return session_info.session

def _get_state(hash_funcs=None):
    session = _get_session()
    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = sessionstate._SessionState(session, hash_funcs)
    return session._custom_session_state

    
def school(state):
    
    state = _get_state()
    
    #back button
    state.back = st.button('Back')
    if state.back:
        state.page=False
        return
    if not state.page: return #go back to app.py

    school = state.page
    st.write(school)
    # st.write(st.get_params())
    st.title(school+' Dataset')

    if school == 'NCHS':

        #for NCHS: 

        df1 = pd.read_excel("NCHS Weather Sensor Data.xlsx")
        df2 = pd.read_excel("NCHS Weather Sensor Data_New.xlsx")
        df =  pd.concat([df1,df2])
        df = df.reset_index(drop=True)


        df['School'] = 'NCHS'

        df['hour'] = ''
        for row in range(len(df)):
            time = (df.loc[row,'Time'])
            df.loc[row,'hour'] = time.hour
        df['Date'] = [datetime.datetime.date(d) for d in df['Time']]

        st.write(df)
        placeholder = st.empty()

        fig = px.bar(df, x='hour', y='Temperature (°C)', color='Device ID')
        st.plotly_chart(fig)

        #date
        state.date_slider = st.sidebar.date_input("Date(s)",[min(df['Date']),max(df['Date'])],min_value=min(df['Date']),max_value=max(df['Date']))
        st.sidebar.write(state.date_slider)

        #devices
        devices = df['Device ID'].unique().tolist()
        state.devices_filter = st.sidebar.multiselect('Device(s)',devices)

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
            st.write(min_date,max_date)
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


        placeholder.table(filtered_df)


    
    
    state.sync()



# if __name__ == "__main__":
#     main()

