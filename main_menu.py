import streamlit as st
import school
st.set_page_config(layout="wide",initial_sidebar_state="collapsed")

state = st.session_state 
school_list = ['Select','NCHS','DSS','SGS','YSS'] #list of schools
state.ph = st.empty() #placeholder

state.query_params = st.experimental_get_query_params()
# st.write(state.query_params)

if 'school' not in state: #initilize
    state.school = 'SchoolSelection' #default url query param for selection
    #st.experimental_set_query_params(school='SelectSchool') ##### - if initialized: doesnt work!

if 'school' in state.query_params.keys(): #if url is changed directly
    #if state.query_params["school"][0] != 'SelectSchool':
    state.school = state.query_params["school"][0] #?school=NCHS
    # st.write(state.school)
# else:

if state.school == 'SchoolSelection':  #if mainmenu page
    state.dropdownMenu = state.ph.selectbox('school',school_list,index = 0,key='page')
    if (state.dropdownMenu != 'Select'): #if selectbox changed
        state.school = state.dropdownMenu
        state.ph.empty()
        school.school(state)
        st.experimental_set_query_params(school=state.school)

else:  #if url changed directly - (state.school!='SchoolSelection')
    school.school(state)