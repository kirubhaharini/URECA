import streamlit as st
import school

state = st.session_state 

state.query_params = st.experimental_get_query_params()
# st.write(state.query_params)

state.school = 'SchoolSelection' #default url query param for selection

if 'school' in state.query_params.keys(): #if url is changed directly
    state.school = state.query_params["school"][0]
else:
    st.experimental_set_query_params(school=state.school)


school_list = ['Select','NCHS','DSS']
state.ph = st.empty()
state.dropdownMenu = state.ph.selectbox('school',school_list,index = 0,key='page')

if (state.school!='SchoolSelection'):
    state.ph.empty()
    school.school(state)
elif (state.dropdownMenu != 'Select'):
    state.school = state.dropdownMenu
    state.ph.empty()
    school.school(state)
    st.experimental_set_query_params(school=state.school)
