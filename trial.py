import streamlit as st
import school

state = st.session_state 

state.query_params = st.experimental_get_query_params()

if 'school' not in state:
    state.school = 'SchoolSelection'

school_list = ['Select','NCHS','DHS']
state.ph = st.empty()
state.dropdownMenu = state.ph.selectbox('school',school_list,index = 0,key='page')

if state.dropdownMenu != 'Select':
    state.school = state.dropdownMenu
    state.ph.empty()
    school.school(state)

st.experimental_set_query_params(school=state.school)
