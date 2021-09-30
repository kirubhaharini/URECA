import streamlit as st
import school

state = st.session_state 

state.query_params = st.experimental_get_query_params()
st.write(state.query_params)

#state.school_change = 'None' #changing url directly
state.school = 'SchoolSelection'

if 'school' in state.query_params.keys(): #if url is changed directly
    state.school = state.query_params["school"][0]

school_list = ['Select','NCHS','DHS']
state.ph = st.empty()
state.dropdownMenu = state.ph.selectbox('school',school_list,index = 0,key='page')

if (state.dropdownMenu != 'Select'): #or (state.school!='SchoolSelection'):
    state.school = state.dropdownMenu
    state.ph.empty()
    school.school(state)
elif (state.school!='SchoolSelection'):
    state.ph.empty()
    school.school(state)

state.set_school = st.experimental_set_query_params(school=state.school)
