import NCHS,DSS #ADD SCHOOLS HERE
import streamlit as st
   
def school(state):
  
    st.title(state.school + ' Dataset')

    if state.school == 'NCHS':
        NCHS.NCHS(state)
    elif state.school == 'DSS':
        DSS.DSS(state)



    # #back button

    # state.back = st.button('Select School')
    # if state.back:
    #     state.check = st.write('okok')
    #     state.dropdownMenu='Select' #reset
    #     state.check2 = st.write('ok')
    #     return #temp_trial.main()
   # if not state.activated: return #temp_trial.main() #go back to app.py