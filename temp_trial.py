import streamlit as st
#import nchs,dhs #to display the other pages
import school
import sessionstate
from streamlit.hashing import _CodeHasher
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server

#@st.cache(hash_funcs={st.delta_generator.DeltaGenerator: my_hash_func})
def main():
    state = _get_state()

    # pages = {
    #         "NCHS": school.school,
    #         "DHS" : school.school
    #     }
    #state.ph = st.sidebar.empty()
    #state.ph1 = st.sidebar.empty()
    

    state.page = st.selectbox('school',['Select','NCHS','DHS'],index = 0)
    st.experimental_set_query_params(school=state.page)
    # Display the selected page with the session state
    st.title("Pages")
    #options = tuple(pages.keys())
    #state.page = st.sidebar.radio("Select your page", options, options.index(state.page) if state.page else 0)
   # pages[state.page](state)
    
    if state.page != 'Select':
        
           # state.ph.empty()
            #state.ph1.empty()
        st.experimental_set_query_params(school=state.page)
        school.school(state)

        # Mandatory to avoid rollbacks with widgets, must be called at the end of your app
    

    state.sync()

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


if __name__ == "__main__":
    main()

