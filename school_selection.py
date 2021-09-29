import streamlit as st
import pandas as pd
import school
import sessionstate
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server

import webbrowser

def main():
    state = _get_state()

    def make_clickable(link):
        # target _blank to open new window
        # extract clickable text to display for your link
        text = link #.split('=')[1]
        return f'<a target="_blank" href="https://share.streamlit.io/kirubhaharini/ureca/main/{text}">{text}</a>'

    # link is the column with hyperlinks

    options = ['NCHS.py','SGS','Dunearn',"Yishun"]
    df = pd.DataFrame(options,columns=['schools'])
    st.write(df)
    #df['schools'] = df['schools'].apply(make_clickable)
    url = 'https://share.streamlit.io/kirubhaharini/ureca/main/school.py'
    for i in range(len(df)):
        if st.button(df["schools"][i], key=df["schools"][i]):
            school_name = df["schools"][i]
            url = 'https://share.streamlit.io/kirubhaharini/ureca/main/school.py' # #+ '/' + school_name
            state.query_username = df["schools"][i]
            state.__setattr__(query_username,school_name)
            state.__setitem__(query_username,school_name)
            st.write(url)
            
            #webbrowser.open(url,new=2)
    
    
#     from bokeh.models.widgets import Div

#     if st.button('Go to Streamlit'):
#         js = "window.open('https://www.streamlit.io/')"  # New tab or window
#         js = "window.location.href = 'https://www.streamlit.io/'"  # Current tab
#         html = '<img src onerror="{}">'.format(js)
#         div = Div(text=html)
#         st.bokeh_chart(div)

    st.write(state.query_username)
    
    if state.query_username: 
        school.school(state)


    # df = df.to_html(escape=False)
    # st.write(df, unsafe_allow_html=True)

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
