import streamlit as st
from comps.knowledge_based import ask_me_anything
from comps.mental import therapy
from comps.email import ai_mail

def navigate():
    st.sidebar.title("메인 페이지")
    page = st.sidebar.radio("Go to", ["Ask", "Therapy", "Resume", "Email"])

    if page == "Ask":
        ask_me_anything()
    elif page == "Therapy":
        therapy()
    elif page == "Resume":
        pass
    elif page == "Email":
        ai_mail()
# Call the navigate function to display the sidebar and handle page navigation
if __name__ == "__main__":
    navigate()