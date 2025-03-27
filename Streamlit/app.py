import streamlit as st
import streamlit_antd_components as sac
from streamlit_option_menu import option_menu
import pandas as pd

# Import other modules
import about
import start
from eda import EDAApp
import results as results

main_color = "#384B70"

# Main App
with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["Start Analysis" ,"EDA", "Results", "About"],
        icons=["bi bi-sliders","bar-chart", "bi bi-clipboard2-data-fill", "bi bi-mortarboard"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"background-color": "#f8f9fa"},
            "icon": {"font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#d9d9d9"},
            "nav-link-selected": {"background-color": main_color, "color": "white", "icon": {"color": "white"} },
        }
    )
  
if selected == "Start Analysis":
    start.show()
elif selected == "EDA":
    EDAApp.show()
elif selected == "Results":
    if st.session_state.get("problem_type") == "forecast":
        results.show_forecast()
    else:
        results.show_supervised()
    #unsupervised_results = unsupervised_model.get_results()
    #results.show_unsupervised(unsupervised_results)
elif selected == "About":
    about.show()


