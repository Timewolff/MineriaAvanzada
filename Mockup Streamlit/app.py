import streamlit as st
import streamlit_antd_components as sac
from streamlit_option_menu import option_menu

# Import other modules
import home
import eda
import results

# Main menu
selected = option_menu(
    menu_title=None,
    options=["Home", "EDA", "Results"],
    icons=["house", "bar-chart", "rocket-takeoff"],  # Bootstrap Icons
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={  # 🎨 Personalización de estilos
        "container": {
            "padding": "0!important",
            "background-color": "#121212",
        },
        "icon": {"color": "white", "font-size": "20px"},
        "nav-link": {
            "font-size": "16px",
            "text-align": "center",
            "margin": "0px",
            "padding": "10px 10px",
            "color": "#fff",
            "border-radius": "10px",
        },
        "nav-link-selected": {
            "background-color": "orange",
            "color": "white",
            "font-weight": "bold",
        },
    } # 🔥 Establece el menú en horizontal
)

if selected == "Home":
    home.show()
elif selected == "EDA":
    eda.show()
elif selected == "Results":
    results.show()