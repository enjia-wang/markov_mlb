import streamlit as st 
from markov_mlb import simulate_mlb_game


"""
streamlit run streamlit_preds.py
"""


bat_proj = "ATC_Batters_2026.csv"
pitch_proj = "ATC_Pitchers_2026.csv"

st.header("Welcome to this website", divider="rainbow")

# get inputs for date and teams
with st.form("game_info"):
    date = st.text_input("Date: ")
    home_team = st.text_input("Home Team (Abbreviated): ").upper()
    away_team = st.text_input("Away Team (Abbreviated): ").upper()

    submitted = st.form_submit_button("Get Probabilities")

# run the simulation, print probs
if submitted: 
    if home_team and away_team and date: 
        prob_dict, stat_lists = simulate_mlb_game(home_team,away_team,date,bat_proj,pitch_proj)
        st.markdown(
            prob_dict
        )