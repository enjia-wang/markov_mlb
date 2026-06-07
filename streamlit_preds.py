import streamlit as st 
from markov_mlb import simulate_mlb_game

bat_proj = "ATC_Batters_2026.csv"
pitch_proj = "ATC_Pitchers_2026.csv"

st.header("Welcome to this website", divider="rainbow")

date = st.text_input("Date: ")
home_team = st.text_input("Home Team (Abbreviated): ")
away_team = st.text_input("Away Team (Abbreviated): ")

prob_dict, stat_lists = simulate_mlb_game(home_team,away_team,date,bat_proj,pitch_proj)

st.markdown(
    prob_dict
)