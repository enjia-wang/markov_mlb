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
        
        # find probabilities within betting range
        if prob_dict["Home_Win_Prob"] >= 0.575 and prob_dict["Home_Win_Prob"] <= 0.625:
            st.markdown(f"Home Will Likely Win, prob: {prob_dict["Home_Win_Prob"]}")
        if prob_dict["Home_Win_Prob"] >= 0.65:
            st.markdown(f"Home Will Likely Win, prob: {prob_dict["Home_Win_Prob"]}")
        elif prob_dict["Home_Win_Prob"] <= 0.375: 
            st.markdown(f"Home Will Likely Lose, prob: {round(1-prob_dict["Home_Win_Prob"],2)}")
        
        if prob_dict["Over_8_Prob"] >= 0.6 and prob_dict["Over_8_Prob"] <= 0.65: 
            st.markdown(f"Total Runs Likely Over 8.5, prob: {prob_dict["Over_8_Prob"]}")
        if prob_dict["Over_8_Prob"] >= 0.725:
            st.markdown(f"Total Runs Likely Over 8.5, prob: {prob_dict["Over_8_Prob"]}")

        if prob_dict["Over_4_Prob"] >= 0.625: 
            st.markdown(f"Total Runs Likely Over 4.5, prob: {prob_dict["Over_4_Prob"]}")
        elif prob_dict["Over_4_Prob"] <= 0.375: 
            st.markdown(f"Total Runs Likely Under 4.5, prob: {round(1-prob_dict["Over_4_Prob"],2)}")

        if prob_dict["RIFI_Prob"] >= 0.6: 
            st.markdown(f"Run in First Inning Likely, prob: {prob_dict["RIFI_Prob"]}")
            st.markdown(f"Run in First Inning Likely, prob: {prob_dict["RIFI_Prob"]}")
