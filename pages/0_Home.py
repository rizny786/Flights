import streamlit as st

st.set_page_config(page_title="Introduction", page_icon="💁🏽", layout="wide", initial_sidebar_state="auto", menu_items=None,)

st.title("✈️ Airline on-time performance")

st.header("Objective 🎯 ")

st.markdown('''
            1. Compare 1991 and 2001 with respect to:
- What characterises flights that are on time?
- What predictive methodologies can be employed to anticipate flight delays based on input parameters?
- How might the optimization of flight schedules be achieved by leveraging insights derived from model outcomes to minimize instances of flight delays?
2. A bigger airport intends to collect data from the travellers’ smart phones, flight plans, passport
control, security control, and visits to shops and restaurants. The intention is to collect data, and
then categorise travellers. In the future, travellers that are categorised as “likely to cause late
departure of flight” will receive reminders via text message and staff around the airport can access a
dashboard that can indicate in which areas the traveller might be in. Future plans also include to
install a facial recognition system in shops and restaurants.
            ''')


st.header("Analysis Steps 📊")
st.subheader("⚙️ Data Processing/Engineering")
st.subheader("🤖 Model Buliding/Selection")
st.subheader("✨ Feature analysis")
st.subheader("💡 Compare and Contrast")
