import streamlit as st
import pandas as pd


st.set_page_config(page_title="Compare & Contrast", page_icon="💡", layout="wide", initial_sidebar_state="auto", menu_items=None)

@st.cache_data
def load_data():
   cols_91 = ['Year','Month','DayofMonth','DayOfWeek','DepTime','CRSDepTime','ArrTime','CRSArrTime','UniqueCarrier','FlightNum','ActualElapsedTime','CRSElapsedTime','ArrDelay','DepDelay','Origin','Dest','Distance','Cancelled','Diverted']
   cols_01 = ['Year','Month','DayofMonth','DayOfWeek','DepTime','CRSDepTime','ArrTime','CRSArrTime','UniqueCarrier','FlightNum','TailNum','ActualElapsedTime','CRSElapsedTime','ArrDelay','DepDelay','Origin','Dest','Distance','TaxiIn','TaxiOut','Cancelled','Diverted']
   return  pd.read_csv("Data/1991.csv.gz", encoding='cp1252', compression="gzip", usecols=cols_91), pd.read_csv("Data/2001.csv.gz", encoding='cp1252', compression="gzip", usecols=cols_01)
   

df91, df01 = load_data()

diff = pd.read_csv('Data/diff.csv', index_col=None, delimiter='|')
simi = pd.read_csv('Data/simi.csv', index_col=None, delimiter='|')
avg_delay_by_airline_91 = df91.groupby('Airline')['ArrDelay'].mean().reset_index()
top_10_airline_delays_91 = avg_delay_by_airline_91.nlargest(12, 'ArrDelay')

avg_delay_by_airline_01 = df01.groupby('Airline')['ArrDelay'].mean().reset_index()
top_10_airline_delays_01 = avg_delay_by_airline_01.nlargest(12, 'ArrDelay')




col_l, col_r = st.columns([1,1], gap="small")
with col_l:
    st.header("Similarities 1991 vs 2001")
    st.table(simi)
    col_c1, col_c2, col_c3 = st.columns([1,1,1], gap="small")
    with col_c1:
        top_10_airline_delays_91.show()

with col_r:
    st.header("Differences 1991 vs 2001")
    st.table(diff)
    col_c1, col_c2, col_c3 = st.columns([1,1,1], gap="small")
    with col_c1:
      top_10_airline_delays_91.show()


st.header("What predictive methodologies can be employed to anticipate flight delays based on input parameters?", divider="grey")
st.markdown('''
##### Model Selection Process:

Upon thorough evaluation of various models, it was observed that the classifier model exhibited superior performance in predicting flight delays based on input parameters. Considering both accuracy and computational efficiency, the Decision Tree classifier emerged as the most suitable model.

##### Decision to Utilize Decision Tree Classifier:

Given its commendable performance metrics and computational cost-effectiveness, the Decision Tree classifier was chosen as the primary model for predicting flight delays. Its accuracy and interpretability make it a reliable choice for anticipating delays based on the provided input parameters.

            ''')


st.header("How might the optimization of flight schedules be achieved by leveraging insights derived from model outcomes to minimize instances of flight delays?")
st.markdown('''
            ### Approach:

 **Optimization of Flight Schedules:**
  - Identifies areas within flight schedules that require efficiency enhancement to reduce delays.
  
**Leveraging Insights from Model Outcomes:**
  - Utilizes predictive models, specifically the Decision Tree classifier, to discern influential factors contributing to flight delays.
  
**Minimizing Instances of Flight Delays:**
  - Formulates strategies based on model insights to mitigate and decrease the frequency of flight delays.

### Actionable Steps:

**Efficiency Enhancement in Flight Schedules:**
   - Focuses on optimizing turnaround times, considering peak hours and weather conditions affecting flight operations.
  
**Influential Factors Identification:**
   - Identifies factors like departure time, distance, carrier, and historical delays as crucial influencers impacting flight delays.
  
**Strategies Implementation:**
   - Implements revised scheduling protocols based on the identified influential factors to reduce instances of delays.

### Resulting Impact:

- **Reduced Delays:** The application of insights derived from the Decision Tree model leads to a noticeable reduction in flight delays.
  
- **Improved Operational Efficiency:** Adjustments in scheduling based on influential factors enhance overall operational efficiency.

            ''')

st.header("Ethical Analysis of Airport Data Collection and Utilization through Solove's Taxonomy of Privacy (2006)", divider="grey")
st.markdown('''
##### Information Collection: Extensive Data Gathering

The airport's intent to collect a vast array of data from travelers' smartphones, flight plans, passport control, security checkpoints, and commercial interactions raises concerns aligned with Solove's "Information Collection." This comprehensive data acquisition potentially infringes upon individuals' privacy rights by amassing extensive personal information without explicit consent or justification for the breadth of surveillance.

##### Information Processing: Categorization and Predictive Modeling

The airport's plan to categorize travelers based on their potential to cause flight delays invokes issues within Solove's "Information Processing" taxonomy. Categorizing individuals using predictive algorithms may introduce biases or errors, impacting privacy by subjecting individuals to potential profiling or unfair treatment based on algorithmic decisions, undermining their privacy and autonomy.

##### Information Dissemination and Invasion: Personalized Notifications and Location Tracking

Utilizing travelers' categorized data to send targeted text message reminders and providing airport staff with a dashboard indicating travelers' locations raises concerns of "Information Dissemination" and "Invasion" as per Solove's framework. Disseminating personalized information and monitoring individuals' movements without transparent consent may infringe upon privacy rights, potentially leading to invasive surveillance practices.

##### Decisional Interference: Facial Recognition Systems in Commercial Spaces

The proposed installation of facial recognition systems in commercial establishments aligns with Solove's concept of "Decisional Interference." Implementing such systems without clear consent mechanisms or oversight may impede individuals' autonomy and choices within these spaces, raising ethical concerns regarding privacy and potential limitations on individuals' freedom.

##### Conclusion

The airport's comprehensive data collection and utilization strategy significantly implicate Solove's taxonomy of privacy, encompassing concerns regarding extensive data gathering, potential biases in categorization, dissemination of personal information, invasive surveillance practices, and interference with individuals' decision-making autonomy. This initiative requires careful ethical consideration to balance security imperatives with the protection of individuals' privacy rights and ethical norms within society.

''')    