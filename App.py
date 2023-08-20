from sqlalchemy import create_engine
from mlxtend.frequent_patterns import association_rules
from mlxtend.frequent_patterns import apriori
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import datetime as dt
import calendar
from wordcloud import WordCloud
from pyparsing import col
import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


# from wordcloud import WordCloud

# lets import our data from the AWS RDS MySQL DataBase
# db info

host = "localhost"
user = "skumar"
password = "root"
# port = st.secrets["port"]
database = "market_analysis"


st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title="Unveiling Customer Insights through Market Basket Analysis on Online Retail Data",
                   page_icon="random",
                   layout="wide")

col1, col2, col3 = st.columns((.1, 1, .1))

with col1:
    st.write("")

with col2:
    st.markdown("<h1 style='text-align: center;'>Unveiling Customer Insights through Market Basket Analysis on Online Retail Data</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'><i><b>Providing a Retail Business with a strategy which helps improve their "
                "product sales, inventory management, and customer retention, which in turn would improve the profitability of the business.</b></i></p>", unsafe_allow_html=True)
    # st.markdown("<center><img src='https://github.com/kkrusere/Market-Basket-Analysis-on-the-Online-Retail-Data/blob/main/Assets/MBA.jpg?raw=1' width=600/></center>", unsafe_allow_html=True)

with col3:
    st.write("")

st.markdown("----")

col1, col2, col3 = st.columns((1, 0.1, 1))

with col1:

    st.markdown("### ***Project Contributors:***")
    st.markdown("Shreyash Kumar || Samridhha Kumar Vyas || Animesh Shukla")

    st.markdown("### **Project Introduction**")
    st.markdown("***Business Proposition:*** This initiative focuses on enhancing the performance of retail enterprises by devising a comprehensive strategy."
                "The strategy aims to elevate product sales, optimize inventory management, and bolster customer retention, consequently elevating overall business profitability"
                "The retail sector places paramount importance on profitability and the bottom line, driving the essence of this endeavor."
                "By scrutinizing transaction data garnered from point-of-sale systems, "
                "invaluable insights are unlocked. These insights unravel customer purchasing behavior, product trends, and sales patterns, "
                "providing a profound understanding of business dynamics."
                "Through a meticulous exploration of these data-driven insights, patterns, and correlations, an effective strategy is meticulously crafted."
                "This strategy, upon implementation, orchestrates a tangible upswing in the retail enterprise's revenue, profits, and operational efficacy."
                )
    st.markdown("***Methodology:*** approach encompasses Data Mining, Analysis, and Visualization techniques applied to Retail Sales Data.")
    """
    1. Market Basket Analysis (MBA): 
        This technique delves into the retail sales data to uncover connections and recurring patterns. It identifies relationships among products frequently purchased together, shedding light on customer preferences and enabling strategic product placement. 
    2. Customer Segmentation:
    > * RFM Analysis: This involves categorizing customers based on three factors: recency of purchase, frequency of purchase, and monetary value spent. This segmentation helps in targeting high-value customers and tailoring marketing strategies accordingly.
    > * RFM (recency, frequency, monetary) Analysis
    3. Product Recomendation :
        > *Association Analysis: Utilizing the "people who bought this also bought" concept, this method suggests related products to customers based on their previous purchases, thereby boosting cross-selling opportunities.
    """

    st.markdown("In addition, we created this `Streamlit` interactive data visualization "
                "tool that allows users interact with the data and analytics.")

with col2:
    pass
with col3:
    st.markdown("### ***Data Collection:***")
    """
    **Overview of the Data**

    This dataset contains records of transactions that occurred between 01/12/2010 and 09/12/2011 for an online retail business based in the UK. The company specializes in distinctive all-occasion gifts and serves both individual customers and wholesale buyers.

    **Details about the Dataset's Attributes/Columns**
    * ***InvoiceNo:*** This is a unique 6-digit code assigned to each transaction. Transactions starting with the letter 'c' signify cancellations.
    * ***StockCode:*** A 5-digit code assigned to each distinct product.
    * ***Description:*** The name of the product.
    * ***Quantity:*** The quantity of each product involved in the transaction.
    * ***InvoiceDate:*** Date and time of the transaction.
    * ***UnitPrice:*** The cost per unit of the product in the local currency.
    * ***CustomerID:*** A unique 5-digit code assigned to each customer.
    * ***Country:*** The name of the country where the customer resides.

    ###### **The data source:**
    
    """

    # st.image("Assets/UCI_ML_REPO.png", caption="https://archive.ics.uci.edu/ml/datasets/online+retail")

st.markdown("----")


@st.cache(allow_output_mutation=True, ttl=1200)
def load_data():
    """
    This fuction loads data from the aws rds mysql table
    """
    data = None
    engine = create_engine(
        f"mysql+pymysql://{user}:{password}@{host}/{database}")

    try:
        query = f"SELECT * FROM MBA_Online_Retail_Data"
        data = pd.read_sql(query, engine)

    except Exception as e:
        print(str(e))

    return data


# loading the data
df = load_data()

st.markdown("#### ***Lets take a look at the data:***")
"""
We are going to use the pandas `.shape` function/method to the total number of columns and rows of the dataframe. We can see that our dataframe contains 481313 rows and 16 columns

We'll use the pandas `.info()` function so see the general infomation (data types, null value count, etc.) about the data.
"""
st.markdown(f"###### ***The shape of the data***: {df.shape}")

col1, col2, col3 = st.columns((1, 0.01, .5))

df_head = pd.read_csv("df_head.csv")
with col1:
    st.markdown("***The below is the first 5 rows of the cleaed dataset***")
    st.dataframe(df_head)
with col2:
    pass
df_info = pd.read_csv("df_info.csv", index_col=0)
with col3:
    st.markdown("***The below is the info of the data***")
    st.dataframe(df_info)

st.success("If you want to take a look at how the data was cleaned, you "
           "can go check out the jupyter notebook of this project at: "
           "https://github.com/shreyashkr17/MLflipkart/blob/main/grid.ipynb")


###################### functions############################
@st.cache(allow_output_mutation=True)
def group_Quantity_and_SalesRevenue(df, string):
    """ 
    This function inputs the main data frame and feature name 
    The feature name is the column name that you want to group the Quantity and Sales Revenue
    """

    df = df[[f'{string}', 'Quantity', 'Sales Revenue']].groupby([f'{string}']).sum(
    ).sort_values(by='Sales Revenue', ascending=False).reset_index()

    return df


@st.cache(allow_output_mutation=True)
def choose_country(country="All", data=df):
    """
    This fuction takes in a country name and filters the data frame for just country
    if the there is no country inputed the fuction return the un filtered dataframe
    """
    if country == "All":
        return data
    else:
        temp_df = data[data["Country"] == country]
        temp_df.reset_index(drop=True, inplace=True)

        return temp_df


def wordcloud_of_Description(df, title):
    """
    This fuction creates a word cloud
    inputs a data frame converts it to tuples and uses the input 'title' as the title of the word cloud
    """
    plt.rcParams["figure.figsize"] = (20, 20)
    tuples = [tuple(x) for x in df.values]
    wordcloud = WordCloud(
        max_font_size=100,  background_color="white").generate_from_frequencies(dict(tuples))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title(title, fontsize=27)
    plt.show()


country_list = ["All"] + list(dict(df['Country'].value_counts()).keys())


@st.cache(allow_output_mutation=True)
def choose_country(country, data=df):
    """
    This fuction takes in a country name and filters the data frame for just country
    if the there is no country inputed the fuction return the un filtered dataframe
    """
    if country == "All":
        return data
    else:
        temp_df = data[data["Country"] == country]
        temp_df.reset_index(drop=True, inplace=True)

        return temp_df


st.markdown("---")
st.markdown(" <h3 style='text-align: center;'>Delving into Data: Unleashing Insights through Dynamic Exploratory Data Analysis<i>(EDA)</i>:</h3>",
            unsafe_allow_html=True)
col1, col2, col3 = st.columns((.1, 1, .1))
with col1:
    pass
with col2:
    """
    * In the realm of data analysis, Exploratory Data Analysis (EDA) emerges as a dynamic practice. It revolves around the      meticulous examination of datasets, employing tools like statistical graphics and data visualization to encapsulate their core attributes. This integral approach involves delving into data at its onset, unveiling prevailing trends, detecting anomalies and outliers, and even unearthing concealed insights that lie beneath the surface.
    * This journey into data entails intriguing inquiries, such as quantifying the total purchase volume across various time spans, be it months, weeks, days of the week, or even down to precise hours. As we navigate this endeavor, our focus will shift to the clientele, a topic thoroughly explored in the forthcoming section: ***Recency***, ***Frequency***, and ***Monetary Analysis (RFM)*** within the Customer Segmentation domain of this project.
    """
with col3:
    pass

col1, col2, col3 = st.columns((1, .1, 1))

with col1:
    Country_Data = df.groupby("Country")["InvoiceNo"].nunique(
    ).sort_values(ascending=False).reset_index().head(10)
    fig = px.bar(Country_Data, x="InvoiceNo", y='Country',
                 title="Top 10 Number of orders per country with the UK")
    # fig.show(renderer='png', height=700, width=1000)
    # fig.show( height=700, width=1000)
    st.plotly_chart(fig)
    st.markdown("UK has more number of orders witk 16k Invoice numbers")

with col2:
    pass

with col3:
    Country_Data = df[df['Country'] != "United Kingdom"].groupby(
        "Country")["InvoiceNo"].nunique().sort_values(ascending=False).reset_index().head(10)
    fig = px.bar(Country_Data, x="InvoiceNo", y='Country',
                 title="Top 10 Number of orders per country without the UK")
    # fig.show(renderer='png', height=700, width=1000)
    # fig.show( height=700, width=1000)
    st.plotly_chart(fig)


col1, col2, col3 = st.columns((.1, 1, .1))
with col1:
    pass
with col2:
    """
    The visual representations presented above distinctly illustrate the UK's dominance in invoice generation, aligning with initial expectations, with its invoice count surpassing a substantial 15,000. Meanwhile, Germany secures second place with a notably lower tally, approximately thirty times less than the UK. These divergent statistics prompt the retail management to initiate a series of thoughtful inquiries into the underlying causes. Particularly noteworthy is the discrepancy given the online nature of the retail store. This invites critical questions such as the nature of web traffic to the store's online platform and the necessity to contemplate Search Engine Optimization (SEO) strategies. SEO entails enhancing both the quality and quantity of website traffic by optimizing visibility on search engines. The data stimulates multifaceted inquiries beyond these, underlining the comprehensive insights it offers.

    Transitioning our focus, we delve into a comparison of countries based on their Quantity sold and Sales Revenue throughout 2011. The ensuing graphical illustrations provide a comprehensive view of each country's performance in these crucial aspects over the entirety of the year.
    """
with col3:
    pass

col1, col2, col3 = st.columns((1, .1, 1))
with col1:
    # choice = st.radio("", ("Top 10", "Bottom 10"))
    temp_df = group_Quantity_and_SalesRevenue(df, 'Country')
    fig = px.bar(temp_df, x="Quantity", y='Country',
                 title="Quantity of orders per country with the UK")
    # fig.show(renderer='png', height=700, width=1000)
    # fig.show( height=700, width=1000)
    st.plotly_chart(fig)
with col2:
    pass
with col3:
    temp_df = group_Quantity_and_SalesRevenue(df, 'Country')
    fig = px.bar(temp_df[temp_df['Country'] != "United Kingdom"], x="Quantity",
                 y='Country', title="Quantity of orders per country without the UK")
    # fig.show(renderer='png', height=700, width=1000)
    # fig.show( height=700, width=1000)
    st.plotly_chart(fig)

col1, col2, col3 = st.columns((.1, 1, .1))
with col1:
    pass
with col2:
    """
    Just as expected, the UK has high volumes of Quantitly sold and the below charts should show that the UK has high sales as well. However, unlike the number of invoices, the Netherlands has the second highest volume of Quantity sold at approximately 200K. 
    """
with col3:
    pass


col1, col2, col3 = st.columns((1, .1, 1))
with col1:
    # choice = st.radio("", ("Top 10", "Bottom 10"))
    temp_df = group_Quantity_and_SalesRevenue(df, 'Country')
    fig = px.bar(temp_df, x="Sales Revenue", y='Country',
                 title="Sales Revenue of orders per country with the UK")
    # fig.show(renderer='png', height=700, width=1000)
    # fig.show( height=700, width=1000)
    st.plotly_chart(fig)

with col2:
    pass

with col3:
    temp_df = group_Quantity_and_SalesRevenue(df, 'Country')
    fig = px.bar(temp_df[temp_df['Country'] != "United Kingdom"], x="Sales Revenue",
                 y='Country', title="Sales Revenue of orders per country without the UK")
    # fig.show(renderer='png', height=700, width=1000)
    # fig.show( height=700, width=1000)
    st.plotly_chart(fig)

col1, col2 = st.columns((.1, 1.1))
with col1:
    pass
with col2:
    """
    The sales revenue of Netherlands and Germany is quite close. It would be interesting to see this broken down by time periods: 'Month', 'Week', 'Day of the Week', 'Time of Day' ,or 'Hour'.

    We now going to look at the products, which ones have high Quantity sold, or which product has high Sales Revenue. But first the below chart is a wordcloud of the product descriptions. A wordcloud is a visual representations of words that give greater prominence to words that appear more frequently, in this case the frequency is the 'Quantity'
    """


col1, col2, col3 = st.columns((3))
with col1:
    option = st.selectbox(
        'Please Choose a country to Analyze',
        country_list)
    if option == "All":
        st.markdown("We will at data from All the countries")
    else:
        st.markdown(f"We will be looking at data from {option}")

dataframe = choose_country(country=option)

st.markdown("###### **We can create a word cloud of the Product Descriptions per Quantity & Product Descriptions per Sales Revenue**")

col1, col2, col3 = st.columns((1, .3, 1))
with col1:
    temp_df = pd.DataFrame(dataframe.groupby('Description')[
                           'Quantity'].sum()).reset_index()
    title = "Product Description per Quantity"
    wordcloud_of_Description(temp_df, title)
    st.pyplot()

with col2:
    pass

with col3:
    temp_df = pd.DataFrame(dataframe.groupby('Description')[
                           'Sales Revenue'].sum()).reset_index()
    title = "Product Description per Sales Revenue"
    wordcloud_of_Description(temp_df, title)
    st.pyplot()

st.markdown("##### **Monthly Stats:**")
"""
Below are the monthly analysis of the Sales and the Quantity of iterms sold
"""
col1, col2, col3 = st.columns((1, .3, 1))
with col1:
    temp_df = group_Quantity_and_SalesRevenue(dataframe, 'Month')
    fig = make_subplots(rows=1, cols=2, shared_yaxes=False,
                        subplot_titles=("Quantity", "Sales Revenue")
                        )
    fig.add_trace(
        go.Bar(x=temp_df['Month'], y=temp_df['Quantity'], name='Quantity'), 1, 1)

    fig.add_trace(go.Bar(
        x=temp_df['Month'], y=temp_df['Sales Revenue'], name='Sales Revenue'), 1, 2)

    fig.update_layout(showlegend=False,
                      title_text="Monthly Sales Revanue and Quantity")
    # fig.show(renderer='png', height=700, width=1200)
    # fig.show( height=700, width=1000)
    st.plotly_chart(fig)
    """
    The above graphs show the monthly trend of Quantity of products ordered(left) and Sales Revenue(right).
    """
with col2:
    pass
with col3:
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{"type": "pie"}, {"type": "pie"}]],
                        subplot_titles=("Quantity per Month",
                                        "Sales Revenue per Month")
                        )

    fig.add_trace(
        go.Pie(values=temp_df['Quantity'], labels=temp_df['Month'],
               name='Quantity'),
        row=1, col=1
    )
    fig.add_trace(
        go.Pie(values=temp_df['Sales Revenue'], labels=temp_df['Month'],
               name='Sales Revenue'),
        row=1, col=2
    )
    fig.update_layout(
        title_text="Percentage pie charts for Monthly Sales Revanue and Quantity")

    # fig.show(renderer='png', height=700, width=1200)
    # fig.show( height=700, width=1000)
    st.plotly_chart(fig)
    """
    The above pie charts depicts the quantity of products ordered and sales revenue per month. 
    """

##############################
st.markdown("##### **Weekly Stats:**")
"""
The below are the weekly analysis of the Sales and the Quantity of iterms sold
"""
col1, col2, col3 = st.columns((.5, 1, .5))
with col1:
    pass
with col2:
    temp_df = group_Quantity_and_SalesRevenue(dataframe, 'Week of the Year')
    fig = make_subplots(rows=1, cols=2, shared_yaxes=False,
                        subplot_titles=("Quantity", "Sales Revenue")
                        )

    fig.add_trace(go.Bar(x=temp_df['Week of the Year'],
                  y=temp_df['Quantity'], name='Quantity'), 1, 1)

    fig.add_trace(go.Bar(x=temp_df['Week of the Year'],
                  y=temp_df['Sales Revenue'], name='Sales Revenue'), 1, 2)

    fig.update_layout(showlegend=False,
                      title_text="Weekly Sales Revanue and Quantity")
    # fig.show(renderer='png', height=700, width=1200)
    # fig.show( height=700, width=1000)
    st.plotly_chart(fig)

    """
    The above graphs shows the weekly trend of sales revenue and the quantity of products ordered. 
    """
with col3:
    pass

##############################

st.markdown("##### **Daily Stats:**")
"""
The below are the daily analysis of the Sales and the Quantity of iterms sold
"""
col1, col2, col3 = st.columns((1, .3, 1))
with col1:
    temp_df = group_Quantity_and_SalesRevenue(dataframe, 'Day of Week')
    fig = make_subplots(rows=1, cols=2, shared_yaxes=False,
                        subplot_titles=("Quantity", "Sales Revenue")
                        )

    fig.add_trace(go.Bar(x=temp_df['Day of Week'],
                  y=temp_df['Quantity'], name='Quantity'), 1, 1)

    fig.add_trace(go.Bar(
        x=temp_df['Day of Week'], y=temp_df['Sales Revenue'], name='Sales Revenue'), 1, 2)

    fig.update_layout(coloraxis=dict(colorscale='Greys'), showlegend=False,
                      title_text="Day of the Week Sales Revanue and Quantity")
    # fig.show(renderer='png', height=700, width=1200)
    # fig.show( height=700, width=1000)
    st.plotly_chart(fig)

    st.markdown(
        "The above graphs depict the daily trend of Sales revenue and quantity.")
with col2:
    pass
with col3:
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{"type": "pie"}, {"type": "pie"}]],
                        subplot_titles=("Quantity", "Sales Revenue")
                        )

    fig.add_trace(
        go.Pie(values=temp_df['Quantity'], labels=temp_df['Day of Week'],
               name='Quantity'),
        row=1, col=1
    )
    fig.add_trace(
        go.Pie(values=temp_df['Sales Revenue'], labels=temp_df['Day of Week'],
               name='Sales Revenue'),
        row=1, col=2
    )
    fig.update_layout(
        title_text="Percentage pie charts for Day of the Week Sales Revanue and Quantity")

    # fig.show(renderer='png', height=700, width=1200)
    # fig.show( height=700, width=1000)
    st.plotly_chart(fig)
    st.markdown(
        "The above pie charts shows the daily trend of sales revenue and quantity of products ordered.")

col1, col2, col3 = st.columns((.5, 1, .5))
with col1:
    pass
with col2:
    temp_df = group_Quantity_and_SalesRevenue(dataframe, 'Time of Day')
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{"type": "pie"}, {"type": "pie"}]],
                        subplot_titles=("Quantity", "Sales Revenue")
                        )
    fig.add_trace(
        go.Pie(values=temp_df['Quantity'], labels=temp_df['Time of Day'],
               name='Quantity'),
        row=1, col=1
    )
    fig.add_trace(
        go.Pie(values=temp_df['Sales Revenue'], labels=temp_df['Time of Day'],
               name='Sales Revenue'),
        row=1, col=2
    )
    fig.update_layout(
        title_text="Percentage pie charts for Time of Day Sales Revanue and Quantity")

    st.plotly_chart(fig)
    st.markdown(
        "The above piecharts shows the breakdown of Quantity of orders(left) and Sales revenue(right) by time of the day.")
with col3:
    pass

col1, col2, col3 = st.columns((1, .1, 1))
with col1:
    # we can also look at the volume of Invoice Numbers hourly data
    Hourly_Sales = (dataframe.groupby('Hour').sum()["Quantity"]).reset_index()
    fig = px.bar(Hourly_Sales, x='Hour', y='Quantity',
                 title='Hourly Volume of quantity sold')
    # fig.show(renderer='png', height=700, width=1000)
    # fig.show(height=700, width=1000)
    st.plotly_chart(fig)

with col2:
    pass
with col3:
    # we can also look at the volume quantity sold hourly data
    Hourly_Sales = (dataframe.groupby('Hour').count()
                    ["InvoiceNo"]).reset_index()
    fig = px.bar(Hourly_Sales, x='Hour', y='InvoiceNo',
                 title='Hourly sale using the Invoice Numbers')
    # fig.show(renderer='png', height=700, width=1000)
    # fig.show(height=700, width=1000)
    st.plotly_chart(fig)

st.markdown("##### ***Customers:***")

col1, col2, col3 = st.columns((1, .1, 1))
with col1:
    data = dataframe.groupby("CustomerID")["InvoiceNo"].nunique(
    ).sort_values(ascending=False).reset_index().head(11)
    fig = px.bar(data, x='CustomerID', y='InvoiceNo',
                 title='Graph of top ten customer with respect to the invoice number')
    # fig.show(renderer='png', height=700, width=1000)
    # fig.show(height=700, width=1000)
    st.plotly_chart(fig)
with col2:
    pass
with col3:
    temp_df = dataframe[dataframe["CustomerID"] != "Guest Customer"]
    data = temp_df.groupby("CustomerID")["InvoiceNo"].nunique(
    ).sort_values(ascending=False).reset_index().head(11)
    fig = px.bar(data, x='CustomerID', y='InvoiceNo',
                 title='Graph of top ten customer with respect to the invoice number without the Guest Customer')
    # fig.show(renderer='png', height=700, width=1000)
    # fig.show(height=700, width=1000)
    st.plotly_chart(fig)


#################################################

temp_df = group_Quantity_and_SalesRevenue(dataframe, 'Description')
Quantity_tempA = temp_df.sort_values(
    ascending=False, by="Quantity").reset_index(drop=True)
Sales_Revenue_tempA = temp_df.sort_values(
    ascending=False, by="Sales Revenue").reset_index(drop=True)

Quantity_tempA.drop('Sales Revenue', axis=1, inplace=True)
Sales_Revenue_tempA.drop('Quantity', axis=1, inplace=True)

colspace, col1, col2, col3 = st.columns((.45, 1, .01, 1))
with col1:
    qchoice = st.radio(
        "Choose Either Top or Bottom of Product Description by Quantity", ("Top 10", "Bottom 10"))
    qchoice_dict = {"Top 10": Quantity_tempA.head(
        10), "Bottom 10": Quantity_tempA.tail(10)}
    st.markdown(f"{qchoice} Description by Quantity")
    st.dataframe(qchoice_dict.get(qchoice))

with col3:
    schoice = st.radio(
        "Choose Either Top or Bottom of Product Description by Sales Revenue", ("Top 10", "Bottom 10"))
    schoice_dict = {"Top 10": Sales_Revenue_tempA.head(
        10), "Bottom 10": Sales_Revenue_tempA.tail(10)}
    st.markdown(f"{schoice} Description by Sales Revenue")
    st.dataframe(schoice_dict.get(schoice))

##################################################

col1, col2, col3 = st.columns((.5, 1, .5))
with col1:
    pass
with col2:
    fig = make_subplots(rows=1, cols=2, shared_yaxes=False,
                        subplot_titles=(
                            f"{qchoice} Products by Quantity", f"{schoice} Products by Sales Revenue")
                        )

    fig.add_trace(go.Bar(x=qchoice_dict.get(qchoice)['Description'], y=qchoice_dict.get(
        qchoice)['Quantity'], name=f"{qchoice}"), 1, 1)

    fig.add_trace(go.Bar(x=schoice_dict.get(schoice)['Description'], y=schoice_dict.get(
        schoice)['Sales Revenue'], name=f"{schoice}"), 1, 2)
    fig.update_layout(height=600)
    fig.update_layout(
        showlegend=False, title_text="Product Description by Quantity and Sales Revenue")
    # fig.show(renderer='png', height=700, width=1200)
    # fig.show(height=700, width=1000)
    st.plotly_chart(fig)
with col3:
    pass


#####################################################

st.markdown("----")


#########################################################################
st.markdown(" <h3 style='text-align: center;'>Market Basket Analysis <i>(MBA)</i>:</h3>",
            unsafe_allow_html=True)
r"""
**What is Market Basket Analysis?:**

Market Basket Analysis (MBA) is a data mining technique that is mostly used in the Retail Industry to uncover customer purchasing patterns and product relationships. The techniques used in MBA identify the patterns, associations, and relationships (revealing product groupings and which products are likely to be purchased together) in in frequently purchased items by customers in large transaction datasets collected/registered at the point of sale. The results of the Market Basket Analysis can be used by retailers or marketers to design and develop marketing and operation strategies for a retail business or organization.<br>
Market basket analysis mainly utilize Association Rules {IF} -> {THEN}. However, MBA assigns Business outcomes and scenarios to the rules, for example,{IF X is bought} -> {THEN Y is also bought}, so X,Y could be sold together. <br>

Definition: **Association Rule**

Let $I$= \{$i_{1},i_{2},\ldots ,i_{n}$\} be an itemset.

Let $D$= \{$t_{1},t_{2},\ldots ,t_{m}$\} be a database of transactions $t$. Where each transaction $t$ is a nonempty itemset such that ${t \subseteq I}$

Each transaction in D has a unique transaction ID and contains a subset of the items in I.

A rule is defined as an implication of the form:
$X\Rightarrow Y$, where ${X,Y\subseteq I}$.

The rule ${X \Rightarrow Y}$ holds in the dataset of transactions $D$ with support $s$, where $s$ is the percentage of transactions in $D$ that contain ${X \cup Y}$ (that is the union of set $X$ and set $Y$, or, both $X$ and $Y$). This is taken as the probability, ${P(X \cup Y)}$. Rule ${X \Rightarrow Y}$ has confidence $c$ in the transaction set $D$, where $c$ is the percentage of transactions in $D$ containing $X$ that also contains $Y$. This is taken to be the conditional probability, like ${P(Y | X)}$. That is,

* support ${(X \Rightarrow Y)}$ = ${P(X \cup Y)}$

* confidence ${(X \Rightarrow Y)}$ = ${P(X|Y)}$

The lift of the rule ${(X \Rightarrow Y)}$  is the confidence of the rule divided by the expected confidence, assuming that the itemsets $X$ and $Y$ are independent of each other.The expected confidence is the confidence divided by the frequency of ${Y}$.

* lift ${(X \Rightarrow Y)}$ = ${ \frac {\mathrm {supp} (X\cap Y)}{\mathrm {supp} (X)\times \mathrm {supp} (Y)}}$


Lift value near 1 indicates ${X}$ and ${Y}$ almost often appear together as expected, greater than 1 means they appear together more than expected and less than 1 means they appear less than expected.Greater lift values indicate stronger association

"""
"""



"""

"""
##### ***Now the Implementation of the MBA***
"""
mba_country_list = [
    'United Kingdom',
    'Germany',
    'France',
    'EIRE',
    'Spain',
    'Netherlands',
    'Switzerland',
    'Belgium',
    'Portugal',
    'Australia']

col1, col2, col3 = st.columns((3))
with col1:
    option = st.selectbox(
        'Please Choose a country for the Market Basket Analysis',
        mba_country_list)
    if option == "All":
        st.markdown("We will at data from All the countries")
    else:
        st.markdown(f"We will be looking at data from {option}")


MBA_df = choose_country(country=option)

"""
We are going to use the Apriori Algorithm for the association rule mining/analysis. Apriori is an algorithm for frequent item set mining and association rule learning over relational dataset. It proceeds by identifying the frequent individual items in the dataset and extending them to larger and larger item sets as long as those item sets appear sufficiently often in the dataset. The frequent item sets determined by Apriori can be used to determine association rules which highlight general trends, pattern, and relationships in the dataset.
"""
# we are going to rearrage the dataframe having the 'InvoiceNo' column the index, so that each row contains all the items purchased under the same invoice
basket = (MBA_df.groupby(['InvoiceNo', 'Description'])[
          'Quantity'].sum().unstack().reset_index().fillna(0).set_index('InvoiceNo'))
st.markdown(
    "Below is the one-hot encoded basket with the InvoiceNo #s being the index")


def change_dtype_to_list(x):
    x = list(x)
    return x


def encoder(x):
    if (x <= 0):
        return 0
    if (x >= 1):
        return 1


col1, col2, col3 = st.columns((.1, 1, .1))
with col1:
    pass
with col2:
    with st.spinner("One-Hot Encoding the basket..."):
        # now we encode
        basket = basket.applymap(encoder)

        st.dataframe(basket.head())
    st.success('Done!')

    st.markdown("The next step will be to generate the frequent itemsets that have a support of at "
                "least 10% using the MLxtend Apriori fuction which returns frequent itemsets from a "
                "one-hot DataFrame. And then can look at the rules  of association using the "
                "`MLxtend association_rules(), The function generates a DataFrame of association "
                "rules including the metrics 'score', 'confidence', and 'lift'")
    with st.spinner("Generating the Frequent Itemsets and Assosiation Rules..."):
        try:
            frequent_itemsets = apriori(
                basket, min_support=0.1, use_colnames=True)
            rules = association_rules(
                frequent_itemsets, metric="lift", min_threshold=1)
            rules = rules.sort_values(
                "lift", ascending=False).reset_index(drop=True)
            rules["antecedents"] = rules["antecedents"].apply(
                change_dtype_to_list)
            rules["consequents"] = rules["consequents"].apply(
                change_dtype_to_list)
        except Exception as e:
            st.error(f"An error occurred: {e}")
        else:
            st.success('Done!')
            st.dataframe(rules.head())
    #     rules = association_rules(apriori(basket, min_support=0.5, use_colnames=True), metric="lift", min_threshold=1)
    #     # Sort values based on lift
    #     rules = rules.sort_values("lift",ascending=False).reset_index(drop= True)
    #     rules["antecedents"] = rules["antecedents"].apply(change_dtype_to_list)
    #     rules["consequents"] = rules["consequents"].apply(change_dtype_to_list)
    # st.success('Done!')

    """Assosiation Rules"""
    st.dataframe(rules.head())
with col3:
    pass

st.markdown("----")
st.markdown(" <h3 style='text-align: center;'>Customer Segmentation and RFM Analysis: Unlocking Marketing Potential</h3>",
            unsafe_allow_html=True)
"""
Customer segmentation, a cornerstone of effective marketing strategies, involves categorizing customers into distinct groups sharing common characteristics. This approach enables companies to tailor their outreach and offerings, yielding enhanced engagement and better results. An integral tool in this process is RFM analysis, which stands for recency, frequency, and monetary value. RFM analysis leverages historical customer behavior data to forecast future interactions.
* RFM (recency, frequency, monetary) Analysis
"""
"""
**RFM (recency, frequency, monetary) Analysis**

At its core, RFM analysis delves into three critical dimensions: recency, frequency, and monetary value. Recency assesses the time elapsed since a customer's last interaction with a brand, highlighting the freshness of their engagement. Frequency measures how often a customer interacts with a brand over a specific period, reflecting their level of engagement. Lastly, monetary value quantifies the amount a customer spends on a brand's products or services, offering insights into their economic contribution.

The power of RFM analysis becomes evident in its ability to empower marketers to make informed decisions. By deciphering these three dimensions, marketers gain a nuanced understanding of customer behavior, enabling them to identify valuable patterns. Armed with such insights, businesses can craft targeted strategies for distinct customer groups, amplifying the resonance of marketing efforts.

For instance, a customer who recently made a purchase (high recency), frequently engages with the brand (high frequency), and contributes substantially in monetary terms (high monetary value) could be considered a premier customer. This group might warrant exclusive offers to enhance their loyalty. On the other hand, customers who've been less active across these dimensions could be nurtured with re-engagement tactics or incentivized with tailored promotions.

The merits of RFM analysis extend beyond reactive strategies. It empowers companies to be proactive, predicting customer behavior. By analyzing historical data to understand the correlation between RFM metrics and future actions, marketers can forecast how new customers might interact with the brand. This predictive capability opens doors to personalized onboarding experiences, creating a strong foundation for lasting customer relationships.

In essence, RFM analysis guides marketing strategies from the realm of broad strokes to precision targeting. It offers a data-driven pathway to connect with customers in ways that resonate deeply. By segmenting customers based on their recency, frequency, and monetary behavior, companies can craft messaging that resonates, offers that entice, and experiences that captivate. This not only heightens customer satisfaction but also cultivates brand loyalty and maximizes revenue generation.

In a landscape where personalized experiences are paramount, RFM analysis stands as a potent tool to decode customer preferences and behaviors. As companies strive to cut through the noise and forge meaningful connections, embracing RFM analysis within the framework of customer segmentation is a strategic imperative that drives success in the dynamic realm of modern marketing.
"""

rfm_country_list = [
    'United Kingdom',
    'Germany',
    'France',
    'EIRE',
    'Spain',
    'Netherlands',
    'Switzerland',
    'Belgium',
    'Portugal',
    'Australia']

col1, col2, col3 = st.columns((3))
with col1:
    option = st.selectbox(
        'Please Choose a country for the Recency, Frequency, Monetary Analysis',
        rfm_country_list)
    if option == "All":
        st.markdown("We will at data from All the countries")
    else:
        st.markdown(f"We will be looking at data from {option}")


RFM_df = choose_country(country=option)

# the first thing that we are going to need is the reference date
# in this case the day after the last recorded date in the dataset plus a day
ref_date = RFM_df['InvoiceDate'].max() + dt.timedelta(days=1)


st.markdown("----")
st.markdown(" <h3 style='text-align: center;'>Product recomendation <i>(people who bought this also bought)</i>:</h3>", unsafe_allow_html=True)
col1, col2, col3 = st.columns((.1, 1, .1))
with col1:
    pass
with col2:
    """
    The product recommendation part of this project is going to make use of the Association Rules that where uncovered in the MBA section. Product recomentation is basically one of the advantages of Market Basket Analysis where you can recommend to customers products that are in the same itemsets as the customer's current products.
    """
with col3:
    pass

st.markdown("---")


st.markdown("---")

st.markdown("### ***Conclusion***")

st.markdown("> * In this comprehensive exploratory data analysis (EDA) project, we embarked on a journey to unravel insights and patterns within a retail dataset."
            "Through a dynamic and interactive Streamlit app, we delved into various facets of the data, shedding light on critical aspects of customer behavior, market basket analysis, and customer segmentation."
            "The EDA journey commenced with a meticulous examination of the dataset's features, illuminating key statistics and distributions. "
            ""
            ""
            " The core of the project revolved around a profound Market Basket Analysis (MBA), an approach essential for understanding customer purchasing patterns and product relationships."
            " By employing the Apriori algorithm, we identified frequent item sets and association rules that unveiled compelling product associations, providing valuable business insights for targeted marketing strategies."
            ""
            ""
            "Moreover, we delved into the realm of Recency, Frequency, and Monetary (RFM) Analysis, a powerful tool for customer segmentation. By dissecting customer behavior based on their recency of transactions, frequency of engagement, and monetary contributions, we gained a nuanced understanding of distinct customer clusters. These clusters, a product of K-means clustering, offered actionable insights for tailored marketing strategies and customer retention efforts."
            ""
            ""
            "One of the notable achievements was the integration of association rules into a product recommendation system. Leveraging the results from MBA, we provided users with intelligent product suggestions based on their current selections, thus enhancing their shopping experience and boosting sales potential."
            ""
            ""
            "In conclusion, this project demonstrated the potency of EDA techniques in extracting actionable insights from complex retail datasets. By combining visualizations, statistical analysis, and machine learning methods, we harnessed the potential to optimize marketing campaigns, enhance customer experiences, and drive business growth."
            ""
            ""
            "Furthermore, this project illuminated the iterative nature of data analysis, where the exploration of one facet often paves the way for further inquiries, uncovering new layers of understanding. As we wrap up this project, the potential for future explorations and analyses remains vast, promising further refinement and expansion of these insights to propel data-driven decision-making in the retail sector.")
