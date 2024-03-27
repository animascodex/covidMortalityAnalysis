import matplotlib
import numpy as np
import csv
import sqlite3
from numpy import genfromtxt
from datetime import datetime
from sqlalchemy import Column, Integer, Float, Date, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt





df = pd.read_csv('covid_19_ECA_dataset.csv', sep=',')
df.drop(df.columns[0], axis=1, inplace=True)
aggregation_functions = {'Date':'first','Country/Region':'first','Confirmed': 'sum', 'Deaths': 'sum','Recovered': 'sum', 'Active': 'sum','WHO Region': 'first','Lat':'first','Long':'first'}
df_new = df.groupby(['Country/Region','Date']).agg(aggregation_functions)
df_new.to_csv('ECA01.csv', sep=',',index=False)

df = pd.read_csv('ECA01.csv', sep=',')
df['New Confirmed Case'] = df.groupby(['Country/Region'])['Confirmed'].diff().fillna(0)
df['New Death'] = df.groupby(['Country/Region'])['Deaths'].diff().fillna(0)
df['Mortality Rate'] = ((df['Deaths']) / (df['Confirmed']/100)).fillna(0).round(2)
df = df.reset_index()
df.to_csv('ECA01.csv', sep=',',index=False)

Base = declarative_base()
conn = sqlite3.connect('ECA_KONGJIAMING.db')
cur = conn.cursor()
class Total_Cases(Base):
    __tablename__ = "Covid_19_Cases"
    #Declare to SQLAlchemy the Names of all the column and its attributes:
    index = Column(Integer, primary_key=True)
    Date = Column(Date)
    Country_Region = Column(String)
    Confirmed = Column(Integer)
    Deaths = Column(Integer)
    Recovered = Column(Integer)
    Active = Column(Integer)
    WHO_Region = Column(String)
    Lat = Column(Float)
    Long = Column(Float)
    New_Confirmed_Case = Column(Integer)
    New_Death = Column(Integer)
    Mortality_Rate = Column(Float)
engine = create_engine('sqlite:///ECA_KONGJIAMING.db')
#Drop Table at the start to allow easier Marking
Base.metadata.drop_all(engine)
#Create the database
Base.metadata.create_all(engine)
session = sessionmaker()
session.configure(bind=engine)
s = session()
try:
    import csv
    with open('ECA01.csv') as f:
        reader = csv.reader(f)
        next(reader, None)
        read_count = 0
        for row in reader:
            read_count+=1
            record = Total_Cases(**{
                'index' : row[0],
                'Date' : datetime.strptime(row[1], '%Y-%m-%d').date(),
                'Country_Region' : row[2],
                'Confirmed' : row[3],
                'Deaths' : row[4],
                'Recovered' : row[5],
                'Active' : row[6],
                'WHO_Region' : row[7],
                'Lat' : row[8],
                'Long' : row[9],
                'New_Confirmed_Case' : row[10],
                'New_Death' : row[11],
                'Mortality_Rate' : row[12]

            })
            s.add(record) #Add all the records

        s.commit() #Attempt to commit all the records

finally:
    cur.execute('SELECT COUNT(*) FROM Covid_19_Cases')
    row_count = cur.fetchone()[0]
    print('Number of Entries entered into Database:{0}'.format(row_count))
s.close() #Close the connection

def Calculate_By_Province(Country,Type):
    total_cases = data.groupby(Country)[Type].max().reset_index(name=Type).sort_values([Type],ascending = [False],ignore_index=True).head(10)
    print(total_cases)

data = df.loc[df.reset_index().groupby('Country/Region')['Confirmed'].idxmax()]
Total_Confirmed = data['Confirmed'].sum()
Total_Deaths = data['Deaths'].sum()
Total_Active = data['Active'].sum()
Mortality_Rate = round(Total_Deaths / (Total_Confirmed/100),2)
print('Total Number of Cases across the world')
print('Confirmed: {0}'.format(Total_Confirmed))
print('Deaths: {0}'.format(Total_Deaths))
print('Active: {0}'.format(Total_Active))
print('Mortality Rate: {0}%'.format(Mortality_Rate))




Calculate_By_Province('Country/Region','Confirmed')
Calculate_By_Province('Country/Region','Deaths')
Calculate_By_Province('Country/Region','Active')

fig = go.Figure(data=go.Choropleth(
    locations = data['Country/Region'],
    locationmode = 'country names',
    z = data['Confirmed'],
    text = data['Country/Region'],
    colorscale = 'Reds',
    autocolorscale=False,
    reversescale=False,
    marker_line_color='darkgray',
    marker_line_width=0.6,
    colorbar_tickprefix = '',
    colorbar_title = 'Confirmed Covid-19 Cases',
))

fig.update_layout(
    title_text='Global Heat Map for Covid-19',
    width=1500, height=1000,
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='equirectangular'
    ),
    annotations = [dict(
        x=0.55,
        y=0.1,
        xref='paper',
        yref='paper',
        text='Global Heat Map for Covid-19',
        showarrow = False
    )]
)

fig.show()

df['Date'].value_counts().rename_axis('Date').reset_index(name='Number of Countries Affected'). \
    sort_values(['Date', 'Number of Countries Affected'], ascending=[True, False], ignore_index=True). \
    plot(kind='line', x='Date', y='Number of Countries Affected', figsize=(19, 7))

data = df[df['Country/Region'] == 'US']
fig = plt.figure()
fig, (ax1, ax2) = plt.subplots(2, 1,sharex=True,figsize=(26,15))
data.plot(x='Date',y='New Confirmed Case', ax=ax1)
ax1.set_title("United States (US)")

data.plot(x='Date',y='Confirmed', ax=ax2)
data.plot(x='Date',y='Deaths', ax=ax2)
data.plot(x='Date',y='Recovered', ax=ax2)


data = df[df['Country/Region'] == 'China']
fig = plt.figure()
fig, (ax1, ax2) = plt.subplots(2, 1,sharex=True,figsize=(26,15))
data.plot(x='Date',y='New Confirmed Case', ax=ax1)
ax1.set_title("China")

data.plot(x='Date',y='Confirmed', ax=ax2)
data.plot(x='Date',y='Deaths', ax=ax2)
data.plot(x='Date',y='Recovered', ax=ax2)
