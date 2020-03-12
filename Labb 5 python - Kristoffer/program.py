import numpy as np
from numpy import percentile
import pandas as pd
import csv, sys
import matplotlib.pyplot as plt
import seaborn as sns

#DF för män
man_df = pd.read_csv('df_table_man.csv', sep=';', names=['år', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019'])
man_df.set_index('år')

år_man = man_df
ålder_man = man_df[['år']]
histo_man = man_df['2007']

#Summan av dataframen
totala = man_df[['2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019']]
p = totala.sum()
print(f'Totala mängden dödsfall för varje år:\n {p}')


#GRAF
år_man.plot(title='Ålder på dödsfall', kind='line')
plt.ylabel('Dödsfall')
plt.xlabel('Ålder')
plt.show()


#SCATTER
fig, scat = plt.subplots(figsize=(6, 4))
scat.scatter(man_df['år'], man_df['2000'], man_df['2009'], man_df['2019'])
scat.set_xlabel('Ålder')
scat.set_ylabel('dödsfall')
plt.show()

#HeatMaps
correlation_matrix = man_df.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()

#Predict
man_df_pre = pd.read_csv('df_table_man_predict.csv', sep=';')
#Df för folkmängd
folk_df = pd.read_csv('df_folk.csv', sep=';')

x_year_folk = np.array(folk_df['år'])
y_age_folk = np.array(folk_df['33'])

x_year = np.array(man_df_pre['år']) #+ np.array(folk_df['år'])
y_age = np.array(man_df_pre['33']) #+ np.array(folk_df['33'])

z_years = np.polyfit(x_year, y_age, 2)
z_folk = np.polyfit(x_year_folk, y_age_folk, 2)
polyfunc = np.poly1d(z_years)
polyfunc2 = np.poly1d(z_folk)
x = np.arange(2000, 2025)
c = np.arange(2000, 2025)
y = polyfunc(x)
z = polyfunc2(c)
plt.plot(x, y, c, z)
plt.title('Prediction')
plt.ylabel('Antal')
plt.xlabel('År')
plt.show()

#RÄKNA MEDIAN
median = man_df['2000']
procent = percentile(median, [25, 50, 75])
print(f"medianen för åren är : ",(procent))
low_count = percentile(median, 1)
high_count = percentile(median, 100)
print(f"Lägsta dödsantalet", low_count, "st")
print(f"Högsta dödsantalet", high_count, "st")
