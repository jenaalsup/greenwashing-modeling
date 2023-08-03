import pandas as pd
import numpy as np
import pickle

file = open("Arraytechinc.pkl",'rb') # 1329
file = open("Cheniere.pkl",'rb') # 726
file = open("Chevron.pkl",'rb') # 3244
file = open("ConEdison.pkl",'rb') # 3250
file = open("conocophillips.pkl",'rb') # 2422
file = open("DevonEnergy.pkl",'rb') # 1434
file = open("Enphase.pkl",'rb') # 3230
file = open("exxonmobil.pkl",'rb') # 3244
file = open("FirstSolar.pkl",'rb') # 1524
file = open("HessCorporation.pkl",'rb') # 1647
file = open("Kinder_Morgan.pkl",'rb') # 1490
file = open("MarathonPetroCo.pkl",'rb') # 2566
file = open("OrmatInc.pkl",'rb') # 185
file = open("Phillips66Co.pkl",'rb') # 1291
file = open("PlugPowerInc.pkl",'rb') # 1872
file = open("PXDtweets.pkl",'rb') # 357
file = open("ShoalsTech.pkl",'rb') # 260
file = open("SolarEdgePV.pkl",'rb') # 3193
file = open("Sunrun.pkl",'rb') # 3215
file = open("ValeroEnergy.pkl",'rb') # 1098
file = open("WeAreOxy.pkl",'rb') # 724
file = open("WilliamsUpdates.pkl",'rb') # 2792

df = pickle.load(file)
for i in range(len(df)):
    print(df[i]["text"])
print(len(df))

# convert from json to pandas

# notes:
# total # of tweets: 41,000
# energy companies not represented: schlumberger, eog resources
# clean energy companies not represented: avangrid, sunnova, clearway,
#                                         green plains, sunpower, fuelcell energy
# 13 energy companies, 9 clean energy companies
# date, author_id, language, public metrics, text id, company name, document type = tweet, 
# doc type sus_report_chapter