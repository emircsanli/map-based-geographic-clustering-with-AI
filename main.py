import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
from matplotlib import pyplot as plt

df = pd.read_csv('train.csv')
df = df.drop(["PdDistrict", "Address", "Resolution", "Descript", "DayOfWeek"], axis=1)

# Tarih sütununu işleme
df["Dates"] = df["Dates"].str.split().str[0]
df["Dates"] = df["Dates"].str.split("-").str[0]
df_2014 = df[df["Dates"] == "2014"].copy()

# X ve Y değerlerini ölçekleme
scaler = MinMaxScaler()
df_2014.loc[:, "X_scaled"] = scaler.fit_transform(df_2014[["X"]])
df_2014.loc[:, "Y_scaled"] = scaler.fit_transform(df_2014[["Y"]])

# K-means kümeleme
k_range = range(1, 15)
list_dist = []

for k in k_range:
    model = KMeans(n_clusters=k)
    model.fit(df_2014[["X_scaled", "Y_scaled"]])
    list_dist.append(model.inertia_)

# Inertia grafiği
plt.xlabel("K")
plt.ylabel("Inertia")
plt.plot(k_range, list_dist)
plt.show()

# K-means kümeleme (5 kümeli)
model = KMeans(n_clusters=5)
df_2014.loc[:, "cluster"] = model.fit_predict(df_2014[["X_scaled", "Y_scaled"]])

# Scatter mapbox grafiği
figure = px.scatter_mapbox(df_2014, lat="Y", lon="X",
                           center=dict(lat=37.8, lon=-122.4),
                           zoom=9,
                           opacity=0.9,
                           mapbox_style='open-street-map',
                           color='cluster',
                           title='San Francisco Crime Districts',
                           width=1100,
                           height=700,
                           hover_data=['cluster', 'Category', 'Y', 'X'])
figure.show()
