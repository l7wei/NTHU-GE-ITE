# 載入套件
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import geopy

# 讀取資料
df = pd.read_csv("data/taipei_traffic_accidents.csv")

# 讀取人口資料
# https://ca.gov.taipei/News_Content.aspx?n=8693DC9620A1AABF&sms=D19E9582624D83CB&s=EE7D5719108F4026
population = pd.read_csv("data/taipei_population.csv")
# 移除總計資料
population = population[population["district"] != "total"]
# 輸出人口資料圖表，並降序排列
population = population.sort_values(by="total", ascending=False)
population.plot.bar(x="district", y="total", rot=0)
plt.title("Population in Taipei")
plt.show()

# 選取經緯度欄位
X = df[["Longitude", "Latitude"]]
X.columns = ["Longitude", "Latitude"]

# 使用 KMeans 進行集群分析
kmeans = KMeans(n_clusters=8, n_init=10, random_state=0)
kmeans.fit(X)

# 將集群標籤添加到原始 DataFrame
df["Cluster"] = kmeans.labels_

# 找出每個集群中事故數量最多的區域
hot_zones = df.groupby("Cluster").size().sort_values(ascending=False)
# 取得集群中心
centers = kmeans.cluster_centers_
# 中心按照事故數量排序
centers = centers[hot_zones.index]

# 根據經緯度查詢地址
geolocator = geopy.Nominatim(user_agent="cluster_traffic_accidents")

for i in range(len(centers)):
    # 反轉經緯度查詢地址
    centers[i][0], centers[i][1] = centers[i][1], centers[i][0]
    address = geolocator.reverse(centers[i], language="zh-TW", exactly_one=True)
    print("Cluster", i)
    print("Center:", centers[i])
    print("Address:", address.address)
    print("Total accidents:", hot_zones[i])
    print()

# 使用 matplotlib 繪製散點圖
plt.scatter(df["Longitude"], df["Latitude"], c=df["Cluster"], cmap="viridis", s=10)
plt.title("Scatter plot of Clusters")
plt.show()

# 繪製集群中心
plt.scatter(centers[:, 0], centers[:, 1], c="black", s=200, alpha=0.5)
plt.title("Scatter plot of Clusters with Centroids")
# 圖會呈現隨機分布
plt.show()

# 繪製集群邊界
x_min, x_max = df["Longitude"].min() - 1, df["Longitude"].max() + 1
y_min, y_max = df["Latitude"].min() - 1, df["Latitude"].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(df["Longitude"], df["Latitude"], c=df["Cluster"])
plt.title("Contour plot of Clusters with Centroids")
plt.show()
