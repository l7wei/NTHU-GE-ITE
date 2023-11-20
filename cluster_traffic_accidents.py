# 載入套件
import geopy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

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
kmeans = KMeans(n_clusters=16, n_init=10, random_state=0)
kmeans.fit(X)

# 將集群標籤添加到原始 DataFrame
df["Cluster"] = kmeans.labels_

# 找出每個集群中事故數量最多的區域
hot_zones = df.groupby("Cluster").size().sort_values(ascending=False)
# 取得集群中心
centers = kmeans.cluster_centers_
# 中心按照事故數量排序
centers = centers[hot_zones.index]

# 將中心點和事故總數轉換為 numpy array
centers_array = np.array(centers)
hot_zones_array = np.array(hot_zones)

# 使用 matplotlib 繪製 scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(centers_array[:, 0], centers_array[:, 1], c=hot_zones_array, cmap="viridis")

# 根據經緯度查詢地址
geolocator = geopy.Nominatim(user_agent="cluster_traffic_accidents")

# 為每個中心點添加地址標籤
for i in range(len(centers)):
    # 經緯度順序要調換
    centers[i][0], centers[i][1] = centers[i][1], centers[i][0]
    # 反轉經緯度查詢地址，並獲取詳細的地址資訊
    address = geolocator.reverse(
        centers[i], language="en-US", exactly_one=True, addressdetails=True
    )
    # 從詳細的地址中選擇需要的部分
    address_str = ", ".join([address.raw["address"].get(key, "") for key in ["suburb"]])
    # 移除 "District" 字樣
    address_str = address_str.replace(" District", "")
    # 在圖表上添加地址標籤
    plt.annotate(address_str, (centers_array[i, 0], centers_array[i, 1]))

plt.colorbar(label="Total accidents")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Accident hot zones")
plt.show()

# 使用 matplotlib 繪製散點圖
plt.scatter(df["Longitude"], df["Latitude"], c=df["Cluster"], cmap="viridis", s=10)
plt.title("Scatter plot of Clusters")
plt.show()

# 繪製集群中心
plt.scatter(centers[:, 0], centers[:, 1], c="black", s=200, alpha=0.5)
plt.title("Scatter plot of Clusters with Centroids")
# 圖會呈現隨機分布
plt.show()
