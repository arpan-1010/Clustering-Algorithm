import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

df = pd.read_csv('Income.csv')
print(df.head(5))

sn.lmplot(data=df, x='Age', y='Income', fit_reg=False);
plt.show()

from sklearn.cluster import KMeans
clusters = KMeans(3)
clusters.fit(df)

df['clusterid'] = clusters.labels_
print(df.head(10))

markers = ['+', '^', '.']
sn.lmplot(data=df, x='Age', y='Income', hue='clusterid',fit_reg=False, markers=markers);
plt.show()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df[['Age', 'Income']])

from sklearn.cluster import KMeans
clusters_new = KMeans(3, random_state=42)
clusters_new.fit(scaled_df)
df['clusters_new'] = clusters_new.labels_
markers = ['+', '^', '.']
sn.lmplot(data=df, x='Age', y='Income', hue='clusters_new',fit_reg=False, markers=markers);
plt.show()

print(df.groupby('clusterid')[['Age', 'Income']].agg(['mean', 'std']).reset_index())