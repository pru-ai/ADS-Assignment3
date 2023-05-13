import pandas as pd
import numpy as np
import cluster_tools as ct
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt
import errors as err
import warnings

warnings.filterwarnings("ignore")

worldbankdata = pd.read_csv("WorldBankCO2Data.csv")

# Dropping the columns with all nan's
cleaned_data = worldbankdata.dropna(axis=1, how='all')

# picking years to extract data
df_ext = cleaned_data[["1999", "2019"]]
df_ext = df_ext.dropna()
df_ext = df_ext.reset_index()
df_ext = df_ext.drop("index", axis=1)

# normalise, store minimum and maximum
df_norm, df_min, df_max = ct.scaler(df_ext)

# loop over number of clusters
print("n score")
for ncluster in range(2, 10):
    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster)
# Fit the data, results are stored in the kmeans object
    kmeans.fit(df_norm)  # fit done on x,y pairs
    labels = kmeans.labels_
# extract the estimated cluster centres
    cen = kmeans.cluster_centers_
# calculate the silhoutte score
    print(ncluster, skmet.silhouette_score(df_ext, labels))


ncluster = 4  # optimal number of clusters
# set up the clusterer with the number of expected clusters
kmeans = cluster.KMeans(n_clusters=ncluster)
# Fit the data, results are stored in the kmeans object
kmeans.fit(df_norm)  # fit done on x,y pairs
labels = kmeans.labels_
# extract the estimated cluster centres
cen = kmeans.cluster_centers_
cen = np.array(cen)
xcen = cen[:, 0]
ycen = cen[:, 1]
# cluster by cluster
plt.figure(figsize=(8.0, 8.0))
cm = plt.cm.get_cmap('tab10')
plt.scatter(df_norm["1999"], df_norm["2019"], 35, labels,
            marker="o", cmap='winter')
plt.scatter(xcen, ycen, 45, "r", marker = "^", alpha = 1)
plt.xlabel("CO2(1999)")
plt.ylabel("CO2(2019)")
plt.title("Normalized Data - 4 clusters")
plt.show()


kmeans = cluster.KMeans(n_clusters=ncluster)
kmeans.fit(df_norm)
# extract labels and cluster centres
labels_orig = kmeans.labels_
cen_orig = kmeans.cluster_centers_


plt.figure(figsize=(8.0, 8.0))
# scatter plot with colours selected using the cluster numbers
# now using the original dataframe
plt.scatter(df_ext["1999"], df_ext["2019"], c=labels_orig, cmap="tab10")
# colour map Accent selected to increase contrast between colours
# rescale and show cluster centres
scen = ct.backscale(cen_orig, df_min, df_max)
xc = scen[:, 0]
yc = scen[:, 1]
plt.scatter(xc, yc, c="k", marker="d", s=80)
plt.xlabel("CO2(1999)")
plt.ylabel("CO2(2019)")
plt.title("Original Data - 4 clusters")
plt.show()

df_new = cleaned_data.drop(
    ['Country Code', 'Indicator Name', 'Indicator Code'], axis=1)
df_co2 = df_new.T
df_co2.reset_index(inplace = True)
df_co2.columns = df_co2.iloc[0]
df_co2.drop(index=0, inplace = True)
df_co2.iloc[:, 0] = pd.to_numeric(df_co2.iloc[:, 0])
df_co2 = df_co2[10:]
df_co2.set_index(df_co2.columns[0], inplace = True)


def poly(t, c0, c1, c2, c3):
    """ Computes a polynominal c0 + c1*t + c2*t^2 + c3*t^3"""
    t = t - 2000
    f = c0 + c1 * t + c2 * t**2 + c3 * t**3
    return f


param, pcorr = opt.curve_fit(
    poly, df_co2.index, df_co2["United States"])
years = np.arange(2000, 2050)
sigmas = np.sqrt(np.diag(pcorr))
lower, upper = err.err_ranges(years, poly, param, sigmas)
forecast = poly(years, *param)
plt.figure()
plt.plot(df_co2.index, df_co2["United States"], label = 'Data')
plt.plot(years, forecast, label="Forecast")
plt.xlabel("Year")
#plt.fill_between(years, upper, upper, alpha=0.9)
ticks_to_use = years[::5]
plt.xticks(ticks_to_use)
plt.legend()
plt.title("CO2 Emissions in kg")
plt.show()
print("C02 emissions (in kg) in")
print("2030:", poly(2025, *param) / 1.0e6, "Mill.")
