import pandas as pd
import numpy as np
import cluster_tools as ct
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt
import errors as err
import warnings
warnings.filterwarnings("ignore")


# Function to read WorldBank data csv files
def read_csv(filename):
    """
    Read CSV file and extract dataframe as per country and year with
    indexes configures as required.

    Parameters:
    filename (String): Name of the CSV file

    Returns:
    Two Dataframes containing Year wise data and Country wise data
    respectively.
    """
    df = pd.read_csv(filename, skiprows = 4)  # Reading file using pandas
    cleaned_data = df.dropna(axis=1, how='all')
    df_ext = cleaned_data[["1999", "2019"]]
    df_ext = df_ext.dropna()
    df_ext = df_ext.reset_index()
    df_ext = df_ext.drop("index", axis=1)
    return df_ext, cleaned_data


# Reading the file using pandas
worldbankdata, cleaned_data = read_csv("WorldBankDataCO2.csv")

# NORMALISATION - Normalising the data and store minimum and maximum
df_norm, df_min, df_max = ct.scaler(worldbankdata)

# loop over number of clusters to arrive at the optimum number of clusters
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
    print(ncluster, skmet.silhouette_score(worldbankdata, labels))


ncluster = 5  # optimal number of clusters as per the score
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

# plotting the clusters[]
plt.figure(figsize=(9.0, 6.0))
plt.scatter(df_norm["1999"], df_norm["2019"], 35, labels,
            marker="o", cmap='winter')
plt.scatter(xcen, ycen, 45, "r", marker = "^", alpha = 1)
plt.xlabel("CO2(1999)")
plt.ylabel("CO2(2019)")
plt.title("Normalized Data - 4 clusters")
plt.show()

# CLUSTERING GRAPH - Using sklearn to form clusters for the original data
kmeans = cluster.KMeans(n_clusters=ncluster)
kmeans.fit(df_norm)
# extract labels and cluster centres
labels_orig = kmeans.labels_
cen_orig = kmeans.cluster_centers_

# scatter plot with colours selected using the cluster numbers
# now using the original dataframe
plt.figure(figsize=(9.0, 6.0))
plt.scatter(worldbankdata["1999"],
            worldbankdata["2019"], c=labels_orig, cmap="tab10")
# colour map Accent selected to increase contrast between colours
# rescale and show cluster centres
scen = ct.backscale(cen_orig, df_min, df_max)
xc = scen[:, 0]
yc = scen[:, 1]
plt.scatter(xc, yc, c="r", marker="^", s=70)
plt.xlabel("CO2(1999)")
plt.ylabel("CO2(2019)")
plt.title("Original Data - 4 clusters")
plt.show()

df_new = cleaned_data.drop(
    ['Country Code', 'Indicator Name', 'Indicator Code'], axis=1)
df_co2 = df_new.T  # Transposing the dataframe to plot timeseries data
df_co2.reset_index(inplace = True)
df_co2.columns = df_co2.iloc[0]
df_co2.drop(index=0, inplace = True)
df_co2.iloc[:, 0] = pd.to_numeric(df_co2.iloc[:, 0])
df_co2 = df_co2[10:]
df_co2.set_index(df_co2.columns[0], inplace = True)


def poly(t, c0, c1, c2, c3):
    """
    Polynomial function to fit the given data and use those parameters to
    predict the future readings

    Parameters
    ----------
    t : TYPE
        DESCRIPTION.
    c0 : TYPE
        DESCRIPTION.
    c1 : TYPE
        DESCRIPTION.
    c2 : TYPE
        DESCRIPTION.
    c3 : TYPE
        DESCRIPTION.

    Returns
    -------
    f : TYPE
        DESCRIPTION.

    """
    """ Computes a polynominal c0 + c1*t + c2*t^2 + c3*t^3"""
    t = t - 2000
    f = c0 + c1 * t + c2 * t**2 + c3 * t**3
    return f


# DATA FITTING - Fitting the data and plotting for forecasting for future years
param, pcorr = opt.curve_fit(
    poly, df_co2.index, df_co2["United States"])
years = np.arange(2000, 2025)
sigmas = np.sqrt(np.diag(pcorr))
lower, upper = err.err_ranges(years, poly, param, sigmas)
forecast = poly(years, *param)
plt.figure(figsize=(9.0, 6.0))
plt.plot(df_co2.index, df_co2["United States"], label = 'Data')
plt.plot(years, forecast, label="Forecast")
plt.xlabel("Year")
ticks_to_use = years[::4]
plt.xticks(ticks_to_use)
plt.legend()
plt.title("C02 EMISSIONS(in kt) in USA")
plt.show()
print(" ")
print("C02 EMISSIONS(in kt) in USA by the year")
print("2024:", poly(2024, *param) / 1.0e6, "Mill.")
