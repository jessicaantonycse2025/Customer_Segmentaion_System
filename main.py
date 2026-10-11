
import os

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# 1. Set up project paths

folder = os.path.dirname(os.path.abspath(__file__))

output = os.path.join(
    os.environ["USERPROFILE"],
    "CustomerSegmentation_Output"
)

os.makedirs(output, exist_ok=True)

if not os.path.isdir(output):
    os.mkdir(output)

# 2. Load the customer dataset

data_file = os.path.join(folder, "customer_data.csv")

if not os.path.exists(data_file):
    raise FileNotFoundError(
        "customer_data.csv was not found. "
        "Place it in the same folder as main.py."
    )

data = pd.read_csv(data_file)

print("\nCUSTOMER SEGMENTATION USING K-MEANS")
print("-----------------------------------")
print("Total customers:", len(data))
print("\nFirst five records:")
print(data.head().to_string(index=False))


# 3. Select customer features

features = [
    "TotalPurchaseAmount",
    "NumberOfOrders",
    "AverageOrderValue",
    "WebsiteVisits",
    "SupportInteractions",
]

missing_columns = [column for column in features if column not in data.columns]

if missing_columns:
    raise ValueError(
        "Required columns are missing from the dataset: "
        + ", ".join(missing_columns)
    )

X = data[features].copy()

# Convert features to numeric values
for column in features:
    X[column] = pd.to_numeric(X[column], errors="coerce")

# Replace missing values with each column's median
print("\nMissing values before preprocessing:")
print(X.isnull().sum())

X = X.fillna(X.median())

# Stop if a feature has no usable numeric values``
if X.isnull().any().any():
    raise ValueError(
        "One or more features contain no usable numeric values."
    )

if len(X) < 3:
    raise ValueError("At least three customers are needed for three clusters.")

print("\nData preprocessing completed.")


# 4. Standardize the features

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Feature standardization completed.")


# 5. Apply the Elbow Method

inertia_values = []
cluster_range = range(1, min(7, len(data) + 1))

for k in cluster_range:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    model.fit(X_scaled)
    inertia_values.append(model.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(list(cluster_range), inertia_values, marker="o")
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.xticks(list(cluster_range))
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output, "elbow_method.png"))
plt.close()

print("\nElbow graph saved.")


# 6. Apply K-Means clustering with three clusters

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

data["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nNumber of clusters:", data["Cluster"].nunique())
print("\nCustomers in each cluster:")
print(data["Cluster"].value_counts().sort_index())


# 7. Summarize customer segments

summary = data.groupby("Cluster")[features].mean().round(2)
summary["CustomerCount"] = data.groupby("Cluster").size()

print("\nCustomer segment characteristics:")
print(summary)


# 8. Identify the segment with the highest average spending

average_spending = data.groupby("Cluster")["TotalPurchaseAmount"].mean()
best_cluster = average_spending.idxmax()

print("\nSegment with the highest average total spending: Cluster", best_cluster)
print("Average total spending:", round(average_spending.max(), 2))


# 9. Visualize the customer clusters using PCA

pca = PCA(n_components=2)
points = pca.fit_transform(X_scaled)

plt.figure(figsize=(7, 5))
plt.scatter(
    points[:, 0],
    points[:, 1],
    c=data["Cluster"],
    cmap="viridis",
    s=60
)
plt.title("Customer Segments Using K-Means")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.tight_layout()
plt.savefig(os.path.join(output, "customer_segments.png"))
plt.close()

print("\nCustomer segment graph saved.")


# 10. Suggest marketing strategies

print("\nMARKETING STRATEGIES")

overall_orders = data["NumberOfOrders"].mean()

for cluster in sorted(data["Cluster"].unique()):
    segment = data[data["Cluster"] == cluster]
    average_orders = segment["NumberOfOrders"].mean()

    print("\nCluster", cluster)

    if cluster == best_cluster:
        print("Strategy: Offer loyalty rewards and exclusive benefits.")
    elif average_orders < overall_orders:
        print("Strategy: Offer discounts to encourage repeat purchases.")
    else:
        print("Strategy: Recommend related products and bundle offers.")


# 11. Save the results

data.to_csv(
    os.path.join(output, "segmented_customers.csv"),
    index=False
)

summary.to_csv(
    os.path.join(output, "cluster_summary.csv")
)

print("\nOUTPUT FILES")
print("1. outputs/elbow_method.png")
print("2. outputs/customer_segments.png")
print("3. outputs/segmented_customers.csv")
print("4. outputs/cluster_summary.csv")

print("\nProject completed successfully!")