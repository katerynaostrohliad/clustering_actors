# 🎬 Actor Clustering and Visualization with t-SNE

This project connects to a PostgreSQL database containing movie and credits data, processes the data to extract actor-level features, clusters them using K-Means, and visualizes the results using t-SNE in both 2D and 3D.

## 📌 Features

- Connects to a PostgreSQL movie database using credentials stored in `.env`.
- Aggregates movie statistics for each actor (e.g., average budget, popularity, revenue, votes).
- Uses **K-Means Clustering** to group similar actors based on movie features.
- Visualizes the actor clusters using **t-SNE** for dimensionality reduction.
- Includes an **Elbow Method** plot to help choose the optimal number of clusters.
