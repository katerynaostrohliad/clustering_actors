import psycopg2
import polars as pl
import os
from dotenv import load_dotenv
from credits_to_db import get_credits_data, transform_credits_data
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def get_movies_data(conn):
    cursor = conn.cursor()
    cursor.execute("select id, budget, popularity, "
                   "revenue, vote_average, vote_count from movies")
    movies = cursor.fetchall()
    return movies

def transform_movies_data(movies):
    data = []
    for row in movies:
        row_dict = {
            'movie_id': row[0],
            'budget': row[1],
            'popularity': row[2],
            'revenue': row[3],
            'vote_average': row[4],
            'vote_count': row[5]
        }
        data.append(row_dict)

    df = pl.DataFrame(data)
    movies = df[['movie_id', 'budget', 'popularity', 'revenue', 'vote_average', 'vote_count']]
    return movies

def merge_movies_cast(movies, cast):
    cast = cast[['movie_id', 'title', 'name']]
    merged_df = cast.join(movies, how='left', on='movie_id')
    agg_df = merged_df.group_by(['name']).agg(pl.col('movie_id').n_unique().alias('number_of_movies'),
                                              pl.col('budget').mean().alias('avg_budget'),
                                              pl.col('popularity').mean().alias('avg_popularity'),
                                              pl.col('revenue').mean().alias('avg_revenue'),
                                              pl.col('vote_average').mean().alias('avg_vote'),
                                              pl.col('vote_count').mean().alias('avg_vote_count'))
    return agg_df

def get_number_of_clusters(data):
    X = data.select(['number_of_movies', 'avg_budget', 'avg_popularity',
                     'avg_revenue', 'avg_vote', 'avg_vote_count']).to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    inertia = []
    for k in range(1, 10):
        model = KMeans(n_clusters=k, random_state=42, n_init='auto')
        model.fit(X_scaled)
        inertia.append(model.inertia_)

    plt.plot(range(1, 10), inertia, 'o-')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for Optimal k")
    plt.show()


def k_means_implementation(data):
    X = data.select(['number_of_movies', 'avg_budget', 'avg_popularity',
              'avg_revenue', 'avg_vote', 'avg_vote_count']).to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(X_scaled)
    data = data.with_columns(pl.Series("cluster", clusters))
    return data


def t_sne(data):
    data = data.sample(5000)
    X = data.select(['number_of_movies', 'avg_budget', 'avg_popularity',
                     'avg_revenue', 'avg_vote', 'avg_vote_count']).to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    tsne = TSNE(n_components=3, perplexity=20, learning_rate='auto', init='pca', random_state=42)
    X_embedded = tsne.fit_transform(X_scaled)

    df_embedded = data.with_columns([
        pl.Series("tsne_1", X_embedded[:, 0]),
        pl.Series("tsne_2", X_embedded[:, 1]),
        pl.Series("cluster", data[:, 6])
    ])

    plt.figure(figsize=(8, 6))
    plt.scatter(
        df_embedded["tsne_1"],
        df_embedded["tsne_2"],
        c=df_embedded["cluster"],
        cmap="viridis",
        s=120
    )
    plt.title("t-SNE Visualization of Actor Clusters")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.show()

def main_clusters():
    load_dotenv(os.path.join(os.path.dirname(os.path.realpath(__file__)), '.env'))
    conn = None
    try:
        conn = psycopg2.connect(
            host=os.getenv('db_host'),
            database=os.getenv('db_name'),
            user=os.getenv('db_user'),
            password=os.getenv('db_password'),
            port=os.getenv('db_port')
        )
        print("Connection to PostgreSQL successful!")
        credits = get_credits_data(conn)
        cast = transform_credits_data(credits)
        movies = get_movies_data(conn)
        movies = transform_movies_data(movies)
        merged = merge_movies_cast(movies, cast)
        get_number_of_clusters(merged)
        clusters = k_means_implementation(merged)
        t_sne(clusters)
    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL: {e}")

    finally:
        if conn:
            conn.close()
            print("PostgreSQL connection closed.")

