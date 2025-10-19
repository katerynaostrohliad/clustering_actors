import psycopg2
import polars as pl
import json
from psycopg2.extras import execute_values
import os
from dotenv import load_dotenv


def get_credits_data(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM credits;")
    credits = cursor.fetchall()
    return credits

def safe_json_load(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def transform_credits_data(credits):
    data = []
    for row in credits:
        json_data_1 = safe_json_load(row[2])  # Assuming first column contains JSON data
        json_data_2 = safe_json_load(row[3])
        row_dict = {
            'movie_id': row[0],
            'title': row[1],
            'cast': json_data_1,
            'crew': json_data_2
        }
        data.append(row_dict)

    # Convert the list of dicts into a Polars DataFrame
    df = pl.DataFrame(data)
    cast = df[['movie_id', 'title', 'cast']]
    cast = cast.explode(['cast'])
    cast = cast.unnest('cast')
    # cast = [tuple(row) for row in cast.rows()]
    return cast


def save_cast_data_to_db(conn, data_iter, batch_size=10000):
    data_iter = [tuple(row) for row in data_iter.rows()]
    cursor = conn.cursor()
    query = (f"INSERT INTO cast_kate (movie_id, title, id, name, order_number, gender, cast_id, character, credit_id) VALUES %s"
             f"ON CONFLICT (credit_id) "
             f"DO UPDATE SET movie_id=EXCLUDED.movie_id,"
             f"title=EXCLUDED.title,"
             f"id=EXCLUDED.id,"
             f"name=EXCLUDED.name,"
             f"order_number=EXCLUDED.order_number,"
             f"gender=EXCLUDED.gender,"
             f"cast_id=EXCLUDED.cast_id,"
             f"character=EXCLUDED.character,"
             f"credit_id=EXCLUDED.credit_id")

    batch = []
    for i, row in enumerate(data_iter, 1):
        batch.append(row)
        if i % batch_size == 0:
            execute_values(cursor, query, batch)
            batch.clear()
    if batch:
        execute_values(cursor, query, batch)
        print('ok')

    conn.commit()

def main():
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
        save_cast_data_to_db(conn, cast)
        print(cast)
    except psycopg2.Error as e:
        print(f"Error connecting to PostgreSQL: {e}")

    finally:
        if conn:
            conn.close()
            print("PostgreSQL connection closed.")