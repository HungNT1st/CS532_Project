# Usage examples:
#   python etl.py \
#     --movies_csv data_raw/movies_metadata.csv \
#     --credits_csv data_raw/credits.csv \
#     --ratings_csv data_raw/ratings.csv \
#     --out_dir data_parquet \
#     --repartition 8 --shuffle_partitions 64
#
#   # Subset timing (20k rows from movies/credits/ratings):
#   python etl.py \
#     --movies_csv data_raw/movies_metadata.csv \
#     --credits_csv data_raw/credits.csv \
#     --ratings_csv data_raw/ratings.csv \
#     --out_dir data_parquet_20k \
#     --limit 20000

import argparse, time
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T


# Create a SparkSession
def build_spark(app_name: str, shuffle_partitions: int = 64):
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.ansi.enabled", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .getOrCreate()
    )

# Context manager that prints how long a code block took.
def timer(label: str):
    class _T:
        def __enter__(self):
            self.t0 = time.perf_counter()
            return self
        def __exit__(self, exc_type, exc, tb):
            dt = time.perf_counter() - self.t0
            print(f"[TIMER] {label}: {dt:.3f}s")
    return _T()


# --- Readers ---

# Read and clean the movies csv
def read_movies(spark, path: str, limit: int | None):
    df = spark.read.option("header", True).option("multiLine", True).csv(path)
    if limit: df = df.limit(limit)

    # Select needed columns 
    df = df.select(
        F.col("id").alias("movieId"),
        F.col("title"),
        F.col("release_date"),
        F.col("genres"),
        F.col("vote_average"),
        F.col("vote_count"),
    )

    # Parse year 
    df = df.withColumn("year", F.regexp_extract(F.col("release_date"), r'^(\d{4})', 1).cast("int"))

    # Parse genres JSON string
    genre_schema = T.ArrayType(T.StructType([
        T.StructField("id", T.LongType()),
        T.StructField("name", T.StringType()),
    ]))
    df = df.withColumn("genres_arr", F.from_json(F.col("genres"), genre_schema))
    df = df.withColumn(
        "genres_names",
        F.transform(F.col("genres_arr"), lambda x: F.initcap(x["name"]))
    )

    # Cast basic numerics to columns
    df = df.withColumn("movieId", F.col("movieId").cast("long"))
    df = df.withColumn("vote_average", F.col("vote_average").cast("double"))
    df = df.withColumn("vote_count", F.col("vote_count").cast("long"))

    movies = df.select(
        "movieId", "title", "year", F.col("genres_names").alias("genres"),
        "vote_average", "vote_count"
    )
    return movies

# Read Credits CSV and produce People and Roles tables.
def read_credits(spark, path: str, limit: int | None):
    df = spark.read.option("header", True).option("multiLine", True).csv(path)
    if limit: df = df.limit(limit)

    # Schema 
    cast_schema = T.ArrayType(T.StructType([
        T.StructField("cast_id", T.LongType()),
        T.StructField("character", T.StringType()),
        T.StructField("credit_id", T.StringType()),
        T.StructField("gender", T.IntegerType()),
        T.StructField("id", T.LongType()),          # personId
        T.StructField("name", T.StringType()),
        T.StructField("order", T.IntegerType()),
        T.StructField("profile_path", T.StringType()),
    ]))
    crew_schema = T.ArrayType(T.StructType([
        T.StructField("credit_id", T.StringType()),
        T.StructField("department", T.StringType()),
        T.StructField("gender", T.IntegerType()),
        T.StructField("id", T.LongType()),          # personId
        T.StructField("job", T.StringType()),
        T.StructField("name", T.StringType()),
        T.StructField("profile_path", T.StringType()),
    ]))

    # Select needed columns
    df = df.select(F.col("id").alias("movieId"), F.col("cast"), F.col("crew")) \
       .withColumn("cast_arr", F.from_json(F.col("cast"), cast_schema)) \
       .withColumn("crew_arr", F.from_json(F.col("crew"), crew_schema))

    # Apply schema for role and crew, clean null fields
    # CAST → actors only, drop null person/movie/name rows
    roles_cast = (
        df.select("movieId", F.explode(F.col("cast_arr")).alias("c"))                 
        .where(F.col("c.id").isNotNull() & F.col("c.name").isNotNull())
        .select(
            F.col("movieId").cast("long").alias("movieId"),
            F.col("c.id").cast("long").alias("personId"),
            F.trim(F.col("c.name")).alias("name"),
            F.lit("actor").alias("role"),
            F.col("c.character").alias("character"),
            F.lit(None).cast("string").alias("job"),
            F.lit(None).cast("string").alias("department"),
        )
        .where(F.col("movieId").isNotNull())
    )

    # CREW → drop null person/movie/name rows
    roles_crew = (
        df.select("movieId", F.explode(F.col("crew_arr")).alias("k"))                
        .where(F.col("k.id").isNotNull() & F.col("k.name").isNotNull())
        .select(
            F.col("movieId").cast("long").alias("movieId"),
            F.col("k.id").cast("long").alias("personId"),
            F.trim(F.col("k.name")).alias("name"),
            F.lit("crew").alias("role"),
            F.lit(None).cast("string").alias("character"),
            F.col("k.job").alias("job"),
            F.col("k.department").alias("department"),
        )
        .where(F.col("movieId").isNotNull())
    )

    # Merge role and crew to the same roles table. 
    roles = roles_cast.unionByName(roles_crew)

    # Avoid duplicate records
    roles = (
        roles
        .withColumn("name", F.initcap(F.col("name")))
        .dropDuplicates(["movieId", "personId", "role", "character"])  
    )

    return roles

# Read ratings csv and return ratings table. 
def read_ratings(spark, path: str, limit: int | None):
    df = spark.read.option("header", True).csv(path)
    if limit: df = df.limit(limit)
    ratings = (df.select(
        F.col("userId").cast("long").alias("userId"),
        F.col("movieId").cast("long").alias("movieId"),
        F.col("rating").cast("double").alias("rating"),
        F.from_unixtime(F.col("timestamp").cast("long")).cast("timestamp").alias("ts")
    ))
    return ratings


# --- Index builders ---

# Explode genres array to one row per (genre, movieId).
# Similar to HW3, we do this for later usage when finding genre
def build_genre_index(movies_df):
    return (
        movies_df
        .withColumn("genre", F.explode_outer(F.col("genres")))
        .select("genre", "movieId")
    )

# Build actor index for quick search by actor name
def build_actor_index(roles_df):
    actor = (roles_df
            .select(
                F.initcap(F.col("name")).alias("name"),
                F.lower(F.trim(F.col("name"))).alias("name_norm"),
                "movieId",
                "character",
            )
            .dropDuplicates(["movieId", "name_norm"])         
         )
    return actor

# Build crew index for quick search by crew name
def build_crew_index(roles_df):
    crew = (roles_df
            .filter(F.col("role") == F.lit("crew"))      
            .select(
                F.initcap(F.col("name")).alias("name"),
                F.lower(F.trim(F.col("name"))).alias("name_norm"),
                "movieId",
                F.col("job"),
                F.col("department"),
            )
            .dropDuplicates(["movieId", "name_norm"])     
         )
    return crew

# --- Main ---

# Write the parquet
def write_parquet(df, path: str, partition_by=None, mode="overwrite"):
    w = df.write.mode(mode)
    if partition_by:
        w = w.partitionBy(*partition_by)
    w.parquet(path)

# Print small preview (n rows + row counts) for saved Parquet tables
def preview_outputs(spark, base: str, n: int = 5):
    for name in ["Movies", "Roles", "GenreIndex", "ActorIndex", "CrewIndex"]:
        path = f"{base}/{name}"
        try:
            df = spark.read.parquet(path)
        except Exception:
            continue
        print(f"\n--- Preview: {name} ---")
        df.show(n, truncate=False)
        print(f"Row count: {df.count():,}")

# Main function
def main():
    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--movies_csv", required=True)
    ap.add_argument("--credits_csv", required=True)
    ap.add_argument("--ratings_csv", required=False)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=None, help="Optional row cap for timing experiments (e.g., 20000)")
    ap.add_argument("--repartition", type=int, default=8)
    ap.add_argument("--shuffle_partitions", type=int, default=32)
    args = ap.parse_args()

    spark = build_spark("Movies-ETL", shuffle_partitions=args.shuffle_partitions)

    # Movies
    with timer("Load+Transform movies"):
        movies = read_movies(spark, args.movies_csv, args.limit)
        movies = movies.repartition(args.repartition, "movieId").cache(); movies.count()

    # Credits to Roles
    with timer("Load+Transform credits"):
        roles = read_credits(spark, args.credits_csv, args.limit)
        # Parition by movieId for easy search later. 
        roles = roles.repartition(args.repartition, "movieId").cache(); roles.count()

    # Ratings
    # This is not neccesary since ratings is the csv with individual user Id ratings => not important for us
    # The ratings is already in Movies table. 
    # with timer("Load+Transform ratings"):
    #     ratings = read_ratings(spark, args.ratings_csv, args.limit)
    #     ratings = ratings.repartition(args.repartition, "movieId").cache(); ratings.count()

    # Index tables
    with timer("Build indexes"):
        genre_index = build_genre_index(movies).repartition(args.repartition, "genre").cache(); genre_index.count()
        actor_index = build_actor_index(roles).repartition(args.repartition, "movieId").cache(); actor_index.count()
        crew_index = build_crew_index(roles).repartition(args.repartition, "movieId").cache(); crew_index.count() 

    base = args.out_dir.rstrip("/")

    with timer("Write Parquet: Movies"):
        # TODO: We can later check whether partition by year partition_by=["year"] is good or no partition is better. 
        write_parquet(movies, f"{base}/Movies")
    with timer("Write Parquet: Roles"):
        write_parquet(roles, f"{base}/Roles")
    # with timer("Write Parquet: Ratings"):
    #     write_parquet(ratings, f"{base}/Ratings")
    with timer("Write Parquet: GenreIndex"):
        write_parquet(genre_index, f"{base}/GenreIndex", partition_by=["genre"])
    with timer("Write Parquet: ActorIndex"):
        write_parquet(actor_index, f"{base}/ActorIndex")
    with timer("Write Parquet: CrewIndex"):
        write_parquet(crew_index, f"{base}/CrewIndex")

    print("\nParquet tables written to:", base)
    preview_outputs(spark, base)
    spark.stop()


if __name__ == "__main__":
    main()

