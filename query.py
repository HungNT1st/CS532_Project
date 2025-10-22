# Usage: 
# python query.py --parquet data_parquet \
#   --actors "Tom Hanks, Meg Ryan" \
#   --genres "Drama, Romance" \
#   --top_k 10

# Query engine: names (actor or crew) + genres => ranked movies with explanations
import argparse, time
from typing import List
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import broadcast

# Weights for score
W_NAME = 0.45   
W_GENRE = 0.35
W_RATING = 0.15
W_RECENCY = 0.05

# Spark setup and small helpers
def build_spark(app_name: str, shuffle_partitions: int = 64):
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.ansi.enabled", "false")
        .getOrCreate()
    )

# Timer utility to measure elapsed time for each step
class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()
    def lap(self, label: str):
        t1 = time.perf_counter()
        dt = t1 - self.t0
        self.t0 = t1
        print(f"[TIMER] {label}: {dt:.3f}s")

# Split comma-separated list and trim spaces for input names and genres
def parse_list(s: str) -> List[str]:
    if not s: return []
    return [x.strip() for x in s.split(',') if x.strip()]

# Normalize names trim space and lowercase
def normalize_name(s: str) -> str:
    return ' '.join(s.strip().split()).lower()

# Fetch candidate movies matching input names and genres (with soft filtering). 
# This is the main filtering step.
def fetch_candidates(spark, base: str, names: List[str], genres: List[str]):
    # Start a timer for this function
    T = Timer()

    # Load tables
    movies = spark.read.parquet(f"{base}/Movies")
    genre_idx = spark.read.parquet(f"{base}/GenreIndex")
    actor_idx = spark.read.parquet(f"{base}/ActorIndex")
    crew_idx = spark.read.parquet(f"{base}/CrewIndex")
    T.lap("load tables")

    # Normalize names and genres
    names_norm = [normalize_name(a) for a in names]
    genres_norm = [g.strip().title() for g in genres]
    T.lap("normalize inputs")

    # Start from all movies
    candidates = movies.select("movieId").distinct()

    # Filter by genres
    if genres_norm:
        # Prepare genre DataFrame
        gdf = spark.createDataFrame([(g,) for g in genres_norm], ["genre"])  

        # Hard filter
        # mg = (genre_idx.join(broadcast(gdf), "genre")
        #         .groupBy("movieId")
        #         .agg(F.countDistinct("genre").alias("matched_genres"))
        #         .where(F.col("matched_genres") == F.lit(len(genres_norm))))
        # candidates = candidates.join(mg, "movieId")

        # Soft filter (for this project, we will use soft filter)
        # Count how many genres matched 
        mg = (genre_idx.join(broadcast(gdf), "genre")
                .groupBy("movieId")
                .agg(F.countDistinct("genre").alias("matched_genres")))
        candidates = (candidates
                        .join(mg, "movieId", "left")
                        .fillna({"matched_genres": 0}))
    else:
        # No genre filtering
        candidates = candidates.withColumn("matched_genres", F.lit(0))
    T.lap("genre filtering")

    # Filter by names (actor or crew),
    # Names can match in either ActorIndex or CrewIndex
    if names_norm:
        # Prepare names DataFrame
        ndf = spark.createDataFrame([(n,) for n in names_norm], ["name_norm"])  
        actor_names = (actor_idx
                        .select("movieId", F.lower(F.trim(F.col("name"))).alias("name_norm"))
                        .dropna(subset=["movieId", "name_norm"]).dropDuplicates())
        crew_names = (crew_idx
                        .select("movieId", F.lower(F.trim(F.col("name"))).alias("name_norm"))
                        .dropna(subset=["movieId", "name_norm"]).dropDuplicates())
        union_names = actor_names.unionByName(crew_names)
        # Hard filter
        # mn = (union_names
        #         .join(broadcast(ndf), "name_norm")
        #         .groupBy("movieId")
        #         .agg(F.countDistinct("name_norm").alias("matched_names"))
        #         .where(F.col("matched_names") == F.lit(len(names_norm))))
        # candidates = candidates.join(mn, "movieId", "inner")

        # Soft filter (for this project, we will use soft filter)
        # Count how many names matched
        mn = (union_names.join(broadcast(ndf), "name_norm")
                .groupBy("movieId")
                .agg(F.countDistinct("name_norm").alias("matched_names")))
        candidates = (candidates
                        .join(mn, "movieId", "left")
                        .fillna({"matched_names": 0}))
    else:
        candidates = candidates.withColumn("matched_names", F.lit(0))
    T.lap("name matching (actor+crew)")

    # For explanations: separate counts for actor vs crew
    if names_norm:
        # Prepare names DataFrame
        adf = spark.createDataFrame([(n,) for n in names_norm], ["name_norm"]) 

        # Actor matches
        ma = (actor_idx.select("movieId", F.lower(F.trim(F.col("name"))).alias("name_norm"))
                .join(broadcast(adf), "name_norm")
                .groupBy("movieId").agg(F.countDistinct("name_norm").alias("matched_actors")))
        
        # Crew matches
        mc = (crew_idx.select("movieId", F.lower(F.trim(F.col("name"))).alias("name_norm"))
                .join(broadcast(adf), "name_norm")
                .groupBy("movieId").agg(F.countDistinct("name_norm").alias("matched_crew")))
        
        # Join to candidates
        candidates = (candidates
                        .join(ma, "movieId", "left")
                        .join(mc, "movieId", "left")
                        .fillna({"matched_actors": 0, "matched_crew": 0}))
    else:
        candidates = candidates.withColumn("matched_actors", F.lit(0)).withColumn("matched_crew", F.lit(0))
    T.lap("explanation counts")

    # Enrich with movie metadata for scoring
    out = (candidates
            .join(movies.select("movieId", "title", "year", "genres", "vote_average", "vote_count"), "movieId"))
    T.lap("enrich metadata")

    # Return candidates and counts
    return out, len(names_norm), len(genres_norm)

# Main function to rank movies and provide explanations
def rank_movies(spark, base: str, names: List[str], genres: List[str], top_k: int = 10):
    # Start timer
    T = Timer()

    # Fetch candidate movies
    # Candidates set contains all movies that match at least one input name or genre
    # We do not hard filter here to allow partial matches, but all candidates must match at least one thing. 
    df, n_names, n_genres = fetch_candidates(spark, base, names, genres)

    # Compute match fractions
    df = (df
        .withColumn("name_match", F.when(F.lit(n_names)>0, F.col("matched_names")/F.lit(n_names)).otherwise(F.lit(0.0)))
        .withColumn("genre_match", F.when(F.lit(n_genres)>0, F.col("matched_genres")/F.lit(n_genres)).otherwise(F.lit(0.0))))
    T.lap("compute match fractions")

    # Rating z using Movies.vote_average across candidate set
    # rating_z = (vote_average - mean(vote_average)) / stddev(vote_average)
    stats = df.agg(F.mean("vote_average").alias("mu"), F.stddev_pop("vote_average").alias("sigma")).collect()[0]
    mu = float(stats.mu) if stats.mu is not None else 5.0
    sigma = float(stats.sigma) if stats.sigma not in (None, 0.0) else 1.0
    df = df.withColumn("rating_z", (F.col("vote_average") - F.lit(mu)) / F.lit(sigma))
    T.lap("rating z-score")

    # Recency scaled to [0,1]
    # recency  = (year - min(year)) / (max(year) - min(year))
    yr = df.agg(F.min("year").alias("ymin"), F.max("year").alias("ymax")).collect()[0]
    ymin = int(yr.ymin) if yr.ymin is not None else 1888
    ymax = int(yr.ymax) if yr.ymax is not None else 2020
    span = (ymax - ymin) if ymax > ymin else 1
    df = df.withColumn("recency", F.when(F.col("year").isNotNull(), (F.col("year")-F.lit(ymin))/F.lit(span)).otherwise(F.lit(0.0)))
    T.lap("recency")

    # Score
    df = df.withColumn(
        "score",
        W_NAME*F.col("name_match") +
        W_GENRE*F.col("genre_match") +
        W_RATING*F.col("rating_z") +
        W_RECENCY*F.col("recency")
    )
    T.lap("score")

    # Explanation
    explain = F.format_string(
        "Matches %s/%s names (actor/crew), %s/%s genres; rating %s (%s votes); %s | hits: actors=%s crew=%s",
        F.col("matched_names"), F.lit(n_names),
        F.col("matched_genres"), F.lit(n_genres),
        F.round(F.col("vote_average"), 1), F.col("vote_count"),
        F.col("year"),
        F.col("matched_actors"), F.col("matched_crew")
    )

    df = df.withColumn("explanation", explain)
    T.lap("explain") 

    # Get top-k results
    res = (df
        .select("score", "movieId", "title", "year", "explanation")
        .orderBy(F.col("score").desc())
        .limit(top_k))

    rows = res.collect()
    T.lap("collect")

    # Print results
    print("\n===== Top Results =====")
    print(f"{'Rank':<5} {'Score':<8} {'MovieID':<10} {'Title (Year)':<50}")
    print("-" * 80)
    for i, r in enumerate(rows, 1):
        print(f"{i:<5} {r.score:0.3f}   {r.movieId:<10d} {r.title} ({r.year})")
        print(f"{r.explanation}\n")


if __name__ == "__main__":
    # Parse commandline arguments
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--actors", default="")   
    p.add_argument("--genres", default="")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--shuffle_partitions", type=int, default=64)
    args = p.parse_args()

    spark = build_spark("MovieSearch-Query", args.shuffle_partitions)
    names = parse_list(args.actors)
    genres = parse_list(args.genres)

    # Overall timer and run
    T0 = Timer()
    rank_movies(spark, args.parquet, names, genres, args.top_k)
    T0.lap("TOTAL query time")
