# Usage: 
# python eval.py --parquet data_parquet --repeats 3

from query import build_spark, fetch_candidates, rank_movies, Timer
import argparse, time
import csv, os, datetime

# Log file path
LOG_PATH = "eval/eval_results.csv"

# Append evaluation result to CSV log file.
def log_result(test_name, detail, duration):
    exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["timestamp", "test", "detail", "duration_sec"])
        w.writerow([datetime.datetime.now().isoformat(timespec="seconds"), test_name, detail, f"{duration:.3f}"])

# Load test: measure time to load and count all Parquet tables
def load_test(spark, base):
    T = Timer()
    movies = spark.read.parquet(f"{base}/Movies"); movies.count()
    actors = spark.read.parquet(f"{base}/ActorIndex"); actors.count()
    genres = spark.read.parquet(f"{base}/GenreIndex"); genres.count()
    crew = spark.read.parquet(f"{base}/CrewIndex"); crew.count()
    dt = T.lap("Load+Count all Parquet tables")
    log_result("load_test", "all_parquet", dt)

# Candidate fetch test: measure time to fetch candidates for various name and genre inputs
def candidate_fetch_test(spark, base):
    tests = [
        (["Tom Hanks"], ["Drama"]),
        (["Tom Hanks", "Meg Ryan"], ["Drama", "Romance", "Comedy"])
    ]
    for i, (names, genres) in enumerate(tests):
        T = Timer()
        df, _, _ = fetch_candidates(spark, base, names, genres)
        cnt = df.count()
        dt = T.lap(f"Candidate Fetch Test {i+1}: {len(names)} names + {len(genres)} genres -> {cnt} rows")
        log_result("candidate_fetch_test", f"test_{i+1}_names_{len(names)}_genres_{len(genres)}_rows_{cnt}", dt)

# Ranking performance test: measure time to rank movies for various name and genre inputs
def rank_perf_test(spark, base):
    tests = [
        (["Tom Hanks", "Meg Ryan"], ["Drama", "Romance"])
    ]
    for i, (names, genres) in enumerate(tests):
        T = Timer()
        rank_movies(spark, base, names, genres, 10)
        dt = T.lap(f"Ranking Test {i+1}: {len(names)} names + {len(genres)} genres")
        log_result("rank_perf_test", f"test_{i+1}_names_{len(names)}_genres_{len(genres)}", dt)

# Cache and broadcast join test: compare query time under different Spark configurations
# (1) no cache/broadcast
# (2) cache only
# (3) cache + broadcast
def cache_broadcast_test(spark, base):
    configs = [
        ("no_cache_broadcast", {"spark.sql.autoBroadcastJoinThreshold": -1}),
        ("cache_only", {"spark.sql.autoBroadcastJoinThreshold": -1}),
        ("cache_and_broadcast", {"spark.sql.autoBroadcastJoinThreshold": 104857600}),  # 100 MB
    ]
    query = (["Tom Hanks", "Meg Ryan"], ["Drama", "Romance"])

    for name, conf in configs:
        print(f"===== Scenario: {name} =====")
        for k, v in conf.items():
            spark.conf.set(k, v)

        # Clear cache before each run
        spark.catalog.clearCache()

        if name != "no_cache_broadcast":
            # Preload and cache all index tables
            spark.read.parquet(f"{base}/Movies").cache().count()
            spark.read.parquet(f"{base}/GenreIndex").cache().count()
            spark.read.parquet(f"{base}/ActorIndex").cache().count()
            spark.read.parquet(f"{base}/CrewIndex").cache().count()

        T = Timer()
        rank_movies(spark, base, query[0], query[1], 10)
        dt = T.lap(f"Query latency ({name})")
        log_result("cache_broadcast_test", name, dt)

# Main evaluation function
def main():
    # Argument parsing
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True)
    p.add_argument("--repeats", type=int, default=1)
    args = p.parse_args()

    # Spark setup
    spark = build_spark("MovieSearch-Eval", 64)

    # Run tests
    for r in range(args.repeats):
        print(f"===== RUN {r+1}/{args.repeats} =====")
        load_test(spark, args.parquet)
        candidate_fetch_test(spark, args.parquet)
        rank_perf_test(spark, args.parquet)
        cache_broadcast_test(spark, args.parquet)

    print("Evaluation completed.")

if __name__ == "__main__":
    main()

