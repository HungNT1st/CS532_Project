# Movie Search ETL and Query

## Overview
Three PySpark scripts:
- **etl.py**: builds clean Parquet tables from raw CSVs.
- **query.py**: searches movies by actors and genres with scoring.
- **eval.py**: evaluates the system 

## ETL
Create Parquet datasets:
```bash
python etl.py \
  --movies_csv data_raw/movies_metadata.csv \
  --credits_csv data_raw/credits.csv \
  --out_dir data_parquet \
  --repartition 8 --shuffle_partitions 64
```

Output tables: Movies, People, Roles, Ratings, GenreIndex, ActorIndex.

## Query
Search ranked movies:
```bash
python query.py --parquet data_parquet \
  --actors "Tom Hanks, Meg Ryan" \
  --genres "Drama, Romance" \
  --top_k 10
```

Score formula for this project: 
score = 0.45*name_match + 0.35*genre_match + 0.15*rating_z + 0.05*recency
rating_z = (vote_average - mean(vote_average)) / stddev(vote_average)
recency  = (year - min(year)) / (max(year) - min(year))

## Eval
Evaluate the system: 
```bash
python eval.py --parquet data_parquet --repeats 3
```