"""
Solutions -- Day 8 : ML System Design Intro

The solutions are provided as detailed docstrings, like in the rest
of the domain. This file is standalone and executable for
syntax verification (python 08-ml-system-design-intro.py).
"""


def solution_exercice_1() -> None:
    """
    Exercise 1 -- Diagnosing training-serving skew.

    +-----+----------------------------+------------------------------------+
    | Case| Cause                      | Mitigation                         |
    +-----+----------------------------+------------------------------------+
    |  1  | Divergent code :           | Single feature store. The same     |
    |     | Python at training, SQL    | Python function serves offline and |
    |     | at serving -> different    | online. If SQL is mandatory,       |
    |     | formulas (rounding,        | generate the SQL from the same     |
    |     | NaN, type casts).          | definition (DSL / DBT / Feast).    |
    +-----+----------------------------+------------------------------------+
    |  2  | Undetected feature drift   | Continuous monitoring (PSI, KL) of |
    |     | or concept drift. The      | the input and output               |
    |     | world has changed, the     | distributions. Automated           |
    |     | model is frozen in time.   | retraining triggered by the        |
    |     |                            | monitoring. At minimum : scheduled |
    |     |                            | retraining (weekly/monthly).       |
    +-----+----------------------------+------------------------------------+
    |  3  | Different data handling.   | Implement the NaN logic            |
    |     | NaNs are handled at        | in the feature definition itself.  |
    |     | training (fillna mean)     | Not in the notebook. What we do    |
    |     | but not at serving.        | at training = what we do           |
    |     |                            | at serving.                        |
    +-----+----------------------------+------------------------------------+
    |  4  | Data leakage / lack of     | Point-in-time correctness :        |
    |     | point-in-time correctness. | when building the dataset,         |
    |     | The model saw the future.  | each example sees the features     |
    |     |                            | as they were at the moment of      |
    |     |                            | the event. Later reviews must      |
    |     |                            | NOT be included in the average.    |
    |     |                            | That is THE raison d'etre of the   |
    |     |                            | feature store on the offline side. |
    +-----+----------------------------+------------------------------------+

    Note : in the real world, the 4 problems often coexist. A well-designed
    feature store solves 3 out of 4. The 4th (drift) requires monitoring.
    """


def solution_exercice_2() -> None:
    """
    Exercise 2 -- Batch vs real-time.

    +-----------------+----------+---------------+-------------------------------+
    | Product         | Archi    | Target latency| Justification                 |
    +-----------------+----------+---------------+-------------------------------+
    | Netflix homepage| Hybrid   | < 500 ms      | Daily batch for the           |
    |                 |          | (re-ranking)  | candidates (1M+ items).       |
    |                 |          |               | Online re-ranking by          |
    |                 |          |               | context (device, time).       |
    +-----------------+----------+---------------+-------------------------------+
    | Payment anti-   | Real-time| p99 < 100 ms  | Blocking rejection = critical |
    | fraud           |          |               | latency. 1000 tx/s = infra    |
    |                 |          |               | sized accordingly             |
    |                 |          |               | (GPU + dynamic batching).     |
    +-----------------+----------+---------------+-------------------------------+
    | Daily email     | Batch    | Several       | Offline generation during     |
    | digests         |          | hours OK      | the night. Read in the morning|
    |                 |          |               | from a DB. Cost divided 10+.  |
    +-----------------+----------+---------------+-------------------------------+
    | Uber pricing    | Real-time| p95 < 500 ms  | Depends on the local          |
    |                 |          |               | supply/demand in real time.   |
    |                 |          |               | Features updated frequently.  |
    +-----------------+----------+---------------+-------------------------------+
    | IoT anomalies   | Hybrid   | 1-60 sec      | Micro-batch every 10-60s      |
    | (5M sensors)    | (micro)  |               | to accumulate the signals.    |
    |                 |          |               | Volume too high for true      |
    |                 |          |               | real-time per sensor.         |
    +-----------------+----------+---------------+-------------------------------+
    | IDE autocorrect | Real-time| < 50 ms       | UX demands responsiveness.    |
    |                 | (local)  |               | Often a distilled model that  |
    |                 |          |               | runs locally inside the IDE.  |
    +-----------------+----------+---------------+-------------------------------+

    Key points :
    - The volume and the change frequency of the context dictate the choice.
    - Real-time is not "better" -- it is 10x more expensive in infra.
    - Hybrid is the norm at scale : batch for the heavy lifting,
      online for the final personalization.
    """


def solution_exercice_3() -> None:
    """
    Exercise 3 -- Feature store for a fashion e-commerce.

    +----------------------------+--------+--------------+----------------+---------+
    | Feature                    | Source | Offline      | Online         | TTL     |
    +----------------------------+--------+--------------+----------------+---------+
    | user_avg_order_value_30d   | batch  | Parquet/BQ   | Redis/Dynamo   | 24h     |
    | user_fav_category          | batch  | Parquet/BQ   | Redis/Dynamo   | 24h     |
    | user_current_cart_size     | stream | Kafka topic  | Redis (stream) | session |
    |                            |        | + archived   |                |         |
    | product_sales_rank_7d      | batch  | BigQuery     | Redis cluster  | 1h      |
    |                            |        |              | (hot items)    |         |
    | user_device_type           | on-    | --           | passed in      | --      |
    |                            | demand |              | the request    |         |
    +----------------------------+--------+--------------+----------------+---------+

    ASCII diagram:

        +-------------+      +------------+     +---------------+
        | Click/order |----->| Kafka topic|---->| Flink stream  |
        | events      |      +-----+------+     | processor     |
        +-------------+            |            +-------+-------+
                                   |                    |
                                   v                    v (cart_size)
                          +---------------+      +-------------+
                          | S3 / Parquet  |      | Redis online|
                          | (offline)     |      | store       |
                          +-------+-------+      +------+------+
                                  |                     ^
                                  v                     |
                          +---------------+             |
                          | Spark batch   |-------------+
                          | jobs          | materialize
                          | (daily)       |
                          +-------+-------+
                                  |
                                  v
                          +---------------+
                          | BigQuery      | <-- training datasets
                          | (offline store)|    point-in-time joins
                          +---------------+

        Model serving flow:
            request ---> FastAPI ---> Redis get (online features)
                                      + device_type from request
                                      ---> model.predict ---> response

    Q3 -- Trickiest point-in-time feature :
    It is `product_sales_rank_7d`. It evolves very fast (a viral item changes
    rank within hours). When building the training dataset, each
    row must contain the product's rank AS IT WAS at the moment of
    the event, not the current rank. Otherwise the model learns that "items
    popular today were popular yesterday", which is data leakage.

    Mitigation : the offline feature store must version the rank values
    by timestamp. The dataset generation does a point-in-time join
    on the largest value <= the event's timestamp.

    Q4 -- Degradation if the online Redis is down :
    Two combinable strategies :
    1. Fallback features : if Redis is unreachable, use default
       values (global mean, 0, etc.) and log a warning. The model
       will keep responding, with degraded precision.
    2. Fallback model : route to a simpler model that does not
       need the missing features (e.g. rule-based "top sellers").
       It's a "graceful degradation" pattern : we accept a worse
       model rather than no response.

    In any case : NEVER return a 500 error. A degraded recommendation
    is better than an empty page.
    """


if __name__ == "__main__":
    print("Day 8 -- Solutions : see the docstrings.")
    for fn in (solution_exercice_1, solution_exercice_2, solution_exercice_3):
        print(f"\n--- {fn.__name__} ---")
        print(fn.__doc__)
