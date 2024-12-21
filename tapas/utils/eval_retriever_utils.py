## Utilities for Evaluating the metrics@k Scores for Table Retriever Predictions
import abc, ast, csv, json, datetime, dataclasses, itertools, collections
import numpy as np
import tensorflow._api.v2.compat.v1 as tf
from absl import logging
from typing import Text, List
from tapas.utils.constants import _NUM_NEIGHBORS


class InnerProductNearestNeighbors:
    """Helper class to perform nearest neighbor search using inner products."""

    def __init__(self, n_neighbors, candidates):
        """Initializes nearest neighbor index.

        Args:
          n_neighbors: Number of candidates to retrieve for each query.
          candidates: 2D Array of embeddings for each candidate.
        """
        self._n_neighbors = n_neighbors
        self._candidates = candidates

    def neighbors(self, queries):
        """Finds nearest nneighbors.

        Args:
          queries: 2D Array of embeddings for each query.

        Returns:
          2D Array of inner product between each query and the nearest neighbors.
          2D Array of nearest neighbors index for each query.
        """
        # Error handling for _n_neighbors due to shapes
        if self._n_neighbors > self._candidates.shape[0]:
            print(
                f"Warning: Requested {self._n_neighbors} neighbors, but only {self._candidates.shape[0]} candidates available"
            )
            self._n_neighbors = self._candidates.shape[0]

        # <float>[num_queries, num_candidates]
        distances = np.matmul(queries, self._candidates.T)
        # <int>[num_queries, n_neighbors]
        indices = np.argpartition(distances, -self._n_neighbors)[
            :, -self._n_neighbors :
        ]
        # This indices aren't sorted so we find the permutation to sort them.
        # <int>[num_queries, n_neighbors]
        permutation = np.argsort(-np.take_along_axis(distances, indices, axis=-1))
        # <int>[num_queries, n_neighbors]
        sorted_indices = np.take_along_axis(indices, permutation, axis=-1)
        # <float>[num_queries, n_neighbors]
        similarities = np.take_along_axis(distances, sorted_indices, axis=-1)

        return similarities, sorted_indices


class Example(abc.ABC):

    @abc.abstractmethod
    def representation(self):
        Ellipsis


@dataclasses.dataclass(frozen=True)
class QueryExample(Example):
    table_ids: List[Text]
    query_id: Text
    query: np.ndarray

    def representation(self):
        return self.query


@dataclasses.dataclass(frozen=True)
class TableExample(Example):
    table_id: Text
    table: np.ndarray

    def representation(self):
        return self.table


def iterate_predictions(prediction_file):
    with tf.io.gfile.GFile(prediction_file, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            yield row


def _to_ndarray(text):
    return np.array(ast.literal_eval(text))


def _read_table_predictions(predictions_path):
    """Reads table predictions from a csv file."""
    for row in iterate_predictions(predictions_path):

        yield TableExample(
            table_id=row["table_id"],
            table=_to_ndarray(row["table_rep"]),
        )


def _read_query_predictions(predictions_path):
    """Reads query predictions from a csv file."""
    for row in iterate_predictions(predictions_path):

        yield QueryExample(
            table_ids=[row["table_id"]],
            query_id=row["query_id"],
            query=_to_ndarray(row["query_rep"]),
        )


def _get_embeddings(examples):
    """Gets a matrix of all embeddings for a certain representation type."""
    embeddings = np.vstack([example.representation() for example in examples])
    return embeddings


def _get_recall_at_k(neighbors, gold_indices):
    """Calculates different recall@k from the nearest neighbors.

    Args:
      neighbors: <int32>[NUM_QUERIES, _NUM_NEIGHBORS], where NUM_QUERIES is the
        total number of queries.
      gold_indices: <int32>[NUM_QUERIES, _MAX_NUM_TABLES_PER_QUERY],
        matrix containing the indices for the gold tables that should be
        retrieved, for queries of size NUM_QUERIES.

    Returns:
      recall_at_k: recall@k results for different k values.
    """
    if gold_indices.shape[0] != neighbors.shape[0]:
        raise ValueError(
            f"Difference in shapes: {gold_indices.shape} {neighbors.shape[0]}"
        )

    # <int32>[num_queries, num_neigbors, 1]
    neighbors = np.expand_dims(neighbors, axis=-1)
    # <int32>[num_queries, 1, _MAX_NUM_TABLES_PER_QUERY]
    gold_indices = np.expand_dims(gold_indices, axis=-2)

    # <bool>[num_queries, num_neigbors, _MAX_NUM_TABLES_PER_QUERY]
    correct = np.equal(neighbors, gold_indices)
    # <bool>[num_queries, num_neigbors]
    correct = np.any(correct, axis=-1)

    total_queries = float(neighbors.shape[0])

    def _calc_recall_at_k(k):
        # <bool>[num_queries, num_neighbors]
        correct_at_k = correct[:, :k]
        # <bool>[num_queries]
        correct_at_k = np.any(correct_at_k, axis=1)
        return np.sum(correct_at_k) / total_queries

    recall_at = [k for k in [1, 5, 10, 15, 50, 100] if k <= _NUM_NEIGHBORS]
    recall_at_k = {"recall_at_{}".format(k): _calc_recall_at_k(k) for k in recall_at}
    logging.info(recall_at_k)

    return recall_at_k


def _get_gold_ids_in_global_indices(
    queries,
    tables,
):
    """Gets the gold tables in terms of their indices in the index.

    Args:
      queries: List of query examples
      tables: List of table examples

    Returns:
     <int32>[num_queries, _MAX_NUM_TABLES_PER_QUERY]
    """
    table_id_to_index = {
        table.table_id: table_index for table_index, table in enumerate(tables)
    }

    max_num_tables_per_query = max(len(query.table_ids) for query in queries)

    indexes = (
        np.zeros(
            shape=(len(queries), max_num_tables_per_query),
            dtype=np.int32,
        )
        - 1
    )
    for query_index, query in enumerate(queries):

        try:
            table_indexes = {
                table_id_to_index[table_id] for table_id in query.table_ids
            }
        except KeyError:
            raise ValueError(
                f"Query with table_id not found in tables: {query.query_id}"
            )

        for i, table_index in enumerate(sorted(table_indexes)):
            indexes[query_index, i] = table_index

    return indexes


def _retrieve(
    queries,
    index,
):
    """Retrieves nearest neighbors for the queries.

    Args:
      queries: Queries for which retrieval is done.
      index: An index containing all the candidate tables.

    Returns:
      <float>[len(queries), _NUM_NEIGHBORS], a matrix of inner products to the
          nearest neighbors per query.
      <int>[len(queries), _NUM_NEIGHBORS], a matrix of nearest neighbors indices.
    """
    query_embeddings = _get_embeddings(queries)
    logging.info("query embeddings size: %s", str(query_embeddings.shape))

    now = datetime.datetime.now()
    similarities, nns = index.neighbors(query_embeddings)
    time = (datetime.datetime.now() - now).total_seconds()
    logging.info("Time required for computing nearest neighbors:= %f secs", time)

    return similarities, nns


def _save_neighbors_to_file(
    queries,
    tables,
    similarities,
    neighbors,
    retrieval_results_file_path,
):
    """Writes to file similarity scores for _NUM_NEIGHBORS best tables."""

    def _get_dict(table_id, score):
        return {"table_id": table_id, "score": score}

    #  write predictions single query per line.
    #  output format: {query_id: ..., table_scores: [{table_id:... score:...}]...}
    with tf.io.gfile.GFile(retrieval_results_file_path, "w") as f:

        for i, example in enumerate(queries):
            query_id = example.query_id
            table_ids = [tables[int(index)].table_id for index in neighbors[i, :]]
            # Negate similarities for backwards compatibility
            scores = [-float(similarity) for similarity in similarities[i, :]]
            query_to_neighbors = {
                "query_id": query_id,
                "table_scores": [
                    _get_dict(table_id, score)
                    for table_id, score in zip(table_ids, scores)
                ],
            }
            json.dump(query_to_neighbors, f)
            f.write("\n")


def process_predictions(
    queries,
    tables,
    index,
    retrieval_results_file_path,
):
    """Processes predictions and calculates p@k.

    Args:
      queries: List of Example objects containing query embeddings.
      tables: List of Example objects containing query embeddings.
      index: Index created from the table embeddings.
      retrieval_results_file_path: File path to write the metrics.

    Returns:
      A dictionary with recall_at_k metrics for different values of k.
    """
    similarities, neighbors = _retrieve(queries, index)

    if retrieval_results_file_path:
        _save_neighbors_to_file(
            queries, tables, similarities, neighbors, retrieval_results_file_path
        )

    gold_indices = _get_gold_ids_in_global_indices(queries, tables)
    recall_at_k = _get_recall_at_k(neighbors, gold_indices=gold_indices)

    return recall_at_k


def build_table_index(
    tables,
):
    """Creates an index for nearest neighbors searcj."""
    table_embeddings = _get_embeddings(tables)
    logging.info("table embeddings size: %s", str(table_embeddings.shape))

    return InnerProductNearestNeighbors(
        n_neighbors=_NUM_NEIGHBORS, candidates=table_embeddings
    )


def read_tables(
    table_prediction_files,
    make_tables_unique,
):
    """Reads files with table embeddings and optionally removes duplicates."""
    tables = list(_read_table_predictions(table_prediction_files))
    logging.info("Read tables.")
    if make_tables_unique:
        tables = list({table.table_id: table for table in tables}.values())
        logging.info("Made tables unique.")

    return tables


def merge_queries(queries):
    """Averages all embeddings to create a new central query embedding.

    When using this to remove duplicates the incoming embedding should be almost
    identical.

    Args:
      queries: List of query embeddings.

    Returns:
      New mean query embedding.
    """
    table_ids = itertools.chain.from_iterable(q.table_ids for q in queries)
    table_ids = sorted(set(table_ids))
    query = np.mean(_get_embeddings(queries), axis=0)

    return QueryExample(
        query_id=queries[0].query_id,
        table_ids=table_ids,
        query=query,
    )


def read_queries(
    query_prediction_files,
):
    """Reads files with query embeddings and removes duplicates."""
    queries = list(_read_query_predictions(query_prediction_files))
    logging.info("Read queries.")
    # Make queries unique.
    query_id_to_queries = collections.defaultdict(list)
    for query in queries:
        query_id_to_queries[query.query_id].append(query)
    queries = [merge_queries(query_list) for query_list in query_id_to_queries.values()]
    logging.info("Made queries unique.")

    return queries


def eval_recall_at_k(
    query_prediction_files,
    table_prediction_files,
    make_tables_unique,
    retrieval_results_file_path=None,
):
    """Reads queries and tables, processes them to produce recall@k metrics."""
    queries = read_queries(query_prediction_files)
    tables = read_tables(table_prediction_files, make_tables_unique)
    index = build_table_index(tables)

    return process_predictions(queries, tables, index, retrieval_results_file_path)


def _get_ndcg_at_k(neighbors, gold_indices):
    """Calculates NDCG@k from the nearest neighbors.

    Args:
        neighbors: <int32>[NUM_QUERIES, _NUM_NEIGHBORS], where NUM_QUERIES is the
        total number of queries.
        gold_indices: <int32>[NUM_QUERIES, _MAX_NUM_TABLES_PER_QUERY],
        matrix containing the indices for the gold tables that should be
        retrieved, for queries of size NUM_QUERIES.

    Returns:
        ndcg_at_k: NDCG@k results for different k values.
    """

    def dcg_at_k(r, k):
        r = np.asfarray(r)[:k]
        if r.size:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        return 0.0

    def ndcg_at_k(r, k):
        dcg_max = dcg_at_k(sorted(r, reverse=True), k)
        if not dcg_max:
            return 0.0
        return dcg_at_k(r, k) / dcg_max

    if gold_indices.shape[0] != neighbors.shape[0]:
        raise ValueError(
            f"Difference in shapes: {gold_indices.shape} {neighbors.shape[0]}"
        )

    # <int32>[num_queries, num_neigbors, 1]
    neighbors = np.expand_dims(neighbors, axis=-1)
    # <int32>[num_queries, 1, _MAX_NUM_TABLES_PER_QUERY]
    gold_indices = np.expand_dims(gold_indices, axis=-2)

    # <bool>[num_queries, num_neigbors, _MAX_NUM_TABLES_PER_QUERY]
    correct = np.equal(neighbors, gold_indices)
    # <bool>[num_queries, num_neigbors]
    correct = np.any(correct, axis=-1)

    ndcg_at = [k for k in [1, 5, 10, 15, 50, 100] if k <= _NUM_NEIGHBORS]
    ndcg_at_k = {
        "ndcg_at_{}".format(k): np.mean(
            [ndcg_at_k(correct[i], k) for i in range(correct.shape[0])]
        )
        for k in ndcg_at
    }
    logging.info(ndcg_at_k)

    return ndcg_at_k


def eval_metrics_at_k(
    query_prediction_files,
    table_prediction_files,
    make_tables_unique,
    retrieval_results_file_path=None,
):
    """Reads queries and tables, processes them to produce recall@k, NDCG@k, and mAP metrics."""
    queries = read_queries(query_prediction_files)
    tables = read_tables(table_prediction_files, make_tables_unique)
    index = build_table_index(tables)

    similarities, neighbors = _retrieve(queries, index)

    if retrieval_results_file_path:
        _save_neighbors_to_file(
            queries, tables, similarities, neighbors, retrieval_results_file_path
        )

    gold_indices = _get_gold_ids_in_global_indices(queries, tables)
    recall_at_k = _get_recall_at_k(neighbors, gold_indices=gold_indices)
    precision_at_k = _get_precision_at_k(neighbors, gold_indices=gold_indices)
    recall_at_k = _get_recall_at_k(neighbors, gold_indices=gold_indices)
    ndcg_at_k = _get_ndcg_at_k(neighbors, gold_indices=gold_indices)
    map_at_k = _get_map_at_k(neighbors, gold_indices=gold_indices)

    return {**precision_at_k, **recall_at_k, **ndcg_at_k, **map_at_k}


def _get_precision_at_k(neighbors, gold_indices):
    """Calculates different precision@k from the nearest neighbors.

    Args:
      neighbors: <int32>[NUM_QUERIES, _NUM_NEIGHBORS], where NUM_QUERIES is the
        total number of queries.
      gold_indices: <int32>[NUM_QUERIES, _MAX_NUM_TABLES_PER_QUERY],
        matrix containing the indices for the gold tables that should be
        retrieved, for queries of size NUM_QUERIES.

    Returns:
      precision_at_k: precision@k results for different k values.
    """
    if gold_indices.shape[0] != neighbors.shape[0]:
        raise ValueError(
            f"Difference in shapes: {gold_indices.shape} {neighbors.shape[0]}"
        )

    # <int32>[num_queries, num_neigbors, 1]
    neighbors = np.expand_dims(neighbors, axis=-1)
    # <int32>[num_queries, 1, _MAX_NUM_TABLES_PER_QUERY]
    gold_indices = np.expand_dims(gold_indices, axis=-2)

    # <bool>[num_queries, num_neigbors, _MAX_NUM_TABLES_PER_QUERY]
    correct = np.equal(neighbors, gold_indices)
    # <bool>[num_queries, num_neigbors]
    correct = np.any(correct, axis=-1)

    total_queries = float(neighbors.shape[0])

    def _calc_precision_at_k(k):
        # <bool>[num_queries, num_neighbors]
        correct_at_k = correct[:, :k]
        # <bool>[num_queries]
        correct_at_k = np.any(correct_at_k, axis=1)
        return np.sum(correct_at_k) / total_queries

    precision_at = [k for k in [1, 5, 10, 15, 50, 100] if k <= _NUM_NEIGHBORS]
    precision_at_k = {
        "precision_at_{}".format(k): _calc_precision_at_k(k) for k in precision_at
    }
    logging.info(precision_at_k)

    return precision_at_k


def _get_recall_at_k(neighbors, gold_indices):
    """Calculates recall@k from the nearest neighbors.

    Args:
        neighbors: <int32>[NUM_QUERIES, _NUM_NEIGHBORS], where NUM_QUERIES is the
        total number of queries.
        gold_indices: <int32>[NUM_QUERIES, _MAX_NUM_TABLES_PER_QUERY],
        matrix containing the indices for the gold tables that should be
        retrieved, for queries of size NUM_QUERIES.

    Returns:
        recall_at_k: recall@k results for different k values.
    """

    def recall(r):
        r = np.asarray(r) != 0
        return np.sum(r) / len(r)

    if gold_indices.shape[0] != neighbors.shape[0]:
        raise ValueError(
            f"Difference in shapes: {gold_indices.shape} {neighbors.shape[0]}"
        )

    # <int32>[num_queries, num_neigbors, 1]
    neighbors = np.expand_dims(neighbors, axis=-1)
    # <int32>[num_queries, 1, _MAX_NUM_TABLES_PER_QUERY]
    gold_indices = np.expand_dims(gold_indices, axis=-2)

    # <bool>[num_queries, num_neigbors, _MAX_NUM_TABLES_PER_QUERY]
    correct = np.equal(neighbors, gold_indices)
    # <bool>[num_queries, num_neigbors]
    correct = np.any(correct, axis=-1)

    recall_at = [k for k in [1, 5, 10, 15, 50, 100] if k <= _NUM_NEIGHBORS]
    recall_at_k = {
        "recall_at_{}".format(k): np.mean(
            [recall(correct[i, :k]) for i in range(correct.shape[0])]
        )
        for k in recall_at
    }
    logging.info(recall_at_k)

    return recall_at_k


def _get_map_at_k(neighbors, gold_indices):
    """Calculates mean average precision (mAP) from the nearest neighbors.

    Args:
        neighbors: <int32>[NUM_QUERIES, _NUM_NEIGHBORS], where NUM_QUERIES is the
        total number of queries.
        gold_indices: <int32>[NUM_QUERIES, _MAX_NUM_TABLES_PER_QUERY],
        matrix containing the indices for the gold tables that should be
        retrieved, for queries of size NUM_QUERIES.

    Returns:
        map_at_k: mAP results for different k values.
    """

    def average_precision(r):
        r = np.asarray(r) != 0
        out = [np.mean(r[: i + 1]) for i in range(len(r)) if r[i]]
        if not out:
            return 0.0
        return np.mean(out)

    if gold_indices.shape[0] != neighbors.shape[0]:
        raise ValueError(
            f"Difference in shapes: {gold_indices.shape} {neighbors.shape[0]}"
        )

    # <int32>[num_queries, num_neigbors, 1]
    neighbors = np.expand_dims(neighbors, axis=-1)
    # <int32>[num_queries, 1, _MAX_NUM_TABLES_PER_QUERY]
    gold_indices = np.expand_dims(gold_indices, axis=-2)

    # <bool>[num_queries, num_neigbors, _MAX_NUM_TABLES_PER_QUERY]
    correct = np.equal(neighbors, gold_indices)
    # <bool>[num_queries, num_neigbors]
    correct = np.any(correct, axis=-1)

    map_at = [k for k in [1, 5, 10, 15, 50, 100] if k <= _NUM_NEIGHBORS]
    map_at_k = {
        "map_at_{}".format(k): np.mean(
            [average_precision(correct[i, :k]) for i in range(correct.shape[0])]
        )
        for k in map_at
    }
    logging.info(map_at_k)

    return map_at_k
