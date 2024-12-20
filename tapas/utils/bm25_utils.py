## Implementation of a Simple TF-IDF Baseline for Table Retrieval

import math, tqdm, logging, collections, dataclasses
import tensorflow._api.v2.compat.v1 as tf
from typing import List
from gensim.summarization import bm25
from tapas.protos import interaction_pb2
from tapas.utils.text_utils import format_text


def iterate_tables(table_file):

    for value in tf.io.tf_record_iterator(table_file):

        table = interaction_pb2.Table()
        table.ParseFromString(value)

        yield table


def iterate_interaction_tables(interaction_file):

    for value in tf.io.tf_record_iterator(interaction_file):

        interaction = interaction_pb2.Interaction()
        interaction.ParseFromString(value)

        yield interaction.table


def iterate_interactions(interactions_file):
    """Get interactions from file."""
    for value in tf.io.tf_record_iterator(interactions_file):

        interaction = interaction_pb2.Interaction()
        interaction.ParseFromString(value)

        yield interaction


def _iterate_table_texts(table, title_multiplicator):

    for _ in range(title_multiplicator):

        if table.document_title:
            yield table.document_title

        for column in table.columns:

            yield column.text

    for row in table.rows:

        for cell in row.cells:
            yield cell.text


def _iterate_tokenized_table_texts(table, title_multiplicator):

    for text in _iterate_table_texts(table, title_multiplicator):

        yield from _tokenize(text)


def _tokenize(text):
    return format_text(text).split()


@dataclasses.dataclass(frozen=True)
class TableFrequency:
    table_index: int
    score: float


@dataclasses.dataclass(frozen=True)
class IndexEntry:
    table_counts: List[TableFrequency]


class InvertedIndex:
    """Inverted Index implementation."""

    def __init__(self, table_ids, index):
        self.table_ids_ = table_ids
        self.index_ = index

    def retrieve(self, question):
        """Retrieves tables sorted by descending score."""
        hits = collections.defaultdict(list)
        num_tokens = 0
        for token in _tokenize(question):

            num_tokens += 1
            index_entry = self.index_.get(token, None)
            if index_entry is None:
                continue

            for table_count in index_entry.table_counts:

                scores = hits[table_count.table_index]
                scores.append(table_count.score)

        scored_hits = []
        for table_index, inv_document_freqs in hits.items():
            score = sum(inv_document_freqs) / num_tokens
            scored_hits.append((self.table_ids_[table_index], score))

        scored_hits.sort(key=lambda name_score: name_score[1], reverse=True)

        return scored_hits


def _remove_duplicates(tables):

    table_id_set = set()
    for table in tables:

        if table.table_id in table_id_set:
            logging.info("Duplicate table ids: %s", table.table_id)
            continue

        table_id_set.add(table.table_id)
        yield table


def create_inverted_index(
    tables, title_multiplicator=1, min_rank=0, drop_term_frequency=True
):
    """Creates an index for some tables.

    Args:
    tables: Tables to index
    title_multiplicator: Emphasize words in title or header.
    min_rank: Word types with a frequency rank lower than this will be ignored.
        Can be useful to remove stop words.
    drop_term_frequency: Don't consider term frequency.

    Returns:
    the inverted index.
    """
    table_ids = []
    token_to_info = collections.defaultdict(lambda: collections.defaultdict(int))
    for table in _remove_duplicates(tables):

        table_index = len(table_ids)
        table_ids.append(table.table_id)

        for token in _iterate_tokenized_table_texts(table, title_multiplicator):

            token_to_info[token][table_index] += 1

    logging.info("Table Ids: %d", len(table_ids))
    logging.info("Num types: %d", len(token_to_info))

    def count_fn(table_counts):
        return sum(table_counts.values())

    token_to_info = list(token_to_info.items())
    token_to_info.sort(key=lambda token_info: count_fn(token_info[1]), reverse=True)

    index = {}
    for freq_rank, (token, table_counts) in enumerate(token_to_info):
        df = count_fn(table_counts)
        if freq_rank < min_rank:
            logging.info('Filter "%s" for index (%d, rank: %d).', token, df, freq_rank)
            continue

        idf = 1.0 / (math.log(df, 2) + 1)
        counts = []
        for table, count in table_counts.items():

            if drop_term_frequency:
                count = 1.0

            counts.append(TableFrequency(table, idf * count))

        index[token] = IndexEntry(counts)

    return InvertedIndex(table_ids, index)


class BM25Index:
    """Index based on gensim BM25."""

    def __init__(self, corpus, table_ids):
        self._table_ids = table_ids
        self._model = bm25.BM25(corpus)

    def retrieve(self, question):
        q_tokens = _tokenize(question)
        scores = self._model.get_scores(q_tokens)
        table_scores = [
            (self._table_ids[index], score)
            for index, score in enumerate(scores)
            if score > 0.0
        ]
        table_scores.sort(key=lambda table_score: table_score[1], reverse=True)

        return table_scores


def create_bm25_index(
    tables,
    title_multiplicator=1,
    num_tables=None,
):
    """Creates a new index."""
    corpus, table_ids = [], []
    for table in tqdm.tqdm(_remove_duplicates(tables), total=num_tables):

        corpus.append(list(_iterate_tokenized_table_texts(table, title_multiplicator)))
        table_ids.append(table.table_id)

    return BM25Index(corpus, table_ids)
