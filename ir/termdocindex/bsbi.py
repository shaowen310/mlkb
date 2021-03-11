import os
import shutil
import collections
import itertools
import csv
import queue
import time
import datetime
import sys

from loggingutil import get_logger


class IntIDGenerator:
    def __init__(self, start=0):
        self.start = start

    def __iter__(self):
        self.id = self.start - 1
        return self

    def __next__(self):
        self.id += 1
        return self.id


class BSBI:
    '''
    Blocked sort-based indexing
    
    Reference: 
        https://nlp.stanford.edu/IR-book/html/htmledition/blocked-sort-based-indexing-1.html
    '''
    id_generator = iter(IntIDGenerator())
    _logger = get_logger('bsbi')

    class BlockFileWriter:
        def __init__(self, fostream):
            self.csvw = csv.writer(fostream)

        def write_pair(self, pair):
            self.csvw.writerow(pair)

    def __init__(self, block_size, block_dir='block_'):
        if not (sys.version_info[0] >= 3 and sys.version_info[1] >= 7):
            # dict preserves insertion order
            # SimpleQueue
            raise RuntimeError('Python version >= 3.7 required.')

        self.block_size = block_size
        self.block_dir = block_dir

        # Clean old block files
        if os.path.exists(block_dir) and os.path.isdir(block_dir):
            shutil.rmtree(block_dir)

        os.makedirs(block_dir)

    def get_block(self, token_stream):
        return list(itertools.islice(token_stream, self.block_size))

    @staticmethod
    def bsbi_invert(tokens):
        sorted_tokens = sorted(tokens)

        # Assume python version >= 3.7, dict preserves insertion order
        postinglists = collections.defaultdict(lambda: [])
        for token in sorted_tokens:
            postinglist = postinglists[token[0]]
            if not len(postinglist) or postinglist[-1] != token[1]:
                postinglist.append(token[1])

        return postinglists

    @staticmethod
    def generate_term_doc_pairs(postinglists):
        for (term, docs) in postinglists.items():
            for doc in docs:
                yield (term, doc)

    def get_block_fp(self, block_id):
        return os.path.join(self.block_dir, str(block_id)) + '.csv'

    def write_block(self, block_id, postinglists):
        term_doc_pairs = __class__.generate_term_doc_pairs(postinglists)

        file = self.get_block_fp(block_id)
        with open(file, 'w') as f:
            w = self.BlockFileWriter(f)
            for pair in term_doc_pairs:
                w.write_pair(pair)

    def read_block(self, block_id):
        file = self.get_block_fp(block_id)
        with open(file, 'r') as f:
            csvr = csv.reader(f)
            for row in csvr:
                yield row

    def merge_blocks(self):
        block_queue = queue.SimpleQueue()

        files = os.listdir(self.block_dir)

        for f in files:
            if not os.path.isfile(os.path.join(self.block_dir, f)):
                continue
            name, ext = os.path.splitext(f)
            if ext != '.csv':
                continue
            block_id = int(name)
            block_queue.put(block_id)

        while block_queue.qsize() > 1:
            block_id0 = block_queue.get()
            block_id1 = block_queue.get()
            block_id2 = next(self.id_generator)
            self.merge_two_blocks(block_id0, block_id1, block_id2)
            block_queue.put(block_id2)

        return block_queue.get()

    def merge_two_blocks(self, block_id0, block_id1, block_id2):
        self._logger.debug('Merging block {} and {} to {}'.format(block_id0, block_id1, block_id2))

        term_doc_stream0 = self.read_block(block_id0)
        term_doc_stream1 = self.read_block(block_id1)

        outfile = self.get_block_fp(block_id2)
        with open(outfile, 'w') as fout:
            writer = self.BlockFileWriter(fout)

            # merge sort
            term_doc0 = next(term_doc_stream0, None)
            term_doc1 = next(term_doc_stream1, None)
            pair_last_written = ('', '')
            while (term_doc0 is not None) and (term_doc1 is not None):
                if term_doc0 < term_doc1:
                    pair_to_be_written = term_doc0
                    term_doc0 = next(term_doc_stream0, None)
                else:
                    pair_to_be_written = term_doc1
                    term_doc1 = next(term_doc_stream1, None)
                if pair_to_be_written != pair_last_written:
                    writer.write_pair(pair_to_be_written)
                    pair_last_written = pair_to_be_written
            while term_doc0 is not None:
                writer.write_pair(term_doc0)
                term_doc0 = next(term_doc_stream0, None)
            while term_doc1 is not None:
                writer.write_pair(term_doc1)
                term_doc1 = next(term_doc_stream1, None)

    def process(self, token_stream):
        # write blocks
        while True:
            block_id = next(self.id_generator)
            self._logger.debug('Processing block {}'.format(block_id))

            tokens = self.get_block(token_stream)
            if not len(tokens):
                break

            sort_time = time.time()
            postinglists = __class__.bsbi_invert(tokens)
            sort_time = time.time() - sort_time
            self._logger.debug('Time to sort a block time: {}'.format(
                datetime.timedelta(seconds=sort_time)))
            self.write_block(block_id, postinglists)

        # merge blocks
        merge_time = time.time()
        merged_block_id = self.merge_blocks()
        merge_time = time.time() - merge_time
        self._logger.debug('Time to merge all blocks: {}'.format(
            datetime.timedelta(seconds=merge_time)))

        merged_file = self.get_block_fp(merged_block_id)

        return merged_file
