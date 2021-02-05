import json
import datetime
import logging
import urllib.parse as urlparse
logging.basicConfig(level=logging.INFO)


from pyspark.sql import DataFrameWriter, HiveContext
from pyspark.sql.functions import udf, col, lit
from pyspark.sql.types import FloatType, StringType, TimestampType, ArrayType

import many_stop_words
import feedparser
from pymorphy2 import MorphAnalyzer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize



def _get_stopwords():
    """
    Extracts stop-words from multiple sources for
    both russian and english languages
    """
    all_stopwords = many_stop_words.get_stop_words('ru')
    all_stopwords.update(many_stop_words.get_stop_words('en'))

    more_stopwords = set(stopwords.words(['russian', 'english']))
    all_stopwords.update(more_stopwords)

    return all_stopwords


def _get_tokens(morph, stopwords, document):
    """
    Converts atricle body into proper list of tokens
    without stop-words and incorrect chars
    """

    PortSt = PorterStemmer()

    invalid_chars = string.punctuation + u'»' + u'«' + u'—' + u'“' + u'„'
    translation_table = {ord(c): None for c in invalid_chars if c != u'-'}

    # parse rss body
    soup = BeautifulSoup(document, 'html.parser')
    body = ' '.join(
        [tag.string.replace('\\n', ' ').replace('\\r', ' ')
         for tag in soup.descendants if tag.string]
    )

    if body == "":
        return []

    body_clean = body.translate(translation_table).lower().strip()
    words = word_tokenize(body_clean)
    tokens = []

    # stemming and text normalization
    for word in words:
        if re.match('^[a-z0-9-]+$', word) is not None:
            tokens.append(PortSt.stem(word))
        elif word.count('-') > 1:
            tokens.append(word)
        else:
            normal_forms = morph.normal_forms(word)
            tokens.append(normal_forms[0] if normal_forms else word)

    tokens = filter(lambda token: token not in stopwords, set(tokens))

    # remove all words with less than 4 chars
    tokens = filter(lambda token: len(token) >= 4, tokens)

    return tokens



class NormalizeTitlesTask:

    def __init__(self, context):
        self.destination_table = 'normalized_titles'
        self.source_table = 'logs_raw'

    @property
    def day(self):
        return self.context.ds

    def execute(self):

        self.log.info('Extracting logs...')
        logs = self._extract_logs()

        self.log.info('Filtering logs...')
        logs = self._filter_logs(logs)

        self.log.info('Normalizing titles...')
        logs = self._normalize_titles(logs)
        
        self.log.info(
            'Uploading processed logs to hive table %s ...',
            self.destination_table
        )
        self._upload_to_hive(logs)

        self.log.info('Finished.')
        
    def _extract_logs(self):

        sql_context = HiveContext()
        logs = sql_context.table(self.source_table)
        
        return logs

    def _filter_logs(self, logs):

        logs = logs.filter(logs.day == self.day)
        logs = logs.filter(logs.uid != '-')
        logs = logs.filter(logs.title != '-')
        logs = logs.select('uid', 'title')
        unquote_udf = udf(urlparse.unquote)
        logs = logs.withColumn('title', unquote_udf(logs.title))

        return logs
    
    def _normalize_titles(self, logs):
        
        def get_tokens(morph, sw):
            _get_tokens = self._get_tokens
            return udf(lambda x: _get_tokens(morph, sw, x), StringType())
        
        morph = MorphAnalyzer(
            result_type=None,
            units=MorphAnalyzer.DEFAULT_UNITS
        )
        sw = self._get_stopwords()

        logs = logs.withColumn(
            'tokens',
            get_tokens(morph=morph, sw=sw)(logs.title)
        )
        return logs

    def _upload_to_hive(self, logs):

        sql_context = HiveContext()
        
        # update Hive table
        df_writer = DataFrameWriter(logs)
        df_writer.insertInto(self.destination_table, overwrite=True)

