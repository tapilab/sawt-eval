"""
usage: OPENAI_API_KEY=xxx python evaluate.py

This will read test queries from queries.csv, get the sawt response, then evaluate the response according
to several metrics as implemented by the deepeval library <https://github.com/confident-ai/deepeval/>

"""
import logging
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))


import csv
from deepeval import assert_test, evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import AnswerRelevancyMetric, BiasMetric, ContextualRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from inquirer import route_question
from helper import get_dbs
from api import RESPONSE_TYPE_DEPTH
from tqdm import tqdm

logger = logging.getLogger(__name__)

# model to use for evaluating responses
MODEL = 'gpt-3.5-turbo-1106'

def get_test_cases():
    """
    Run sawt on all test queries and create LLMTestCases for each.
    """
    test_cases = []
    db_fc, db_cj, db_pdf, db_pc, db_news, voting_roll_df = get_dbs()
    logger.info('generating answers to all test queries...')
    for query in open('queries.csv'):
        query = query.strip()
        actual_output, retrieval_context = route_question(
            voting_roll_df,
            db_fc,
            db_cj,
            db_pdf,
            db_pc,
            db_news,
            query,
            RESPONSE_TYPE_DEPTH,
            k=5,
            return_context=True
        )
        # get single string for text response.
        actual_output = ' '.join(i['response'] for i in actual_output['responses'])    
        test_cases.append(LLMTestCase(input=query, actual_output=actual_output, retrieval_context=[retrieval_context]))
    return EvaluationDataset(test_cases=test_cases)


dataset = get_test_cases()
dataset.evaluate([
                    AnswerRelevancyMetric(threshold=0.2, model=MODEL),
                    BiasMetric(threshold=0.5, model=MODEL),
                    ContextualRelevancyMetric(threshold=0.7, include_reason=True, model=MODEL),
                    FaithfulnessMetric(threshold=0.7, include_reason=True, model=MODEL),
                    GEval(name="Readability",
                        criteria="Determine whether the text in 'actual output' is easy to read for those with a high school reading level.",
                        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
                        model=MODEL)
                  ])

