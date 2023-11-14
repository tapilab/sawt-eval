import logging
import time
import math

import google.cloud.logging
import functions_framework
from supabase import create_client

from helper import parse_field, get_dbs
from inquirer import answer_query
import os

logging_client = google.cloud.logging.Client()
logging_client.setup_logging()

API_VERSION = "0.0.1"

db_general, db_in_depth, voting_roll_df = get_dbs()

# Setup Supabase client
supabase_url = os.environ.get("SUPABASE_URL_STAGING")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY_STAGING")
supabase = create_client(supabase_url, supabase_key)

def update_supabase(response, query):
    # Assume you have a table named 'answers' with a column named 'answer'
    response = supabase.table('cards').insert({'title': query, 'responses': response}).execute()
    if response.error:
        logging.error(f"Failed to update Supabase: {response.error}")

@functions_framework.http
def getanswer(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        A success message and status, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """
    # Set CORS headers for the preflight request
    if request.method == "OPTIONS":
        # Allows POST requests from any origin with the Content-Type
        # header and caches preflight response for an 3600s
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "3600",
        }

        return ("", 204, headers)

    # Set CORS headers for the main request
    headers = {"Access-Control-Allow-Origin": "*"}

    logging.info(f"getanswer {API_VERSION}")
    start = time.time()

    # Parse args
    content_type = request.headers["Content-Type"]
    if content_type == "application/json":
        request_json = request.get_json(silent=True)
        logging.info(request_json)

        query = parse_field(request_json, "query")
        response_type = parse_field(request_json, "response_type")
    else:
        raise ValueError("Unknown content type: {}".format(content_type))
    
    logging.info("Request parsed")

    answer = answer_query(query, response_type, voting_roll_df, db_general, db_in_depth)

    # Update Supabase instead of returning the answer to the client
    update_supabase(answer, query)

    end = time.time()
    elapsed = math.ceil(end - start)
    logging.info(f"Completed getanswer in {elapsed} seconds")
    print(f"\n\t--------- Completed getanswer in {elapsed} seconds --------\n")

    return ("Answer successfully submitted to Supabase", 200, headers)
