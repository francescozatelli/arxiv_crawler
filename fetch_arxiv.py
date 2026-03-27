"""
Usage: python fetch_arxiv.py config_filename
add -d for debugging (log beginning of each stage)
"""

import json
import os
import re
import sys
import unicodedata
from datetime import datetime
from urllib.parse import quote, unquote_plus

import feedparser
import pandas as pd

import database_manipulation as dbmanip


def _convert_time(val):
    """Changes the date-time string format"""
    date = datetime.strptime(val,'%Y-%m-%dT%H:%M:%SZ')
    return date.strftime("%Y-%m-%d %H:%M:%S")

def _remove_newlines(val):
    """Strips line breaks from the title string"""
    return val.replace('\n  ', ' ')

def _join_authors(val):
    """Makes a single string as the author list"""
    return ', '.join([val[i]['name'] for i in range(len(val))])


def _normalize_tokens(text):
    """Normalize a string into lowercase alphanumeric tokens."""
    normalized = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    return re.findall(r"[a-z0-9]+", normalized.lower())


def _extract_author_terms(search_query):
    """Extract au: terms from one search query string."""
    terms = []
    for chunk in search_query.split('+AND+'):
        if not chunk.startswith('au:'):
            continue

        raw = unquote_plus(chunk[3:].strip())
        raw = raw.replace('"', '').replace('%22', '')
        # Guard against malformed lines like au:katsaros+cat:cond-mat.supr-con
        raw = raw.split('+cat:')[0].strip()
        if raw:
            terms.append(raw)
    return terms


def _author_term_matches_name(author_term, author_name):
    """Check whether one au: term matches one concrete author name."""
    term_tokens = _normalize_tokens(author_term)
    name_tokens = _normalize_tokens(author_name)

    if not term_tokens or not name_tokens:
        return False

    surname = name_tokens[-1]
    given_tokens = name_tokens[:-1]
    given_initials = {token[0] for token in given_tokens if token}

    # Single-token author query is treated as surname query only.
    if len(term_tokens) == 1:
        token = term_tokens[0]
        # Ignore one-letter surname queries (too broad, e.g. "g").
        if len(token) == 1:
            return False
        return token == surname

    # Multi-token query: last token must be surname.
    if term_tokens[-1] != surname:
        return False

    for token in term_tokens[:-1]:
        if len(token) == 1:
            if token not in given_initials:
                return False
        else:
            if token not in given_tokens:
                return False

    return True


def _entry_matches_author_terms(entry_authors, author_terms):
    """Ensure every author query term matches at least one listed author."""
    if not author_terms:
        return True

    author_names = [entry_authors[i]['name'] for i in range(len(entry_authors))]
    return all(
        any(_author_term_matches_name(term, author_name) for author_name in author_names)
        for term in author_terms
    )


def _split_middle_names(middle_names):
    """Normalize optional middle names to a list of tokens."""
    if not middle_names:
        return []
    if isinstance(middle_names, list):
        out = []
        for item in middle_names:
            if isinstance(item, str):
                out.extend(item.split())
        return [token for token in out if token]
    if isinstance(middle_names, str):
        return [token for token in middle_names.split() if token]
    return []


def _build_author_name_variants(author):
    """Build name variants such as 'Vandersypen', 'L Vandersypen', etc."""
    last_name = str(author.get('last_name', '')).strip()
    first_name = str(author.get('first_name', '')).strip()
    middle_names = _split_middle_names(author.get('middle_name', author.get('middle_names')))

    if not last_name:
        return []

    variants = []

    def _add_variant(name):
        clean = ' '.join(name.split())
        if clean and clean not in variants:
            variants.append(clean)

    # Last name only.
    _add_variant(last_name)

    if first_name:
        first_initial = first_name[0]

        # Full first + last, and first initial + last.
        _add_variant(f"{first_name} {last_name}")
        _add_variant(f"{first_initial} {last_name}")

        if middle_names:
            # Full names.
            _add_variant(f"{first_name} {' '.join(middle_names)} {last_name}")

            middle_initials = [name[0] for name in middle_names if name]
            # Progressive initial variants: L M Last, L M K Last, ...
            for idx in range(1, len(middle_initials) + 1):
                _add_variant(f"{first_initial} {' '.join(middle_initials[:idx])} {last_name}")

            # First name + middle initials + last.
            _add_variant(f"{first_name} {' '.join(middle_initials)} {last_name}")

    return variants


def _author_variant_to_query_term(variant):
    """Encode one author variant into an arXiv quoted author query term."""
    return f'au:%22{quote(variant, safe="")}%22'


def _sanitize_query_string(query):
    """Fix common malformed query fragments before calling arXiv API."""
    cleaned = query.strip()
    # Fix typo pattern like "...+au:katsaros+cat:cond-mat.supr-con".
    cleaned = cleaned.replace('+cat:', '+AND+cat:')
    # Avoid accidental duplicate AND after repeated sanitization.
    cleaned = cleaned.replace('+AND+AND+', '+AND+')
    return cleaned


def _load_search_queries(query_input):
    """Load queries from legacy txt format or structured json format.

    JSON schema:
    {
      "custom_queries": ["all:majorana+AND+cat:cond-mat.mes-hall", ...],
      "authors": [
        {
          "last_name": "Vandersypen",
          "first_name": "Lieven",
          "middle_name": "M K",   # optional, string or list
          "categories": ["cond-mat.mes-hall", "cond-mat.supr-con"]
        }
      ]
    }
    """
    _, ext = os.path.splitext(query_input)
    ext = ext.lower()

    queries = []

    if ext == '.json':
        with open(query_input) as file:
            payload = json.load(file)

        for query in payload.get('custom_queries', []):
            if isinstance(query, str) and query.strip():
                queries.append(_sanitize_query_string(query))

        for author in payload.get('authors', []):
            if not isinstance(author, dict):
                continue

            categories = author.get('categories', [])
            if isinstance(categories, str):
                categories = [categories]

            categories = [cat.strip() for cat in categories if isinstance(cat, str) and cat.strip()]
            if not categories:
                continue

            variants = _build_author_name_variants(author)
            for category in categories:
                for variant in variants:
                    term = _author_variant_to_query_term(variant)
                    queries.append(f'{term}+AND+cat:{category}')
    else:
        with open(query_input) as file:
            for line in file.readlines():
                stripped = line.strip()
                if stripped:
                    queries.append(_sanitize_query_string(stripped))

    # Deduplicate while preserving order.
    unique_queries = []
    seen = set()
    for query in queries:
        if query not in seen:
            seen.add(query)
            unique_queries.append(query)

    return unique_queries

def query_arxiv_org(query_input):
    """Search for query items on arXiv and return the list of results"""

    # Construct elements of the query string sent to arxiv.org:
    # Base api query url
    base_url = 'https://export.arxiv.org/api/query?'
    # each search item (legacy txt or structured json)
    search_keywords = _load_search_queries(query_input)
    # some options
    start = 0
    max_results = 50 # see arXiv API for max result limits
    sorting_order = '&sortBy=submittedDate&sortOrder=descending'

    result_list = []

    # search for the keywords/authors one by one
    for search_query in search_keywords:
        author_terms = _extract_author_terms(search_query)
        query = f'search_query={search_query}&start={start}&max_results={max_results}'

        d = feedparser.parse(base_url+query+sorting_order) # actual querying

        for entry in d.entries:
            if not _entry_matches_author_terms(entry.authors, author_terms):
                continue

            dic_stored = {}
            dic_stored['id'] = entry.id.split('/')[-1].split('v')[0]
            dic_stored['author_list'] = _join_authors(entry.authors)
            dic_stored['title'] = _remove_newlines(entry.title)
            dic_stored['arxiv_primary_category'] = entry.arxiv_primary_category['term']
            dic_stored['published'] = _convert_time(entry.published)

            # replace the query with legible terms
            query = (
                unquote_plus(search_query)
                .replace('"', '')
                .replace('%22', '')
                .replace("+", " ")
                .replace("AND", " ")
                .replace("all:", " Content : ")
                .replace("au:", "Author : ")
                .replace("\n", "")
                .replace("cat:", " ")
                .replace("cond-mat.supr-con", "")
                .replace("cond-mat.mes-hall" , "")
                .replace("ti:", "Title : ")
            )

            
            dic_stored['search_query'] = str(query)
            dic_stored['link'] = entry.link
            result_list.append(dic_stored)

    return result_list


def main():
    """
    Usage: python fetch_arxiv.py config_filename
    """

    debug_mode = bool('-d' in sys.argv)

    # read config file
    config_file = sys.argv[1]
    assert os.path.exists(config_file), "Config file not found."
    with open(config_file) as c_f:
        configs = json.load(c_f)

    if debug_mode:
        print('Beginning query: ', datetime.now())
    result_list = query_arxiv_org(configs['query_input'])
    if debug_mode:
        print('Query successful: ', datetime.now())

    # create a new empty data frame if failed to read an existing DB with the same name
    try:
        old_db = pd.read_pickle(configs['db_output'])
    except FileNotFoundError:
        old_db = pd.DataFrame()
    
    new_db = pd.DataFrame(result_list)
    updated_db = dbmanip.update_database(old_db, new_db)
    if debug_mode:
        print('Database updated: ', datetime.now())

    pd.to_pickle(updated_db, configs['db_output'])
    if debug_mode:
        print('pkl written: ', datetime.now())

    dbmanip.create_html(updated_db, configs['html_output'])
    print(f"Done writing {configs['html_output']}: ", datetime.now())


if __name__ == '__main__':
    main()
