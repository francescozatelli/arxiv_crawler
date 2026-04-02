"""
Usage: python fetch_arxiv.py config_filename
add -d for debugging (log beginning of each stage)
"""

import json
import os
import random
import re
import ssl
import sys
import time
import unicodedata
from datetime import datetime
from urllib.parse import quote, unquote_plus
from urllib.request import Request, urlopen

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


def _extract_category_terms(search_query):
    """Extract cat: terms from one search query string."""
    categories = []
    for chunk in search_query.split('+AND+'):
        if not chunk.startswith('cat:'):
            continue

        category = chunk[4:].strip()
        if category:
            categories.append(category)
    return categories


def _build_author_token_fallback_query(search_query, author_terms):
    """Build an author-token fallback query for zero-result quoted author searches.

    Example:
      au:%22John%20Preskill%22+AND+cat:quant-ph
      -> au:john+AND+au:preskill+AND+cat:quant-ph
    """
    if not author_terms:
        return None

    fallback_terms = []
    for term in author_terms:
        term_tokens = _normalize_tokens(term)
        if not term_tokens:
            continue

        # Keep the query narrow: first token + surname token.
        if len(term_tokens) == 1:
            fallback_terms.append(f'au:{quote(term_tokens[0], safe="")}')
        else:
            fallback_terms.append(f'au:{quote(term_tokens[0], safe="")}')
            fallback_terms.append(f'au:{quote(term_tokens[-1], safe="")}')

    category_terms = [f'cat:{cat}' for cat in _extract_category_terms(search_query)]
    if not fallback_terms:
        return None

    parts = fallback_terms + category_terms
    return '+AND+'.join(parts)


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

    # Multi-token query: support both "Given Surname" and "Surname Given".
    if term_tokens[-1] == surname:
        given_query_tokens = term_tokens[:-1]
    elif term_tokens[0] == surname:
        given_query_tokens = term_tokens[1:]
    else:
        return False

    for token in given_query_tokens:
        if len(token) == 1:
            if token not in given_initials:
                return False
        else:
            # Accept full-token equality and full-name vs initial equivalence.
            if token not in given_tokens and token[0] not in given_initials:
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
    """Build balanced name variants for arXiv author queries.

    Default variants are:
    - FirstName LastName
    - F LastName
    Optionally include LastName-only when require_given_name is false.
    """
    last_name = str(author.get('last_name', '')).strip()
    first_name = str(author.get('first_name', '')).strip()
    require_given_name = bool(author.get('require_given_name', False))

    if not last_name:
        return []

    variants = []

    def _add_variant(name):
        clean = ' '.join(name.split())
        if clean and clean not in variants:
            variants.append(clean)

    # Default behavior includes surname-only query, but this can be disabled for
    # common surnames where false positives are likely.
    if not require_given_name:
        _add_variant(last_name)

    if first_name:
        first_initial = first_name[0]

        # Full first + last, and first initial + last.
        _add_variant(f"{first_name} {last_name}")
        _add_variant(f"{first_initial} {last_name}")

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
                    "require_given_name": false,  # optional; when true skip surname-only
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


def _is_ssl_cert_error(exc):
    """Return True when the parser exception is an SSL cert verification failure."""
    if exc is None:
        return False
    return 'CERTIFICATE_VERIFY_FAILED' in repr(exc)


def _is_rate_limited_error(exc):
    """Return True when the parser exception indicates HTTP 429 rate limiting."""
    if exc is None:
        return False
    text = repr(exc)
    return 'HTTPError 429' in text or '429' in text


def _is_transient_parse_error(exc):
    """Return True for transient malformed XML/stream errors."""
    if exc is None:
        return False
    text = repr(exc)
    return 'SAXParseException' in text or 'not well-formed' in text or 'syntax error' in text


def _download_bytes(url, timeout=30, insecure_ssl=False):
    """Download raw response bytes with a stable User-Agent."""
    request = Request(url, headers={'User-Agent': 'arxiv-crawler/1.0 (+github-actions)'})
    if insecure_ssl:
        insecure_ctx = ssl.create_default_context()
        insecure_ctx.check_hostname = False
        insecure_ctx.verify_mode = ssl.CERT_NONE
        with urlopen(request, context=insecure_ctx, timeout=timeout) as response:
            return response.read()

    with urlopen(request, timeout=timeout) as response:
        return response.read()


def _parse_arxiv_feed(url, max_attempts=3, base_sleep_seconds=1.0):
    """Parse one arXiv API URL with retry and targeted SSL fallback.

    Returns:
        feedparser.FeedParserDict

    Raises:
        RuntimeError: if parsing fails after retries/fallback.
    """
    last_error = None

    for attempt in range(1, max_attempts + 1):
        parsed = feedparser.parse(url)
        bozo_exception = getattr(parsed, 'bozo_exception', None)
        entry_count = len(getattr(parsed, 'entries', []))

        if not getattr(parsed, 'bozo', False):
            return parsed

        # Some bozo parser states still contain valid entries; accept those.
        if entry_count > 0:
            print(f'Warning: bozo parse with {entry_count} entries, continuing: {bozo_exception!r}')
            return parsed

        # Handle local trust-store issues with a targeted fallback.
        if _is_ssl_cert_error(bozo_exception):
            try:
                parsed_insecure = feedparser.parse(_download_bytes(url, insecure_ssl=True))
                if not getattr(parsed_insecure, 'bozo', False):
                    print('Warning: SSL verification failed, used insecure fallback for arXiv API.')
                    return parsed_insecure
                last_error = getattr(parsed_insecure, 'bozo_exception', bozo_exception)
            except Exception as exc:
                last_error = exc
        elif _is_rate_limited_error(bozo_exception):
            # arXiv API throttling: wait longer before retrying.
            last_error = bozo_exception
            if attempt < max_attempts:
                time.sleep(max(10.0, base_sleep_seconds * (2 ** attempt)))
                continue
        elif _is_transient_parse_error(bozo_exception):
            # Retry once with explicit byte download and then back off.
            try:
                parsed_retry = feedparser.parse(_download_bytes(url))
                if not getattr(parsed_retry, 'bozo', False):
                    return parsed_retry
                if len(getattr(parsed_retry, 'entries', [])) > 0:
                    print('Warning: transient parse error, recovered entries on byte retry.')
                    return parsed_retry
                last_error = getattr(parsed_retry, 'bozo_exception', bozo_exception)
            except Exception as exc:
                last_error = exc

            if attempt < max_attempts:
                time.sleep(max(10.0, base_sleep_seconds * (2 ** attempt)))
                continue
        else:
            last_error = bozo_exception

        if attempt < max_attempts:
            sleep_seconds = base_sleep_seconds * attempt
            time.sleep(sleep_seconds)

    raise RuntimeError(f'arXiv API parse failed after {max_attempts} attempts: {last_error!r}')


def _truncate_for_log(text, max_len=140):
    """Truncate long query strings to keep CI logs readable."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + '...'


def _sleep_with_jitter(base_seconds, jitter_seconds):
    """Sleep with symmetric jitter while keeping delay non-negative."""
    delay = max(0.0, base_seconds + random.uniform(-jitter_seconds, jitter_seconds))
    time.sleep(delay)


def _execute_query(base_url, search_query, start, max_results, sorting_order):
    """Execute one arXiv query with fallback and return structured results."""
    author_terms = _extract_author_terms(search_query)
    api_query = f'search_query={search_query}&start={start}&max_results={max_results}'

    try:
        parsed = _parse_arxiv_feed(base_url + api_query + sorting_order, max_attempts=5, base_sleep_seconds=3.0)
    except RuntimeError as exc:
        return {
            'status': 'failed',
            'query': search_query,
            'error': repr(exc),
            'raw_entry_count': 0,
            'matched_entry_count': 0,
            'sample_ids': [],
            'rows': [],
            'author_terms': author_terms,
            'suspicious_zero': False,
        }

    raw_entry_count = len(parsed.entries)

    # arXiv occasionally returns empty result sets for quoted author queries.
    # Retry once with tokenized author terms while preserving strict author
    # post-filtering to avoid false positives.
    if raw_entry_count == 0 and author_terms:
        fallback_query = _build_author_token_fallback_query(search_query, author_terms)
        if fallback_query and fallback_query != search_query:
            fallback_api_query = f'search_query={fallback_query}&start={start}&max_results={max_results}'
            try:
                parsed_fallback = _parse_arxiv_feed(
                    base_url + fallback_api_query + sorting_order,
                    max_attempts=3,
                    base_sleep_seconds=3.0,
                )
                fallback_count = len(parsed_fallback.entries)
                if fallback_count > 0:
                    print(
                        'Warning: recovered zero-result author query with token fallback'
                        f' | query={_truncate_for_log(search_query)}'
                        f' | fallback_query={_truncate_for_log(fallback_query)}'
                        f' | recovered_entries={fallback_count}'
                    )
                    parsed = parsed_fallback
                    raw_entry_count = fallback_count
            except RuntimeError:
                # Keep original zero-result response if fallback also fails.
                pass

    rows = []
    matched_entry_count = 0
    sample_ids = []

    for entry in parsed.entries:
        if not _entry_matches_author_terms(entry.authors, author_terms):
            continue

        dic_stored = {}
        dic_stored['id'] = entry.id.split('/')[-1].split('v')[0]
        matched_entry_count += 1
        if len(sample_ids) < 3:
            sample_ids.append(dic_stored['id'])
        dic_stored['author_list'] = _join_authors(entry.authors)
        dic_stored['title'] = _remove_newlines(entry.title)
        dic_stored['arxiv_primary_category'] = entry.arxiv_primary_category['term']
        dic_stored['published'] = _convert_time(entry.published)

        # replace the query with legible terms
        legible_query = (
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
            .replace("cond-mat.mes-hall", "")
            .replace("ti:", "Title : ")
        )

        dic_stored['search_query'] = str(legible_query)
        dic_stored['link'] = entry.link
        rows.append(dic_stored)

    suspicious_zero = bool(author_terms) and raw_entry_count == 0
    return {
        'status': 'ok',
        'query': search_query,
        'raw_entry_count': raw_entry_count,
        'matched_entry_count': matched_entry_count,
        'sample_ids': sample_ids,
        'rows': rows,
        'author_terms': author_terms,
        'suspicious_zero': suspicious_zero,
    }

def query_arxiv_org(query_input):
    """Search for query items on arXiv and return the list of results"""

    # Construct elements of the query string sent to arxiv.org:
    # Base api query url
    base_url = 'https://export.arxiv.org/api/query?'
    # each search item (legacy txt or structured json)
    search_keywords = _load_search_queries(query_input)
    # arXiv recommends keeping requests slow to avoid throttling.
    request_delay_seconds = 4.0
    request_delay_jitter_seconds = 0.8
    second_pass_delay_seconds = 10.0
    second_pass_delay_jitter_seconds = 1.5
    # some options
    start = 0
    max_results = 50 # see arXiv API for max result limits
    sorting_order = '&sortBy=submittedDate&sortOrder=descending'

    result_list = []
    final_failed_queries = []
    final_suspicious_zero_queries = []
    query_audit = []
    second_pass_candidates = []

    # First pass: search for the keywords/authors one by one.
    for query_idx, search_query in enumerate(search_keywords, start=1):
        if query_idx > 1:
            _sleep_with_jitter(request_delay_seconds, request_delay_jitter_seconds)

        query_result = _execute_query(base_url, search_query, start, max_results, sorting_order)

        audit_item = {
            'phase': 'pass1',
            'status': query_result['status'],
            'query': search_query,
        }

        if query_result['status'] == 'failed':
            audit_item['error'] = query_result['error']
            second_pass_candidates.append(search_query)
            print(f'Warning: scheduling second-pass retry after failure: {search_query} | {query_result["error"]}')
            query_audit.append(audit_item)
            continue

        result_list.extend(query_result['rows'])
        audit_item['raw_entry_count'] = query_result['raw_entry_count']
        audit_item['matched_entry_count'] = query_result['matched_entry_count']
        audit_item['sample_ids'] = query_result['sample_ids']
        query_audit.append(audit_item)

        if query_result['suspicious_zero']:
            second_pass_candidates.append(search_query)

    # Second pass: retry failed and suspicious zero-result author queries.
    if second_pass_candidates:
        print(
            f'Info: second-pass retry for {len(second_pass_candidates)} queries '
            '(failed or suspicious zero-result author queries).'
        )

    for retry_idx, search_query in enumerate(second_pass_candidates, start=1):
        if retry_idx > 1:
            _sleep_with_jitter(second_pass_delay_seconds, second_pass_delay_jitter_seconds)

        query_result = _execute_query(base_url, search_query, start, max_results, sorting_order)
        audit_item = {
            'phase': 'pass2',
            'status': query_result['status'],
            'query': search_query,
        }

        if query_result['status'] == 'failed':
            final_failed_queries.append((search_query, query_result['error']))
            audit_item['error'] = query_result['error']
            print(f'Warning: skipping query after second-pass retries: {search_query} | {query_result["error"]}')
            query_audit.append(audit_item)
            continue

        audit_item['raw_entry_count'] = query_result['raw_entry_count']
        audit_item['matched_entry_count'] = query_result['matched_entry_count']
        audit_item['sample_ids'] = query_result['sample_ids']
        query_audit.append(audit_item)

        if query_result['rows']:
            result_list.extend(query_result['rows'])

        if query_result['suspicious_zero']:
            final_suspicious_zero_queries.append(search_query)

    # Per-query diagnostics: helps detect silent misses in successful runs.
    for item in query_audit:
        if item['status'] == 'failed':
            print(
                'Query audit | status=failed'
                f' | phase={item.get("phase", "pass1")}'
                f' | query={_truncate_for_log(item["query"])}'
                f' | error={item["error"]}'
            )
            continue

        sample_ids_text = ','.join(item['sample_ids']) if item['sample_ids'] else '-'
        print(
            'Query audit | status=ok'
            f' | phase={item.get("phase", "pass1")}'
            f' | raw_entries={item["raw_entry_count"]}'
            f' | matched_entries={item["matched_entry_count"]}'
            f' | sample_ids={sample_ids_text}'
            f' | query={_truncate_for_log(item["query"])}'
        )

    if final_suspicious_zero_queries:
        print(
            'Warning: '
            f'{len(final_suspicious_zero_queries)} author queries returned zero raw entries '
            'even after second pass.'
        )

    if final_failed_queries:
        print(f'Warning: {len(final_failed_queries)} queries failed and were skipped.')
        # If every query failed, fail loudly because output is unusable.
        if len(final_failed_queries) == len(search_keywords):
            raise RuntimeError('All arXiv queries failed; aborting crawl.')

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
