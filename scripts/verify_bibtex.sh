#!/usr/bin/env bash

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'
REQUEST_DELAY="${REQUEST_DELAY:-1}"

verify_paper() {
    local dir="$1"
    local readme="$dir/README.md"
    local name
    local arxiv_id
    local bib_title
    local api_url
    local response

    name="$(basename "$dir")"

    if [ ! -f "$readme" ]; then
        printf "%b\n" "${YELLOW}WARN  $name: no README.md${NC}"
        return
    fi

    arxiv_id="$(sed -nE 's/.*(arXiv:|eprint[[:space:]]*=[[:space:]]*\{)([0-9]{4}\.[0-9]{4,5}).*/\2/p' "$readme" | head -1)"
    bib_title="$(sed -nE 's/^[[:space:]]*title[[:space:]]*=[[:space:]]*\{(.*)\},?[[:space:]]*$/\1/p' "$readme" | head -1)"

    if [ -z "$arxiv_id" ] && [ -z "$bib_title" ]; then
        printf "%b\n" "${YELLOW}WARN  $name: BibTeX has no arXiv ID or title${NC}"
        return
    fi

    if [ -n "$arxiv_id" ]; then
        api_url="https://api.semanticscholar.org/graph/v1/paper/ArXiv:${arxiv_id}?fields=title,authors,year,venue,externalIds,citationCount"
    else
        local encoded_title
        encoded_title="$(python3 -c 'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1]))' "$bib_title")"
        api_url="https://api.semanticscholar.org/graph/v1/paper/search?query=${encoded_title}&limit=1&fields=title,authors,year,venue,externalIds,citationCount"
    fi

    if ! response="$(curl --fail --silent --show-error --max-time 30 "$api_url")"; then
        printf "%b\n\n" "${RED}FAIL  $name: Semantic Scholar request failed${NC}"
        return
    fi

    if ! printf '%s' "$response" | python3 -c "import json,sys; d=json.load(sys.stdin); assert 'title' in d or d.get('data')" 2>/dev/null; then
        printf "%b\n" "${RED}FAIL  $name: invalid or empty Semantic Scholar response${NC}"
        printf "  Response: %.200s\n\n" "$response"
        return
    fi

    local api_data
    local api_title
    local api_authors
    local api_year
    local api_arxiv
    local api_citations
    local api_venue
    local bib_year
    local bib_authors

    api_data="$(printf '%s' "$response" | python3 -c '
import json, sys
d = json.load(sys.stdin)
if "data" in d:
    d = d["data"][0]
authors = ", ".join(a["name"] for a in d.get("authors", [])[:5])
if len(d.get("authors", [])) > 5:
    authors += ", ..."
print("TITLE: " + str(d.get("title", "N/A")))
print("AUTHORS: " + authors)
print("YEAR: " + str(d.get("year", "N/A")))
print("VENUE: " + str(d.get("venue", "N/A")))
print("ARXIV: " + str(d.get("externalIds", {}).get("ArXiv", "N/A")))
print("CITATIONS: " + str(d.get("citationCount", 0)))
')"

    api_title="$(printf '%s\n' "$api_data" | sed -n 's/^TITLE: //p')"
    api_authors="$(printf '%s\n' "$api_data" | sed -n 's/^AUTHORS: //p')"
    api_year="$(printf '%s\n' "$api_data" | sed -n 's/^YEAR: //p')"
    api_arxiv="$(printf '%s\n' "$api_data" | sed -n 's/^ARXIV: //p')"
    api_citations="$(printf '%s\n' "$api_data" | sed -n 's/^CITATIONS: //p')"
    api_venue="$(printf '%s\n' "$api_data" | sed -n 's/^VENUE: //p')"
    bib_year="$(sed -nE 's/.*year[[:space:]]*=[[:space:]]*\{?([0-9]{4}).*/\1/p' "$readme" | head -1)"
    bib_authors="$(sed -nE 's/^[[:space:]]*author[[:space:]]*=[[:space:]]*\{(.*)\},?[[:space:]]*$/\1/p' "$readme" | head -1)"

    printf "%b\n" "${GREEN}PAPER $name${NC}"
    printf "  BibTeX title: %s\n  API title:    %s\n" "$bib_title" "$api_title"

    if [ -n "$bib_year" ] && [ "$bib_year" = "$api_year" ]; then
        printf "%b\n" "  Year: ${GREEN}OK $bib_year${NC}"
    else
        printf "%b\n" "  Year: ${RED}MISMATCH BibTeX=${bib_year:-N/A} API=$api_year${NC}"
    fi

    if [ -n "$arxiv_id" ]; then
        if [ "$arxiv_id" = "$api_arxiv" ]; then
            printf "%b\n" "  arXiv: ${GREEN}OK $arxiv_id${NC}"
        else
            printf "%b\n" "  arXiv: ${RED}MISMATCH BibTeX=$arxiv_id API=$api_arxiv${NC}"
        fi
    fi

    printf "  API authors: %s\n" "$api_authors"
    printf "  BibTeX authors: %.80s%s\n" "$bib_authors" "$([ "${#bib_authors}" -gt 80 ] && printf '...')"
    printf "  Venue: %s\n  Citations: %s\n\n" "$api_venue" "$api_citations"
}

printf '%s\n\n' 'BibTeX verification via Semantic Scholar'

if [ "$#" -gt 0 ]; then
    if [ ! -d "$1" ]; then
        printf "%b\n" "${RED}Directory not found: $1${NC}" >&2
        exit 2
    fi
    verify_paper "$(cd "$1" && pwd)"
else
    while IFS= read -r -d '' readme; do
        verify_paper "$(dirname "$readme")"
        sleep "$REQUEST_DELAY"
    done < <(find "$REPO_ROOT" -type f -name README.md -not -path '*/.git/*' -print0 | sort -z)
fi
