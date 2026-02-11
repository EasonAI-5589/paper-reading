#!/bin/bash
# BibTeX 验证脚本 - 使用 Semantic Scholar API 交叉验证
# 用法: ./verify_bibtex.sh [论文目录]
# 如果不传参数，验证所有论文

BASE="/mnt/eason/paper-reading"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

verify_paper() {
    local dir="$1"
    local readme="$dir/README.md"
    local name=$(basename "$dir")
    
    if [ ! -f "$readme" ]; then
        echo -e "${YELLOW}⚠️  $name: 无 README.md${NC}"
        return
    fi
    
    # 提取 BibTeX 中的 arXiv ID
    local arxiv_id=$(grep -oP 'arXiv:\K[0-9.]+|eprint=\{\K[0-9.]+|arXiv:.*?(\K[0-9]{4}\.[0-9]{4,5})' "$readme" 2>/dev/null | head -1)
    
    # 提取 BibTeX 中的标题
    local bib_title=$(grep 'title=' "$readme" 2>/dev/null | head -1 | sed 's/.*{\(.*\)}.*/\1/' | sed 's/,$//')
    
    if [ -z "$arxiv_id" ] && [ -z "$bib_title" ]; then
        echo -e "${YELLOW}⚠️  $name: BibTeX 中无 arXiv ID 或标题${NC}"
        return
    fi
    
    # 用 Semantic Scholar API 查询
    local api_url=""
    if [ -n "$arxiv_id" ]; then
        api_url="https://api.semanticscholar.org/graph/v1/paper/ArXiv:${arxiv_id}?fields=title,authors,year,venue,externalIds,citationCount"
    else
        # 用标题搜索
        local encoded_title=$(python3 -c "import urllib.parse; print(urllib.parse.quote('$bib_title'))")
        api_url="https://api.semanticscholar.org/graph/v1/paper/search?query=${encoded_title}&limit=1&fields=title,authors,year,venue,externalIds,citationCount"
    fi
    
    local response=$(curl -s "$api_url")
    
    # 解析返回结果
    if echo "$response" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'title' in d or 'data' in d" 2>/dev/null; then
        # 提取 API 返回的元数据
        local api_data=$(echo "$response" | python3 -c "
import sys, json
d = json.load(sys.stdin)
if 'data' in d and d['data']:
    d = d['data'][0]
title = d.get('title', 'N/A')
year = d.get('year', 'N/A')
venue = d.get('venue', 'N/A')
citations = d.get('citationCount', 0)
authors = ', '.join([a['name'] for a in d.get('authors', [])[:5]])
if len(d.get('authors', [])) > 5:
    authors += ', ...'
arxiv = d.get('externalIds', {}).get('ArXiv', 'N/A')
print(f'TITLE: {title}')
print(f'AUTHORS: {authors}')
print(f'YEAR: {year}')
print(f'VENUE: {venue}')
print(f'ARXIV: {arxiv}')
print(f'CITATIONS: {citations}')
" 2>/dev/null)
        
        local api_title=$(echo "$api_data" | grep "^TITLE:" | sed 's/TITLE: //')
        local api_authors=$(echo "$api_data" | grep "^AUTHORS:" | sed 's/AUTHORS: //')
        local api_year=$(echo "$api_data" | grep "^YEAR:" | sed 's/YEAR: //')
        local api_arxiv=$(echo "$api_data" | grep "^ARXIV:" | sed 's/ARXIV: //')
        local api_citations=$(echo "$api_data" | grep "^CITATIONS:" | sed 's/CITATIONS: //')
        local api_venue=$(echo "$api_data" | grep "^VENUE:" | sed 's/VENUE: //')
        
        # 提取 BibTeX 中的作者和年份
        local bib_year=$(grep 'year=' "$readme" 2>/dev/null | head -1 | grep -oP '\d{4}')
        local bib_authors=$(grep 'author=' "$readme" 2>/dev/null | head -1 | sed 's/.*{\(.*\)}.*/\1/' | sed 's/,$//')
        
        echo -e "${GREEN}📄 $name${NC}"
        echo "  BibTeX  标题: $bib_title"
        echo "  API     标题: $api_title"
        
        # 比对年份
        if [ "$bib_year" = "$api_year" ]; then
            echo -e "  年份: ${GREEN}✅ $bib_year${NC}"
        else
            echo -e "  年份: ${RED}❌ BibTeX=$bib_year API=$api_year${NC}"
        fi
        
        # 比对 arXiv ID
        if [ -n "$arxiv_id" ] && [ "$arxiv_id" = "$api_arxiv" ]; then
            echo -e "  arXiv: ${GREEN}✅ $arxiv_id${NC}"
        elif [ -n "$arxiv_id" ]; then
            echo -e "  arXiv: ${RED}❌ BibTeX=$arxiv_id API=$api_arxiv${NC}"
        fi
        
        echo "  API 作者: $api_authors"
        echo "  BibTeX 作者: $(echo $bib_authors | cut -c1-80)..."
        echo "  Venue: $api_venue"
        echo "  Citations: $api_citations"
        echo ""
    else
        echo -e "${RED}❌ $name: Semantic Scholar API 查询失败${NC}"
        echo "  arXiv ID: $arxiv_id"
        echo "  Response: $(echo $response | head -c 200)"
        echo ""
    fi
}

echo "=========================================="
echo "  BibTeX 验证 (via Semantic Scholar API)"
echo "=========================================="
echo ""

if [ -n "$1" ]; then
    verify_paper "$1"
else
    # 验证所有论文
    for d in "$BASE"/Agent-Memory/*/; do
        verify_paper "$d"
        sleep 1  # API rate limit
    done
    for d in "$BASE"/\[*\]/; do
        verify_paper "$d"
        sleep 1
    done
fi
