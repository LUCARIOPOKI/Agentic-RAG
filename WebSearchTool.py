from duckduckgo_search import DDGS

def search_DDG(query, no_result = 10):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, region="wt-wt", safesearch='moderate',max_results=no_result):
            results.append(
                {
                    'title':r.get('title'),
                    'href': r.get('href'),
                    'body':r.get('body')
                }
            )
            return results

# print(search_DDG("CEO of google"))    