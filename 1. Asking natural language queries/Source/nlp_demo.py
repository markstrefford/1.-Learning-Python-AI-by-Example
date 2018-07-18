from crawler.crawler import Crawler, CrawlerCache
#import crawler.crawler

cache = CrawlerCache('crawler.db')
domain = 'arxiv.org'
base_url = 'list/cs/1807'

# *** ONLY run this if the cache is empty or it's a new domain!
crawler = Crawler(cache, depth=2)
crawler.crawl('https://{}/{}'.format(domain,base_url))

for key in crawler.content[domain].keys():
    print (key)

#page = crawler.content['www.gov.uk']['/vat-record-keeping']
#print (page)