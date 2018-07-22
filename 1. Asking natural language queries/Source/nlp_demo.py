from crawler.crawler import Crawler, CrawlerCache
#import crawler.crawler

# https://en.wikipedia.org/wiki/Artificial_intelligence
cache = CrawlerCache('crawler.db')
domain = 'en.wikipedia.org'
base_url = 'wiki/Artificial_intelligence'

# *** ONLY run this if the cache is empty or it's a new domain!
crawler = Crawler(cache, depth=2) #, save='pdf')
crawler.crawl('https://{}/{}'.format(domain,base_url))

for key in crawler.content[domain].keys():
    print (key)

#page = crawler.content['www.gov.uk']['/vat-record-keeping']
#print (page)