"""
Learning Python AI by Example: Asking Natural Language Queries

Using Wikipedia Artificial Intelligence content to create a model
that we can query using natural language to get rich information
and responses from.


"""

# Imports
from crawler.crawl_wikipedia import CrawlWikipedia

# Configuration:
# Start with the top level Artificial Intelligence category on Wikipedia
category = 'Category:Artificial_intelligence'

# Only go down 1 level in subcategories
depth = 1

# Set up a simple database so we can use this data later
crawler = CrawlWikipedia('content.db')

# Now use the Wikipedia API to populate our database
crawler.get_categories_and_members(category, depth)



