
import json
import pandas as pd

from similarity import Similarities

raw_articles = []
with open('../../3_production/output/20180523_articles.json', encoding='utf-8') as f:
	for line in f:
		raw_articles.append(json.loads(line))

df = pd.DataFrame(raw_articles)

print("\nTesting similarities with 20180523 articles ...")

sim = Similarities(df)
sim.build_similarity_matrix()
# sim.find_similar_articles()
sim.return_similar_articles(threshold=0.5, verbose=True)