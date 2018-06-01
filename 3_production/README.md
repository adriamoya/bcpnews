# Production

### Crawl and download

Spiders, multiple crawling running and downloading articles

* expansion
* cincodias
* elconfidencial
* eleconomista

Output results report included

```shell
python main.py YYYYMMDD
```
### Validation

To check results only open `python` console and type

 ```python
from utils.check_output import check_output

check_output('./output/YYYYMMDD_articles.json')
 ```
