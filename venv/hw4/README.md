## NLP HW4 README
* There are three python files: `crawler.py`, `tester.py` and `CS372_HW4_code_20150860.py`. The last one creates the extractor and uses the test script defined in `tester.py`. 
* Run `crawler.py` to crawl PUBMED and search for articles that contain relevant verbs in their abstracts, starting from year 2020. This may take a few hours.
* Required pip modules are listed in the `requirements.txt` file.
* The output csv file contains a list of articles and their ids, years, organizations, sentences, triples and the titles.
* In the tagged corpus, triples have types that range from -1 to 5. Each of them mean the following:  
-1: unidentified case (because I already had enough sentence from that year)  
0: no triple contained in the sentence  
1: verb `activate`  
2: verb `inhibit`  
3: verb `bind`  
4: positive verbs (one of `accelerate`, `augment`, `induce`, `stimulate`, `require`, `up-regulate`)  
5: negative verbs (one of `abolish`, `block`, `down-regulate`, `prevent`)