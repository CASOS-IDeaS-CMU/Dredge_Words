# dredge_words
Bridging Social Media and Search Engines: Dredge Words and the Detection of Unreliable Domains

```
@article{williams2024bridging,
  title={Bridging Social Media and Search Engines: Dredge Words and the Detection of Unreliable Domains},
  author={Williams, Evan M and Carragher, Peter and Carley, Kathleen M},
  journal={arXiv preprint arXiv:2406.11423},
  year={2024}
}
```
### Data

## Any usage of this data must also site `ahrefs.com`, the originator of many of these fields.

Consistent with Twitter/X ToS, we cannot release any Twitter data used in this study. However, we release domain attribute data and dredge word SERPs extracted from Ahrefs.
 
In data/DredgeWordsData, we release the following three files:

`attributes.csv` are website level attributes extracted from the SEO toolkit Ahrefs.

`backlinks.csv` we extracted the 10 domains that most backlink to each labeled target domain. This is an edge list where each row contains an edge from `domain_from` to `domain_to`. Links is the number of links/ domain-level edge weight. Unique_pages contain the number of unique webpages on `domain_from`.

`dredge_serps.csv` contains dredge words (in `qry` column) and their top 10 SERP results. Most columns are identical to those in WebSearcher (https://github.com/gitronald/WebSearcher). We added a final conditional column `target_domain` which is 1 when the returned domain is the same as the domain associated with the dredge word phrase.

## GNNs

We release GNN scripts in gnn_models. We release the code for transparency, but these models cannot be run without the requisite Twitter data.

### Discovery Process
```
cd dredge_discovery/
conda create -n dredge_words python=3.10
conda activate dredge_words
pip3 install -r requirements.txt
```

The [discovery process](dredge_discovery/discovery.ipynb) requires several components:
1. List of unreliable domains
2. The list of keyphrases for which those domains rank highly
3. SERP results scraped from Google for those keyphrases ([serp scraping script](dredge_discovery/serp_pull.py))
4. SEO attributes for each URL on these SERP results ([data collection script](https://github.com/CASOS-IDeaS-CMU/Detection-and-Discovery-of-Misinformation-Sources/tree/master/data_collection))
5. Classifiers trained on to label domains as reliable or unreliable based on SEO data ([training script](https://github.com/CASOS-IDeaS-CMU/Detection-and-Discovery-of-Misinformation-Sources/blob/master/flat_models/train_classifiers.ipynb)). [Pretrained models](pretrained_models/) from this codebase are supplied.

If you use the SEO data collection, classifier training scripts, or pretrained classifiers from [the previous paper](https://github.com/CASOS-IDeaS-CMU/Detection-and-Discovery-of-Misinformation-Sources/tree/master), please consider citing it:
```
@article{carragher2024detection,
  title={Detection and Discovery of Misinformation Sources using Attributed Webgraphs},
  author={Carragher, Peter and Williams, Evan M and Carley, Kathleen M},
  journal={arXiv preprint arXiv:2401.02379},
  year={2024}
}
```