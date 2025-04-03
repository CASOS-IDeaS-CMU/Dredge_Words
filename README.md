# Dredge Words: Bridging Social Media and Search Engines for Detecting Unreliable Domains

## Data Availability

Any usage of this dataset must also cite `ahrefs.com`, the source of many of the included fields.

In compliance with Twitter/X Terms of Service, we cannot release Twitter data used in this study. However, we provide domain attribute data and Dredge Word SERP (Search Engine Results Page) extractions obtained from Ahrefs.

### Data Files
The dataset is available in `data/DredgeWordsData/`, which contains the following files:

- **`attributes.csv`**: Website-level attributes extracted from the SEO toolkit Ahrefs.
- **`backlinks.csv`**: The top 10 domains that most frequently backlink to each labeled target domain. This is structured as an edge list, where each row contains:
  - `domain_from`: The referring domain.
  - `domain_to`: The target domain receiving backlinks.
  - `links`: The number of links (domain-level edge weight).
  - `unique_pages`: The number of unique pages on `domain_from` that link to `domain_to`.
- **`dredge_serps.csv`**: Contains Dredge Words (`qry` column) and their top 10 SERP results. Most columns align with those in [WebSearcher](https://github.com/gitronald/WebSearcher). A final column, `target_domain`, is set to `1` when the returned domain matches the domain associated with the Dredge Word phrase.

## Graph Neural Networks (GNNs)

We provide GNN model scripts in the `gnn_models/` directory. These scripts are shared for transparency; however, they cannot be executed without the requisite Twitter data.

## Discovery Process

To set up the discovery process, run:

```
cd dredge_discovery/
conda create -n dredge_words python=3.10
conda activate dredge_words
pip3 install -r requirements.txt
```

The [discovery process](dredge_discovery/discovery.ipynb) involves the following steps:
1. Compile a list of unreliable domains.
2. Identify keyphrases for which these domains rank highly.
3. Scrape SERP results from Google for these keyphrases ([SERP scraping script](dredge_discovery/serp_pull.py)).
4. Collect SEO attributes for each URL appearing in these SERP results ([data collection script](https://github.com/CASOS-IDeaS-CMU/Detection-and-Discovery-of-Misinformation-Sources/tree/master/data_collection)).
5. Train classifiers to label domains as reliable or unreliable based on SEO data ([training script](https://github.com/CASOS-IDeaS-CMU/Detection-and-Discovery-of-Misinformation-Sources/blob/master/flat_models/train_classifiers.ipynb)). Pretrained models are available in `pretrained_models/`.


## Citation
If you use this work, please cite:

```
@article{williams2024bridging,
  title={Bridging Social Media and Search Engines: Dredge Words and the Detection of Unreliable Domains},
  author={Williams, Evan M and Carragher, Peter and Carley, Kathleen M},
  journal={arXiv preprint arXiv:2406.11423},
  year={2024}
}
```

### Additional Citation
If you use the SEO data collection, classifier training scripts, or pretrained classifiers from our previous work, please consider citing:
```
@article{carragher2024detection,
  title={Detection and Discovery of Misinformation Sources using Attributed Webgraphs},
  author={Carragher, Peter and Williams, Evan M and Carley, Kathleen M},
  journal={arXiv preprint arXiv:2401.02379},
  year={2024}
}
```
