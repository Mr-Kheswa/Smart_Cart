# Smart_Cart
A very small content based Artificial Intelligency application recommender named Smart Cart, it is a minimal content based product recommender. It models each product with two features which are **tags** which will be based on set of keywords and **description** based on short texts. Recommendations combine Jaccard similarity on tags with cosine similarity on **Term frequency - inverse document frequency** [TF-IDF] descriptions & combined score ranks recommendations.

___

## Quick overview
* Tags are parsed into sets of keywords.
* Descriptions are vectorized with TF - IDF and compared with cosine similarity.
* Jaccard similarity is computed on tag sets.
* Combine score = alpha * jaccard + (1 - alpha) * cosine (default alpha = 0.5).
* Recommendations return the top most similar products with the score breakdown.

___

## Files in this project root
* **README.md** - current file.
* **requirements.txt** - Python dependencies.
* **data/products.csv** - sample product dataset.
* **smart_cart.py** - single file recommender implementation.
* **rush.sh** - optional use but a small runner script to demo the recommender.

[Setup and Basic Project Files](https://github.com/users/Mr-Kheswa/projects/4?pane=issue&itemId=132728546&issue=Mr-Kheswa%7CSmart_Cart%7C1)
___
