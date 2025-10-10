# Smart_Cart
A very small content based Artificial Intelligency application recommender named Smart Cart, it is a minimal content based product recommender. It models each product with two features which are **tags** which will be based on set of keywords and **description** based on short texts. Recommendations combine Jaccard similarity on tags with cosine similarity on **Term frequency - inverse document frequency** [TF-IDF] descriptions & combined score ranks recommendations.

___

## Quick overview
* Tags are parsed into sets of keywords.
* Descriptions are vectorized with TF - IDF and compared with cosine similarity.
* Jaccard similarity is computed on tag sets.
* Recommendations return the top most similar products with the score breakdown.

___

## Files in this project root
* **data/products.csv** - sample product dataset.
* **app.py** - single file recommender implementation to define and configure the core application logic.
* **main.py** - offline entry point of the application.
* **paths.py** - driving the path linkage via the folders to align to its calling.
* **README.md** - current file.
* **requirements.txt** - Python dependencies.
* **rush.sh** - optional use but a small runner script to demo the recommender.

[Setup and Basic Project Files](https://github.com/users/Mr-Kheswa/projects/4?pane=issue&itemId=132728546&issue=Mr-Kheswa%7CSmart_Cart%7C1)
___

## Data format (data/products.csv)
CSV columns:
* **product_id** - integer unique ID
* **name** - product name string
* **category** - specified field to find the product.
* **tags** - keywords as semicolon separated tokens or JSON like list
* **description** - short text used for TF - IDF
* **price** - value of the product.
* **rating** - recommendation of people who purchased.
* **stock** - checks the stock number and range availability.
* **image_url** - image intercetion link to csv

**Example row:**
101,Wireless Mouse,Electronics,mouse;wireless;peripheral,Compact wireless mouse with USB receiver and ergonomic design,199.99,4.5,25

[Create a Sample Dataset](https://github.com/Mr-Kheswa/Smart_Cart/issues/2)
___

## Key similarities
1. **Content Based Filtering**
* **Smart Cart uses product metadata** which are tags and description to recommend similar items.
* it does not reply on user history or collaborative data, its focus it is the product's own features.
2. **Tag Modeling with Jaccard Similarity**
* **Tags** are treated as keyword sets. **For instance:** "wireles, mouse."
* **Jaccard similarity** compares overlap between tag sets to find related products.
3. **Text Based Modeling with TF-IDF**
* Product desccriptions are vectorized using **Term Frequency - Inverse Document Frequency (TF -IDF**.
* **Cosine similarity** is used to measure semantic closeness between descriptions.
4. **Modular, Explainable Arichitecture**
* Each product is modeled with clear features: ID, name,, category, tags, description, price, rating, stock, image.
* The assistant can explain why a product is recommended, and this aligns with explainable AI principles.
5. **Image Integration and UI Personalization**
* Local images are displayed alongside recommendations, enhancing visual context.

**In summary**, Smart Cart is a compact, contect based recommender that combines structured tags and semantic descriptions to deliver explainable.

[Core Similarity](https://github.com/Mr-Kheswa/Smart_Cart/issues/3)

___
