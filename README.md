# Smart_Cart
A very small content based Artificial Intelligency application recommender named Smart Cart, it is a minimal content based product recommender. It models each product with two features which are **tags** which will be based on set of keywords and **description** based on short texts. Recommendations combine Jaccard similarity on tags with cosine similarity on **Term frequency - inverse document frequency** [TF-IDF] descriptions & combined score ranks recommendations.

___

## Quick overview
* Tags are parsed into sets of keywords.
* Descriptions are vectorized with TF - IDF and compared with cosine similarity.
* Jaccard similarity is computed on tag sets.
* Recommendations return the top most similar products with the score breakdown.

___

## Application Features
* **CSV ingestion:** reads products data from data/products.csv.
* **Tag similarity:** uses jaccard index to compare product tags.
* **Description similarity:** applies TF - IDF vectorization and cosine similarity.
* **Weighted scoring:** combines tag and description scores with custamizable weights.
* **Command line interface:** accepts product ID and scoring parameters via CLI

### Web Application interface features
* **interactive UI:** broswe products, view images, and get recommendations in real time.
* **Live scoring:** combines jaccard and TF - IDF similarity with adjustable weights.
* **Explainable output:** displays score breakdowns and highlights why products are recommended.
* **Image integration:** shows local or remote product images for visual context.
* **Responsive layout:** clean design for intuitive navigation.

**NB:** Main features of this application are **tags** and **description.**
___

## Application Requirements
Insatll dependencies using: **pip install -r requirements.txt**

**Dependencies:** which will installed.
* pandas
* scikit-learn
* streamlit

___

## Application Usage
Run the recommender from terminal:
**CLI / PowerShell**

**python main.py --product 101 --top 5 --w_tag 0.6 --w_desc 0.4**

### Web Application Interface
[Web Interface:](https://github.com/Mr-Kheswa/Smart_Cart/issues/6)
is a **streamlit powered web interface** for interactive product exploration and recommendations.

To launch the web application:

**streamlit run app.py**
**python -m streamlit run app.py**

The above of these each command will opens a browser:
* Select a product by ID or name.
* Adjust tag / description weights.
* View top recommended items with score breakdowns and images.

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

# Simple Addition
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
