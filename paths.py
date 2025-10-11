# paths.py
# import necessary libraries
import pandas as pd
from pathlib import Path

# define the path to the csv file containing product data
csv = Path("data/products.csv") 
img_dir = "data/images" # directories where product images are stored
df = pd.read_csv(csv) # load the csv file into a dataframe

# define a function to clean and standardize image paths
def fix(v):
    if not isinstance(v, str) or v.strip() == "":
        return ""
    v = v.strip().replace("\\", "/")
    if v.startswith("data/") or v.startswith("/") or v.startswith("http"):
        return v
    return f"{img_dir}/{v}" # prepend the image directory to create a valid relative path

df["image_url"] = df["image_url"].apply(fix) # apply the fix funtion to the image_url column
df.to_csv(csv, index=False) # save the updated dataframe

# print confirmation message
print("Updated image_url values in", csv)
