'''
PART 2: Embedding Generation with OpenAI

Objective: Generate semantic embeddings from airline Twitter
comments using an OpenAI embedding model to enable downstream clustering
and trend discovery. 

Description: This stage focuses on transforming the cleaned text data into 
vector embeddings using an OpenAI embedding model optimized for semantic similarity tasks.

To optimize API usage and reduce costs, the pipeline checks whether embeddings have already been
generated and stored locally. If the embeddings file exists, it is loaded directly instead of recomputing
them. Otherwise, the embeddings are generated through the OpenAI API and saved for future reuse. 

This caching mechanism prevents unnecessary API calls and helps preserve OpenAI API credits while ensuring
reproducibility of the embedding process. 
'''

# import libraries 
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from openai import OpenAI

# load env to read .env
load_dotenv()
# create client and read API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text_list, batch_size=256): 
    embeddings = []
    
    for i in range(0, len(text_list), batch_size): 
        batch = text_list[i: i+batch_size].tolist()
        response = client.embeddings.create(
            model = "text-embedding-3-small",
            input = batch
        )
        
        batch_embeddings = [item.embedding for item in response.data]
        embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)

# load dataset
df = pd.read_parquet("tweets_clean.parquet")
df.head()

# Check if embeddings exist.
emb_path = "tweet_embeddings.npy"
if os.path.exists(emb_path): 
    print("Loading cached embeddings...")
    emb_matrix = np.load(emb_path)
else: 
    print("Computing embeddings...")
    emb_matrix = get_embedding(df["text"])
    np.save(emb_path, emb_matrix)