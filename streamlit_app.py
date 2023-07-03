import streamlit as st
from sentence_transformers import SentenceTransformer
import pinecone
from scrape_resume import scrape_resume


# Initialize sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Pinecone
pinecone.init(api_key="01f083ab-5384-4110-b7fa-4c8b64981e59")
index_name = "auto-hr"
pinecone.deindex(index_name)
pinecone.create_index(index_name, metric="dotproduct")
index = pinecone.Index(index_name)

# Streamlit
st.title('PDF Embedding App')
uploaded_files = st.file_uploader("Upload PDFs", type=['pdf'], accept_multiple_files=True)

for file_id, uploaded_file in enumerate(uploaded_files):
    pdf_file = scrape_resume(uploaded_file)
    # compute embeddings
    sentence_embeddings = model.encode(pdf_file)
    # upsert the embeddings to PineCone
    for i, emb in enumerate(sentence_embeddings):
        unique_id = f"{uploaded_file.name}_fileId_{file_id}"
        index.upsert(vectors={unique_id: emb})

st.write('Done!')

# Deinitialize Pinecone when done
pinecone.deinit()