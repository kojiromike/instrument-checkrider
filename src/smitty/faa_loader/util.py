from pathlib import Path

import voyageai
from anthropic import Anthropic
from decouple import config
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from pinecone.core.openapi.shared.exceptions import PineconeApiException
from platformdirs import user_cache_dir

ANTHROPIC_API_KEY = config("ANTHROPIC_API_KEY")
PINECONE_API_KEY = config("PINECONE_API_KEY")
VOYAGE_API_KEY = config("VOYAGE_API_KEY")

CACHE_DIR = Path(user_cache_dir(__package__))
(CACHE_DIR / "loaders").mkdir(parents=True, exist_ok=True)
(CACHE_DIR / "vectors").mkdir(parents=True, exist_ok=True)

SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=[
        "\n\n",
        "\n",
        " ",
        "",
    ],
)

VOYAGE = voyageai.Client(api_key=VOYAGE_API_KEY)
# https://docs.voyageai.com/docs/embeddings#model-choices
VOYAGE_MODEL = "voyage-3-large"
VOYAGE_DIMENSIONS = 1024

ANTHROPIC = Anthropic(api_key=ANTHROPIC_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "instrument-checkride-knowledge"
try:
    pc.create_index(
        name=INDEX_NAME,
        dimension=VOYAGE_DIMENSIONS,
        metric="cosine",  # Replace with your model metric
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
except PineconeApiException as e:
    if e.status != 409:
        raise
INDEX = pc.Index(INDEX_NAME)
