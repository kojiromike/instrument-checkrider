import pickle
from collections.abc import Generator, Iterable, Mapping
from functools import cache
from itertools import chain, groupby
from pathlib import Path
from typing import TypedDict
from urllib.parse import parse_qs, urlparse

import requests
import voyageai
from anthropic import Anthropic
from decouple import config
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredXMLLoader as XMLLoader
from langchain_core.documents.base import Document
from pinecone import Pinecone, ServerlessSpec
from pinecone.core.openapi.shared.exceptions import PineconeApiException
from platformdirs import user_cache_dir

from .data import SOURCES

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


def make_filename(url: str) -> str:
    parsed = urlparse(url)
    date = parse_qs(parsed.query).get("date", [""])[0] or parsed.path.split("/")[4]
    part = parse_qs(parsed.query)["part"][0]
    return f"ecfr_title14_part{part}_{date}.xml"


def download_ref(ref: str) -> Path | None:
    print(f"downloading {ref}")
    url = SOURCES[ref]
    parsed = urlparse(url)
    if parsed.scheme != "https":
        print(f"url {url} does not start with https. Skipping.")
        return None
    path = Path(parsed.path)
    suffix = path.suffix
    dest = CACHE_DIR / f"{ref}{suffix}"
    if dest.exists():
        print(f"{dest} already exists")
        return dest
    if suffix == ".zip":
        print(f"url {url} is the big zipfile. Skipping")
        return None
    download_url(url, dest)
    return dest


@cache
def download_url(url: str, dest: Path):
    _, dot, suffix_ = url.rpartition(".")
    dot + suffix_
    response = requests.get(url)
    response.raise_for_status()
    dest.write_bytes(response.content)


print("downloading documents")
downloads = {
    ref: dl
    for ref, dl in {ref: download_ref(ref) for ref in SOURCES if ref}.items()
    if dl
}


def load(ref: str) -> Iterable[Document]:
    print(f"loading {ref}")
    file = downloads[ref]
    cache_file = file.parent / "loaders" / file.name
    if cache_file.exists():
        print(f"found cached loader for {ref}")
        return pickle.loads(cache_file.read_bytes())
    suffix = file.suffix
    loader = {
        ".pdf": PyPDFLoader,
        ".xml": XMLLoader,
    }
    doc = loader[suffix](file).load()
    cache_file.write_bytes(pickle.dumps(doc))
    return doc


print("loading documents")
docs = {
    ref: loaded
    for ref, loaded in {ref: load(ref) for ref in downloads}.items()
    if loaded
}
# "Airman Certification Standards, Instrument Rating"
# FAA-S-ACS-8C
print("splitting docs")
split_docs = {ref: SPLITTER.split_documents(docs_) for ref, docs_ in docs.items()}

Vector = list[list[float]] | list[list[int]]


# Store with metadata
def create_embeddings(chunks: Iterable[Document]) -> Vector:
    return VOYAGE.embed(
        [chunk.page_content for chunk in chunks],
        model=VOYAGE_MODEL,
        input_type="document",
    ).embeddings


class VectorMapping(TypedDict):
    id: str
    values: Vector
    metadata: Mapping[str, object]


def map_vectors(
    ref: str, chunks: Iterable[Document]
) -> Generator[VectorMapping, None, None]:
    print(f"creating vectors for {ref}")
    file = downloads[ref]
    page_groups = groupby(chunks, key=lambda ch: ch.metadata["page"])
    try:
        for page, group in page_groups:
            id_ = f"{ref}-{page}"
            cache_file = file.parent / f"vectors/{file.stem}-{page}{file.suffix}"
            if cache_file.exists():
                print(f"found cached vectors for {ref} page {page}")
                yield pickle.loads(cache_file.read_bytes())
            else:
                vectors = create_embeddings(group)
                result = VectorMapping(
                    id=id_,
                    values=vectors,
                    metadata={
                        "page": page,
                        "ref": ref,
                    },
                )
                cache_file.write_bytes(pickle.dumps(result))
                yield result
    except KeyError:
        raise


vectors = list(
    chain.from_iterable(map_vectors(ref, chunks) for ref, chunks in split_docs.items())
)

# INDEX.upsert(vectors=vectors)
# metadata = {"ref": ref}
#   for i, chunk in enumerate(chunks):
#       metadata["page"] = chunk.metadata["page"]
#       vectors = create_embedding(chunk)
#       INDEX.upsert([(str(i), vectors, metadata)])


# def query_knowledge_base(query, acs_section=None):
#     query_vector = embeddings.embed_query(query)
#
#     filter = {}
#     if acs_section:
#         filter["acs_reference"] = acs_section
#
#     results = index.query(
#         query_vector,
#         filter=filter,
#         top_k=3
#     )
#     return process_results(results)
