import pickle
from collections.abc import Generator, Iterable, Mapping
from functools import cache
from itertools import chain, groupby
from pathlib import Path
from typing import TypedDict
from urllib.parse import parse_qs, urlparse

import requests
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredXMLLoader as XMLLoader
from langchain_core.documents.base import Document
from more_itertools import batched
from tqdm import tqdm

from .data import SOURCES
from .util import CACHE_DIR, INDEX, SPLITTER, VOYAGE, VOYAGE_MODEL


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
def create_embeddings(chunks: Iterable[Document]) -> Vector | None:
    content = [chunk.page_content for chunk in chunks]
    if not content:
        return None
    return VOYAGE.embed(
        content,
        model=VOYAGE_MODEL,
        input_type="document",
    ).embeddings


class VectorMapping(TypedDict):
    id: str
    values: Vector
    metadata: Mapping[str, object]


def getpage(ch):
    try:
        return ch.metadata["page"]
    except KeyError:
        return None


def map_vectors(
    ref: str, chunks: Iterable[Document]
) -> Generator[VectorMapping, None, None]:
    print(f"creating vectors for {ref}")
    file = downloads[ref]
    page_groups = list(groupby(chunks, key=getpage))
    last_page = page_groups[-1][0]
    with tqdm(total=last_page) as progress_bar:
        for page, group in page_groups:
            id_ = f"{ref}-{page}"
            cache_file = file.parent / f"vectors/{file.stem}-{page}{file.suffix}"
            if cache_file.exists():
                # print(f"found cached vectors for {ref} page {page}")
                results = pickle.loads(cache_file.read_bytes())
                yield results
            else:
                # print(f"creating new vectors for {ref} page {page}")
                vectors = create_embeddings(group)
                if not vectors:
                    continue
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
            progress_bar.update(1)
        progress_bar.update(1)


vector_chain = chain.from_iterable(
    map_vectors(ref, chunks) for ref, chunks in split_docs.items()
)
vectors = [
    {**v, "values": embedding} for v in vector_chain for embedding in v["values"]
]
num_batches = 20
batch_size, last_batch_size = divmod(len(vectors), num_batches)
print("uploading batches")
batches = batched(vectors, batch_size)
with tqdm(total=num_batches) as progress_bar:
    for batch in batches:
        INDEX.upsert(vectors=batch)
        progress_bar.update(batch_size)
