import numpy as np
import weaviate
from sentence_transformers import SentenceTransformer
import time


# Load model once at module level
model = None


def embed_text(text: str) -> np.ndarray:
    """Return a 384-dim float32 numpy vector for the input string."""

    global model

    retries = 3
    wait_time = 5

    for attempt in range(retries):

        try:
            if model is None:
                model = SentenceTransformer("all-MiniLM-L6-v2")

            v = model.encode(
                text,
                convert_to_numpy=True
            ).astype(np.float32)

            return v

        except Exception as e:

            if "429" in str(e) and attempt < retries - 1:
                print(f"Rate limited. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= 2
            else:
                raise e


def weaviate_ready(url: str) -> bool:
    """Return True if Weaviate at `url` is reachable and ready."""

    try:
        client = weaviate.Client(url)
        return client.is_ready()

    except Exception:
        return False


def ingest_corpus(
    client: weaviate.Client,
    class_name: str,
    items: list[dict]
) -> int:
    """Ingest items into the named class."""

    # Check existing schema/classes
    existing_classes = [
        c["class"]
        for c in client.schema.get()["classes"]
    ] if client.schema.get().get("classes") else []

    # Create class if it doesn't exist
    if class_name not in existing_classes:

        schema = {
            "class": class_name,
            "vectorizer": "none",
            "properties": [
                {
                    "name": "title",
                    "dataType": ["text"]
                },
                {
                    "name": "text",
                    "dataType": ["text"]
                }
            ]
        }

        client.schema.create_class(schema)

    # Batch ingest
    with client.batch as batch:

        for item in items:

            batch.add_data_object(
                data_object={
                    "title": item["title"],
                    "text": item["text"]
                },
                class_name=class_name,
                vector=item["vector"]
            )

    # Verify count
    result = (
        client.query
        .aggregate(class_name)
        .with_meta_count()
        .do()
    )

    count = result["data"]["Aggregate"][class_name][0]["meta"]["count"]

    return count