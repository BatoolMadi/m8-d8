import numpy as np
import weaviate
from sentence_transformers import SentenceTransformer


# Load model once at module level
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_text(text: str) -> np.ndarray:
    """Return a 384-dim float32 numpy vector for the input string."""

    v = model.encode(
        text,
        convert_to_numpy=True
    ).astype(np.float32)

    return v


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