import os
import sys
import warnings
from typing import List, Dict, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer
from scipy.sparse import csr_matrix
from pymilvus import connections, Collection

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

import streamlit as st

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG
# =============================================================================
class Config:
    """Centralized configuration management."""

    # Milvus / Zilliz
    MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
    MILVUS_URI = os.getenv("MILVUS_URI")        # For Zilliz Cloud
    MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")    # For Zilliz Cloud
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "product_hybrid_search_mvp2")

    # Embedding Models
    CLIP_MODEL = "clip-ViT-B-32"
    SPLADE_MODEL = "naver/splade-cocondenser-ensembledistil"
    SPLADE_TOP_K = 256   # Keep top 256 dimensions for sparse vectors

    # LLM
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

    # LangSmith (optional)
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "prod-rag")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

    # Search
    DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
    RRF_K = int(os.getenv("RRF_K", "60"))
    TEXT_WEIGHT = float(os.getenv("TEXT_WEIGHT", "0.6"))

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def validate(cls):
        """Validate required configuration."""
        errors = []
        if not cls.MILVUS_URI and not cls.MILVUS_HOST:
            errors.append("MILVUS_HOST or MILVUS_URI must be set")
        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY must be set")
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")

    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("Current Configuration:")
        print(f"  Milvus: {cls.MILVUS_URI or f'{cls.MILVUS_HOST}:{cls.MILVUS_PORT}'}")
        print(f"  Collection: {cls.COLLECTION_NAME}")
        print(f"  CLIP Model: {cls.CLIP_MODEL}")
        print(f"  SPLADE Model: {cls.SPLADE_MODEL}")
        print(f"  LLM Model: {cls.LLM_MODEL}")
        print(f"  Device: {cls.DEVICE}")
        print(f"  LangSmith Tracing: {cls.LANGCHAIN_TRACING_V2}")
        print()


# =============================================================================
# BACKEND INITIALIZATION
# =============================================================================
def connect_to_milvus() -> Collection:
    """Connect to Milvus / Zilliz and return collection."""
    if Config.MILVUS_URI:
        # Zilliz Cloud
        connections.connect(
            alias="default",
            uri=Config.MILVUS_URI,
            token=Config.MILVUS_TOKEN,
        )
    else:
        # Local Milvus
        connections.connect(
            alias="default",
            host=Config.MILVUS_HOST,
            port=Config.MILVUS_PORT,
        )

    collection = Collection(Config.COLLECTION_NAME)
    collection.load()

    print(f"✓ Connected to Milvus collection: {Config.COLLECTION_NAME}")
    print(f"✓ Collection entities: {collection.num_entities}")
    return collection


def load_clip_model() -> SentenceTransformer:
    """Load CLIP model for dense embeddings."""
    model = SentenceTransformer(Config.CLIP_MODEL, device=Config.DEVICE)
    model.eval()
    torch.set_num_threads(2)
    print(f"✓ CLIP model loaded on {Config.DEVICE}")
    return model


def load_splade_model() -> Optional[SentenceTransformer]:
    """Load SPLADE model for sparse embeddings."""
    try:
        model = SentenceTransformer(Config.SPLADE_MODEL, device=Config.DEVICE)
        model.eval()
        print(f"✓ SPLADE model loaded on {Config.DEVICE}")
        return model
    except Exception as e:
        print(f"⚠ SPLADE model not available: {e}")
        print("⚠ Continuing with CLIP-only mode")
        return None


def load_llm() -> ChatOpenAI:
    """Load LLM for generation."""
    llm = ChatOpenAI(
        model=Config.LLM_MODEL,
        temperature=Config.LLM_TEMPERATURE,
        api_key=Config.OPENAI_API_KEY,
    )
    print(f"✓ LLM initialized: {Config.LLM_MODEL}")
    return llm


def convert_to_sparse_csr(dense_array: np.ndarray, top_k: int = 256) -> csr_matrix:
    """
    Convert dense array to scipy CSR sparse matrix.

    Keeps only top-K dimensions.
    """
    top_indices = np.argsort(dense_array)[-top_k:]
    top_values = dense_array[top_indices]

    n_cols = len(dense_array)
    sparse_matrix = csr_matrix(
        (top_values, (np.zeros(len(top_indices), dtype=int), top_indices)),
        shape=(1, n_cols),
    )
    return sparse_matrix


# =============================================================================
# ENCODING / RETRIEVAL
# =============================================================================
def encode_text_query(
    text: str,
    backend: Dict,
) -> Tuple[np.ndarray, Optional[csr_matrix]]:
    """Encode text query with CLIP (dense) and SPLADE (sparse)."""
    clip_model: SentenceTransformer = backend["clip_model"]
    splade_model: Optional[SentenceTransformer] = backend.get("splade_model")

    # Dense
    dense = clip_model.encode([text], convert_to_numpy=True)
    dense = dense / np.linalg.norm(dense)

    # Sparse
    sparse_csr = None
    if splade_model is not None:
        sparse_array = splade_model.encode([text], convert_to_numpy=True)[0]
        sparse_csr = convert_to_sparse_csr(sparse_array, top_k=Config.SPLADE_TOP_K)

    return dense[0], sparse_csr


def encode_image_query(
    image: Image.Image,
    backend: Dict,
) -> np.ndarray:
    """Encode image query with CLIP."""
    clip_model: SentenceTransformer = backend["clip_model"]
    dense = clip_model.encode([image], convert_to_numpy=True)
    dense = dense / np.linalg.norm(dense)
    return dense[0]


def encode_hybrid_query(
    text: str,
    image: Image.Image,
    backend: Dict,
    text_weight: float = None,
) -> Tuple[np.ndarray, Optional[csr_matrix]]:
    """Encode hybrid query (text + image) with weighted fusion."""
    if text_weight is None:
        text_weight = Config.TEXT_WEIGHT

    text_dense, text_sparse = encode_text_query(text, backend)
    image_dense = encode_image_query(image, backend)

    fused_dense = text_weight * text_dense + (1 - text_weight) * image_dense
    fused_dense = fused_dense / np.linalg.norm(fused_dense)

    return fused_dense, text_sparse


def reciprocal_rank_fusion(result_lists: List[List], k: int = None) -> List[Dict]:
    """
    Combine multiple result lists using Reciprocal Rank Fusion.
    """
    if k is None:
        k = Config.RRF_K

    combined_scores: Dict[str, Dict] = {}

    for result_list in result_lists:
        for rank, hit in enumerate(result_list):
            product_id = hit.entity.get("product_id")
            rrf_score = 1.0 / (k + rank + 1)

            if product_id not in combined_scores:
                combined_scores[product_id] = {
                    "entity": hit.entity,
                    "score": 0.0,
                    "sources": [],
                }

            combined_scores[product_id]["score"] += rrf_score
            combined_scores[product_id]["sources"].append(
                {"rank": rank + 1, "distance": hit.distance}
            )

    sorted_results = sorted(
        combined_scores.values(),
        key=lambda x: x["score"],
        reverse=True,
    )
    return sorted_results


def retrieve_products(
    backend: Dict,
    query_text: Optional[str] = None,
    query_image: Optional[Image.Image] = None,
    top_k: int = None,
    text_weight: float = None,
) -> List[Dict]:
    """
    Retrieve products using hybrid search (CLIP + SPLADE) over Milvus.
    """
    if not query_text and not query_image:
        raise ValueError("Must provide either query_text or query_image")

    if top_k is None:
        top_k = Config.DEFAULT_TOP_K

    collection: Collection = backend["collection"]

    # Encode query
    if query_text and query_image:
        dense_emb, sparse_csr = encode_hybrid_query(
            text=query_text,
            image=query_image,
            backend=backend,
            text_weight=text_weight,
        )
    elif query_text:
        dense_emb, sparse_csr = encode_text_query(query_text, backend)
    else:
        dense_emb = encode_image_query(query_image, backend)
        sparse_csr = None

    # Search all vector fields
    search_params = {"metric_type": "COSINE", "params": {"ef": 100}}
    result_lists = []

    # text_dense_embedding
    text_dense_results = collection.search(
        data=[dense_emb.tolist()],
        anns_field="text_dense_embedding",
        param=search_params,
        limit=top_k * 2,
        output_fields=[
            "product_id",
            "product_name",
            "image_url",
            "selling_price",
            "clip_text",
            "about_product",
        ],
    )[0]
    result_lists.append(text_dense_results)

    # image_dense_embedding
    image_dense_results = collection.search(
        data=[dense_emb.tolist()],
        anns_field="image_dense_embedding",
        param=search_params,
        limit=top_k * 2,
        output_fields=[
            "product_id",
            "product_name",
            "image_url",
            "selling_price",
            "clip_text",
            "about_product",
        ],
    )[0]
    result_lists.append(image_dense_results)

    # text_sparse_embedding (if SPLADE available)
    splade_model: Optional[SentenceTransformer] = backend.get("splade_model")
    if sparse_csr is not None and splade_model is not None:
        sparse_search_params = {"metric_type": "IP", "params": {}}
        sparse_results = collection.search(
            data=[sparse_csr],
            anns_field="text_sparse_embedding",
            param=sparse_search_params,
            limit=top_k * 2,
            output_fields=[
                "product_id",
                "product_name",
                "image_url",
                "selling_price",
                "clip_text",
                "about_product",
            ],
        )[0]
        result_lists.append(sparse_results)

    combined_results = reciprocal_rank_fusion(result_lists)
    return combined_results[:top_k]


# =============================================================================
# GENERATION
# =============================================================================
def create_context_string(products: List[Dict]) -> str:
    """Format products into context string for LLM."""
    if not products:
        return "No relevant products found."

    context_parts = []
    for i, product in enumerate(products, 1):
        entity = product["entity"]
        context_parts.append(
            f"[Product {i}]\n"
            f"Name: {entity.get('product_name')}\n"
            f"Price: {entity.get('selling_price')}\n"
            f"Description: {entity.get('about_product', 'N/A')[:300]}\n"
            f"Image URL: {entity.get('image_url')}\n"
        )
    return "\n".join(context_parts)


def create_query_description(
    query_text: Optional[str],
    query_image: Optional[Image.Image],
) -> str:
    """Describe the query for the prompt."""
    if query_text and query_image:
        return f"Text query: '{query_text}' (User also provided a reference image)"
    elif query_text:
        return f"Text query: '{query_text}'"
    elif query_image:
        return "Visual search query (User provided an image)"
    else:
        return "Query"


generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful shopping assistant for an e-commerce platform.
Your role is to recommend products based on the user's query and retrieved product information.

Guidelines:
- Be concise and helpful
- Recommend products that best match the user's needs
- Mention key features like price, description, and unique attributes
- If multiple good options exist, explain the differences
- If no good matches found, suggest alternatives or ask clarifying questions
- Do NOT mention image URLs in your response (those are for display purposes)
- Do NOT make up product information - only use what's provided in the context""",
        ),
        (
            "human",
            """User Query: {query_description}

Retrieved Products:
{context}

Based on the above products, please provide helpful recommendations to the user.""",
        ),
    ]
)


def generate_answer(
    query_text: Optional[str],
    query_image: Optional[Image.Image],
    retrieved_products: List[Dict],
    backend: Dict,
) -> str:
    """Generate answer text from retrieved products."""
    context = create_context_string(retrieved_products)
    query_description = create_query_description(query_text, query_image)

    llm: ChatOpenAI = backend["llm"]
    chain = generation_prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "query_description": query_description})
    return answer


class RAGResponse:
    """Response object from RAG pipeline."""

    def __init__(
        self,
        answer: Optional[str],
        products: List[Dict],
        query_type: str,
        query_text: Optional[str] = None,
        query_image: Optional[Image.Image] = None,
    ):
        self.answer = answer
        self.products = products
        self.query_type = query_type
        self.query_text = query_text
        self.query_image = query_image

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "products": [
                {
                    "product_name": p["entity"].get("product_name"),
                    "price": p["entity"].get("selling_price"),
                    "image_url": p["entity"].get("image_url"),
                    "score": p["score"],
                }
                for p in self.products
            ],
            "query_type": self.query_type,
        }


def rag_query(
    backend: Dict,
    query_text: Optional[str] = None,
    query_image: Optional[Image.Image] = None,
    top_k: int = None,
    return_mode: str = "full",
) -> RAGResponse:
    """
    Complete RAG pipeline: retrieve + generate.
    """
    if not query_text and not query_image:
        raise ValueError("Must provide either query_text or query_image")

    if return_mode not in ["full", "retrieval_only"]:
        raise ValueError("return_mode must be 'full' or 'retrieval_only'")

    if top_k is None:
        top_k = Config.DEFAULT_TOP_K

    # Determine query type
    if query_text and query_image:
        query_type = "hybrid"
    elif query_text:
        query_type = "text"
    else:
        query_type = "image"

    products = retrieve_products(
        backend=backend,
        query_text=query_text,
        query_image=query_image,
        top_k=top_k,
    )

    answer = None
    if return_mode == "full":
        answer = generate_answer(query_text, query_image, products, backend)

    return RAGResponse(
        answer=answer,
        products=products,
        query_type=query_type,
        query_text=query_text,
        query_image=query_image,
    )


# =============================================================================
# STREAMLIT UI
# =============================================================================
st.set_page_config(
    page_title="Product Chatbot (Hybrid RAG on Milvus)",
    page_icon="🛒",
    layout="wide",
)


@st.cache_resource(show_spinner=True)
def init_backend() -> Dict:
    """
    Initialize and cache backend resources (Milvus, models, LLM).
    """
    Config.validate()
    Config.print_config()

    collection = connect_to_milvus()
    clip_model = load_clip_model()
    splade_model = load_splade_model()
    llm = load_llm()

    backend = {
        "collection": collection,
        "clip_model": clip_model,
        "splade_model": splade_model,
        "llm": llm,
    }
    return backend


def main():
    backend = init_backend()

    st.title("🛒 Product Chatbot (Hybrid RAG on Milvus)")
    st.write(
        "Ask questions about products using **text**, optionally upload a **product image**, "
        "and get an answer that combines retrieval from Milvus/Zilliz and LLM generation."
    )

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        top_k = st.slider(
            "Number of products to retrieve",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
        )
        mode = st.selectbox(
            "RAG mode",
            options=["full (retrieve + generate)", "retrieval_only"],
            index=0,
        )
        return_mode = "full" if mode.startswith("full") else "retrieval_only"

        st.markdown("### Backend info")
        st.write(f"Collection: `{Config.COLLECTION_NAME}`")
        st.write(f"Device: `{Config.DEVICE}`")
        st.write(f"LLM: `{Config.LLM_MODEL}`")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input widgets
    uploaded_file = st.file_uploader(
        "Optional: upload a product image",
        type=["png", "jpg", "jpeg"],
    )

    user_input = st.chat_input(
        "Ask a question about a product (you can also say 'show me a picture of ...')"
    )

    if user_input:
        # Show user message
        with st.chat_message("user"):
            st.markdown(user_input)
            if uploaded_file is not None:
                st.image(
                    uploaded_file,
                    caption="Uploaded image",
                    use_container_width=True,
                )

        st.session_state["messages"].append({"role": "user", "content": user_input})

        # Convert uploaded file to PIL image
        query_image = None
        if uploaded_file is not None:
            query_image = Image.open(uploaded_file).convert("RGB")

        # RAG pipeline
        try:
            response = rag_query(
                backend=backend,
                query_text=user_input.strip() or None,
                query_image=query_image,
                top_k=top_k,
                return_mode=return_mode,
            )
        except ValueError as e:
            with st.chat_message("assistant"):
                st.error(str(e))
            st.session_state["messages"].append(
                {"role": "assistant", "content": f"Error: {e}"}
            )
            return

        # Assistant message
        with st.chat_message("assistant"):
            if response.answer:
                st.markdown(response.answer)
                answer_text_for_history = response.answer
            else:
                if not response.products:
                    answer_text_for_history = (
                        "I couldn't find any matching products for your query."
                    )
                    st.markdown(answer_text_for_history)
                else:
                    answer_text_for_history = (
                        "I retrieved some products matching your query (generation is disabled)."
                    )
                    st.markdown(answer_text_for_history)

            # Main product image (single image to mirror your sample Q&A)
            if response.products:
                main_product = response.products[0]
                entity = main_product["entity"]
                img_url = entity.get("image_url")
                name = entity.get("product_name", "Product image")

                if img_url:
                    st.markdown("#### Product image")
                    st.image(
                        img_url,
                        caption=name,
                        use_container_width=True,
                    )

            # Debug view of retrieved products
            if response.products:
                with st.expander("Retrieved products (debug view)"):
                    for i, p in enumerate(response.products, start=1):
                        e = p["entity"]
                        st.markdown(
                            f"**{i}. {e.get('product_name')}**  "
                            f"(Price: {e.get('selling_price')}, Score: {p['score']:.4f})"
                        )
                        if e.get("about_product"):
                            st.markdown(
                                f"<small>{str(e['about_product'])[:200]}...</small>",
                                unsafe_allow_html=True,
                            )
                        st.markdown("---")

        st.session_state["messages"].append(
            {"role": "assistant", "content": answer_text_for_history}
        )


if __name__ == "__main__":
    main()
