import os
import io
import warnings
from typing import List, Dict, Optional

import numpy as np
from PIL import Image
import torch
from sentence_transformers import SentenceTransformer
from scipy.sparse import csr_matrix
from pymilvus import connections, Collection, MilvusException

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
        """Print current configuration (for debugging)."""
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
    try:
        if Config.MILVUS_URI:
            connections.connect(
                alias="default",
                uri=Config.MILVUS_URI,
                token=Config.MILVUS_TOKEN,
            )
        else:
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

    except MilvusException as e:
        raise RuntimeError(
            "❌ Failed to connect to Milvus.\n\n"
            "Please check:\n"
            "- MILVUS_URI and MILVUS_TOKEN (or MILVUS_HOST/MILVUS_PORT)\n"
            "- Zilliz cluster is running\n"
            "- Network/firewall allows outbound HTTPS to Zilliz\n\n"
            f"Details from Milvus: {e}"
        )


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
    """Convert dense array to scipy CSR sparse matrix (top-K dims)."""
    top_indices = np.argsort(dense_array)[-top_k:]
    top_values = dense_array[top_indices]
    n_cols = len(dense_array)
    sparse_matrix = csr_matrix(
        (top_values, (np.zeros(len(top_indices), dtype=int), top_indices)),
        shape=(1, n_cols),
    )
    return sparse_matrix


def reciprocal_rank_fusion(result_lists: List[List], k: int = None) -> List[Dict]:
    """
    Combine multiple result lists using Reciprocal Rank Fusion.

    Uses Milvus primary key (hit.id) as the unique key for fusion.
    """
    if k is None:
        k = Config.RRF_K

    combined_scores: Dict[str, Dict] = {}

    for result_list in result_lists:
        for rank, hit in enumerate(result_list):
            unique_id = str(hit.id)
            rrf_score = 1.0 / (k + rank + 1)

            if unique_id not in combined_scores:
                combined_scores[unique_id] = {
                    "entity": hit.entity,
                    "score": 0.0,
                    "sources": [],
                }

            combined_scores[unique_id]["score"] += rrf_score
            combined_scores[unique_id]["sources"].append(
                {"rank": rank + 1, "distance": hit.distance}
            )

    sorted_results = sorted(
        combined_scores.values(),
        key=lambda x: x["score"],
        reverse=True,
    )
    return sorted_results


# =============================================================================
# RETRIEVAL
# =============================================================================
def retrieve_products(
    backend: Dict,
    query_text: Optional[str] = None,
    query_image: Optional[Image.Image] = None,
    top_k: int = None,
) -> List[Dict]:
    """
    Retrieve products from Milvus.

    - Image provided (no text): image-only retrieval (image_dense_embedding).
    - Text provided (no image): text-only retrieval (text_dense_embedding + SPLADE).
    - Both: retrieval uses image; text is only for LLM generation.
    """
    if not query_text and not query_image:
        raise ValueError("Must provide either query_text or query_image")

    if top_k is None:
        top_k = Config.DEFAULT_TOP_K

    collection: Collection = backend["collection"]
    clip_model: SentenceTransformer = backend["clip_model"]
    splade_model: Optional[SentenceTransformer] = backend.get("splade_model")

    search_params = {"metric_type": "COSINE", "params": {"ef": 100}}

    # CASE 1: IMAGE-ONLY
    if query_image is not None and query_text is None:
        image_dense = clip_model.encode([query_image], convert_to_numpy=True)
        image_dense = image_dense / np.linalg.norm(image_dense)
        image_dense = image_dense[0]

        image_results = collection.search(
            data=[image_dense.tolist()],
            anns_field="image_dense_embedding",
            param=search_params,
            limit=top_k,
            output_fields=[
                "product_id",
                "product_name",
                "image_url",
                "selling_price",
                "clip_text",
                "about_product",
            ],
        )[0]

        return [
            {
                "entity": hit.entity,
                "score": 1.0 / (rank + 1),
                "sources": [{"rank": rank + 1, "distance": hit.distance}],
            }
            for rank, hit in enumerate(image_results)
        ]

    # CASE 2: TEXT-ONLY
    if query_text and query_image is None:
        result_lists = []

        text_dense = clip_model.encode([query_text], convert_to_numpy=True)
        text_dense = text_dense / np.linalg.norm(text_dense)
        text_dense = text_dense[0]

        text_results = collection.search(
            data=[text_dense.tolist()],
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
        result_lists.append(text_results)

        if splade_model is not None:
            sparse_array = splade_model.encode([query_text], convert_to_numpy=True)[0]
            sparse_csr = convert_to_sparse_csr(sparse_array, top_k=Config.SPLADE_TOP_K)

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

        if not result_lists:
            return []

        if len(result_lists) == 1:
            single_list = result_lists[0]
            return [
                {
                    "entity": hit.entity,
                    "score": 1.0 / (rank + 1),
                    "sources": [{"rank": rank + 1, "distance": hit.distance}],
                }
                for rank, hit in enumerate(single_list)
            ][:top_k]

        combined_results = reciprocal_rank_fusion(result_lists)
        return combined_results[:top_k]

    # CASE 3: HYBRID (TEXT + IMAGE) → image-driven retrieval
    image_dense = clip_model.encode([query_image], convert_to_numpy=True)
    image_dense = image_dense / np.linalg.norm(image_dense)
    image_dense = image_dense[0]

    image_results = collection.search(
        data=[image_dense.tolist()],
        anns_field="image_dense_embedding",
        param=search_params,
        limit=top_k,
        output_fields=[
            "product_id",
            "product_name",
            "image_url",
            "selling_price",
            "clip_text",
            "about_product",
        ],
    )[0]

    return [
        {
            "entity": hit.entity,
            "score": 1.0 / (rank + 1),
            "sources": [{"rank": rank + 1, "distance": hit.distance}],
        }
        for rank, hit in enumerate(image_results)
    ]


# =============================================================================
# GENERATION (TOP PRODUCT ONLY)
# =============================================================================
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

You will receive context for a single product labeled [Product 1].
Always focus ONLY on this top product when answering.

Your job:
- Identify what the product is
- Describe its main purpose and how to use it
- Highlight key features, materials, options, or accessories
- Keep the answer clear and concise (1–3 short paragraphs)

Do NOT invent features that are not in the context.
Do NOT mention other products or ranking/order.""",
        ),
        (
            "human",
            """User Query: {query_description}

Top Retrieved Product:
{context}

Based on this product only, answer the user's question and describe the product and its usage.""",
        ),
    ]
)


def generate_answer_for_top_product(
    query_text: Optional[str],
    query_image: Optional[Image.Image],
    retrieved_products: List[Dict],
    backend: Dict,
) -> str:
    """Generate answer focusing only on the top retrieved product."""
    if not retrieved_products:
        return "I couldn't find any matching product for your query."

    top_product = retrieved_products[0]["entity"]
    context = (
        f"[Product 1]\n"
        f"Name: {top_product.get('product_name')}\n"
        f"Price: {top_product.get('selling_price')}\n"
        f"Description: {str(top_product.get('about_product', 'N/A'))[:400]}\n"
        f"Image URL: {top_product.get('image_url')}\n"
    )

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


def rag_query(
    backend: Dict,
    query_text: Optional[str] = None,
    query_image: Optional[Image.Image] = None,
    top_k: int = None,
    return_mode: str = "full",
) -> RAGResponse:
    """Complete RAG pipeline: retrieve + generate (or retrieval only)."""
    if not query_text and not query_image:
        raise ValueError("Must provide either query_text or query_image")

    if return_mode not in ["full", "retrieval_only"]:
        raise ValueError("return_mode must be 'full' or 'retrieval_only'")

    if top_k is None:
        top_k = Config.DEFAULT_TOP_K

    if query_text and query_image:
        query_type = "hybrid"
    elif query_text:
        query_type = "text"
    else:
        query_type = "image"

    products = retrieve_products(
        backend=backend,
        query_text=query_text if query_image is None else None,
        query_image=query_image,
        top_k=top_k,
    )

    answer = None
    if return_mode == "full":
        answer = generate_answer_for_top_product(
            query_text, query_image, products, backend
        )

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
    """Initialize and cache backend resources (Milvus, models, LLM)."""
    Config.validate()
    Config.print_config()

    try:
        collection = connect_to_milvus()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

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


def render_product_card(entity: Dict, score: float):
    """Render a product in an Amazon-like card."""
    name = entity.get("product_name", "Unnamed product")
    price = entity.get("selling_price")
    img_url = entity.get("image_url")
    desc = entity.get("about_product", "")
    desc_snip = (str(desc)[:130] + "…") if desc else ""

    st.markdown(
        """
        <div class="product-card">
            <div class="product-img-wrapper">
        """,
        unsafe_allow_html=True,
    )
    if img_url:
        st.image(img_url, width=180)
    else:
        st.markdown(
            '<div class="product-placeholder">No image</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        f"""
            </div>
            <div class="product-info">
                <div class="product-title">{name}</div>
                <div class="product-price">
                    {f"${price:.2f}" if isinstance(price, (int, float)) else (price or "")}
                </div>
                <div class="product-desc">{desc_snip}</div>
                <div class="product-meta">Relevance score: {score:.4f}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    backend = init_backend()

    # --- CSS theme ---
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #ffffff;
        }
        .main-container {
            max-width: 1200px;
            margin: 0 auto;
            padding-bottom: 2rem;
        }
        .amazon-header {
            background-color: #131921;
            color: #ffffff;
            padding: 0.5rem 1.5rem;
            margin: -1rem -1.5rem 1rem -1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .amazon-logo {
            font-weight: 700;
            font-size: 1.3rem;
            letter-spacing: 0.03em;
            color: #ff9900;
        }
        .amazon-tagline {
            font-size: 0.9rem;
            opacity: 0.85;
        }
        .left-panel, .right-panel {
            background-color: transparent;
            border-radius: 0;
            padding: 0;
            box-shadow: none;
            margin-bottom: 0.5rem;
        }
        .chat-card {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.6rem;
            box-shadow: 0 1px 3px rgba(15,17,17,0.1);
        }
        .assistant-bubble {
            border-left: 3px solid #ffa41c;
        }
        .user-bubble {
            border-left: 3px solid #007185;
        }
        .product-panel-title {
            font-weight: 600;
            margin-bottom: 0.4rem;
        }
        .product-card {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 0.5rem;
            margin-bottom: 0.6rem;
            box-shadow: 0 1px 3px rgba(15,17,17,0.12);
        }
        .product-img-wrapper {
            border-bottom: 1px solid #f0f0f0;
            margin-bottom: 0.35rem;
            text-align: center;
        }
        .product-placeholder {
            font-size: 0.8rem;
            color: #555;
            padding: 1rem 0;
        }
        .product-info {
            font-size: 0.8rem;
        }
        .product-title {
            font-weight: 600;
            margin-bottom: 0.2rem;
            color: #007185;
        }
        .product-price {
            color: #b12704;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }
        .product-desc {
            font-size: 0.8rem;
            margin-bottom: 0.2rem;
        }
        .product-meta {
            font-size: 0.7rem;
            color: #555;
        }
        .stButton > button {
            background-color: #ffa41c;
            color: #111111;
            border-radius: 999px;
            border: 1px solid #fcd200;
            padding: 0.25rem 0.9rem;
            font-weight: 500;
        }
        .stButton > button:hover {
            background-color: #f7a50c;
            border-color: #f0c14b;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="amazon-header">
            <div class="amazon-logo">Store Chatbot</div>
            <div class="amazon-tagline">-Product Q&A assistant powered by hybrid search</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        top_k = st.slider(
            "Results per query",
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

        if st.button("🧹 Start new chat"):
            st.session_state["messages"] = []
            st.session_state["last_products"] = None
            st.rerun()

        st.markdown("### Backend")
        st.write(f"Collection: `{Config.COLLECTION_NAME}`")
        st.write(f"Device: `{Config.DEVICE}`")
        st.write(f"LLM: `{Config.LLM_MODEL}`")

    # Session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "last_products" not in st.session_state:
        st.session_state["last_products"] = None

    # Layout: chat (left) + product results (right)
    col_main, col_right = st.columns([2.1, 1.2])

    # LEFT: CHAT
    with col_main:
        st.markdown('<div class="left-panel">', unsafe_allow_html=True)

        st.subheader("Chat with your product assistant")

        for msg in st.session_state["messages"]:
            role = msg["role"]
            css_class = "assistant-bubble" if role == "assistant" else "user-bubble"
            with st.chat_message(role):
                st.markdown(
                    f'<div class="chat-card {css_class}">{msg["content"]}</div>',
                    unsafe_allow_html=True,
                )

        uploaded_file = st.file_uploader(
            "Upload a product image (optional)",
            type=["png", "jpg", "jpeg"],
            key="image_uploader",
        )

        use_image = st.checkbox(
            "Use uploaded image for this question",
            value=(uploaded_file is not None),
            help="Uncheck to ignore the image and run a text-only search.",
        )

        user_input = st.chat_input(
            "Ask about a product (e.g., 'What is this product and how do I use it?')"
        )

        if user_input:
            # Show user message
            with st.chat_message("user"):
                content_html = f'<div class="chat-card user-bubble">{user_input}</div>'
                st.markdown(content_html, unsafe_allow_html=True)
                if uploaded_file is not None and use_image:
                    st.image(uploaded_file, caption="Uploaded image", width=220)

            st.session_state["messages"].append(
                {"role": "user", "content": user_input}
            )

            # Build query image
            query_image = None
            if uploaded_file is not None and use_image:
                image_bytes = uploaded_file.getvalue()
                query_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Run RAG
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
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                return

            # Show assistant answer (top product only) + top product image
            with st.chat_message("assistant"):
                if response.answer:
                    ans_html = (
                        f'<div class="chat-card assistant-bubble">{response.answer}</div>'
                    )
                    st.markdown(ans_html, unsafe_allow_html=True)
                    answer_text_for_history = response.answer

                    # Show top product image under the answer
                    if response.products:
                        top_entity = response.products[0]["entity"]
                        img_url = top_entity.get("image_url")
                        if img_url:
                            st.image(
                                img_url,
                                caption=top_entity.get("product_name", "Top product"),
                                width=220,
                            )
                else:
                    if not response.products:
                        answer_text_for_history = (
                            "I couldn't find any matching products for your query."
                        )
                    else:
                        answer_text_for_history = (
                            "I retrieved some products matching your query (generation is disabled)."
                        )
                    ans_html = (
                        f'<div class="chat-card assistant-bubble">{answer_text_for_history}</div>'
                    )
                    st.markdown(ans_html, unsafe_allow_html=True)

            st.session_state["messages"].append(
                {"role": "assistant", "content": answer_text_for_history}
            )
            st.session_state["last_products"] = response.products

        st.markdown("</div>", unsafe_allow_html=True)

    # RIGHT: MATCHING PRODUCTS
    with col_right:
        st.markdown('<div class="right-panel">', unsafe_allow_html=True)

        st.subheader("Recommended Products")

        products = st.session_state.get("last_products") or []
        if not products:
            st.markdown(
                "No products yet. Ask a question or upload an image to see matches here."
            )
        else:
            st.markdown(
                "<div class='product-panel-title'>Top results</div>",
                unsafe_allow_html=True,
            )
            # Show ALL retrieved products, ranked by relevance score
            for p in products:
                render_product_card(p["entity"], p["score"])

            with st.expander("Debug: raw product list"):
                for i, p in enumerate(products, start=1):
                    e = p["entity"]
                    st.markdown(
                        f"**{i}. {e.get('product_name')}**  "
                        f"(Price: {e.get('selling_price')}, Score: {p['score']:.4f})"
                    )
                    st.markdown(
                        f"<small>product_id: {e.get('product_id')} | "
                        f"image_url: {e.get('image_url')}</small>",
                        unsafe_allow_html=True,
                    )
                    if e.get("about_product"):
                        st.markdown(
                            f"<small>{str(e['about_product'])[:200]}...</small>",
                            unsafe_allow_html=True,
                        )
                    st.markdown("---")

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

