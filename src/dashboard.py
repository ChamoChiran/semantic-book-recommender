import pandas as pd
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModel
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.embeddings.base import Embeddings
from langchain_text_splitters import CharacterTextSplitter

import gradio as gr

books = pd.read_csv("../data/books_with_emotions.csv")
books['large_thumbnail'] = books['thumbnail'] + '&file=w800'

books['large_thumbnail'] = np.where(
    books['large_thumbnail'].isna(),
    "../images/cover-not-found.png",
    books['large_thumbnail']
)


# Create Embedding model
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
device = "cpu"
model.to(device)
model.eval()

def create_embeddings(texts, batch_size=32):
    all_embeds = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        with torch.no_grad():
            tokens = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)
            outputs = model(**tokens)
            token_embeddings = outputs.last_hidden_state
            attention_mask = tokens["attention_mask"]
            mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = (token_embeddings * mask_expanded).sum(1) / mask_expanded.sum(1)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeds.extend(embeddings.cpu().tolist())
    return all_embeds

class SimpleEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return create_embeddings(texts, batch_size=32)
    def embed_query(self, text):
        return create_embeddings([text], batch_size=1)[0]

emb = SimpleEmbeddings()
import os

persist_dir = "./chroma_db_cpu"

if not os.path.exists(persist_dir):
    # First time: build and persist
    raw_documents = TextLoader(
        '../data/tagged_descriptions.txt',
        encoding='utf-8'
    ).load()

    text_splitter = CharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=0,
        separator="\n",
        keep_separator=False
    )
    documents = text_splitter.split_documents(raw_documents)

    db_books = Chroma.from_documents(
        documents,
        embedding=emb,
        persist_directory=persist_dir
    )
else:
    # Later runs: just load
    db_books = Chroma(
        embedding_function=emb,
        persist_directory=persist_dir
    )

def retrieve_sementic_recommendations(
        query: str,
        category: str = None,
        tone = None,
        initial_top_k: int = 10,
        final_top_k: int = 16
) -> pd.DataFrame:

    recs = db_books.similarity_search(
        query,
        k = initial_top_k,
    )

    book_list = [int(rec.page_content.strip('""').split()[0]) for rec in recs]
    books_recs = books[books['isbn13'].isin(book_list)].head(final_top_k)

    if category != 'All':
        book_recs = books_recs[books_recs['simple_categories'] == category][:final_top_k]
    else:
        book_recs = books_recs.head(final_top_k)

    if tone == 'Happy':
        book_recs.sort_values(by='joy', ascending=False, inplace=True)
    elif tone == 'Surprising':
        book_recs.sort_values(by='surprise', ascending=True, inplace=True)
    elif tone == 'Angry':
        book_recs.sort_values(by='anger', ascending=False, inplace=True)
    elif tone == 'Suspenseful':
        book_recs.sort_values(by='fear', ascending=False, inplace=True)
    elif tone == 'Sad':
        book_recs.sort_values(by='sadness', ascending=False, inplace=True)


    return book_recs

def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_sementic_recommendations(query, category, tone)

    results = []

    for _, row in recommendations.iterrows():
        description = row['description']

        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row['authors'].split(';')

        if len(authors_split) == 2:
            authors_str = f'{authors_split[0]} and {authors_split[1]}'

        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row['authors']

        caption = f"{row['title']} by {authors_str}: {truncated_description}"

        results.append((row['large_thumbnail'], caption))
    return results

categories = ['All'] + sorted(books['simple_categories'].unique())
tones = ['All'] + ['Happy', 'Surprising', 'Angry', 'Suspenseful', 'Sad']

# Custom CSS for black-and-white metallic glassmorphic theme
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Dark mode variables (default) */
:root {
    --primary-bg: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0a0a0a 100%);
    --glass-bg: rgba(255, 255, 255, 0.03);
    --glass-bg-hover: rgba(255, 255, 255, 0.06);
    --glass-border: rgba(255, 255, 255, 0.12);
    --metallic-accent: linear-gradient(135deg, #505050 0%, #808080 50%, #505050 100%);
    --metallic-hover: linear-gradient(135deg, #606060 0%, #909090 50%, #606060 100%);
    --metallic-shine: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.05) 50%, rgba(255,255,255,0.15) 100%);
    --text-primary: #ffffff;
    --text-secondary: #b0b0b0;
    --text-tertiary: #808080;
    --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.4);
    --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.5);
    --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.6);
    --input-bg: rgba(255, 255, 255, 0.04);
    --input-border: rgba(255, 255, 255, 0.15);
}

/* Light mode variables */
.light-mode {
    --primary-bg: linear-gradient(135deg, #f5f5f5 0%, #e8e8e8 50%, #f5f5f5 100%);
    --glass-bg: rgba(0, 0, 0, 0.03);
    --glass-bg-hover: rgba(0, 0, 0, 0.06);
    --glass-border: rgba(0, 0, 0, 0.12);
    --metallic-accent: linear-gradient(135deg, #d0d0d0 0%, #a0a0a0 50%, #d0d0d0 100%);
    --metallic-hover: linear-gradient(135deg, #e0e0e0 0%, #b0b0b0 50%, #e0e0e0 100%);
    --metallic-shine: linear-gradient(135deg, rgba(255,255,255,0.8) 0%, rgba(255,255,255,0.4) 50%, rgba(255,255,255,0.8) 100%);
    --text-primary: #000000;
    --text-secondary: #404040;
    --text-tertiary: #707070;
    --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.1);
    --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.15);
    --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.2);
    --input-bg: rgba(0, 0, 0, 0.04);
    --input-border: rgba(0, 0, 0, 0.15);
}

body, .gradio-container {
    background: var(--primary-bg) !important;
    background-attachment: fixed !important;
    transition: background 0.4s ease !important;
}

.main {
    background: transparent !important;
}

/* Glassmorphic container */
.container {
    backdrop-filter: blur(20px) saturate(180%);
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 20px !important;
    box-shadow: var(--shadow-lg) !important;
    padding: 2rem !important;
    margin: 1rem auto !important;
    max-width: 1400px !important;
    transition: all 0.3s ease !important;
}

/* Header styling */
.header-section {
    text-align: center;
    padding: 2rem 0;
    margin-bottom: 2rem;
    position: relative;
}

.header-section h1 {
    background: var(--metallic-shine);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 3rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.5rem !important;
    letter-spacing: -0.02em;
    text-shadow: 0 2px 10px rgba(255, 255, 255, 0.1);
}

.header-section p {
    color: var(--text-secondary) !important;
    font-size: 1.1rem !important;
    font-weight: 300 !important;
}

/* Theme toggle button */
.theme-toggle {
    position: absolute;
    top: 1rem;
    right: 1rem;
    background: var(--glass-bg) !important;
    backdrop-filter: blur(10px);
    border: 1px solid var(--glass-border) !important;
    border-radius: 50px !important;
    padding: 0.6rem 1.2rem !important;
    color: var(--text-primary) !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    cursor: pointer;
    transition: all 0.3s ease !important;
    box-shadow: var(--shadow-sm) !important;
}

.theme-toggle:hover {
    background: var(--glass-bg-hover) !important;
    box-shadow: var(--shadow-md) !important;
    transform: translateY(-2px);
}

/* Input card */
.input-card {
    backdrop-filter: blur(20px) saturate(180%);
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 16px !important;
    padding: 2rem !important;
    box-shadow: var(--shadow-md) !important;
    transition: all 0.3s ease !important;
    position: relative;
    overflow: hidden;
}

.input-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: var(--metallic-shine);
    transition: left 0.5s ease;
    pointer-events: none;
}

.input-card:hover::before {
    left: 100%;
}

.input-card:hover {
    background: var(--glass-bg-hover) !important;
    box-shadow: var(--shadow-lg) !important;
    transform: translateY(-2px);
}

/* Textbox & inputs */
.input-card textarea,
.input-card input[type="text"] {
    background: var(--input-bg) !important;
    border: 1px solid var(--input-border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    padding: 1rem !important;
    font-size: 0.95rem !important;
    transition: all 0.3s ease !important;
    box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.2) !important;
}

.input-card textarea:focus,
.input-card input[type="text"]:focus {
    background: var(--input-bg) !important;
    border-color: var(--text-secondary) !important;
    box-shadow: 0 0 0 3px rgba(128, 128, 128, 0.15), inset 0 2px 8px rgba(0, 0, 0, 0.2) !important;
    outline: none !important;
}

.input-card textarea::placeholder,
.input-card input[type="text"]::placeholder {
    color: var(--text-tertiary) !important;
}

/* Dropdown styling */
.input-card select,
.input-card .dropdown-container,
.input-card .dropdown-container > div {
    background: var(--input-bg) !important;
    border: 1px solid var(--input-border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    transition: all 0.3s ease !important;
}

.input-card select:focus,
.input-card .dropdown-container:focus-within {
    border-color: var(--text-secondary) !important;
    box-shadow: 0 0 0 3px rgba(128, 128, 128, 0.15) !important;
}

.input-card select option {
    background: var(--input-bg) !important;
    color: var(--text-primary) !important;
}

/* Button metallic */
.recommend-btn {
    background: var(--metallic-accent) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 1rem 2rem !important;
    box-shadow: var(--shadow-md) !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    position: relative;
    overflow: hidden;
}

.recommend-btn::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: var(--metallic-shine);
    transition: left 0.5s ease;
}

.recommend-btn:hover::before {
    left: 100%;
}

.recommend-btn:hover {
    background: var(--metallic-hover) !important;
    box-shadow: var(--shadow-lg) !important;
    transform: translateY(-2px);
}

.recommend-btn:active {
    transform: translateY(0);
    box-shadow: var(--shadow-sm) !important;
}

/* Gallery */
.gallery-container {
    backdrop-filter: blur(20px) saturate(180%);
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 16px !important;
    padding: 2rem !important;
    box-shadow: var(--shadow-md) !important;
    margin-top: 2rem !important;
    transition: all 0.3s ease !important;
}

.gallery-header {
    color: var(--text-primary) !important;
    margin-bottom: 1rem;
    font-weight: 600;
    font-size: 1.3rem;
    letter-spacing: 0.01em;
}

#gallery {
    gap: 1.5rem !important;
}

#gallery .thumbnail-item {
    backdrop-filter: blur(10px);
    background: var(--glass-bg) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    transition: all 0.3s ease !important;
    box-shadow: var(--shadow-sm) !important;
    animation: fadeInUp 0.5s ease-out;
}

#gallery .thumbnail-item:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: var(--shadow-lg) !important;
    border-color: var(--text-secondary) !important;
}

#gallery img {
    transition: all 0.3s ease !important;
    border-radius: 12px 12px 0 0 !important;
    filter: grayscale(20%);
}

#gallery .thumbnail-item:hover img {
    transform: scale(1.05);
    filter: grayscale(0%);
}

#gallery .caption {
    background: var(--input-bg) !important;
    backdrop-filter: blur(10px);
    color: var(--text-primary) !important;
    padding: 1rem !important;
    font-size: 0.85rem !important;
    line-height: 1.5 !important;
    border-top: 1px solid var(--glass-border) !important;
}

/* Label styling */
label {
    color: var(--text-primary) !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    margin-bottom: 0.5rem !important;
    letter-spacing: 0.01em;
}

/* Loading spinner */
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.loading-spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid var(--glass-border);
    border-top-color: var(--text-primary);
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin-left: 0.5rem;
}

/* Shimmer effect */
@keyframes shimmer {
    0% { background-position: -1000px 0; }
    100% { background-position: 1000px 0; }
}

.loading {
    background: linear-gradient(90deg, 
        var(--glass-bg) 25%, 
        var(--glass-bg-hover) 50%, 
        var(--glass-bg) 75%);
    background-size: 1000px 100%;
    animation: shimmer 2s infinite;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-track {
    background: var(--glass-bg);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: var(--metallic-accent);
    border-radius: 10px;
    transition: all 0.3s ease;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--metallic-hover);
}

/* Responsive design */
@media (max-width: 768px) {
    .header-section h1 {
        font-size: 2rem !important;
    }
    
    .header-section p {
        font-size: 0.95rem !important;
    }
    
    #gallery {
        grid-template-columns: repeat(2, 1fr) !important;
        gap: 1rem !important;
    }
    
    .container {
        padding: 1rem !important;
    }
    
    .input-card {
        padding: 1rem !important;
    }
    
    .gallery-container {
        padding: 1rem !important;
    }
    
    .theme-toggle {
        position: static;
        margin: 1rem auto;
        display: block;
        width: fit-content;
    }
}

@media (max-width: 480px) {
    #gallery {
        grid-template-columns: repeat(1, 1fr) !important;
    }
    
    .header-section h1 {
        font-size: 1.5rem !important;
    }
}

/* Fade-in animation */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.fade-in {
    animation: fadeInUp 0.6s ease-out;
}

/* Pulse animation for loading state */
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

.pulse {
    animation: pulse 2s ease-in-out infinite;
}
"""

# JavaScript for theme toggle
theme_toggle_js = """
<script>
function toggleTheme() {
    const body = document.body;
    const container = document.querySelector('.gradio-container');
    const button = document.getElementById('theme-toggle-btn');
    
    if (body.classList.contains('light-mode')) {
        body.classList.remove('light-mode');
        container.classList.remove('light-mode');
        button.textContent = 'Light Mode';
        localStorage.setItem('theme', 'dark');
    } else {
        body.classList.add('light-mode');
        container.classList.add('light-mode');
        button.textContent = 'Dark Mode';
        localStorage.setItem('theme', 'light');
    }
}

// Load saved theme preference
window.addEventListener('DOMContentLoaded', (event) => {
    const savedTheme = localStorage.getItem('theme');
    const body = document.body;
    const container = document.querySelector('.gradio-container');
    const button = document.getElementById('theme-toggle-btn');
    
    if (savedTheme === 'light') {
        body.classList.add('light-mode');
        container.classList.add('light-mode');
        if (button) button.textContent = 'Dark Mode';
    }
});
</script>
"""

# Create custom theme with monochrome palette
custom_theme = gr.themes.Monochrome(
    primary_hue="slate",
    secondary_hue="zinc",
    neutral_hue="stone",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0a0a0a 100%)",
    body_background_fill_dark="linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0a0a0a 100%)",
)

with gr.Blocks(theme=custom_theme, css=custom_css, title="Semantic Book Recommender") as dashboard:
    # Header section with theme toggle
    gr.HTML(f"""
        <div class="header-section fade-in">
            <button id="theme-toggle-btn" class="theme-toggle" onclick="toggleTheme()">Light Mode</button>
            <h1>Semantic Book Recommender</h1>
            <p>Discover your next favorite book using AI-powered semantic search</p>
        </div>
        {theme_toggle_js}
    """)

    # Main content in centered column
    with gr.Column(elem_classes="container fade-in"):
        # Input card
        with gr.Column(elem_classes="input-card"):
            query_input = gr.Textbox(
                label="What kind of book are you looking for?",
                placeholder="E.g., A thrilling mystery novel set in a small town with unexpected plot twists...",
                lines=4,
                max_lines=6,
            )

            with gr.Row():
                category_input = gr.Dropdown(
                    label="Category",
                    choices=categories,
                    value="All",
                    scale=1
                )
                tone_input = gr.Dropdown(
                    label="Emotional Tone",
                    choices=tones,
                    value="All",
                    scale=1
                )

            recommend_button = gr.Button(
                "Find Books",
                elem_classes="recommend-btn",
                size="lg"
            )

        # Gallery section
        with gr.Column(elem_classes="gallery-container"):
            gr.HTML("<h3 class='gallery-header'>Recommended for You</h3>")
            output_gallery = gr.Gallery(
                label="",
                show_label=False,
                elem_id="gallery",
                columns=4,
                rows=2,
                height="auto",
                object_fit="cover",
                allow_preview=True,
            )

    # Footer
    gr.HTML("""
        <div style="text-align: center; padding: 2rem 0; color: var(--text-tertiary); font-size: 0.85rem;">
            <p>Powered by semantic embeddings and emotion analysis</p>
        </div>
    """)

    # Event binding
    recommend_button.click(
        fn=recommend_books,
        inputs=[query_input, category_input, tone_input],
        outputs=output_gallery
    )

if __name__ == "__main__":
    dashboard.launch()




































