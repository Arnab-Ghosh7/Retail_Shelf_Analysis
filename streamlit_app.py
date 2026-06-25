import streamlit as st
import requests
import io
from PIL import Image


st.set_page_config(
    page_title="SS Pharmacy - Retail Shelf Analysis",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main container styling */
    .main {
        background-color: #0f172a;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }

    /* Title and Header customization */
    .title-text {
        font-size: 3rem !important;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    .subtitle-text {
        font-size: 1.25rem;
        color: #94a3b8;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* Card design for metrics and information */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 1.5rem;
    }

    .metric-header {
        font-size: 1rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.25rem;
    }

    .metric-value {
        font-size: 2.25rem;
        font-weight: 700;
        color: #38bdf8;
    }

    /* Sidebar customization */
    .css-1d391tw {
        background-color: #1e293b;
    }

    /* Highlight tags */
    .brand-tag {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #ffffff;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://images.unsplash.com/photo-1542838132-92c53300491e?auto=format&fit=crop&q=80&w=400", width="stretch")
    st.markdown("### ⚙️ API Configuration")


    api_url = st.text_input(
        "Modal API Base URL",
        value="https://jitghosh81287--sku110k-api-api.modal.run",
        help="Paste the deployed Modal URL here. E.g. https://<username>--sku110k-api-api.modal.run"
    )

    st.markdown("---")
    st.markdown("""
    ### 📝 Assignment Objective
    Build an AI system that:
    - Detects individual products on retail shelves.
    - Groups detected products by brand similarity.
    - Generates visualization outputs.
    - Returns structured inference results via an API.

    ### 🔬 Technical Stack
    - **YOLOv8 Nano** (Object Detection)
    - **ResNet-50** (Visual Embedding Extractor)
    - **DBSCAN** (Cosine distance clustering)
    - **FastAPI** & **Modal** (GPU Deployment)
    """)


st.markdown("<h1 class='title-text'>Retail Shelf Analysis System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle-text'>Deep Learning-powered shelf monitoring & brand clustering pipeline</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📥 Input Source")

    input_method = st.radio("Choose input method:", ("Upload Image File", "Provide Image URL"))

    image_to_process = None

    if input_method == "Upload Image File":
        uploaded_file = st.file_uploader("Choose a retail shelf image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image_to_process = Image.open(uploaded_file).convert("RGB")
            st.image(image_to_process, caption="Uploaded Image", width="stretch")
    else:
        url_input = st.text_input("Enter Image URL:")
        if url_input:
            try:
                response = requests.get(url_input, timeout=10)
                image_to_process = Image.open(io.BytesIO(response.content)).convert("RGB")
                st.image(image_to_process, caption="Fetched Image", width="stretch")
            except Exception as e:
                st.error(f"Error fetching image: {e}")

    st.markdown("---")


    run_button = st.button("🚀 Analyze Shelf Layout", type="primary", width="stretch")

with col2:
    st.markdown("### 📊 Inference Results")

    if run_button:
        if image_to_process is None:
            st.warning("Please upload an image or provide a valid URL first.")
        else:
            with st.spinner("Invoking Modal GPU Backend (YOLO Object Detection & ResNet Clustering)..."):
                try:

                    base_url = api_url.rstrip("/")
                    predict_endpoint = f"{base_url}/predict"


                    response = None
                    if input_method == "Upload Image File":
                        img_byte_arr = io.BytesIO()
                        image_to_process.save(img_byte_arr, format='JPEG')
                        img_byte_arr = img_byte_arr.getvalue()

                        files = {'file': ('image.jpg', img_byte_arr, 'image/jpeg')}
                        response = requests.post(predict_endpoint, files=files, timeout=45)
                    else:
                        data = {'image_url': url_input}
                        response = requests.post(predict_endpoint, data=data, timeout=45)

                    if response.status_code == 200:
                        results = response.json()
                        objects = results.get("objects", [])
                        vis_path = results.get("visualization_path", "")


                        total_products = len(objects)
                        brand_groups = set([obj.get("group_id") for obj in objects])
                        total_brands = len(brand_groups)


                        m_col1, m_col2 = st.columns(2)
                        with m_col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-header">Products Detected</div>
                                <div class="metric-value">{total_products}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with m_col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-header font-sans">Brand Groups (Clusters)</div>
                                <div class="metric-value" style="color: #a855f7;">{total_brands}</div>
                            </div>
                            """, unsafe_allow_html=True)


                        if vis_path:

                            vis_filename = vis_path.split("/")[-1]
                            vis_url = f"{base_url}/outputs/{vis_filename}"

                            st.markdown("#### Bounding Boxes & Brand Grouping")
                            try:
                                vis_response = requests.get(vis_url, timeout=10)
                                if vis_response.status_code == 200:
                                    vis_image = Image.open(io.BytesIO(vis_response.content))
                                    st.image(vis_image, caption="Color-coded Product Groups", width="stretch")
                                else:
                                    st.error(f"Failed to fetch visualization image. Status code: {vis_response.status_code}")
                            except Exception as e:
                                st.error(f"Error fetching visualization image: {e}")


                        st.markdown("#### Detected Products Detail")

                        color_palette = [
                            "#FF3366", "#33FF66", "#3366FF", "#FFFF33", "#FF33FF", "#33FFFF",
                            "#FF9933", "#9933FF", "#33FF99", "#FF3399", "#99FF33", "#3399FF"
                        ]

                        unique_brands_list = sorted(list(brand_groups))
                        brand_colors_html = ""
                        for idx, b_id in enumerate(unique_brands_list):
                            color = color_palette[idx % len(color_palette)]
                            brand_colors_html += f'<span class="brand-tag" style="background-color: {color};">{b_id}</span>'

                        st.markdown(brand_colors_html, unsafe_allow_html=True)

                        with st.expander("Show Detailed Object List"):
                            for i, obj in enumerate(objects):
                                st.write(f"Product #{i+1}: BBox: {obj['bbox']} | Confidence: {obj['confidence']:.2f} | Brand: **{obj['group_id']}**")


                        with st.expander("Show Raw API Response"):
                            st.json(results)

                    else:
                        st.error(f"Inference failed. Status code: {response.status_code}")
                        st.text(response.text)

                except Exception as e:
                    st.error(f"Error connecting to backend API: {e}")
                    st.info("Check if your Modal API is deployed and the URL in the sidebar is correct.")
    else:
        st.info("Upload an image on the left and click 'Analyze Shelf Layout' to run inference.")
