"""
AdaIN Neural Style Transfer Studio — Interactive Web Application
Built with Streamlit and PyTorch
Author: Senior AI/ML Engineer
"""

import io
import time
from pathlib import Path
from typing import Optional

import streamlit as st
import torch
from PIL import Image
from torchvision.transforms import functional as TF

import adain_model as am

# ---------------------------------------------------------------------------
# 1. Page Configuration & UI Theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AdaIN Neural Style Transfer Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF4B4B, #FF8F00, #1E88E5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #888888;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #FF4B4B 0%, #FF8F00 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.65rem 1rem;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 2. Cached Model Loading (Ensures models are loaded only ONCE in memory)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing VGG19 & AdaIN Decoder...")
def get_models(device_name: str):
    """
    Loads VGG19 feature extractor and trained AdaIN decoder.
    Cached by Streamlit so models stay in GPU memory between user interactions.
    """
    device = torch.device(device_name)
    encoder = am.VGG19Encoder().to(device).eval()
    decoder = am.Decoder().to(device).eval()

    ckpt_path = Path("project/checkpoints/adain_decoder_100_batches.pth")
    if ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        decoder.load_state_dict(ckpt["decoder_state_dict"])

    return encoder, decoder


# ---------------------------------------------------------------------------
# 3. Sidebar: Configuration & Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    # Hardware Detection
    has_cuda = torch.cuda.is_available()
    device_opt = st.radio(
        "Compute Device",
        options=["cuda", "cpu"] if has_cuda else ["cpu"],
        index=0,
        help="CUDA leverages your NVIDIA RTX GPU for sub-second style transfer."
    )
    device = torch.device(device_opt)

    if has_cuda and device_opt == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        st.success(f"🎮 GPU: {gpu_name}\n💾 VRAM: {vram_gb:.1f} GB")
    else:
        st.info("💻 Running on CPU")

    st.markdown("---")
    st.subheader("🛠️ Algorithm Mode")
    mode = st.radio(
        "Select Mode:",
        options=[
            "⚡ Fast AdaIN Feed-Forward (Instant ~0.3s - Recommended)",
            "✨ High-Fidelity Perceptual NST (Iterative Optimization)"
        ],
        index=0,
        help="Fast AdaIN runs in a single forward pass (~0.3s). Perceptual NST runs multi-step backpropagation (takes ~15-25s on CPU, 5s on GPU)."
    )

    st.markdown("---")
    st.subheader("📐 Resolution & Parameters")
    default_res = 256 if device_opt == "cpu" else 384
    image_size = st.select_slider(
        "Resolution (Pixels)",
        options=[256, 384, 512],
        value=default_res,
        help="256px is optimal for fast CPU execution; 384px-512px for GPU."
    )

    if "Fast AdaIN" in mode:
        alpha = st.slider(
            "Style Strength (α)",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
            help="0.0 = Preserve Content Only, 1.0 = Full AdaIN Style Transfer"
        )
    else:
        style_weight = st.select_slider(
            "Style Weight",
            options=[1e6, 2e6, 4e6, 8e6],
            value=4e6,
            format_func=lambda x: f"{x:.0e}",
            help="Higher values force stronger brushstrokes and painterly textures."
        )
        default_steps = 40 if device_opt == "cpu" else 180
        steps = st.slider(
            "Optimization Steps",
            min_value=20,
            max_value=200,
            value=default_steps,
            step=10,
            help="40 steps is optimal for fast cloud CPU (~15s); 180+ on local GPU."
        )
        color_transfer = st.checkbox("Transfer Style Color Palette", value=True,
                                     help="Adapts scene lighting and colors to match the style artwork.")
        if device_opt == "cpu":
            st.caption("💡 *Cloud CPU Tip: 40 steps takes ~15 seconds. For instant generation in 0.3s, select 'Fast AdaIN' above.*")

# Load models from cache
encoder, decoder = get_models(device_opt)

# ---------------------------------------------------------------------------
# 4. Main Header & Architectural Explanation
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">🎨 AdaIN Neural Style Transfer Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Arbitrary Neural Style Transfer in Real-Time with PyTorch & GPU Acceleration</div>', unsafe_allow_html=True)

with st.expander("📖 Architecture & Technology Overview (Click to expand)", expanded=False):
    st.markdown(r"""
    **How It Works:**
    1. **VGG19 Feature Extraction**: Content and style images are passed through a frozen VGG19 encoder to extract feature maps across multiple layers (`relu1_1` to `relu5_1`).
    2. **Adaptive Instance Normalization (AdaIN)**: Matches the channel-wise mean and variance of content features to the style features:
       $$\text{AdaIN}(x, y) = \sigma(y) \left( \frac{x - \mu(x)}{\sigma(x)} \right) + \mu(y)$$
    3. **Decoder Network**: A 7-layer convolutional decoder reconstructs the stylized feature map back into an RGB image.
    4. **High-Fidelity Perceptual NST**: Uses multi-scale Gram matrices ($G = \frac{F F^T}{C \cdot H \cdot W}$) and Total Variation loss to render vibrant oil-painting textures and canvas brushstrokes.
    """)

# ---------------------------------------------------------------------------
# 5. Image Input Selection (Sample Gallery or Upload)
# ---------------------------------------------------------------------------
st.subheader("🖼️ Select Content & Style Images")

tab_presets, tab_upload = st.tabs(["📁 Sample Gallery (One-Click Demos)", "📤 Upload Custom Images"])

# Sample image directories (checks standalone samples/ first, then project/dataset/)
c_dir = Path("samples/content") if Path("samples/content").exists() else Path("project/dataset/test2017/test2017/test2017")
s_dir = Path("samples/style") if Path("samples/style").exists() else Path("project/dataset/style/train_8")

sample_contents = {
    "Train under Stone Bridge (000000006499.jpg)": c_dir / "000000006499.jpg",
    "City Street Taxi (000000000001.jpg)": c_dir / "000000000001.jpg",
    "Girl with Phone (000000000554.jpg)": c_dir / "000000000554.jpg",
}

sample_styles = {
    "Dutch Baroque Lute Player (80695.jpg)": s_dir / "80695.jpg",
    "Oil Painting of Horses (8.jpg)": s_dir / "8.jpg",
    "Impressionist Portrait (80037.jpg)": s_dir / "80037.jpg",
    "Abstract Artwork (80.jpg)": s_dir / "80.jpg",
}

content_pil: Optional[Image.Image] = None
style_pil: Optional[Image.Image] = None

with tab_presets:
    col_c, col_s = st.columns(2)
    with col_c:
        chosen_c = st.selectbox("Choose Content Image:", list(sample_contents.keys()))
        c_path = sample_contents[chosen_c]
        if c_path.is_file():
            content_pil = Image.open(c_path).convert("RGB")
            st.image(content_pil, caption=f"Content: {c_path.name}", use_container_width=True)
    with col_s:
        chosen_s = st.selectbox("Choose Style Artwork:", list(sample_styles.keys()))
        s_path = sample_styles[chosen_s]
        if s_path.is_file():
            style_pil = Image.open(s_path).convert("RGB")
            st.image(style_pil, caption=f"Style: {s_path.name}", use_container_width=True)

with tab_upload:
    col_up_c, col_up_s = st.columns(2)
    with col_up_c:
        up_c = st.file_uploader("Upload Content Image", type=["jpg", "jpeg", "png"])
        if up_c is not None:
            content_pil = Image.open(up_c).convert("RGB")
            st.image(content_pil, caption="Custom Content Image", use_container_width=True)
    with col_up_s:
        up_s = st.file_uploader("Upload Style Artwork", type=["jpg", "jpeg", "png"])
        if up_s is not None:
            style_pil = Image.open(up_s).convert("RGB")
            st.image(style_pil, caption="Custom Style Artwork", use_container_width=True)

# ---------------------------------------------------------------------------
# 6. Execution & Results Display
# ---------------------------------------------------------------------------
st.markdown("---")
if st.button("✨ Generate Stylized Artwork", use_container_width=True):
    if content_pil is None or style_pil is None:
        st.error("Please select or upload both Content and Style images.")
    else:
        # Preprocess directly using PIL bicubic resize (clean & fast)
        c_resized = content_pil.resize((image_size, image_size), Image.Resampling.BICUBIC)
        s_resized = style_pil.resize((image_size, image_size), Image.Resampling.BICUBIC)

        c_tensor = TF.to_tensor(c_resized).unsqueeze(0).to(device)
        s_tensor = TF.to_tensor(s_resized).unsqueeze(0).to(device)

        status = st.empty()
        t_start = time.time()

        if "Fast AdaIN" in mode:
            status.info("⚡ Running AdaIN real-time feed-forward transfer...")
            with torch.inference_mode():
                out_tensor = am.style_transfer(encoder, decoder, c_tensor, s_tensor, alpha=alpha)
                if device.type == "cuda":
                    torch.cuda.synchronize()
            # Normalize visualization range
            out_norm = (out_tensor - out_tensor.min()) / (out_tensor.max() - out_tensor.min() + 1e-8)
            res_pil = TF.to_pil_image(out_norm[0].clamp(0, 1).cpu())
            elapsed = time.time() - t_start
        else:
            status.info(f"🎨 Running High-Fidelity Perceptual NST ({steps} steps) on {device}...")
            res_tensor = am.perceptual_style_transfer(
                content_tensor=c_tensor,
                style_tensor=s_tensor,
                steps=steps,
                lr=0.035,
                style_weight=style_weight,
                content_weight=0.6,
                tv_weight=1e-5,
                color_transfer=color_transfer,
                device=device,
                encoder=encoder
            )
            res_pil = TF.to_pil_image(res_tensor[0].cpu().clamp(0, 1))
            elapsed = time.time() - t_start

        status.empty()
        st.success(f"🎉 Stylization Completed in **{elapsed:.2f} seconds** on **{device_opt.upper()}**!")

        # 3-Column side-by-side visualization
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(c_resized, caption="Original Content Image", use_container_width=True)
        with col2:
            st.image(s_resized, caption="Style Artwork", use_container_width=True)
        with col3:
            st.image(res_pil, caption="Generated Stylized Artwork", use_container_width=True)

        # Download Button
        buf = io.BytesIO()
        res_pil.save(buf, format="PNG")
        st.download_button(
            label="💾 Download Stylized Artwork (PNG)",
            data=buf.getvalue(),
            file_name="stylized_artwork.png",
            mime="image/png",
            use_container_width=True
        )

# ---------------------------------------------------------------------------
# 7. Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 0.88rem;'>"
    "🎨 <b>AdaIN Neural Style Transfer Studio</b> • Developed by "
    "<a href='https://github.com/mohit3737/Neural-Style-Transfer-AdaIN' target='_blank' style='color: #FF4B4B; text-decoration: none; font-weight: bold;'>Mohit Ramoliya</a> "
    "• PyTorch & Streamlit"
    "</div>",
    unsafe_allow_html=True
)
