import requests
import streamlit as st
from config import BACKEND_URL

st.set_page_config(
    page_title="VERIONA",
    layout="wide",
    initial_sidebar_state="collapsed"
)
# st.markdown("""
# <style>
# .stApp{
#     background:white;
# }
# </style>
# """, unsafe_allow_html=True)

st.markdown("""
<style>
.header{
    background:#F3F4F6;
    padding:18px 30px;
    border-radius:18px;
    margin-bottom:20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

div[data-testid="stFileUploader"] section{
    border:2px dashed #8B8B8B;
    border-radius:18px;
    background:#1B1B1B;
    padding:45px;
}

div[data-testid="stFileUploader"]{
    width:100%;
}

div[data-testid="stFileUploader"] button{
    background:#3B82F6;
    color:white;
    border-radius:10px;
    font-size:18px;
    padding:12px 30px;
}

</style>
""", unsafe_allow_html=True)

#session state variables
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None

if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

st.markdown('<div class="header">', unsafe_allow_html=True)
logo_col, menu_col = st.columns([2, 8])

with logo_col:
    st.markdown("## VERIONA")

with menu_col:
    tools_col, ai_col, solutions_col = st.columns([1, 1, 1])

    with tools_col:
        with st.popover("TOOLS"):
            left, right = st.columns(2)

            with left:
                st.subheader("Audio Extraction")
                st.write("Extracts speech from uploaded videos while preserving audio quality.")

                st.subheader("Translation")
                st.write("Translates Urdu speech into natural English.")

                st.subheader("Video Dubbing")
                st.write("Synchronizes generated speech with the original video.")

            with right:
                st.subheader("Speech Recognition")
                st.write("Converts spoken Urdu into machine-readable text.")

                st.subheader("Voice Synthesis")
                st.write("Produces English speech while preserving the speaker's voice.")

    with ai_col:
        with st.popover("AI"):
            st.subheader("VISION")

            st.divider()
            st.write("""Our vision is to remove language barriers through intelligent
            video dubbing while preserving speaker identity and context.""")

            st.divider()
            st.subheader("GOAL")
            st.write("""Deliver high-quality Urdu-to-English video dubbing using speech 
            recognition, translation, voice preservation and automatic video generation.""")

    with solutions_col:
        with st.popover("SOLUTIONS"):
            left, right = st.columns(2)

            with left:
                st.subheader("ENTERTAINMENT")
                st.write("English dubbed Pakistani movies and dramas.")

                st.subheader("CONTENT CREATION")
                st.write("Expand audience reach across multiple languages.")

            with right:
                st.subheader("BUSINESS")
                st.write("Training videos for a better user experience.")

st.markdown('<div class="header">', unsafe_allow_html=True)

st.markdown("""
<style>

.hero{
    background:#111111;
    color:white;
    border-radius:25px;
    padding:70px 60px;
    margin-top:20px;
    margin-bottom:40px;
}

.hero h1{
    font-size:64px;
    font-weight:800;
    text-align:center;
    margin-bottom:8px;
}

.hero h3{
    text-align:center;
    color:#CFCFCF;
    font-size:24px;
    font-weight:400;
}

.upload-box{
    max-width:700px;
    margin:auto;
    margin-top:40px;
}

</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero">', unsafe_allow_html=True)

st.markdown("""
<h1>AUTOMATED VIDEO DUBBING</h1>

<h3>
Urdu to English Dubbing with Voice Preservation
</h3>
""", unsafe_allow_html=True)

st.markdown('<div class="upload-box">', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload your video here",
    type=["mp4"],
    label_visibility="collapsed"
)

st.markdown("</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file
    st.success("Video uploaded successfully.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Filename", uploaded_file.name)

    with c2:
        st.metric("Size", f"{round(uploaded_file.size/(1024*1024),2)} MB")

    with c3:
        st.metric("Format", "MP4")

    with c4:
        st.metric("Status", "Ready")

left, right = st.columns([7, 2])
with right:
    start_pipeline = st.button(
        "START DUBBING",
        type="primary",
        use_container_width=True
    )

if start_pipeline:
    progress = st.progress(0)
    status = st.empty()

    try:
        status.info("Uploading video")
        progress.progress(10)
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file,
                uploaded_file.type
            )
        }

        response = requests.post(f"{BACKEND_URL}/pipeline/process",
            files=files, timeout=3600)
        progress.progress(100)

        if response.status_code != 200:
            raise Exception(response.text)

        st.session_state.pipeline_result = response.json()

        status.success("Video dubbing completed successfully.")

    except Exception as e:
        progress.empty()
        status.error(str(e))

#results
if st.session_state.pipeline_result is not None:
    result = st.session_state.pipeline_result

    if result["status"] == "success":
        data = result["data"]
        st.divider()
        st.header("PIPELINE RESULTS")

        info1, info2 = st.columns(2)
        with info1:
            st.metric("VIDEO ID", data["video_id"])

        with info2:
            if "processing_time" in data:
                st.metric("PROCESSING TIME", f"{data['processing_time']} sec")

        st.divider()
        st.subheader("ENGLISH TRANSLATION")
        st.write(data["translation"])

        st.divider()
        st.subheader("GENERATED ENGLISH AUDIO")
        try:
            st.audio( data["generated_audio"])
        except:
            st.warning("Generated audio not found.")

        st.divider()
        st.subheader("DUBBED VIDEO")
        try:
            st.video(data["dubbed_video"])
        except:
            st.warning("Dubbed video not found.")

        try:
            with open(data["dubbed_video"], "rb") as file:
                st.download_button(
                    "DOWNLOAD DUBBED VIDEO",
                    data=file,
                    file_name="dubbed_video.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
        except:
            st.error("Unable to prepare the download.")

#side bar
with st.sidebar:
    st.title("VERIONA")
    st.divider()

    st.header("AI MODELS")
    st.success("Speech Recognition\n\nWhisper Large-v3")
    st.success("Translation\n\nNLLB-200")
    st.success("Voice Generation\n\nF5-TTS")
    st.success("Video Dubbing\n\nFFmpeg")

    st.divider()
    st.header("SUPPORTED LANGUAGES")

    st.write("**Source Language**")
    st.write("Urdu")

    st.write("")

    st.write("**Target Language**")
    st.write("English")

#footer
st.divider()

left, middle, right = st.columns(3)
with left:
    st.caption("VERIONA")

with middle:
    st.caption("AI VIDEO DUBBING SYSTEM")

with right:
    st.caption("FINAL YEAR PROJECT")