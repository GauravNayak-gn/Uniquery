
import os
import logging
import json
import wave
import sys
import piper

logger = logging.getLogger(__name__)

# Constants
# Using a high-quality but fast voice: en_US-lessac-medium
VOICE_NAME = "en_US-lessac-medium"
MODEL_URL = f"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/{VOICE_NAME}.onnx"
CONFIG_URL = f"https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/{VOICE_NAME}.onnx.json"

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "piper_models")
MODEL_PATH = os.path.join(MODEL_DIR, f"{VOICE_NAME}.onnx")
CONFIG_PATH = os.path.join(MODEL_DIR, f"{VOICE_NAME}.onnx.json")

_voice = None # Singleton voice instance

def ensure_model_exists():
    """Checks if the Piper model exists, otherwise downloads it."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    if not os.path.exists(MODEL_PATH) or not os.path.exists(CONFIG_PATH):
        logger.info(f"⬇️ Piper model '{VOICE_NAME}' not found. Downloading...")
        try:
            import requests
            
            # Download ONNX
            logger.info(f"Downloading model binary from {MODEL_URL}...")
            r_onnx = requests.get(MODEL_URL, allow_redirects=True)
            r_onnx.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(r_onnx.content)
            
            # Download JSON Config
            logger.info(f"Downloading model config from {CONFIG_URL}...")
            r_json = requests.get(CONFIG_URL, allow_redirects=True)
            r_json.raise_for_status()
            with open(CONFIG_PATH, "wb") as f:
                f.write(r_json.content)
                
            logger.info("✅ Piper model downloaded successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to download Piper model: {e}")
            if os.path.exists(MODEL_PATH): os.remove(MODEL_PATH)
            if os.path.exists(CONFIG_PATH): os.remove(CONFIG_PATH)
            raise e
    else:
        logger.info(f"✅ Piper model '{VOICE_NAME}' found locally.")

def generate_audio_stream(text: str):
    """
    Generates audio for the given text using the Piper CLI via subprocess.
    Returns a BytesIO buffer containing the WAV data.
    """
    ensure_model_exists()
    
    import subprocess
    import io
    import tempfile
    import shutil
    
    # Locate piper executable
    piper_cmd = "piper"
    
    # Create a temp file path (we close the file handle so piper can open it)
    # We use a context manager for the prompt creation, but we need a named file for output
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        tmp_wav_path = tmp_wav.name
    
    try:
        # Command: echo "text" | piper -m <model> -f <temp_file>
        cmd = [piper_cmd, "-m", MODEL_PATH, "-f", tmp_wav_path]
        
        logger.info(f"🎤 Running Piper CLI: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        encoded_text = text.encode("utf-8")
        stdout_data, stderr_data = process.communicate(input=encoded_text)
        
        if process.returncode != 0:
            err_msg = stderr_data.decode(errors='replace')
            logger.error(f"❌ Piper CLI failed (code {process.returncode}): {err_msg}")
            raise RuntimeError(f"Piper CLI failed: {err_msg}")
            
        # Read the generated WAV file
        with open(tmp_wav_path, "rb") as f:
            wav_data = f.read()
            
        if len(wav_data) < 100:
             logger.warning(f"⚠️ Piper produced suspiciously small output ({len(wav_data)} bytes). stderr: {stderr_data.decode()}")
        
        return io.BytesIO(wav_data)

    except Exception as e:
        logger.error(f"❌ Failed to run Piper CLI: {e}")
        import shutil
        if not shutil.which(piper_cmd):
            logger.error("❌ 'piper' executable not found in PATH.")
        raise e
    finally:
        # Cleanup
        if os.path.exists(tmp_wav_path):
            try:
                os.remove(tmp_wav_path)
            except Exception:
                pass


