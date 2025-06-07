import streamlit as st
import whisper
import torch
import os
import subprocess
import numpy as np
from moviepy.editor import VideoFileClip, TextClip, ColorClip, CompositeVideoClip
from moviepy.config import change_settings
from textblob import TextBlob
import emoji
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import spacy.cli
import cv2
import mediapipe as mp
from pathlib import Path
import librosa
import scipy
from scipy.signal import find_peaks
import statistics
import numpy as np

# Initialize ImageMagick
possible_paths = [
    # r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe",
    # r"C:\Program Files\ImageMagick-7.1.1-Q16\magick.exe",
    # r"C:\Program Files (x86)\ImageMagick-7.1.1-Q16-HDRI\magick.exe",
    # r"C:\Program Files (x86)\ImageMagick-7.1.1-Q16\magick.exe",
    r"D:\Dikha_workspace\Dikha_workspace\videodb-reel-main\project\ImageMagick-7.1.1-47-Q16-HDRI-x64-dll.exe"
]


# change_settings({"FFMPEG_BINARY": r"D:\Dikha_workspace\videodb-reel-main\ffmpeg-master-latest-win64-gpl-shared\ffmpeg-master-latest-win64-gpl-shared"})
# Initialize magick_path
magick_path = None

# First check if we have a local installation
local_magick = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagemagick")
if os.path.exists(local_magick):
    for root, dirs, files in os.walk(local_magick):
        if "magick.exe" in files:
            magick_path = os.path.join(root, "magick.exe")
            break

# If not found locally, check system paths
if not magick_path:
    for path in possible_paths:
        if os.path.exists(path):
            magick_path = path
            break

if magick_path:
    change_settings({"IMAGEMAGICK_BINARY": magick_path})
    st.sidebar.success(f"✅ Found ImageMagick at: {magick_path}")
else:
    st.sidebar.error("⚠️ ImageMagick not found. Installing locally...")
    try:
        from setup_imagemagick import setup_imagemagick
        magick_path = setup_imagemagick()
        if magick_path and os.path.exists(magick_path):
            change_settings({"IMAGEMAGICK_BINARY": magick_path})
            st.sidebar.success(f"✅ ImageMagick installed at: {magick_path}")
        else:
            st.error("Failed to install ImageMagick. Please install it manually from https://imagemagick.org/script/download.php#windows")
    except Exception as e:
        st.error(f"Error installing ImageMagick: {str(e)}")
        st.info("Please install ImageMagick manually from https://imagemagick.org/script/download.php#windows")

# Check CUDA availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.info(f"Using device: {DEVICE}")

# Try to enable GPU for spaCy if available
SPACY_GPU = False
if DEVICE == "cuda":
    try:
        import cupy # type: ignore
        spacy.require_gpu()
        spacy.prefer_gpu()
        SPACY_GPU = True
        st.sidebar.success("✅ spaCy GPU acceleration enabled")
    except (ImportError, ValueError, ModuleNotFoundError) as e:
        st.sidebar.warning("⚠️ spaCy running on CPU (install cupy for GPU acceleration)")
        SPACY_GPU = False

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

def process_with_gpu(texts, batch_size=32):
    """Process texts in batches"""
    if not texts:
        return []
    try:
        return list(nlp.pipe(texts, batch_size=batch_size))
    except Exception as e:
        st.warning(f"Batch processing failed, falling back to single processing: {str(e)}")
        return [nlp(text) for text in texts]

def simple_tokenize(text):
    """Simple tokenization without NLTK"""
    # Split on spaces and punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    return [word.strip() for word in text.split() if word.strip()]

def analyze_topic_importance(text_segments):
    """
    Analyze importance of topics in text segments using simplified NLP
    """
    # Preprocess text
    def preprocess_text(text):
        return ' '.join(simple_tokenize(text))
    
    processed_texts = [preprocess_text(text) for text in text_segments]
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    
    # Process all texts at once with GPU
    docs = process_with_gpu(text_segments)
    
    # Calculate importance scores
    importance_scores = []
    for idx, (text, doc) in enumerate(zip(text_segments, docs)):
        # Entity score (presence of named entities)
        entity_score = len([ent for ent in doc.ents]) / max(len(text.split()), 1)
        
        # Key phrase score (noun chunks)
        phrase_score = len(list(doc.noun_chunks)) / max(len(text.split()), 1)
        
        # TF-IDF score
        tfidf_score = np.mean(tfidf_matrix[idx].toarray())
        
        # Sentiment intensity
        blob = TextBlob(text)
        sentiment_score = abs(blob.sentiment.polarity)
        
        # Question-Answer pattern score (simple regex)
        has_question = bool(re.search(r'\?', text))
        has_answer = len(text.split()) > 5
        qa_score = 1.0 if (has_question and has_answer) else 0.0
        
        # Length score (normalized)
        length_score = min(len(text.split()) / 50, 1.0)  # Cap at 50 words
        
        # Combine scores with weights
        importance = (
            0.25 * entity_score +      # Weight for named entities
            0.20 * phrase_score +      # Weight for key phrases
            0.20 * tfidf_score +       # Weight for term importance
            0.15 * sentiment_score +   # Weight for sentiment strength
            0.10 * qa_score +          # Weight for Q&A patterns
            0.10 * length_score        # Weight for length
        )
        
        importance_scores.append(importance)
    
    return importance_scores

def analyze_segment_context(text, prev_text="", next_text=""):
    """Analyze how well a segment fits in context"""
    doc = nlp(text)
    prev_doc = nlp(prev_text) if prev_text else None
    next_doc = nlp(next_text) if next_text else None
    
    score = 0
    
    # Check for sentence completeness
    if text.strip().endswith(('.', '!', '?')):
        score += 0.2
    
    # Check for context continuity with previous text
    if prev_doc:
        # Check for shared entities
        prev_ents = set(e.text.lower() for e in prev_doc.ents)
        curr_ents = set(e.text.lower() for e in doc.ents)
        shared_ents = prev_ents.intersection(curr_ents)
        if shared_ents:
            score += 0.2 * (len(shared_ents) / len(curr_ents) if curr_ents else 0)
        
        # Check for pronoun references that make sense
        has_pronouns = any(token.pos_ == "PRON" for token in doc)
        if not has_pronouns or (has_pronouns and shared_ents):
            score += 0.1
    else:
        # If no previous text, prefer segments that start with proper context
        if not any(token.pos_ == "PRON" for token in list(doc)[:2]):
            score += 0.3
    
    # Check for context continuity with next text
    if next_doc:
        # Check for shared entities
        next_ents = set(e.text.lower() for e in next_doc.ents)
        curr_ents = set(e.text.lower() for e in doc.ents)
        shared_ents = next_ents.intersection(curr_ents)
        if shared_ents:
            score += 0.2 * (len(shared_ents) / len(curr_ents) if curr_ents else 0)
    
    # Check for topic coherence
    topic_words = set()
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN", "VERB"] and not token.is_stop:
            topic_words.add(token.lemma_.lower())
    
    if len(topic_words) >= 3:  # Has enough topic-related content
        score += 0.2
    
    # Check for meaningful content
    has_entities = any(doc.ents)
    has_key_phrases = any(doc.noun_chunks)
    if has_entities and has_key_phrases:
        score += 0.1
    
    return score

def get_important_segments(transcription, min_duration=30, max_duration=60):
    """
    Get important segments based on topic relevance and conversation patterns
    Ensures segments are between min_duration and max_duration seconds
    """
    segments = []
    current_segment = {"text": "", "start": 0, "end": 0}
    
    # Split into basic segments based on natural breaks
    for segment in transcription["segments"]:
        text = segment.get("text", "").strip()
        if not text:
            continue
            
        # Start new segment if current one is too long or contains natural break
        if (len(current_segment["text"].split()) > 50 or 
            any(text.endswith(p) for p in ['.', '?', '!']) or 
            len(text.split()) > 15):
            
            if current_segment["text"]:
                segments.append(current_segment.copy())
            current_segment = {
                "text": text,
                "start": segment["start"],
                "end": segment["end"]
            }
        else:
            current_segment["text"] += " " + text
            current_segment["end"] = segment["end"]
    
    # Add final segment
    if current_segment["text"]:
        segments.append(current_segment)
    
    # Combine short segments or split long ones to meet duration requirements
    final_segments = []
    current = None
    
    for segment in segments:
        duration = segment["end"] - segment["start"]
        
        if duration < min_duration:
            # Combine with previous if possible
            if current and (current["end"] - current["start"]) + duration <= max_duration:
                current["text"] += " " + segment["text"]
                current["end"] = segment["end"]
            else:
                current = segment.copy()
                final_segments.append(current)
        elif duration > max_duration:
            # Split into multiple segments
            num_splits = int(duration / max_duration) + 1
            split_duration = duration / num_splits
            
            for i in range(num_splits):
                start_time = segment["start"] + (i * split_duration)
                end_time = start_time + split_duration
                
                # For text, split roughly by word count
                words = segment["text"].split()
                words_per_split = len(words) // num_splits
                start_idx = i * words_per_split
                end_idx = start_idx + words_per_split if i < num_splits - 1 else len(words)
                
                split_segment = {
                    "text": " ".join(words[start_idx:end_idx]),
                    "start": start_time,
                    "end": end_time
                }
                final_segments.append(split_segment)
        else:
            final_segments.append(segment.copy())
    
    return final_segments

def analyze_sentiment(text):
    """Analyze sentiment of text using TextBlob"""
    analysis = TextBlob(text)
    return (analysis.sentiment.polarity + 1) / 2

def check_ffmpeg():
    """Check if FFmpeg is installed and has GPU support"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True)
        has_gpu = 'enable-cuda' in result.stdout or 'enable-nvenc' in result.stdout
        if has_gpu:
            st.sidebar.success("FFmpeg GPU acceleration available")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def extract_audio(video_path):
    """Extract audio from video file using GPU-accelerated FFmpeg if available"""
    try:
        if not os.path.exists(video_path):
            st.error("Video file not found")
            return None
            
        audio_path = video_path.replace('.mp4', '.wav')
        
        # Use FFmpeg directly instead of MoviePy for more reliable extraction
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',
            '-ar', '16000',
            '-ac', '1',
            audio_path
        ]
        
        # Add CUDA acceleration if available
        if torch.cuda.is_available():
            cmd.insert(1, '-hwaccel')
            cmd.insert(2, 'cuda')
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            st.error(f"FFmpeg error: {result.stderr}")
            return None
            
        if not os.path.exists(audio_path):
            st.error("Audio extraction failed")
            return None
            
        return audio_path
        
    except Exception as e:
        st.error(f"Error extracting audio: {str(e)}")
        return None

def transcribe_audio(video_path):
    """Transcribe audio and get timestamps using GPU if available"""
    st.write("Starting transcription process...")
    
    if not check_ffmpeg():
        st.error("FFmpeg is not installed. Please install FFmpeg to continue.")
        st.markdown("""
        To install FFmpeg:
        1. Download from: https://github.com/BtbN/FFmpeg-Builds/releases
        2. Extract the zip file
        3. Add the bin folder to your system PATH
        """)
        return None
    
    audio_path = None
    try:
        st.write("Extracting audio...")
        # Extract audio first
        audio_path = extract_audio(video_path)
        if audio_path is None:
            st.error("Failed to extract audio")
            return
            
        if not os.path.exists(audio_path):
            st.error("Audio file not found after extraction")
            return None
            
        st.write("Loading Whisper model...")
        # Load model with GPU support
        model = whisper.load_model("base", device=DEVICE)
        
        # Add progress indicator
        with st.spinner("Transcribing audio..."):
            st.write("Starting transcription...")
            result = model.transcribe(audio_path, language="hi")
            st.write("Transcription completed")
            
        if not result:
            st.error("Transcription returned no results")
            return None
            
        # Convert whisper result to segments format
        segments = []
        current_time = 0
        for segment in result["segments"]:
            segments.append({
                "text": segment["text"],
                "start": segment["start"],
                "end": segment["end"]
            })
            current_time = segment["end"]
            
        result["segments"] = segments
        st.write(f"Found {len(segments)} segments")
        return result
        
    except Exception as e:
        st.error(f"Error in transcription: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return None
        
    finally:
        # Clean up temporary audio file
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception as e:
                st.warning(f"Could not remove temporary audio file: {str(e)}")

def detect_active_speaker(frame, audio_segment, sample_rate=44100):
    """
    Detect if there is active speech in the current audio segment
    Returns a confidence score between 0 and 1
    """
    if audio_segment is None:
        return 0.5  # Neutral score when no audio
        
    try:
        # Ensure audio_segment is a proper numpy array
        if not isinstance(audio_segment, np.ndarray):
            audio_segment = np.array(audio_segment)
        
        # Handle empty or invalid audio
        if len(audio_segment) == 0:
            return 0.5
            
        # Ensure audio is the right shape
        if len(audio_segment.shape) > 1:
            audio_segment = audio_segment.flatten()
            
        # Convert to float32 if needed
        if audio_segment.dtype != np.float32:
            audio_segment = audio_segment.astype(np.float32)
        
        # Normalize audio
        if np.max(np.abs(audio_segment)) > 1.0:
            audio_segment = audio_segment / np.max(np.abs(audio_segment))
            
        # Ensure audio segment is long enough
        min_samples = int(0.02 * sample_rate)  # Minimum 20ms
        if len(audio_segment) < min_samples:
            return 0.5
            
        # Calculate mel spectrogram
        mels = librosa.feature.melspectrogram(
            y=audio_segment, 
            sr=sample_rate,
            n_mels=40,
            n_fft=int(0.025 * sample_rate),  # 25ms window
            hop_length=int(0.010 * sample_rate),  # 10ms hop
            fmax=4000
        )
        
        # Convert to dB scale
        mels_db = librosa.power_to_db(mels, ref=np.max)
        
        # Focus on speech frequencies (100-400 Hz)
        speech_bands = mels_db[1:5, :]
        speech_energy = np.mean(speech_bands)
        
        # Calculate total energy for normalization
        total_energy = np.mean(mels_db)
        
        # Calculate relative speech energy
        if total_energy != 0:
            relative_speech_energy = speech_energy / total_energy
        else:
            relative_speech_energy = 0
            
        # Convert to confidence score
        confidence = np.clip((relative_speech_energy + 1) / 2, 0, 1)
        
        return float(confidence)  # Ensure we return a scalar
        
    except Exception as e:
        print(f"Warning: Error in speech detection: {str(e)}")
        return 0.5  # Neutral score on error

MOVEMENT_HISTORY_SIZE = 5  # Number of frames to keep in history
movement_history = []      # Store recent movement scores
center_history = []       # Store recent lip centers

def detect_lip_movement(face_landmarks):
    """
    Detect lip movement using facial landmarks.
    Enhanced with temporal smoothing to reduce jitter.
    Returns movement score and lip center position.
    """
    global movement_history, center_history
    
    try:
        # Comprehensive set of lip landmarks for better coverage
        upper_lip_indices = [
            0, 37, 39, 40, 61, 185, 267, 269, 270, 409,  # Outer upper lip
            13, 78, 80, 81, 82, 191, 310, 311, 312, 415  # Inner upper lip
        ]
        lower_lip_indices = [
            17, 84, 91, 146, 181, 314, 321, 375, 405, 409,  # Outer lower lip
            14, 78, 87, 88, 95, 178, 317, 318, 324, 402     # Inner lower lip
        ]
        
        # Extract all lip points
        upper_lip = [face_landmarks.landmark[idx] for idx in upper_lip_indices]
        lower_lip = [face_landmarks.landmark[idx] for idx in lower_lip_indices]
        
        # Calculate vertical distances between corresponding points
        vertical_gaps = []
        for up, low in zip(upper_lip, lower_lip):
            dy = abs(up.y - low.y)
            # Weight by z-coordinate to handle side views
            # Points closer to camera (smaller z) get more weight
            weight = 1.0 / (1.0 + abs(up.z))
            vertical_gaps.append(dy * weight)
        
        # Use weighted average of top vertical gaps
        vertical_gaps.sort(reverse=True)
        top_gaps = vertical_gaps[:5]  # Use top 5 largest gaps
        vertical_gap = np.mean(top_gaps)
        
        # Calculate mouth width using outer corners
        left_points = [p for p in upper_lip + lower_lip if p.x < 0.5]
        right_points = [p for p in upper_lip + lower_lip if p.x >= 0.5]
        
        if left_points and right_points:
            left_x = max(p.x for p in left_points)
            right_x = min(p.x for p in right_points)
            mouth_width = right_x - left_x
        else:
            # Fallback for extreme side views
            all_points = upper_lip + lower_lip
            mouth_width = max(p.x for p in all_points) - min(p.x for p in all_points)
        
        # Calculate visibility score based on z-coordinates
        z_values = [p.z for p in upper_lip + lower_lip]
        z_range = max(z_values) - min(z_values)
        visibility = 1.0 / (1.0 + z_range * 5.0)  # Reduce score for large z variations
        
        # Calculate raw movement score
        if mouth_width > 0:
            raw_score = (vertical_gap / mouth_width) * visibility
        else:
            raw_score = 0
        
        # Apply non-linear scaling with reduced sensitivity
        raw_score = np.power(raw_score * 6.0, 0.7)  # Reduced multiplier and adjusted power
        raw_score = np.clip(raw_score, 0, 1)
        
        # Calculate lip center with z-weighting
        weighted_x = sum(p.x * (1.0 / (1.0 + abs(p.z))) for p in upper_lip + lower_lip)
        weighted_y = sum(p.y * (1.0 / (1.0 + abs(p.z))) for p in upper_lip + lower_lip)
        total_weight = sum(1.0 / (1.0 + abs(p.z)) for p in upper_lip + lower_lip)
        
        if total_weight > 0:
            raw_center_x = weighted_x / total_weight
            raw_center_y = weighted_y / total_weight
        else:
            raw_center_x = np.mean([p.x for p in upper_lip + lower_lip])
            raw_center_y = np.mean([p.y for p in upper_lip + lower_lip])
        
        # Update movement history
        movement_history.append(raw_score)
        if len(movement_history) > MOVEMENT_HISTORY_SIZE:
            movement_history.pop(0)
        
        # Update center history
        center_history.append((raw_center_x, raw_center_y))
        if len(center_history) > MOVEMENT_HISTORY_SIZE:
            center_history.pop(0)
        
        # Apply temporal smoothing with exponential weights
        weights = np.exp(np.linspace(-1, 0, len(movement_history)))
        weights /= weights.sum()
        
        # Calculate smoothed movement score
        smoothed_score = np.sum(weights * np.array(movement_history))
        
        # Apply hysteresis thresholding to reduce jitter
        if len(movement_history) > 1:
            prev_score = movement_history[-2]
            if abs(smoothed_score - prev_score) < 0.1:  # Small change threshold
                smoothed_score = prev_score * 0.7 + smoothed_score * 0.3
        
        # Calculate smoothed center position
        smoothed_x = np.average([x for x, _ in center_history], weights=weights)
        smoothed_y = np.average([y for _, y in center_history], weights=weights)
        
        # Apply minimum movement threshold
        if smoothed_score < 0.15:  # Increased minimum threshold
            smoothed_score = 0
        
        return smoothed_score, (smoothed_x, smoothed_y)
        
    except Exception as e:
        print(f"Warning: Error in lip movement detection: {str(e)}")
        return 0.0, (0.5, 0.5)

def calculate_face_size(face_landmarks):
    """Calculate relative face size in the frame"""
    try:
        # Key facial landmarks for size calculation
        landmarks = [
            face_landmarks.landmark[10],   # Top of forehead
            face_landmarks.landmark[152],  # Bottom of chin
            face_landmarks.landmark[234],  # Right temple
            face_landmarks.landmark[454]   # Left temple
        ]
        
        # Get face bounding box coordinates
        x_coords = [landmark.x for landmark in landmarks]
        y_coords = [landmark.y for landmark in landmarks]
        
        # Calculate face dimensions relative to frame
        face_width = max(x_coords) - min(x_coords)
        face_height = max(y_coords) - min(y_coords)
        
        # Use the larger dimension as the face size
        face_size = max(face_width, face_height)
        
        return face_size
        
    except Exception as e:
        print(f"Warning: Error calculating face size: {str(e)}")
        return 0.2  # Return reasonable default size

def calculate_face_angle(face_landmarks):
    """Calculate approximate face rotation angle"""
    try:
        # Key facial landmarks for angle calculation
        left_eye = face_landmarks.landmark[33]    # Left eye outer corner
        right_eye = face_landmarks.landmark[263]  # Right eye outer corner
        nose_top = face_landmarks.landmark[168]   # Upper nose bridge
        nose_bottom = face_landmarks.landmark[2]  # Lower nose bridge
        
        # Calculate horizontal angle using eye positions
        dx_eyes = right_eye.x - left_eye.x
        dy_eyes = right_eye.y - left_eye.y
        horizontal_angle = np.degrees(np.arctan2(dy_eyes, dx_eyes))
        
        # Calculate vertical angle using nose bridge
        dx_nose = nose_bottom.x - nose_top.x
        dy_nose = nose_bottom.y - nose_top.y
        vertical_angle = np.degrees(np.arctan2(dx_nose, dy_nose))
        
        # Combine angles (give more weight to horizontal angle)
        total_angle = np.sqrt((horizontal_angle * 0.7)**2 + (vertical_angle * 0.3)**2)
        
        return total_angle
        
    except Exception as e:
        print(f"Warning: Error calculating face angle: {str(e)}")
        return 0

def process_faces_and_audio(frame, faces_data, frame_state, t, audio_segment=None, sample_rate=None):
    """Process detected faces and audio to determine the active speaker"""
    
    if not faces_data:
        # Reset split screen timer if no faces
        frame_state['split_screen_start'] = 0
        frame_state['last_split_change'] = 0
        return None, 0.0

    frame_height, frame_width = frame.shape[:2]
    aspect_ratio = 9/16  # Target aspect ratio
    
    # Initialize split screen state if not present
    if 'split_screen_start' not in frame_state:
        frame_state['split_screen_start'] = 0
    if 'last_split_change' not in frame_state:
        frame_state['last_split_change'] = 0
    
    # Check for split screen conditions
    current_time = t
    if len(faces_data) == 2:
        # Sort faces left to right
        faces_data.sort(key=lambda x: x['center_x'])
        left_face, right_face = faces_data
        
        # Calculate horizontal distance between faces
        distance = right_face['center_x'] - left_face['center_x']
        
        # Check if faces are sufficiently separated horizontally
        if distance > 0.3:
            # Check if we're not in cooldown period
            if current_time - frame_state['last_split_change'] > 2.0:
                if frame_state['split_screen_start'] == 0:
                    frame_state['split_screen_start'] = current_time
                    frame_state['last_split_change'] = current_time
        else:
            # Only reset split screen if we've passed the persistence period
            if frame_state['split_screen_start'] > 0:
                if current_time - frame_state['split_screen_start'] > 3.0:
                    frame_state['split_screen_start'] = 0
                    frame_state['last_split_change'] = current_time

    max_confidence = 0.0
    active_speaker = None
    
    # Calculate scores for each face
    face_scores = []
    for face_data in faces_data:
        # Get face metrics
        lip_movement = face_data.get('lip_movement', 0.0)
        face_center = face_data.get('face_center', (0.5, 0.5))
        face_size = face_data.get('face_size', 0.0)
        face_angle = face_data.get('face_angle', 0)  # Angle of face rotation
        
        # Calculate face visibility score (1.0 for front face, lower for side face)
        angle_penalty = np.cos(np.radians(min(abs(face_angle), 90)))
        visibility_score = max(0.3, angle_penalty)
        
        # Adjust lip movement confidence based on face angle and size
        if face_size < 0.15:  # Small face
            size_boost = face_size / 0.15  # Proportional boost
            lip_movement *= size_boost * 1.5  # Boost small face detection
        else:
            size_boost = min(face_size / 0.3, 1.5)  # Cap the boost
        
        lip_movement *= visibility_score * size_boost
        
        # Calculate frame coverage for sliding window
        x, y = face_center
        # x *= frame_width
        # y *= frame_height
        
        # Calculate optimal crop window maintaining 9:16 ratio
        crop_height = frame_height
        crop_width = int(crop_height * aspect_ratio)
        
        # Ensure crop window stays within frame bounds
        max_x = frame_width - crop_width
        window_x = np.clip(x - crop_width/2, 0, max_x)
        
        # Calculate window score based on face position
        window_center_x = window_x + crop_width/2
        position_score = 1.0 - abs(x - window_center_x) / crop_width
        
        # Boost confidence for single face
        if len(faces_data) == 1:
            base_confidence = 0.99
        else:
            # Calculate weighted score
            lip_weight = 0.70  # Reduced slightly to account for other factors
            audio_weight = 0.15
            position_weight = 0.15
            
            # Get audio score
            audio_score = 0.0
            if audio_segment is not None:
                try:
                    audio_score = detect_active_speaker(frame, audio_segment, sample_rate)
                except Exception:
                    audio_score = 0.0
            
            # Calculate combined score
            base_confidence = (
                lip_movement * lip_weight +
                audio_score * audio_weight +
                position_score * position_weight
            )
        
        # Apply temporal stability
        if frame_state['prev_speaker'] is not None:
            prev_x, prev_y = frame_state['prev_speaker'].get('face_center', (0.5, 0.5))
            # prev_x *= frame_width
            # prev_y *= frame_height
            
            distance = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
            temporal_score = 1.0 - min(distance / (frame_width * 0.2), 1.0)
            
            if temporal_score > 0.8:  # Close to previous speaker
                base_confidence *= 1.2  # Boost confidence for stability
            
            # Maintain speaker if confidence is close
            if (0.8 * frame_state['speaker_confidence'] <= base_confidence <= 1.2 * frame_state['speaker_confidence']):
                base_confidence = frame_state['speaker_confidence']
        
        # Store enhanced face data
        enhanced_face = face_data.copy()
        enhanced_face.update({
            'window_x': window_x,
            'crop_width': crop_width,
            'crop_height': crop_height,
            'visibility_score': visibility_score,
            'position_score': position_score
        })
        
        face_scores.append((enhanced_face, base_confidence))
    
    # Sort faces by confidence
    face_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Get the most confident face
    if face_scores:
        best_face, best_confidence = face_scores[0]
        
        # Always update to the current active speaker if one is detected
        active_speaker = best_face
        max_confidence = best_confidence
        
    return active_speaker, max_confidence

# def create_split_screen(frame, faces_data, target_width=1080, target_height=1920):
#     """Create a split-screen layout when exactly two different people are detected"""
#     if len(faces_data) != 2:
#         print(f"create_split_screen called with incorrect number of faces: {len(faces_data)}")
#         return None
    
#     # Sort people from left to right
#     faces_data.sort(key=lambda x: x['center_x'])
    
#     # Get original frame dimensions
#     frame_height, frame_width = frame.shape[:2]
    
#     # Calculate PPI (pixels per inch) of original video
#     original_ppi = frame_width / (frame_width / 96)  # Assuming 96 DPI as base
    
#     # Calculate new dimensions maintaining PPI
#     new_width = int(target_width * (original_ppi / 96))
#     new_height = int(target_height * (original_ppi / 96))
    
#     # Create the split screen frame
#     split_frame = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    
#     try:
#         # For each person
#         for i, person in enumerate(faces_data):
#             # Calculate crop dimensions (wider for better framing)
#             crop_width = int(frame_width * 0.6)  # 60% of frame width for each person
#             crop_height = int(crop_width * (new_height / (new_width/2)))  # Maintain aspect ratio
            
#             # Calculate crop region centered on person
#             center_x = person['center_x']
#             center_y = person['center_y']
            
#             # Ensure crop region stays within frame bounds
#             start_x = max(0, min(center_x - crop_width//2, frame_width - crop_width))
#             start_y = max(0, min(center_y - crop_height//2, frame_height - crop_height))
            
#             # Adjust if too close to edges
#             if i == 0:  # Left person
#                 start_x = min(start_x, frame_width//2 - crop_width)
#             else:  # Right person
#                 start_x = max(start_x, frame_width//2)
            
#             # Crop and resize the region
#             crop = frame[int(start_y):int(start_y + crop_height),
#                        int(start_x):int(start_x + crop_width)]
            
#             # Resize maintaining PPI
#             resized_crop = cv2.resize(crop, 
#                                     (new_width//2, new_height),
#                                     interpolation=cv2.INTER_LANCZOS4)
            
#             # Place in split screen
#             split_frame[:, i * (new_width//2):(i + 1) * (new_width//2)] = resized_crop
        
#         return split_frame
        
#     except Exception as e:
#         print(f"Error creating split screen: {str(e)}")
#         return None
# def create_split_screen(frame, faces_data, target_width=1080, target_height=1920):
#     """
#     Create a split-screen layout that intelligently frames two people,
#     regardless of their position in the original frame.
#     """
#     if len(faces_data) != 2:
#         # This function should only be called when we are certain there are two subjects.
#         return None

#     # Sort people from left to right based on their detected center
#     faces_data.sort(key=lambda x: x['center_x'])
    
#     frame_height, frame_width = frame.shape[:2]
    
#     # The final split-screen will be composed of two vertical panes.
#     # Each pane will have half the target width.
#     pane_width = target_width // 2
#     pane_height = target_height

#     # We will create two separate panes and then combine them.
#     all_panes = []

#     for person_data in faces_data:
#         # For each person, we create an ideal crop from the original frame.
#         # The crop should have the same aspect ratio as a single pane.
#         aspect_ratio = pane_width / pane_height
        
#         # We'll make the crop a bit wider than the detected face for better framing.
#         # Let's base the crop height on a factor of the face height.
#         # This makes the framing adaptive to how close the person is to the camera.
#         person_face_height = person_data.get('height', frame_height * 0.4) # Use a default if no height
#         crop_h = int(person_face_height * 2.5) # Crop height is 2.5x the face height
#         crop_w = int(crop_h * aspect_ratio)

#         # Now, center this crop box on the person's face center.
#         center_x = person_data['center_x']
#         center_y = person_data['center_y']
        
#         # Calculate top-left corner of the crop box
#         start_x = int(center_x - crop_w / 2)
#         start_y = int(center_y - crop_h / 2)

#         # --- IMPORTANT: Boundary Checks ---
#         # Ensure the crop box does not go outside the original frame dimensions.
#         start_x = max(0, min(start_x, frame_width - crop_w))
#         start_y = max(0, min(start_y, frame_height - crop_h))

#         # Extract the cropped region from the original frame
#         person_crop = frame[start_y : start_y + crop_h, start_x : start_x + crop_w]

#         # Resize this crop to fit perfectly into one of the output panes.
#         resized_pane = cv2.resize(person_crop, (pane_width, pane_height), interpolation=cv2.INTER_LANCZOS4)
#         all_panes.append(resized_pane)

#     # Combine the two generated panes side-by-side.
#     if len(all_panes) == 2:
#         split_frame = np.hstack(all_panes)
#         return split_frame
    
#     return None # Return None if something went wrong

def create_split_screen(frame, person1_center, person2_center, target_width=1080, target_height=1920):
    """
    Creates a robust split-screen by generating two stable, well-framed panes
    and placing them side-by-side.
    """
    frame_height, frame_width = frame.shape[:2]
    
    # Define the dimensions for each person's pane in the final output
    pane_width = target_width // 2
    pane_height = target_height
    pane_aspect_ratio = pane_width / pane_height
    
    all_panes = []
    
    # Process each person using their smoothed center coordinates
    for center_x, center_y in [person1_center, person2_center]:
        # --- ROBUST CROPPING LOGIC ---
        # Define a generous crop size based on the original frame's width.
        # This is more stable than using detected face size.
        crop_w = int(frame_width * 0.45) # Crop 45% of the frame's width for each person
        crop_h = int(crop_w / pane_aspect_ratio) # Calculate height to match the pane's aspect ratio
        
        # Ensure crop height isn't larger than the frame itself
        crop_h = min(crop_h, frame_height)
        
        # Calculate the top-left corner of this crop, centered on the person
        start_x = int(center_x - crop_w / 2)
        start_y = int(center_y - crop_h / 2)
        
        # --- Boundary Checks to keep the crop inside the frame ---
        start_x = max(0, min(start_x, frame_width - crop_w))
        start_y = max(0, min(start_y, frame_height - crop_h))
        
        # Extract the crop from the original frame
        person_crop = frame[start_y : start_y + crop_h, start_x : start_x + crop_w]
        
        # Resize this stable crop to fit the final pane dimensions
        resized_pane = cv2.resize(person_crop, (pane_width, pane_height), interpolation=cv2.INTER_LANCZOS4)
        all_panes.append(resized_pane)
        
    # Combine the two panes horizontally
    if len(all_panes) == 2:
        return np.hstack(all_panes)
        
    return None

def count_unique_people(people_data, frame_shape):
    """Count unique people in frame using grid-based position clustering"""
    if not people_data:
        return []
    
    frame_width, frame_height = frame_shape
    
    # Create a 3x3 grid for position-based clustering
    unique_positions = {}
    
    for person in people_data:
        # Calculate grid position (divide frame into 3x3 sections)
        grid_x = int(person['center_x'] / (frame_width / 3))
        grid_y = int(person['center_y'] / (frame_height / 3))
        
        # Create unique position key
        pos_key = f"{grid_x}_{grid_y}"
        
        # Store highest confidence detection for each grid position
        if pos_key not in unique_positions or person.get('confidence', 0) > unique_positions[pos_key].get('confidence', 0):
            unique_positions[pos_key] = person
    
    return list(unique_positions.values())

# def focus_on_speaker(clip, target_width=1080, target_height=1920):
#     """Process video to focus on active speakers with direct cuts"""
    
#     # Initialize frame state
#     frame_state = {
#         'prev_speaker': None,
#         'speaker_confidence': 0.0,
#         'current_speaker_id': None,
#         'last_switch_time': 0,
#         'min_switch_interval': 0.001,
#         'is_split_screen': False,
#         'split_screen_start': 0,
#         'last_split_change': 0,
#         'current_crop_center': None,
#         'speaker_positions': {},  # Dictionary to store speaker positions
#         'position_history': {},   # Dictionary to store position history for each speaker
#         'max_history_per_speaker': 10,  # Keep last N positions for each speaker
#         'prev_crop_center': None,  # Previous frame's crop center
#         'movement_history': [],    # Store recent movement amounts
#         'movement_window': 5,      # Number of frames to analyze for jitter
#         'max_movement_threshold': 0.05,  # Maximum allowed movement (5% of frame dimension)
#         'face_mesh': mp.solutions.face_mesh.FaceMesh(
#             max_num_faces=4,
#             min_detection_confidence=0.5,
#             min_tracking_confidence=0.5
#         ),
#         'pose_detector': mp.solutions.pose.Pose(
#             min_detection_confidence=0.5,
#             min_tracking_confidence=0.5
#         ),
#         'transition_state': None,  # Track transition progress
#         'transition_start_pos': None,  # Starting position for transition
#         'transition_target_pos': None,  # Target position for transition
#         'current_transition_frame': 0,  # Current frame in transition
#         'pending_position': None,  # Store detected out-of-frame position
#         'skip_frame': False,      # Flag to skip transition frame
#     }

#     SPLIT_SCREEN_PERSISTENCE = 3.0  # Duration to persist split screen
#     MOVEMENT_THRESHOLD = 0.15  # Face must move 15% of frame dimension to trigger update
#     POSITION_MATCH_THRESHOLD = 100  # Pixel distance to consider positions matching

#     def update_speaker_position(speaker_data, frame_dims):
#         """Update speaker position database"""
#         if not speaker_data:
#             return None

#         w, h = frame_dims
#         current_pos = (speaker_data['center_x'], speaker_data['center_y'])
        
#         # Generate unique ID based on face landmarks if available
#         if 'face_landmarks' in speaker_data:
#             # Use a subset of stable landmarks for identification
#             landmarks = speaker_data['face_landmarks'][:5]  # Use first 5 landmarks
#             speaker_id = hash(tuple([tuple(p) for p in landmarks]))
#         else:
#             # Fallback to position-based ID
#             speaker_id = str(hash(f"{current_pos[0]}_{current_pos[1]}"))
        
#         # Update position history
#         if speaker_id not in frame_state['position_history']:
#             frame_state['position_history'][speaker_id] = []
        
#         history = frame_state['position_history'][speaker_id]
#         history.append(current_pos)
        
#         # Keep only recent positions
#         if len(history) > frame_state['max_history_per_speaker']:
#             history.pop(0)
        
#         # Update current position
#         frame_state['speaker_positions'][speaker_id] = current_pos
        
#         return speaker_id

#     def should_update_crop(current_center, face_center, frame_width, frame_height, crop_width, crop_height):
#         """Calculate if crop window should be updated based on face position"""
#         if current_center is None:
#             return True, face_center

#         current_x, current_y = current_center
        
#         # Calculate margins (15% of crop dimensions)
#         margin_x = crop_width * 0.1
#         margin_y = crop_height * 0.1
        
#         # Calculate boundaries with margins
#         left = current_x - (crop_width / 2)
#         right = current_x + (crop_width / 2)
#         top = current_y - (crop_height / 2)
#         bottom = current_y + (crop_height / 2)
        
#         # Get face position
#         face_x, face_y = face_center
        
#         # Check if face is outside margins
#         outside_x = face_x < (left + margin_x) or face_x > (right - margin_x)
#         outside_y = face_y < (top + margin_y) or face_y > (bottom - margin_y)
        
#         # If face is outside margins, update immediately
#         if outside_x or outside_y:
#             return True, face_center
        
#         return False, current_center

#     def process_frame(get_frame, t):
#         frame = get_frame(t)
#         if frame is None:
#             return None
        
#         # Get frame dimensions
#         h, w = frame.shape[:2]
#         frame_dims = (w, h)
        
#         # Calculate target dimensions maintaining aspect ratio
#         crop_width = int(h * (9/16))  # Maintain aspect ratio
#         crop_height = h
        
#         # Initialize crop position to center (default)
#         left = (w - crop_width) // 2
#         top = (h - crop_height) // 2
        
#         # Process frame for face and pose detection
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
#         # Detect faces and poses in parallel for speed
#         face_results = frame_state['face_mesh'].process(frame_rgb)
#         pose_results = frame_state['pose_detector'].process(frame_rgb)
        
#         face_detections = []
#         if face_results.multi_face_landmarks:
#             for landmarks in face_results.multi_face_landmarks:
#                 face_data = process_face_detection(landmarks, (h, w))
#                 if face_data:
#                     face_detections.append(face_data)
        
#         pose_detections = []
#         if pose_results.pose_landmarks:
#             pose_detections.append(process_pose_detection(pose_results.pose_landmarks, (h, w)))
        
#         # Match and process detections
#         matched_people = match_face_and_pose(face_detections, pose_detections, (w, h))
#         all_detections = face_detections + pose_detections + matched_people
#         unique_people = count_unique_people(all_detections, (w, h))
        
#         # Handle split screen for exactly 2 faces
#         if len(unique_people) == 2:
#             split_frame = create_split_screen(frame, unique_people, target_width, target_height)
#             if split_frame is not None:
#                 return split_frame
        
#         # Process active speaker detection
#         active_speaker, speaker_confidence = process_faces_and_audio(frame, unique_people, frame_state, t)
        
#         if active_speaker:
#             speaker_id = update_speaker_position(active_speaker, (w, h))
#             face_center = (active_speaker['center_x'], active_speaker['center_y'])
            
#             should_update, new_pos = should_update_crop(
#                 frame_state['current_crop_center'], 
#                 face_center,
#                 w, h, 
#                 crop_width, 
#                 crop_height
#             )
            
#             if should_update:
#                 # Immediately update to new position
#                 frame_state['current_crop_center'] = new_pos
#                 left, top, _ = get_crop_coordinates(new_pos, w, h, crop_width, crop_height)
#             else:
#                 # Use current position
#                 left, top, _ = get_crop_coordinates(frame_state['current_crop_center'], w, h, crop_width, crop_height)
#         else:
#             # If no active speaker but we have faces, use the best face
#             best_face = find_best_face(all_detections)
#             if best_face:
#                 face_center = (best_face['center_x'], best_face['center_y'])
#                 should_update, new_pos = should_update_crop(
#                     frame_state['current_crop_center'],
#                     face_center,
#                     w, h,
#                     crop_width,
#                     crop_height
#                 )
#                 if should_update:
#                     # Immediately update to new position
#                     frame_state['current_crop_center'] = new_pos
#                     left, top, _ = get_crop_coordinates(new_pos, w, h, crop_width, crop_height)
#                 elif frame_state['current_crop_center']:
#                     left, top, _ = get_crop_coordinates(frame_state['current_crop_center'], w, h, crop_width, crop_height)
    
#         # Ensure crop coordinates are within frame bounds
#         left = max(0, min(w - crop_width, left))
#         top = max(0, min(h - crop_height, top))
        
#         # Crop and resize the frame
#         cropped = frame[top:top + crop_height, left:left + crop_width]
#         return cv2.resize(cropped, (target_width, target_height))

#     def get_crop_coordinates(face_center, frame_width, frame_height, crop_width, crop_height):
#         """Calculate crop coordinates based on face center"""
#         face_x, face_y = face_center
        
#         # Calculate crop window boundaries
#         left = max(0, min(frame_width - crop_width, int(face_x - crop_width//2)))
#         top = max(0, min(frame_height - crop_height, int(face_y - crop_height//2)))
        
#         # Calculate new center
#         new_center_x = left + crop_width // 2
#         new_center_y = top + crop_height // 2
        
#         return left, top, (new_center_x, new_center_y)

#     # Process the clip
#     processed_clip = clip.fl(process_frame)
    
#     # Create a cleanup callback
#     def cleanup():
#         if frame_state['face_mesh'] is not None:
#             frame_state['face_mesh'].close()
#             frame_state['face_mesh'] = None
#         if frame_state['pose_detector'] is not None:
#             frame_state['pose_detector'].close()
#             frame_state['pose_detector'] = None
    
#     # Attach cleanup to the clip
#     processed_clip.cleanup = cleanup
    
#     return processed_clip

# def focus_on_speaker(clip, target_width=1080, target_height=1920):
#     """
#     Process video to focus on active speakers with SMOOTH, STABLE cuts and pans.
#     """
    
#     # --- State variables for smooth camera movement ---
#     frame_state = {
#         'face_mesh': mp.solutions.face_mesh.FaceMesh(
#             max_num_faces=4,
#             min_detection_confidence=0.5,
#             min_tracking_confidence=0.5
#         ),
#         'active_speaker_data': None,
#         'current_crop_center': None,  # The actual center of the crop window on the current frame
#         'target_crop_center': None,   # The desired target center we are moving towards

#         # --- Panning/Interpolation State ---
#         'is_panning': False,
#         'pan_start_center': None,
#         'pan_progress': 0.0, # A float from 0.0 to 1.0
#         'pan_speed': 0.05, # Controls how fast the pan is. Smaller = slower. (e.g., 1.0 / (fps * 0.5) for a 0.5 sec pan)
        
#         # --- Smoothing/Damping State ---
#         'smoothed_face_center': None,
#         'smoothing_factor': 0.6, # Alpha for exponential moving average. Higher = more smoothing, but more lag.
        
#         # --- Other state ---
#         'frame_count': 0, # To get an idea of the clip's FPS
#         'last_detection_time': -1,
#         'split_screen_start': 0,
#         'last_split_change': 0,
#     }

#     def get_crop_coordinates(center_pos, frame_width, frame_height, crop_width, crop_height):
#         """Calculates crop coordinates from a center point, ensuring they are within bounds."""
#         center_x, center_y = center_pos
        
#         left = int(center_x - crop_width / 2)
#         top = int(center_y - crop_height / 2)

#         # Ensure crop coordinates are within the frame
#         left = max(0, min(left, frame_width - crop_width))
#         top = max(0, min(top, frame_height - crop_height))
        
#         return left, top

#     def process_frame(get_frame, t):
#         frame = get_frame(t)
#         if frame is None:
#             return None
        
#         h, w = frame.shape[:2]
#         crop_height = h
#         crop_width = int(h * (9/16))

#         # Initialize crop center on the first frame
#         if frame_state['current_crop_center'] is None:
#             initial_center = (w / 2, h / 2)
#             frame_state['current_crop_center'] = initial_center
#             frame_state['target_crop_center'] = initial_center
#             frame_state['smoothed_face_center'] = initial_center

#         # --- Run Face Detection (can be optimized to run less frequently) ---
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         face_results = frame_state['face_mesh'].process(frame_rgb)
        
#         all_detections = []
#         if face_results.multi_face_landmarks:
#             for landmarks in face_results.multi_face_landmarks:
#                 # Simplified face data extraction for this example
#                 x_coords = [lm.x * w for lm in landmarks.landmark]
#                 y_coords = [lm.y * h for lm in landmarks.landmark]
#                 center_x = sum(x_coords) / len(x_coords)
#                 center_y = sum(y_coords) / len(y_coords)
#                 face_width = max(x_coords) - min(x_coords)
#                 all_detections.append({'center_x': center_x, 'center_y': center_y, 'width': face_width})

#         # --- Speaker Selection Logic (Simplified - yours is more complex and can be kept) ---
#         # For simplicity, we'll focus on the largest face as the 'active speaker'
#         active_speaker = None
#         if all_detections:
#             active_speaker = max(all_detections, key=lambda x: x['width'])

#         # --- CORE STABILIZATION LOGIC ---
#         if active_speaker:
#             # 1. Get raw detected center of the active speaker
#             raw_face_center = (active_speaker['center_x'], active_speaker['center_y'])
            
#             # 2. Apply Temporal Smoothing (Exponential Moving Average) to the raw detection
#             # This absorbs the high-frequency jitter from the detector.
#             alpha = frame_state['smoothing_factor']
#             prev_smoothed = frame_state['smoothed_face_center']
            
#             smoothed_x = (prev_smoothed[0] * alpha) + (raw_face_center[0] * (1 - alpha))
#             smoothed_y = (prev_smoothed[1] * alpha) + (raw_face_center[1] * (1 - alpha))
            
#             frame_state['smoothed_face_center'] = (smoothed_x, smoothed_y)
            
#             # 3. Implement a Dead Zone to decide IF we should move the camera
#             # We only update the target if the smoothed face position moves far enough away from the *current target*.
#             target_dist = np.sqrt(
#                 (frame_state['smoothed_face_center'][0] - frame_state['target_crop_center'][0])**2 +
#                 (frame_state['smoothed_face_center'][1] - frame_state['target_crop_center'][1])**2
#             )

#             # Dead zone is 10% of the crop window's width.
#             dead_zone_radius = crop_width * 0.10
            
#             if target_dist > dead_zone_radius:
#                 # The speaker has moved significantly. Set a new target and start panning.
#                 frame_state['target_crop_center'] = frame_state['smoothed_face_center']
#                 if not frame_state['is_panning']:
#                     frame_state['is_panning'] = True
#                     frame_state['pan_start_center'] = frame_state['current_crop_center']
#                     frame_state['pan_progress'] = 0.0

#         # 4. Update camera position (Pan if needed)
#         if frame_state['is_panning']:
#             # We are in a panning motion. Interpolate the current crop center towards the target.
#             frame_state['pan_progress'] += frame_state['pan_speed']
            
#             if frame_state['pan_progress'] >= 1.0:
#                 # Panning is complete. Snap to the final target and stop panning.
#                 frame_state['pan_progress'] = 1.0
#                 frame_state['is_panning'] = False
#                 frame_state['current_crop_center'] = frame_state['target_crop_center']
#             else:
#                 # Linearly interpolate (Lerp) the position
#                 start_x, start_y = frame_state['pan_start_center']
#                 target_x, target_y = frame_state['target_crop_center']
#                 progress = frame_state['pan_progress']
                
#                 current_x = start_x + (target_x - start_x) * progress
#                 current_y = start_y + (target_y - start_y) * progress
#                 frame_state['current_crop_center'] = (current_x, current_y)

#         # --- Cropping the Frame ---
#         left, top = get_crop_coordinates(
#             frame_state['current_crop_center'], w, h, crop_width, crop_height
#         )
        
#         cropped_frame = frame[top : top + crop_height, left : left + crop_width]
        
#         # Your split-screen logic can be integrated here if len(unique_people) == 2
#         # For now, we focus on the single-speaker case.
        
#         return cv2.resize(cropped_frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

#     # Process the clip
#     processed_clip = clip.fl(process_frame)
    
#     # Create a cleanup callback
#     def cleanup():
#         if frame_state.get('face_mesh') is not None:
#             frame_state['face_mesh'].close()
#             frame_state['face_mesh'] = None
    
#     # Attach cleanup to the clip
#     processed_clip.cleanup = cleanup
    
#     return processed_clip

# def calculate_pose_center(pose_landmarks):
#     """Calculate the center point of a detected pose"""
#     x_coords = [lm.x for lm in pose_landmarks.landmark]
#     y_coords = [lm.y for lm in pose_landmarks.landmark]
#     return (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))

# def calculate_pose_size(pose_landmarks):
#     """Calculate the relative size of a detected pose"""
#     x_coords = [lm.x for lm in pose_landmarks.landmark]
#     y_coords = [lm.y for lm in pose_landmarks.landmark]
#     width = max(x_coords) - min(x_coords)
#     height = max(y_coords) - min(y_coords)
#     return max(width, height)

# def calculate_pose_confidence(pose_landmarks):
#     """Calculate confidence score for pose detection"""
#     visible_landmarks = sum(1 for lm in pose_landmarks.landmark if lm.visibility > 0.5)
#     return visible_landmarks / len(pose_landmarks.landmark)

# def extract_pose_keypoints(pose_landmarks):
#     """Extract key pose points for tracking and matching"""
#     keypoints = {
#         'nose': pose_landmarks.landmark[0],
#         'left_shoulder': pose_landmarks.landmark[11],
#         'right_shoulder': pose_landmarks.landmark[12],
#         'left_hip': pose_landmarks.landmark[23],
#         'right_hip': pose_landmarks.landmark[24],
#         'left_ear': pose_landmarks.landmark[7],
#         'right_ear': pose_landmarks.landmark[8]
#     }
#     return keypoints

# def calculate_head_center(pose_landmarks):
#     """Calculate head center from pose landmarks"""
#     nose = pose_landmarks.landmark[0]
#     left_ear = pose_landmarks.landmark[7]
#     right_ear = pose_landmarks.landmark[8]
    
#     # Average the positions
#     head_x = (nose.x + left_ear.x + right_ear.x) / 3
#     head_y = (nose.y + left_ear.y + right_ear.y) / 3
    
#     return (head_x, head_y)

# def calculate_pose_movement(keypoints):
#     """Calculate a movement score based on key pose points"""
#     left_shoulder = keypoints['left_shoulder']
#     right_shoulder = keypoints['right_shoulder']
#     left_hip = keypoints['left_hip']
#     right_hip = keypoints['right_hip']
    
#     # Calculate shoulder and hip angles
#     shoulder_angle = np.arctan2(right_shoulder.y - left_shoulder.y,
#                                right_shoulder.x - left_shoulder.x)
#     hip_angle = np.arctan2(right_hip.y - left_hip.y,
#                           right_hip.x - left_hip.x)
    
#     # Movement score based on angle difference and visibility
#     movement_score = abs(shoulder_angle - hip_angle)
#     visibility_score = min(left_shoulder.visibility, right_shoulder.visibility,
#                          left_hip.visibility, right_hip.visibility)
    
#     return movement_score * visibility_score

# def focus_on_speaker(clip, target_width=1080, target_height=1920):
#     """
#     Process video to focus on speakers with smooth pans and intelligent split-screen.
#     """
    
#     # --- Constants for Split-Screen Logic ---
#     SPLIT_SCREEN_CONFIRM_DURATION = 1.0  # Must see 2 people for this long to switch TO split-screen
#     SPLIT_SCREEN_EXIT_DURATION = 1.5     # Must see NOT 2 people for this long to switch FROM split-screen

#     frame_state = {
#         'face_mesh': mp.solutions.face_mesh.FaceMesh(
#             max_num_faces=4, min_detection_confidence=0.5, min_tracking_confidence=0.5
#         ),
#         'current_crop_center': None,
#         'target_crop_center': None,
#         'is_panning': False,
#         'pan_start_center': None,
#         'pan_progress': 0.0,
#         'pan_speed': 0.05,
#         'smoothed_face_center': None,
#         'smoothing_factor': 0.7, # Increased smoothing slightly
        
#         # --- NEW: State for Smart Split-Screen ---
#         'is_in_split_screen_mode': False,
#         'split_screen_entry_timestamp': -1, # Time we first saw 2 people
#         'split_screen_exit_timestamp': -1,  # Time we first lost 2 people
#     }

#     def get_crop_coordinates(center_pos, frame_width, frame_height, crop_width, crop_height):
#         # (This helper function remains the same as before)
#         center_x, center_y = center_pos
#         left = int(center_x - crop_width / 2)
#         top = int(center_y - crop_height / 2)
#         left = max(0, min(left, frame_width - crop_width))
#         top = max(0, min(top, frame_height - crop_height))
#         return left, top

#     def process_frame(get_frame, t):
#         frame = get_frame(t)
#         if frame is None: return None
        
#         h, w = frame.shape[:2]
#         crop_height = h
#         crop_width = int(h * (9/16))

#         if frame_state['current_crop_center'] is None:
#             initial_center = (w / 2, h / 2)
#             frame_state.update({
#                 'current_crop_center': initial_center,
#                 'target_crop_center': initial_center,
#                 'smoothed_face_center': initial_center
#             })

#         # --- Face Detection and Person Counting ---
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         face_results = frame_state['face_mesh'].process(frame_rgb)
        
#         all_detections = []
#         if face_results.multi_face_landmarks:
#             for landmarks in face_results.multi_face_landmarks:
#                 x_coords = [lm.x * w for lm in landmarks.landmark]
#                 y_coords = [lm.y * h for lm in landmarks.landmark]
#                 all_detections.append({
#                     'center_x': sum(x_coords) / len(x_coords),
#                     'center_y': sum(y_coords) / len(y_coords),
#                     'width': max(x_coords) - min(x_coords),
#                     'height': max(y_coords) - min(y_coords)
#                 })
        
#         # --- NEW: Smart Split-Screen State Machine ---
#         num_people = len(all_detections)

#         if num_people == 2:
#             # We see two people. Start or continue the confirmation timer.
#             if frame_state['split_screen_entry_timestamp'] == -1:
#                 frame_state['split_screen_entry_timestamp'] = t
#             frame_state['split_screen_exit_timestamp'] = -1 # Reset exit timer

#             if t - frame_state['split_screen_entry_timestamp'] >= SPLIT_SCREEN_CONFIRM_DURATION:
#                 frame_state['is_in_split_screen_mode'] = True
#         else:
#             # We see 1 or 3+ people. Start or continue the exit timer.
#             if frame_state['split_screen_exit_timestamp'] == -1:
#                 frame_state['split_screen_exit_timestamp'] = t
#             frame_state['split_screen_entry_timestamp'] = -1 # Reset entry timer

#             if t - frame_state['split_screen_exit_timestamp'] >= SPLIT_SCREEN_EXIT_DURATION:
#                 frame_state['is_in_split_screen_mode'] = False

#         # --- Frame Processing Decision ---
#         if frame_state['is_in_split_screen_mode']:
#             # --- RENDER SPLIT SCREEN ---
#             split_frame = create_split_screen(frame, all_detections, target_width, target_height)
#             if split_frame is not None:
#                 return split_frame
#             else:
#                 # If split-screen fails for any reason, fall back to single-person logic
#                 frame_state['is_in_split_screen_mode'] = False

#         # --- RENDER SINGLE SPEAKER (FALLBACK) ---
#         # (This is the same stabilization logic from the previous step)
#         active_speaker = None
#         if all_detections:
#             active_speaker = max(all_detections, key=lambda x: x['width'])

#         if active_speaker:
#             raw_face_center = (active_speaker['center_x'], active_speaker['center_y'])
#             alpha = frame_state['smoothing_factor']
#             prev_smoothed = frame_state['smoothed_face_center']
            
#             smoothed_x = (prev_smoothed[0] * alpha) + (raw_face_center[0] * (1 - alpha))
#             smoothed_y = (prev_smoothed[1] * alpha) + (raw_face_center[1] * (1 - alpha))
#             frame_state['smoothed_face_center'] = (smoothed_x, smoothed_y)
            
#             target_dist = np.sqrt(
#                 (smoothed_x - frame_state['target_crop_center'][0])**2 +
#                 (smoothed_y - frame_state['target_crop_center'][1])**2
#             )
#             dead_zone_radius = crop_width * 0.10
            
#             if target_dist > dead_zone_radius:
#                 frame_state['target_crop_center'] = (smoothed_x, smoothed_y)
#                 if not frame_state['is_panning']:
#                     frame_state['is_panning'] = True
#                     frame_state['pan_start_center'] = frame_state['current_crop_center']
#                     frame_state['pan_progress'] = 0.0

#         if frame_state['is_panning']:
#             frame_state['pan_progress'] += frame_state['pan_speed']
#             if frame_state['pan_progress'] >= 1.0:
#                 frame_state['pan_progress'] = 1.0
#                 frame_state['is_panning'] = False
#                 frame_state['current_crop_center'] = frame_state['target_crop_center']
#             else:
#                 start_x, start_y = frame_state['pan_start_center']
#                 target_x, target_y = frame_state['target_crop_center']
#                 progress = frame_state['pan_progress']
#                 current_x = start_x + (target_x - start_x) * progress
#                 current_y = start_y + (target_y - start_y) * progress
#                 frame_state['current_crop_center'] = (current_x, current_y)

#         left, top = get_crop_coordinates(frame_state['current_crop_center'], w, h, crop_width, crop_height)
#         cropped_frame = frame[top:top + crop_height, left:left + crop_width]
#         return cv2.resize(cropped_frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

#     processed_clip = clip.fl(process_frame)
    
#     def cleanup():
#         if frame_state.get('face_mesh') is not None:
#             frame_state['face_mesh'].close()
#             frame_state['face_mesh'] = None
    
#     processed_clip.cleanup = cleanup
#     return processed_clip

def focus_on_speaker(clip, target_width=1080, target_height=1920):
    """
    Processes video using a "Virtual Director" to create smooth pans,
    intelligent split-screens, and seamless cross-fade transitions.
    """
    # --- Director Tuning Parameters ---
    TRANSITION_DURATION = 0.5  # Duration of the cross-fade in seconds
    SPLIT_SCREEN_CONFIRM_SEC = 1.0
    SPLIT_SCREEN_EXIT_SEC = 1.5
    
    frame_state = {
        'face_mesh': mp.solutions.face_mesh.FaceMesh(max_num_faces=4, min_detection_confidence=0.5, min_tracking_confidence=0.5),
        
        # --- Speaker Tracking State ---
        'speaker_positions': {}, # Tracks smoothed position of each speaker, e.g., {0: (x,y), 1: (x,y)}
        'smoothing_factor': 0.9, # More smoothing to reduce jitter further
        
        # --- Camera and Shot State ---
        'current_shot_type': 'single_speaker', # Can be 'single_speaker' or 'split_screen'
        'active_speaker_idx': 0,
        'current_pan_center': None,
        'target_pan_center': None,
        'pan_speed': 0.02, # Slower pan for more cinematic feel
        'is_panning': False,
        'pan_start_center': None,
        'pan_progress': 0.0,
        
        # --- State Machine Timers ---
        'last_shot_decision_time': -1,
        'time_in_current_state': 0,

        # --- Transition State ---
        'is_in_transition': False,
        'transition_progress': 0.0,
        'transition_from_frame': None,
    }

    def render_single_speaker_view(frame, center_pos):
        h, w = frame.shape[:2]
        crop_height = h
        crop_width = int(h * (9/16))
        left = int(center_pos[0] - crop_width / 2)
        top = int(center_pos[1] - crop_height / 2)
        left = max(0, min(left, w - crop_width))
        top = max(0, min(top, h - crop_height))
        cropped = frame[top:top+crop_height, left:left+crop_width]
        return cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

    def process_frame(get_frame, t):
        frame = get_frame(t)
        if frame is None: return None

        h, w = frame.shape[:2]
        fps = clip.fps if clip.fps else 30
        
        # --- Initialize on first frame ---
        if frame_state['current_pan_center'] is None:
            initial_center = (w / 2, h / 2)
            frame_state.update({
                'current_pan_center': initial_center,
                'target_pan_center': initial_center,
            })

        # --- 1. Face Detection and Position Smoothing ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = frame_state['face_mesh'].process(frame_rgb)
        
        detected_centers = []
        if face_results.multi_face_landmarks:
            for landmarks in face_results.multi_face_landmarks:
                x_coords = [lm.x * w for lm in landmarks.landmark]
                y_coords = [lm.y * h for lm in landmarks.landmark]
                detected_centers.append((sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords)))
        
        # Update smoothed positions for all detected people
        # This simple matching by order is okay for stable scenes. More complex tracking could be added.
        for i, center in enumerate(detected_centers):
            if i in frame_state['speaker_positions']:
                alpha = frame_state['smoothing_factor']
                prev_pos = frame_state['speaker_positions'][i]
                smoothed_pos = (prev_pos[0] * alpha + center[0] * (1-alpha), prev_pos[1] * alpha + center[1] * (1-alpha))
                frame_state['speaker_positions'][i] = smoothed_pos
            else: # New person
                frame_state['speaker_positions'][i] = center
        
        num_people = len(detected_centers)

        # --- 2. The Virtual Director: Decide the Shot ---
        previous_shot_type = frame_state['current_shot_type']
        new_shot_type = previous_shot_type
        
        dt = t - frame_state['last_shot_decision_time'] if frame_state['last_shot_decision_time'] > -1 else 0
        frame_state['last_shot_decision_time'] = t
        frame_state['time_in_current_state'] += dt

        if num_people == 2 and previous_shot_type == 'single_speaker':
            if frame_state['time_in_current_state'] > SPLIT_SCREEN_CONFIRM_SEC:
                new_shot_type = 'split_screen'
        elif num_people != 2 and previous_shot_type == 'split_screen':
            if frame_state['time_in_current_state'] > SPLIT_SCREEN_EXIT_SEC:
                new_shot_type = 'single_speaker'
        
        # If the shot changes, reset the timer and trigger a transition
        if new_shot_type != previous_shot_type:
            frame_state['time_in_current_state'] = 0
            frame_state['current_shot_type'] = new_shot_type
            frame_state['is_in_transition'] = True
            frame_state['transition_progress'] = 0.0
            # Store the last fully rendered frame to transition FROM
            frame_state['transition_from_frame'] = frame_state.get('last_rendered_frame')

        # --- 3. Panning Logic for Single Speaker Mode ---
        if frame_state['current_shot_type'] == 'single_speaker' and frame_state['speaker_positions']:
            # For simplicity, target the first detected speaker. Your audio logic could refine this.
            frame_state['active_speaker_idx'] = 0 
            target_pos = frame_state['speaker_positions'][frame_state['active_speaker_idx']]
            
            # Only start a new pan if not already panning and target is far enough away
            dist_to_target = np.linalg.norm(np.array(target_pos) - np.array(frame_state['target_pan_center']))
            if not frame_state['is_panning'] and dist_to_target > (w * 0.05): # 5% of screen width dead zone
                frame_state['is_panning'] = True
                frame_state['pan_start_center'] = frame_state['current_pan_center']
                frame_state['target_pan_center'] = target_pos
                frame_state['pan_progress'] = 0.0
        
        if frame_state['is_panning']:
            frame_state['pan_progress'] += frame_state['pan_speed']
            if frame_state['pan_progress'] >= 1.0:
                frame_state['is_panning'] = False
                frame_state['current_pan_center'] = frame_state['target_pan_center']
            else:
                p = frame_state['pan_progress']
                start = np.array(frame_state['pan_start_center'])
                end = np.array(frame_state['target_pan_center'])
                frame_state['current_pan_center'] = tuple(start + p * (end - start))
        
        # --- 4. Render the Target Frame for this moment in time ---
        target_frame = None
        if frame_state['current_shot_type'] == 'single_speaker':
            if frame_state['speaker_positions']:
                target_frame = render_single_speaker_view(frame, frame_state['current_pan_center'])
        elif frame_state['current_shot_type'] == 'split_screen':
            if len(frame_state['speaker_positions']) >= 2:
                p1 = frame_state['speaker_positions'][0]
                p2 = frame_state['speaker_positions'][1]
                target_frame = create_split_screen(frame, p1, p2, target_width, target_height)

        if target_frame is None: # Fallback if rendering fails
            target_frame = render_single_speaker_view(frame, frame_state['current_pan_center'])

        # --- 5. Apply Cross-Fade Transition if necessary ---
        final_frame = target_frame
        if frame_state['is_in_transition'] and frame_state['transition_from_frame'] is not None:
            frame_state['transition_progress'] += 1 / (TRANSITION_DURATION * fps)
            alpha = frame_state['transition_progress']

            if alpha >= 1.0:
                frame_state['is_in_transition'] = False
                final_frame = target_frame
            else:
                # Blend the 'from' frame and the 'to' frame
                final_frame = cv2.addWeighted(frame_state['transition_from_frame'], 1 - alpha, target_frame, alpha, 0)

        # Store the final rendered frame for the next transition
        frame_state['last_rendered_frame'] = final_frame.copy()
        
        return final_frame

    processed_clip = clip.fl(process_frame)
    
    def cleanup():
        if frame_state.get('face_mesh') is not None:
            frame_state['face_mesh'].close()
    
    processed_clip.cleanup = cleanup
    return processed_clip


def process_face_detection(landmarks, frame_dims):
    """Process face landmarks to extract face data"""
    h, w = frame_dims
    
    # Extract face landmarks as normalized coordinates
    face_landmarks = []
    for landmark in landmarks.landmark:
        x, y = int(landmark.x * w), int(landmark.y * h)
        face_landmarks.append([x, y])
    
    if not face_landmarks:
        return None
    
    # Calculate face center
    center_x = sum(x for x, y in face_landmarks) / len(face_landmarks)
    center_y = sum(y for x, y in face_landmarks) / len(face_landmarks)
    
    # Calculate face size using bounding box
    x_coords = [x for x, y in face_landmarks]
    y_coords = [y for x, y in face_landmarks]
    face_width = max(x_coords) - min(x_coords)
    face_height = max(y_coords) - min(y_coords)
    
    return {
        'center_x': center_x,
        'center_y': center_y,
        'width': face_width,
        'height': face_height,
        'face_landmarks': face_landmarks
    }

def process_pose_detection(landmarks, frame_dims):
    """Process pose landmarks to extract pose data"""
    h, w = frame_dims
    
    # Extract pose landmarks as normalized coordinates
    pose_landmarks = []
    for landmark in landmarks.landmark:
        x, y = int(landmark.x * w), int(landmark.y * h)
        pose_landmarks.append([x, y])
    
    if not pose_landmarks:
        return None
    
    # Calculate pose center (using upper body landmarks)
    upper_body_indices = [11, 12, 23, 24]  # Shoulders and hips
    center_x = sum(pose_landmarks[i][0] for i in upper_body_indices) / len(upper_body_indices)
    center_y = sum(pose_landmarks[i][1] for i in upper_body_indices) / len(upper_body_indices)
    
    return {
        'center_x': center_x,
        'center_y': center_y,
        'pose_landmarks': pose_landmarks
    }

def match_face_and_pose(faces, poses, frame_dims):
    """Match face detections with pose detections"""
    if not faces or not poses:
        return []
    
    matched = []
    w, h = frame_dims
    max_dist = min(w, h) * 0.2  # Maximum distance for matching
    
    for face in faces:
        face_center = (face['center_x'], face['center_y'])
        best_pose = None
        min_dist = float('inf')
        
        for pose in poses:
            pose_center = (pose['center_x'], pose['center_y'])
            dist = ((face_center[0] - pose_center[0])**2 + 
                   (face_center[1] - pose_center[1])**2)**0.5
            
            if dist < min_dist and dist < max_dist:
                min_dist = dist
                best_pose = pose
        
        if best_pose:
            matched.append({
                **face,
                'pose_landmarks': best_pose['pose_landmarks']
            })
    
    return matched

def find_best_face(detections):
    """Find the best face from all detections"""
    if not detections:
        return None
            
    # Prefer detections with both face and pose
    full_detections = [d for d in detections if 'pose_landmarks' in d]
    if full_detections:
        # Return the largest face with pose detection
        return max(full_detections, key=lambda x: x['width'] * x['height'])
    
    # If no full detections, return the largest face
    face_detections = [d for d in detections if 'face_landmarks' in d]
    if face_detections:
        return max(face_detections, key=lambda x: x['width'] * x['height'])
    
    return None

def create_reel(video_path, segment, output_path):
    """Create a reel from the video segment with captions"""
    try:
        video = VideoFileClip(video_path)
        start_time = segment["start"]
        end_time = segment["end"]
        
        st.write(f"Extracting segment: {start_time:.2f}s to {end_time:.2f}s")
        try:
            # Extract the segment
            clip = video.subclip(start_time, end_time)
            
            # Process with face detection and dynamic cropping
            clip = focus_on_speaker(clip)
            
            caption_clips = []

            # Combine everything
            final_clip = CompositeVideoClip([clip] + caption_clips)
            
            # Verify final clip
            if not hasattr(final_clip, 'size') or not all(final_clip.size):
                st.error("Failed to create final video - invalid dimensions")
                return
                
            if not hasattr(final_clip, 'duration') or final_clip.duration <= 0:
                st.error("Failed to create final video - invalid duration")
                return
                
            st.write("Video and captions combined successfully")
            
            # Ensure audio duration matches video duration
            if final_clip.audio is not None:
                final_clip = final_clip.set_duration(min(final_clip.duration, final_clip.audio.duration))
                # Add a small safety margin
                final_clip = final_clip.subclip(0, final_clip.duration - 0.1)
                
            # Use GPU acceleration for encoding if available
            if torch.cuda.is_available():
                output_params = ['-c:v', 'h264_nvenc', '-preset', 'fast']
                st.write("Using GPU acceleration for encoding")
            else:
                output_params = ['-c:v', 'libx264', '-preset', 'medium']
                st.write("Using CPU for encoding")
                
            # Write the final clip
            with st.spinner(f"Creating reel... {os.path.basename(output_path)}"):
                st.write("Writing final video file...")
                final_clip.write_videofile(
                    output_path,
                    codec='libx264' if not torch.cuda.is_available() else 'h264_nvenc',
                    audio_codec='aac',
                    ffmpeg_params=output_params
                )
                st.success(f"Created: {os.path.basename(output_path)}")
                
        except Exception as e:
            st.error(f"Error in video processing: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            return
            
    except Exception as e:
        st.error(f"Error loading video: {str(e)}")
        return None

def main():
    st.title("Podcast Video to Reels Converter")
    st.write("Upload a podcast video to automatically create reels from important segments")
    
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4'])
    
    if uploaded_file is not None:
        try:
            # Save uploaded file to temp location
            temp_path = "temp_video.mp4"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if st.button("Process Video"):
                with st.spinner("Processing video..."):
                    # Process the video
                    transcription = transcribe_audio(temp_path)
                    if transcription:
                        segments = get_important_segments(transcription)
                        
                        if not os.path.exists("output_reels"):
                            os.makedirs("output_reels")
                        
                        for i, segment in enumerate(segments):
                            output_path = f"output_reels/reel_{i+1}.mp4"
                            create_reel(temp_path, segment, output_path)
                        
                        st.success("Reels generated successfully!")
                        st.info("Check the output_reels directory for your generated reels.")
            
            # Cleanup temp files
            try:
                if 'video' in locals():
                    video.close()
                if os.path.exists(temp_path):
                    os.close(os.open(temp_path, os.O_RDONLY))  # Release any file handles
                    os.remove(temp_path)
            except Exception as e:
                st.warning(f"Note: Could not remove temporary file: {str(e)}")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass

if __name__ == "__main__":
    main()
