# --- START OF FILE app7.py ---
import streamlit as st
import whisper
import torch
import os
import subprocess
import numpy as np
from moviepy.editor import VideoFileClip, CompositeVideoClip
from moviepy.config import change_settings
import cv2
import mediapipe as mp
import librosa
import numpy as np

# --- Setup Code ---
st.set_page_config(layout="wide")
possible_paths = ["D:\Dikha_workspace\Dikha_workspace\videodb-reel-main\project\ImageMagick-7.1.1-47-Q16-HDRI-x64-dll.exe"]
magick_path = None
local_magick = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imagemagick")
if os.path.exists(local_magick):
    for root, dirs, files in os.walk(local_magick):
        if "magick.exe" in files:
            magick_path = os.path.join(root, "magick.exe")
            break
if not magick_path:
    for path in possible_paths:
        if os.path.exists(path):
            magick_path = path
            break
if magick_path:
    change_settings({"IMAGEMAGICK_BINARY": magick_path})

# <<< FIX: The DEVICE variable definition is restored here >>>
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.sidebar.title("Reel Creator Controls")
st.sidebar.info(f"Using device: {DEVICE}")
if magick_path: st.sidebar.success(f"✅ ImageMagick Linked")
else: st.sidebar.warning("⚠️ ImageMagick not found.")

# <<< NEW: A more robust and sensitive lip movement detection function >>>
LIP_MOVEMENT_HISTORY_SIZE = 4
lip_movement_history = {}
def detect_lip_movement_v2(face_id, landmarks):
    global lip_movement_history
    try:
        # Define landmark indices for mouth and eyes
        upper_lip_y = np.mean([landmarks.landmark[i].y for i in [13, 81, 82, 312, 311]])
        lower_lip_y = np.mean([landmarks.landmark[i].y for i in [14, 87, 88, 318, 317]])
        mouth_opening = abs(upper_lip_y - lower_lip_y)

        # Use eye distance as a stable normalization reference
        left_eye_x = landmarks.landmark[133].x
        right_eye_x = landmarks.landmark[362].x
        eye_distance = abs(left_eye_x - right_eye_x)
        
        if eye_distance == 0: return 0.0
        
        normalized_opening = mouth_opening / eye_distance

        # Use a dictionary to track history per face_id
        if face_id not in lip_movement_history:
            lip_movement_history[face_id] = []
        
        history = lip_movement_history[face_id]
        history.append(normalized_opening)
        if len(history) > LIP_MOVEMENT_HISTORY_SIZE:
            history.pop(0)
        
        smoothed_opening = np.mean(history)
        
        # Calibrate the score to be more sensitive
        score = min(smoothed_opening * 5.0, 1.0)
        return score
    except Exception:
        return 0.0

# --- Get Segments ---
def get_important_segments(transcription, target_duration=60.0):
    st.write(f"Creating a reel of up to {target_duration} seconds.")
    segments = transcription['segments']
    if not segments: return []
    video_end_time = segments[-1]['end']
    reel_end_time = min(video_end_time, target_duration)
    return [{'text': transcription['text'], 'start': 0, 'end': reel_end_time}]

# --- Audio Analysis ---
def pre_analyze_audio(audio_clip):
    st.write("Pre-analyzing audio for voice activity...")
    try:
        y = audio_clip.to_soundarray(fps=16000)
        if y.ndim > 1: y = y.mean(axis=1)
        frame_length, hop_length = 512, 256
        if y.shape[0] < frame_length: return lambda t: False
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        if np.max(rms) > 0: rms = rms / np.max(rms)
        threshold = np.clip(np.mean(rms) * 1.5, 0.1, 0.8)
        is_speech_frames = rms > threshold
        def is_speech_at_time(t):
            frame_index = int(t * 16000 / hop_length)
            return is_speech_frames[frame_index] if 0 <= frame_index < len(is_speech_frames) else False
        st.write("✅ Audio analysis complete.")
        return is_speech_at_time
    except Exception as e:
        st.warning(f"Could not perform VAD: {e}.")
        return lambda t: False

# <<< The definitive Virtual Director function with Debug Mode >>>
def focus_on_speaker(clip, target_width=1080, target_height=1920, is_speech_at_time=None, debug_mode=False):
    if is_speech_at_time is None: is_speech_at_time = lambda t: False
    
    TALKING_THRESHOLD = 0.05 # Lowered threshold to be more sensitive
    
    frame_state = {
        'face_mesh': mp.solutions.face_mesh.FaceMesh(max_num_faces=4, min_detection_confidence=0.5, min_tracking_confidence=0.5),
        'speaker_data': {},
        'last_valid_target_center': None,
        'current_pan_center': None,
        'shot_mode': 'single',
        'smoothing_factor': 0.92,
    }

    def render_two_shot_view(frame, p1_pos, p2_pos):
        h, w = frame.shape[:2]
        aspect_ratio = target_width / target_height
        mid_x, mid_y = (p1_pos[0] + p2_pos[0]) / 2, (p1_pos[1] + p2_pos[1]) / 2
        span_w = abs(p1_pos[0] - p2_pos[0])
        crop_w = int(span_w * 1.4)
        crop_w = max(crop_w, int(h * aspect_ratio * 0.7))
        crop_h = int(crop_w / aspect_ratio)
        top, left = int(mid_y - crop_h / 2), int(mid_x - crop_w / 2)
        top, left = max(0, min(top, h - crop_h)), max(0, min(left, w - crop_w))
        crop_h, crop_w = min(crop_h, h - top), min(crop_w, w - left)
        cropped = frame[top:top+crop_h, left:left+crop_w]
        return cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

    def render_single_shot_view(frame, center_pos):
        h, w = frame.shape[:2]
        aspect_ratio = target_width / target_height
        crop_h, crop_w = h, int(h * aspect_ratio)
        face_y, face_x = center_pos[1], center_pos[0]
        top, left = int(face_y - (crop_h * 0.35)), int(face_x - crop_w / 2)
        top, left = max(0, min(top, h - crop_h)), max(0, min(left, w - crop_w))
        cropped = frame[top:top+crop_h, left:left+crop_w]
        return cv2.resize(cropped, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)

    def process_frame(get_frame, t):
        original_frame = get_frame(t)
        if original_frame is None: return np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        frame_for_detection = original_frame.copy()
        h, w = frame_for_detection.shape[:2]

        if frame_state['current_pan_center'] is None:
            initial_center = (w / 2, h / 2)
            frame_state.update({ 'current_pan_center': initial_center, 'last_valid_target_center': initial_center })
        
        is_speech_now = is_speech_at_time(t)
        frame_rgb = cv2.cvtColor(frame_for_detection, cv2.COLOR_BGR2RGB)
        face_results = frame_state['face_mesh'].process(frame_rgb)
        
        current_faces = {}
        if face_results.multi_face_landmarks:
            for i, landmarks in enumerate(face_results.multi_face_landmarks):
                x_coords = [lm.x * w for lm in landmarks.landmark]
                y_coords = [lm.y * h for lm in landmarks.landmark]
                raw_center = (sum(x_coords) / len(x_coords), sum(y_coords) / len(y_coords))
                
                if i in frame_state['speaker_data'] and 'position' in frame_state['speaker_data'][i]:
                    alpha = frame_state['smoothing_factor']
                    prev_pos = frame_state['speaker_data'][i]['position']
                    smoothed_pos = (prev_pos[0] * alpha + raw_center[0] * (1-alpha), prev_pos[1] * alpha + raw_center[1] * (1-alpha))
                else:
                    smoothed_pos = raw_center

                score = detect_lip_movement_v2(i, landmarks) if is_speech_now else 0.0
                current_faces[i] = {'position': smoothed_pos, 'score': score, 'raw_landmarks': landmarks}
        
        frame_state['speaker_data'] = current_faces
        num_people = len(current_faces)
        
        target_center = frame_state['last_valid_target_center']
        
        if num_people == 1:
            frame_state['shot_mode'] = 'single'
            target_center = list(current_faces.values())[0]['position']
        elif num_people >= 2:
            best_id = max(current_faces, key=lambda id: current_faces[id]['score'])
            if current_faces[best_id]['score'] > TALKING_THRESHOLD:
                frame_state['shot_mode'] = 'single'
                target_center = current_faces[best_id]['position']
            else:
                frame_state['shot_mode'] = 'two_shot'
        else:
             frame_state['shot_mode'] = 'lost_track'

        if target_center: frame_state['last_valid_target_center'] = target_center
        
        current_pos = np.array(frame_state['current_pan_center'])
        target_pos = np.array(frame_state['last_valid_target_center'])
        if np.linalg.norm(current_pos - target_pos) > 10:
             frame_state['current_pan_center'] = tuple(current_pos * 0.95 + target_pos * 0.05)

        if frame_state['shot_mode'] == 'two_shot' and num_people >= 2:
            people = sorted(current_faces.values(), key=lambda p: p['position'][0])
            final_frame = render_two_shot_view(original_frame, people[0]['position'], people[1]['position'])
        else:
            final_frame = render_single_shot_view(original_frame, frame_state['current_pan_center'])
            
        if debug_mode:
            debug_frame = final_frame.copy()
            for i, face_data in current_faces.items():
                lms = face_data['raw_landmarks'].landmark
                h_debug, w_debug = debug_frame.shape[:2]
                x_coords = [lm.x * w_debug for lm in lms]
                y_coords = [lm.y * h_debug for lm in lms]
                cv2.rectangle(debug_frame, (int(min(x_coords)), int(min(y_coords))), (int(max(x_coords)), int(max(y_coords))), (0, 255, 0), 2)
                score_text = f"Score: {face_data['score']:.2f}"
                cv2.putText(debug_frame, score_text, (int(min(x_coords)), int(min(y_coords))-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
            debug_text = f"Mode: {frame_state['shot_mode']} | People: {num_people}"
            cv2.putText(debug_frame, debug_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
            return debug_frame

        return final_frame

    processed_clip = clip.fl(process_frame)
    
    def cleanup():
        if frame_state.get('face_mesh') is not None: frame_state['face_mesh'].close()
    
    processed_clip.cleanup = cleanup
    return processed_clip

# --- Main Application Logic ---
def create_reel(video_path, segment, output_path, debug_mode):
    try:
        video = VideoFileClip(video_path)
        clip = video.subclip(segment["start"], segment["end"])
        
        is_speech_at_time = pre_analyze_audio(clip.audio) if clip.audio else lambda t: False
        processed_clip = focus_on_speaker(clip, is_speech_at_time=is_speech_at_time, debug_mode=debug_mode)
        final_clip = processed_clip.set_audio(clip.audio)
        
        st.write("Encoding reel...")
        codec = 'libx264'
        ffmpeg_params = []
        if torch.cuda.is_available():
            codec, ffmpeg_params = 'h264_nvenc', ['-preset', 'fast']
        
        with st.spinner(f"Creating reel... {os.path.basename(output_path)}"):
            final_clip.write_videofile(output_path, codec=codec, audio_codec='aac', ffmpeg_params=ffmpeg_params)
            st.success(f"Created: {os.path.basename(output_path)}")
    except Exception as e:
        st.error(f"Error in video processing: {e}")
        import traceback
        st.error(traceback.format_exc())
    finally:
        for c in [video, clip, processed_clip, final_clip]:
            if 'c' in locals() and c is not None:
                try: c.close()
                except Exception: pass

def transcribe_audio(video_path):
    st.write("Starting transcription...")
    audio_path = None
    try:
        audio_path = video_path.replace('.mp4', '.wav')
        cmd = ['ffmpeg', '-y', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_path]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        model = whisper.load_model("base", device=DEVICE)
        with st.spinner("Transcribing audio..."):
            result = model.transcribe(audio_path, language="en")
        st.write("Transcription completed.")
        return result
    except Exception as e:
        st.error(f"Error in transcription: {e}")
        return None
    finally:
        if audio_path and os.path.exists(audio_path):
            try: os.remove(audio_path)
            except Exception as e: st.warning(f"Could not remove temp audio file: {e}")

def main():
    st.title("Podcast Video to Reels Converter")
    
    debug_mode = st.sidebar.checkbox("Enable Debug Visualizations", value=True)
    st.sidebar.info("Debug mode will draw boxes and scores on the output video to help diagnose camera issues.")
    
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4'])
    
    if uploaded_file is not None:
        temp_path = "temp_video.mp4"
        try:
            with open(temp_path, "wb") as f: f.write(uploaded_file.getbuffer())
            
            if st.button("Create Reel"):
                with st.spinner("Processing... This may take a while."):
                    transcription = transcribe_audio(temp_path)
                    if transcription and transcription.get('segments'):
                        segments = get_important_segments(transcription, target_duration=60.0)
                        output_dir = "output_reels"
                        if not os.path.exists(output_dir): os.makedirs(output_dir)
                        
                        if segments:
                            st.info(f"Generated {len(segments)} reel segment(s). Now processing video...")
                            for i, segment in enumerate(segments):
                                output_path = os.path.join(output_dir, f"reel_{i+1}.mp4")
                                create_reel(temp_path, segment, output_path, debug_mode)
                            st.success("Reels generated successfully!")
                        else:
                            st.warning("Could not generate a valid segment.")
                    else:
                        st.error("Transcription failed.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
        finally:
            if os.path.exists(temp_path):
                try: os.remove(temp_path)
                except Exception: pass

if __name__ == "__main__":
    main()