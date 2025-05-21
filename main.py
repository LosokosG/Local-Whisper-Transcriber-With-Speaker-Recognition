import sys
import os
import time
import tempfile
import math
import subprocess
import warnings
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel,
                             QFileDialog, QTextEdit, QProgressBar, QVBoxLayout,
                             QHBoxLayout, QWidget, QComboBox, QLineEdit, QMessageBox,
                             QTabWidget, QSplitter, QCheckBox, QSpinBox, QGroupBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
import whisper
import torch
import torchaudio
import numpy as np
from pydub import AudioSegment
import uuid

# Suppress reproducibility warning about TF32.
warnings.filterwarnings("ignore", category=UserWarning,
                        message="TensorFloat-32 \\(TF32\\) has been disabled")

# Force PyTorch to check for CUDA again and enable TF32 for better performance
if torch.cuda.is_available():
    print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
    CUDA_AVAILABLE = True
    CUDA_VERSION = torch.version.cuda
    DEVICE = torch.device("cuda")
    # Enable TF32 for better performance on Ampere GPUs (like RTX 3050)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    print("CUDA is not available. Using CPU.")
    CUDA_AVAILABLE = False
    CUDA_VERSION = "Not available"
    DEVICE = torch.device("cpu")


class AudioPreprocessor:
    """Class to handle audio preprocessing and splitting"""

    @staticmethod
    def convert_to_wav_for_diarization(input_path, output_path=None):
        """
        Convert any audio format to WAV format for diarization only
        Maintains original quality when possible
        """
        if output_path is None:
            # Create temp file if no output path is specified
            tmp_dir = tempfile.gettempdir()
            output_path = os.path.join(tmp_dir, f"diar_{str(uuid.uuid4())[:8]}.wav")

        try:
            # Use pydub to convert the file
            audio = AudioSegment.from_file(input_path)

            # Convert to mono (diarization works better with mono)
            audio = audio.set_channels(1)

            # If sample rate is very low, increase to at least 16kHz
            if audio.frame_rate < 16000:
                audio = audio.set_frame_rate(16000)

            # Export as WAV with high quality settings
            audio.export(
                output_path,
                format="wav",
                parameters=["-ar", str(audio.frame_rate), "-ac", "1"]
            )
            return output_path
        except Exception as e:
            print(f"Error converting audio: {str(e)}")
            # If conversion fails, return the original file
            return input_path

    @staticmethod
    def split_audio(input_path, max_duration_minutes=10):
        """Split a large audio file into smaller chunks"""
        try:
            audio = AudioSegment.from_file(input_path)

            # Convert to milliseconds
            max_duration_ms = max_duration_minutes * 60 * 1000

            # If file is smaller than threshold, return the original
            if len(audio) <= max_duration_ms:
                return [input_path]

            # Create temp directory for chunks
            tmp_dir = tempfile.gettempdir()
            base_name = os.path.splitext(os.path.basename(input_path))[0]

            # Calculate number of chunks
            num_chunks = math.ceil(len(audio) / max_duration_ms)
            chunk_paths = []

            # Split and save chunks
            for i in range(num_chunks):
                start_ms = i * max_duration_ms
                end_ms = min((i + 1) * max_duration_ms, len(audio))

                chunk = audio[start_ms:end_ms]
                chunk_path = os.path.join(tmp_dir, f"{base_name}_chunk{i + 1}.wav")
                chunk.export(chunk_path, format="wav")
                chunk_paths.append(chunk_path)

            return chunk_paths
        except Exception as e:
            print(f"Error splitting audio: {str(e)}")
            # If splitting fails, return the original file
            return [input_path]


class WhisperTranscriber:
    """Class to handle Whisper transcription with real-time output"""

    def __init__(self, model_name="base", device=DEVICE, language="auto"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.language = language

    def load_model(self):
        """Load the Whisper model"""
        self.model = whisper.load_model(self.model_name, device=self.device)
        return self.model

    def transcribe_file(self, audio_path, callback=None):
        """
        Transcribe an audio file with progress updates

        Args:
            audio_path: Path to the audio file
            callback: Function to call with partial results

        Returns:
            Complete transcription
        """
        if self.model is None:
            self.load_model()

        # Load the full audio file
        audio = whisper.load_audio(audio_path)

        # Get audio duration
        duration = len(audio) / 16000  # Audio is 16kHz

        # First detect the language if set to auto
        detected_language = self.language
        if detected_language == "auto":
            # Use a short segment to detect language
            audio_sample = audio[:min(len(audio), 30 * 16000)]  # 30 sec sample
            result = self.model.transcribe(
                audio_sample,
                fp16=(self.device.type == "cuda"),
                language=None,  # Auto-detect
                task="transcribe"
            )
            detected_language = result.get("language", "en")
            if callback:
                callback(f"Detected language: {detected_language}", 0.1, [])

        # Now transcribe the entire file with the detected language
        result = self.model.transcribe(
            audio,
            fp16=(self.device.type == "cuda"),
            language=detected_language,
            task="transcribe",
            verbose=True
        )

        # Process the segments
        segments = []
        all_text = ""

        for i, segment in enumerate(result["segments"]):
            segments.append(segment)

            # Format time
            timestamp = time.strftime('%H:%M:%S', time.gmtime(segment["start"]))
            formatted_text = f"[{timestamp}] {segment['text'].strip()}\n"
            all_text += formatted_text

            # Call the callback with progress updates
            if callback:
                progress = (i + 1) / len(result["segments"])
                callback(formatted_text, progress, segments)

        # Return the complete transcription
        return {"segments": segments, "text": all_text, "language": detected_language}


class PyAnnoteDiarizer:
    """Class to handle speaker diarization using PyAnnote"""

    def __init__(self, device=DEVICE, hf_token=None, fast_mode=False):
        self.device = device
        self.pipeline = None
        self.hf_token = hf_token
        self.fast_mode = fast_mode
        self.progress_callback = None
        self.start_time = None

    def check_pyannote_available(self, log_func=print):
        """Check if PyAnnote is available"""
        try:
            from pyannote.audio import Pipeline
            log_func("PyAnnote is already installed")
            return True
        except ImportError:
            log_func("PyAnnote not found. Please install pyannote.audio")
            return False

    def set_progress_callback(self, callback):
        """Set a callback function to report progress"""
        self.progress_callback = callback

    def load_pipeline(self, log_func=print):
        """Load the PyAnnote diarization pipeline"""
        if not self.check_pyannote_available(log_func):
            log_func("Cannot load pipeline: PyAnnote is not installed")
            return False

        try:
            from pyannote.audio import Pipeline
            log_func("Loading PyAnnote diarization pipeline...")

            if not self.hf_token:
                log_func("Error: Hugging Face token is required")
                return False

            # Initialize the pyannote pipeline
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token
            )

            # If using CUDA (GPU)
            if torch.cuda.is_available():
                log_func("Setting pipeline to use GPU")
                self.pipeline = self.pipeline.to(self.device)

            # Set optimized parameters for faster processing if fast mode is enabled
            if self.fast_mode:
                log_func("Using fast mode with optimized parameters")
                try:
                    # These parameters optimize for speed over accuracy
                    self.pipeline.instantiate({
                        "clustering": {
                            "method": "average",
                            "min_cluster_size": 15,
                            "threshold": 0.7
                        }
                    })
                except Exception as e:
                    log_func(f"Warning: Could not set optimized parameters: {str(e)}")

            log_func("PyAnnote diarization pipeline loaded successfully")
            return True
        except Exception as e:
            log_func(f"Error loading PyAnnote pipeline: {str(e)}")
            import traceback
            log_func(traceback.format_exc())
            return False

    def log_progress(self, message):
        """Log progress during diarization"""
        if self.progress_callback:
            # Calculate approximate progress
            if self.start_time:
                elapsed = time.time() - self.start_time
                self.progress_callback(f"Diarization in progress... {message} (Elapsed: {elapsed:.1f}s)")

    def diarize(self, audio_path, log_func=print):
        """
        Perform speaker diarization on an audio file

        Args:
            audio_path: Path to the audio file
            log_func: Function for logging

        Returns:
            List of speaker segments with start/end times and speaker IDs
        """
        if self.pipeline is None:
            if not self.load_pipeline(log_func):
                log_func("Cannot perform diarization: Pipeline not loaded")
                return []

        try:
            self.start_time = time.time()
            log_func(f"Processing file for diarization: {os.path.basename(audio_path)}")

            # Make sure the file is in WAV format
            if not audio_path.lower().endswith('.wav'):
                log_func(f"Converting {os.path.basename(audio_path)} to WAV format for diarization...")
                wav_path = AudioPreprocessor.convert_to_wav_for_diarization(audio_path)
                log_func(f"Converted to: {os.path.basename(wav_path)}")
                audio_path = wav_path

            # Get audio duration for progress estimation
            audio_duration = 0
            try:
                audio = AudioSegment.from_file(audio_path)
                audio_duration = len(audio) / 1000.0  # in seconds
                log_func(f"Audio duration: {audio_duration:.2f} seconds")

                # Estimate diarization time based on duration
                estimated_time = audio_duration * (0.5 if CUDA_AVAILABLE else 2.0)
                log_func(f"Estimated diarization time: {estimated_time:.1f} seconds")
            except Exception:
                pass

            # Log the start of actual diarization
            log_func("Starting diarization process - this can take several minutes...")

            # Setup progress reporting
            last_update_time = time.time()
            update_interval = 10  # seconds

            # Create a timer to update progress periodically
            def update_progress():
                nonlocal last_update_time
                current_time = time.time()
                if current_time - last_update_time >= update_interval:
                    last_update_time = current_time
                    elapsed = current_time - self.start_time
                    log_func(f"Diarization still running... {elapsed:.1f}s elapsed")
                    if self.progress_callback:
                        self.progress_callback(f"Processing... {elapsed:.1f}s elapsed")

            # Start a periodic timer for progress updates
            timer = QTimer()
            timer.timeout.connect(update_progress)
            timer.start(5000)  # Check every 5 seconds

            # Apply the pipeline to the audio file - DON'T use a hook since it's causing errors
            diarization = self.pipeline(audio_path)

            # Stop the timer
            timer.stop()

            # Calculate total time
            total_time = time.time() - self.start_time

            # Process the results
            segments = []
            unique_speakers = set()

            # Get speaker segments
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
                unique_speakers.add(speaker)

            log_func(f"Diarization completed in {total_time:.2f} seconds")
            log_func(f"Found {len(segments)} speaker segments with {len(unique_speakers)} unique speakers")

            # Show the first few segments and speakers
            if segments:
                log_func("Sample speaker segments:")
                for i, seg in enumerate(segments[:5]):
                    log_func(f"  - {seg['speaker']}: {seg['start']:.2f}s to {seg['end']:.2f}s")
                if len(segments) > 5:
                    log_func(f"  - ... and {len(segments) - 5} more segments")

            return segments

        except Exception as e:
            log_func(f"Error performing diarization: {str(e)}")
            import traceback
            log_func(traceback.format_exc())
            return []


class TranscriptionThread(QThread):
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    log_update = pyqtSignal(str)
    text_update = pyqtSignal(str)  # For real-time updates
    transcription_complete = pyqtSignal(str)

    def __init__(self, audio_path, hf_token, whisper_model="base",
                 split_files=True, max_chunk_minutes=10,
                 language="auto", fast_diarization=False):
        super().__init__()
        self.audio_path = audio_path
        self.hf_token = hf_token
        self.whisper_model = whisper_model
        self.split_files = split_files
        self.max_chunk_minutes = max_chunk_minutes
        self.language = language
        self.fast_diarization = fast_diarization
        self.is_cancelled = False

    def run(self):
        try:
            start_time = time.time()
            temp_files = []

            # Log system information
            file_size_mb = os.path.getsize(self.audio_path) / (1024 * 1024)
            self.log_update.emit(f"Processing file: {os.path.basename(self.audio_path)}")
            self.log_update.emit(f"File size: {file_size_mb:.2f} MB")
            self.log_update.emit(f"Using Whisper model: {self.whisper_model}")
            self.log_update.emit(f"CUDA available: {CUDA_AVAILABLE}")
            if CUDA_AVAILABLE:
                self.log_update.emit(f"CUDA version: {CUDA_VERSION}")
                self.log_update.emit(f"GPU: {torch.cuda.get_device_name(0)}")
                self.log_update.emit(
                    f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
                self.log_update.emit(f"TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")

            # Check if token is provided
            if not self.hf_token:
                self.status_update.emit("Error: Hugging Face token is required")
                return

            # STEP 1: Preprocess audio (5%)
            self.status_update.emit("Preprocessing audio...")
            self.progress_update.emit(5)

            # Always preprocess to WAV for PyAnnote compatibility
            self.log_update.emit("Converting audio to WAV format for diarization...")
            diarization_wav = AudioPreprocessor.convert_to_wav_for_diarization(self.audio_path)
            temp_files.append(diarization_wav)
            self.log_update.emit(f"Converted to: {os.path.basename(diarization_wav)}")

            # STEP 2: Create smaller chunks if needed (10%)
            self.progress_update.emit(10)
            processed_files = [diarization_wav]

            # Only split if user enabled it and file is large enough
            if self.split_files and file_size_mb > 10:  # Split larger files into chunks
                self.log_update.emit(f"Splitting audio into {self.max_chunk_minutes}-minute chunks...")
                processed_files = AudioPreprocessor.split_audio(diarization_wav, self.max_chunk_minutes)
                temp_files.extend(processed_files)
                self.log_update.emit(f"Split into {len(processed_files)} chunks")
            else:
                self.log_update.emit("Using whole-file approach for better speaker recognition")

            # STEP 3: Load Whisper model (15%)
            self.status_update.emit("Loading Whisper model...")
            self.progress_update.emit(15)

            transcriber = WhisperTranscriber(self.whisper_model, DEVICE, self.language)
            t0 = time.time()
            self.log_update.emit(f"Loading Whisper {self.whisper_model} model...")
            transcriber.load_model()
            load_time = time.time() - t0
            self.log_update.emit(f"Whisper model loaded in {load_time:.2f} seconds")

            # STEP 4: Transcribe all chunks (40%)
            # Track all segments with their timestamps
            all_segments = []
            detected_language = None

            # Transcribe each chunk
            for i, file_path in enumerate(processed_files):
                if self.is_cancelled:
                    break

                chunk_name = os.path.basename(file_path)
                self.status_update.emit(f"Transcribing chunk {i + 1}/{len(processed_files)}...")
                self.log_update.emit(f"Transcribing chunk {i + 1}/{len(processed_files)}: {chunk_name}")

                # Calculate progress for this step (from 20% to 40%)
                progress_step = 20 + (i * (20 / len(processed_files)))
                self.progress_update.emit(int(progress_step))

                # For first chunk, use auto language detection or specified language
                # For subsequent chunks, use previously detected language
                if i == 0 or detected_language is None:
                    transcriber.language = self.language
                else:
                    transcriber.language = detected_language

                # Define callback for real-time updates
                def update_transcription(text, progress, _):
                    if not self.is_cancelled:
                        # Update overall progress (20-40% range for transcription)
                        overall_progress = int(20 + progress * 20)
                        self.progress_update.emit(overall_progress)
                        # Update text in real-time
                        self.text_update.emit(text)

                # Transcribe this chunk - use original file for best quality
                t0 = time.time()
                # For first chunk, transcribe from original file for best quality
                if i == 0 and len(processed_files) == 1:
                    transcription = transcriber.transcribe_file(self.audio_path, update_transcription)
                else:
                    transcription = transcriber.transcribe_file(file_path, update_transcription)

                transcribe_time = time.time() - t0
                self.log_update.emit(f"Transcription completed in {transcribe_time:.2f} seconds")

                # Store detected language from first chunk
                if i == 0:
                    detected_language = transcription.get("language", "en")
                    self.log_update.emit(f"Detected language: {detected_language}")

                # Calculate time offset for chunks after the first one
                time_offset = 0
                if i > 0:
                    # Calculate based on previous audio duration
                    prev_audio = AudioSegment.from_file(processed_files[i - 1])
                    time_offset = sum(
                        AudioSegment.from_file(f).duration_seconds
                        for f in processed_files[:i]
                    )

                # Add segments with corrected timestamps
                for segment in transcription["segments"]:
                    all_segments.append({
                        "start": segment["start"] + time_offset,
                        "end": segment["end"] + time_offset,
                        "text": segment["text"].strip()
                    })

            if self.is_cancelled:
                self.status_update.emit("Transcription cancelled")
                return

            # STEP 5: Initialize PyAnnote diarization (45%)
            self.status_update.emit("Loading PyAnnote diarization pipeline...")
            self.progress_update.emit(45)

            # Initialize PyAnnote diarizer
            diarizer = PyAnnoteDiarizer(DEVICE, self.hf_token, self.fast_diarization)

            # Set up progress updates for diarization
            def update_diarization_progress(message):
                self.status_update.emit(f"Diarizing: {message}")
                self.log_update.emit(message)

            diarizer.set_progress_callback(update_diarization_progress)

            # Load the diarization pipeline
            t0 = time.time()
            if not diarizer.load_pipeline(self.log_update.emit):
                self.log_update.emit("Warning: Could not load PyAnnote pipeline. Continuing with transcription only.")

                # Skip diarization and create transcript with timestamps only
                result_text = ""
                for segment in sorted(all_segments, key=lambda x: x["start"]):
                    timestamp = time.strftime('%H:%M:%S', time.gmtime(segment["start"]))
                    result_text += f"[{timestamp}] {segment['text']}\n"

                # Clean up and return
                for temp_file in temp_files:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    except Exception as e:
                        pass

                total_time = time.time() - start_time
                self.log_update.emit(f"Total processing time: {total_time:.2f} seconds")
                self.status_update.emit("Transcription complete! (No speaker detection)")
                self.progress_update.emit(100)
                self.transcription_complete.emit(result_text)
                return

            load_time = time.time() - t0
            self.log_update.emit(f"PyAnnote pipeline loaded in {load_time:.2f} seconds")

            # STEP 6: Process speaker diarization on all chunks (50-80%)
            all_speaker_turns = []

            # Start with a higher progress value for feedback
            self.progress_update.emit(50)

            # Timer for periodic updates during long-running diarization
            diar_timer = QTimer()
            diar_start = time.time()

            # Setup progress reporting
            def update_progress():
                elapsed = time.time() - diar_start
                self.log_update.emit(f"Diarization in progress... {elapsed:.1f}s elapsed")
                self.status_update.emit(f"Diarizing audio... {elapsed:.1f}s elapsed (this can take several minutes)")
                self.progress_update.emit(50 + min(int(elapsed / 60), 25))  # Increment progress slowly up to 75%

            for i, file_path in enumerate(processed_files):
                if self.is_cancelled:
                    break

                chunk_name = os.path.basename(file_path)
                self.status_update.emit(f"Analyzing speakers in chunk {i + 1}/{len(processed_files)}...")
                self.log_update.emit(
                    f"Performing speaker diarization on chunk {i + 1}/{len(processed_files)}: {chunk_name}")

                # Update status with more details
                duration_info = ""
                try:
                    audio = AudioSegment.from_file(file_path)
                    duration_sec = len(audio) / 1000.0
                    duration_info = f" (Duration: {duration_sec:.1f}s)"
                except:
                    pass

                self.status_update.emit(f"Diarizing chunk {i + 1}/{len(processed_files)}{duration_info}...")

                # Process diarization
                diar_start = time.time()

                # Start a timer to provide periodic updates
                diar_timer.timeout.connect(update_progress)
                diar_timer.start(5000)  # Update every 5 seconds

                # IMPORTANT: Verify this is a WAV file - if not, it won't work
                if not file_path.lower().endswith('.wav'):
                    self.log_update.emit(f"Converting to WAV format for PyAnnote compatibility...")
                    file_path = AudioPreprocessor.convert_to_wav_for_diarization(file_path)
                    temp_files.append(file_path)

                self.log_update.emit(f"Starting diarization on {os.path.basename(file_path)}...")
                t0 = time.time()
                speaker_segments = diarizer.diarize(file_path, self.log_update.emit)

                # Stop the timer
                diar_timer.stop()

                diarize_time = time.time() - t0
                self.log_update.emit(f"Diarization completed in {diarize_time:.2f} seconds")

                # Calculate time offset for chunks after the first one
                time_offset = 0
                if i > 0:
                    # Calculate based on previous audio duration
                    time_offset = sum(
                        AudioSegment.from_file(f).duration_seconds
                        for f in processed_files[:i]
                    )

                # Add segments with time offset
                for segment in speaker_segments:
                    all_speaker_turns.append({
                        "start": segment["start"] + time_offset,
                        "end": segment["end"] + time_offset,
                        "speaker": segment["speaker"]
                    })

                # Update progress after this chunk
                progress = 50 + (30 * (i + 1) / len(processed_files))
                self.progress_update.emit(int(progress))

                self.log_update.emit(f"Found {len(speaker_segments)} speaker segments in this chunk")

            if self.is_cancelled:
                self.status_update.emit("Transcription cancelled")
                return

            # STEP 7: Merge transcription with speaker info (90%)
            self.status_update.emit("Merging transcription with speaker information...")
            self.progress_update.emit(90)

            # Sort all data by timestamps
            all_segments = sorted(all_segments, key=lambda x: x["start"])

            # Process speaker data if available
            if all_speaker_turns:
                self.log_update.emit("Matching speakers with transcription segments...")
                all_speaker_turns = sorted(all_speaker_turns, key=lambda x: x["start"])

                # Process speaker turns to create speaker profiles
                speakers_dict = {}
                for turn in all_speaker_turns:
                    speaker = turn["speaker"]
                    if speaker not in speakers_dict:
                        speakers_dict[speaker] = {
                            "segments": [],
                            "total_duration": 0
                        }

                    segment_duration = turn["end"] - turn["start"]
                    speakers_dict[speaker]["segments"].append(turn)
                    speakers_dict[speaker]["total_duration"] += segment_duration

                # Log speaker information
                self.log_update.emit(f"Speaker analysis: {len(speakers_dict)} unique speakers detected")
                for speaker, data in speakers_dict.items():
                    self.log_update.emit(
                        f"  - {speaker}: {len(data['segments'])} segments, {data['total_duration']:.2f}s total")

                # Initialize result
                result_text = ""
                current_speaker = None

                # Improved speaker assignment algorithm with overlap calculation
                for segment in all_segments:
                    seg_start, seg_end = segment["start"], segment["end"]
                    seg_duration = seg_end - seg_start

                    # Find all speaker turns that overlap with this segment
                    best_speaker = None
                    best_overlap = 0

                    for turn in all_speaker_turns:
                        # Calculate overlap
                        overlap_start = max(seg_start, turn["start"])
                        overlap_end = min(seg_end, turn["end"])

                        if overlap_end > overlap_start:  # There is overlap
                            overlap_duration = overlap_end - overlap_start
                            overlap_percentage = overlap_duration / seg_duration

                            if overlap_percentage > best_overlap:
                                best_overlap = overlap_percentage
                                best_speaker = turn["speaker"]

                    # If we found a good overlap, use that speaker
                    if best_speaker and best_overlap > 0.1:  # At least 10% overlap
                        assigned_speaker = best_speaker
                    # Otherwise use the previous speaker or default
                    else:
                        assigned_speaker = current_speaker if current_speaker else "SPEAKER_01"

                    # Format timestamp and add to result
                    timestamp = time.strftime('%H:%M:%S', time.gmtime(seg_start))

                    # Only show speaker when it changes
                    if assigned_speaker != current_speaker:
                        result_text += f"\n[{timestamp}] {assigned_speaker}: {segment['text']}\n"
                        current_speaker = assigned_speaker
                    else:
                        result_text += f"[{timestamp}] {segment['text']}\n"
            else:
                # No speaker data - just format the transcription with timestamps
                self.log_update.emit("No speaker data available, creating transcript with timestamps only")
                result_text = ""
                for segment in all_segments:
                    timestamp = time.strftime('%H:%M:%S', time.gmtime(segment["start"]))
                    result_text += f"[{timestamp}] {segment['text']}\n"

            # STEP 8: Final processing (100%)
            self.status_update.emit("Finalizing transcription...")
            self.progress_update.emit(95)

            # Clean up temp files
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception as e:
                    self.log_update.emit(f"Warning: Could not remove temp file {temp_file}: {str(e)}")

            total_time = time.time() - start_time
            self.log_update.emit(f"Total processing time: {total_time:.2f} seconds")
            self.status_update.emit("Transcription complete!")
            self.progress_update.emit(100)
            self.transcription_complete.emit(result_text)

        except Exception as e:
            self.log_update.emit(f"ERROR: {str(e)}")
            self.status_update.emit(f"Error: {str(e)}")
            import traceback
            self.log_update.emit(traceback.format_exc())

    def cancel(self):
        self.is_cancelled = True


class TranscriptionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.audio_path = None
        self.elapsed_timer = None
        self.start_time = None
        self.check_cuda()
        self.initUI()

    def check_cuda(self):
        """Attempt to diagnose CUDA issues"""
        try:
            # Force check for CUDA
            if not torch.cuda.is_available():
                print("CUDA not available. Checking why...")
                if not hasattr(torch, 'cuda') or not torch.cuda:
                    print("PyTorch was not built with CUDA support")
                    print("Run: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118")
                else:
                    print("PyTorch has CUDA support, but no CUDA device is available")
                    if hasattr(torch.cuda, 'is_available'):
                        print(f"torch.cuda.is_available() returned: {torch.cuda.is_available()}")
                        print("This could be due to a driver issue or not having a CUDA-capable GPU")
            else:
                print(f"CUDA is available. Device count: {torch.cuda.device_count()}")
                print(f"Current device: {torch.cuda.current_device()}")
                print(f"Device name: {torch.cuda.get_device_name(0)}")
                print(f"CUDA version: {torch.version.cuda}")
        except Exception as e:
            print(f"Error checking CUDA: {str(e)}")

    def initUI(self):
        self.setWindowTitle('Audio Transcription with Speaker Recognition')
        self.setMinimumSize(1000, 700)

        # Main layout
        main_layout = QVBoxLayout()

        # Hardware info - prominent display
        gpu_info = "⚡ GPU ACTIVE" if CUDA_AVAILABLE else "⚠️ CPU ONLY (slower)"
        gpu_details = f" - {torch.cuda.get_device_name(0)}" if CUDA_AVAILABLE else " - No CUDA GPU detected"

        hardware_group = QGroupBox("Hardware Status")
        hardware_layout = QVBoxLayout()

        hardware_label = QLabel(f"{gpu_info}{gpu_details}")
        hardware_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        hardware_layout.addWidget(hardware_label)

        if not CUDA_AVAILABLE:
            cuda_help = QLabel(
                "For faster processing, install PyTorch with CUDA support by running:\npip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118")
            cuda_help.setStyleSheet("color: #666;")
            hardware_layout.addWidget(cuda_help)

        hardware_group.setLayout(hardware_layout)
        main_layout.addWidget(hardware_group)

        # Token input section
        token_layout = QHBoxLayout()
        token_label = QLabel("Hugging Face Token:")
        self.token_input = QLineEdit()
        self.token_input.setEchoMode(QLineEdit.Password)  # Hide token for security

        # Check for token in environment variable
        env_token = os.environ.get('HF_TOKEN', '')
        if env_token:
            self.token_input.setText(env_token)

        token_layout.addWidget(token_label)
        token_layout.addWidget(self.token_input)
        main_layout.addLayout(token_layout)

        # File selection section
        file_layout = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.file_btn = QPushButton("Select Audio File")
        self.file_btn.clicked.connect(self.select_file)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_btn)
        main_layout.addLayout(file_layout)

        # Options Group Box
        options_group = QGroupBox("Processing Options")
        options_layout = QVBoxLayout()

        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Whisper Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.model_combo.setCurrentText("base")  # Default model
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        options_layout.addLayout(model_layout)

        # Language selection
        language_layout = QHBoxLayout()
        language_label = QLabel("Language:")
        self.language_combo = QComboBox()
        self.language_combo.addItems(["auto", "en", "pl", "de", "fr", "es", "it", "nl", "pt", "ja", "zh", "ru"])
        self.language_combo.setCurrentText("auto")  # Default to auto-detect
        language_layout.addWidget(language_label)
        language_layout.addWidget(self.language_combo)
        options_layout.addLayout(language_layout)

        # Splitting options
        split_layout = QHBoxLayout()
        self.split_check = QCheckBox("Split Large Files")
        self.split_check.setChecked(True)
        split_layout.addWidget(self.split_check)

        chunk_label = QLabel("Max Chunk Size (minutes):")
        self.chunk_spin = QSpinBox()
        self.chunk_spin.setMinimum(1)
        self.chunk_spin.setMaximum(30)
        self.chunk_spin.setValue(5)  # Default to 5 minutes
        split_layout.addWidget(chunk_label)
        split_layout.addWidget(self.chunk_spin)
        options_layout.addLayout(split_layout)

        # Fast diarization mode option
        diarization_layout = QHBoxLayout()
        self.fast_diarization_check = QCheckBox("Fast Speaker Detection Mode (Less Accurate)")
        self.fast_diarization_check.setChecked(False)
        diarization_layout.addWidget(self.fast_diarization_check)
        options_layout.addLayout(diarization_layout)

        options_group.setLayout(options_layout)
        main_layout.addWidget(options_group)

        # Start transcription button
        button_layout = QHBoxLayout()
        self.transcribe_btn = QPushButton("Start Transcription")
        self.transcribe_btn.clicked.connect(self.start_transcription)
        self.transcribe_btn.setEnabled(False)  # Disabled until file is selected

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_transcription)
        self.cancel_btn.setEnabled(False)

        button_layout.addWidget(self.transcribe_btn)
        button_layout.addWidget(self.cancel_btn)
        main_layout.addLayout(button_layout)

        # Progress bar and timer
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        self.time_label = QLabel("Time: 00:00:00")

        progress_layout.addWidget(self.progress_bar)
        progress_layout.addWidget(self.time_label)
        main_layout.addLayout(progress_layout)

        # Status label
        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)

        # Create tabs for results and logs
        self.tabs = QTabWidget()

        # Result text area
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.tabs.addTab(self.result_text, "Transcription")

        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.tabs.addTab(self.log_text, "Processing Log")

        main_layout.addWidget(self.tabs)

        # Save button
        save_layout = QHBoxLayout()
        self.save_btn = QPushButton("Save Transcription")
        self.save_btn.clicked.connect(self.save_transcription)
        self.save_btn.setEnabled(False)  # Disabled until transcription is done

        # Copy to clipboard button
        self.copy_btn = QPushButton("Copy to Clipboard")
        self.copy_btn.clicked.connect(self.copy_to_clipboard)
        self.copy_btn.setEnabled(False)  # Disabled until transcription is done

        save_layout.addWidget(self.save_btn)
        save_layout.addWidget(self.copy_btn)
        main_layout.addLayout(save_layout)

        # Set main widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def select_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self,
            "Select Audio File",
            "",
            "Audio Files (*.mp3 *.mp4 *.m4a *.wav);;All Files (*)"
        )

        if file_path:
            self.audio_path = file_path
            self.file_label.setText(os.path.basename(file_path))
            self.transcribe_btn.setEnabled(True)
            self.status_label.setText("File selected. Ready to transcribe.")

    def start_transcription(self):
        if not self.audio_path:
            QMessageBox.warning(self, "Error", "Please select an audio file first.")
            return

        hf_token = self.token_input.text().strip()
        if not hf_token:
            QMessageBox.warning(self, "Error", "Please enter your Hugging Face token.")
            return

        # Disable UI elements during transcription
        self.transcribe_btn.setEnabled(False)
        self.file_btn.setEnabled(False)
        self.token_input.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.language_combo.setEnabled(False)
        self.split_check.setEnabled(False)
        self.chunk_spin.setEnabled(False)
        self.fast_diarization_check.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        # Clear previous results
        self.result_text.clear()
        self.log_text.clear()

        # Log the start time
        now = datetime.now()
        self.log_text.append(f"=== Transcription started at {now.strftime('%Y-%m-%d %H:%M:%S')} ===")

        # Start the elapsed timer
        self.start_time = time.time()
        self.elapsed_timer = QTimer()
        self.elapsed_timer.timeout.connect(self.update_elapsed_time)
        self.elapsed_timer.start(1000)  # update every second

        # Start transcription in a separate thread
        self.thread = TranscriptionThread(
            self.audio_path,
            hf_token,
            self.model_combo.currentText(),
            self.split_check.isChecked(),
            self.chunk_spin.value(),
            self.language_combo.currentText(),
            self.fast_diarization_check.isChecked()
        )

        # Connect signals
        self.thread.progress_update.connect(self.update_progress)
        self.thread.status_update.connect(self.update_status)
        self.thread.log_update.connect(self.update_log)
        self.thread.text_update.connect(self.update_text)
        self.thread.transcription_complete.connect(self.show_transcription)

        # Start the thread
        self.thread.start()

    def update_text(self, text):
        """Update the transcription text in real-time"""
        cursor = self.result_text.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(text)
        self.result_text.setTextCursor(cursor)
        # Switch to transcription tab to show real-time updates
        self.tabs.setCurrentIndex(0)

    def cancel_transcription(self):
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.cancel()
            self.update_status("Cancelling transcription...")
            self.update_log("Transcription process is being cancelled, please wait...")

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def update_status(self, message):
        self.status_label.setText(message)

    def update_log(self, message):
        self.log_text.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        # Auto-scroll to the bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def update_elapsed_time(self):
        if self.start_time:
            elapsed = time.time() - self.start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            self.time_label.setText(f"Time: {hours:02d}:{minutes:02d}:{seconds:02d}")

    def show_transcription(self, text):
        self.result_text.setText(text)

        # Stop the timer
        if self.elapsed_timer:
            self.elapsed_timer.stop()

        # Re-enable UI elements
        self.transcribe_btn.setEnabled(True)
        self.file_btn.setEnabled(True)
        self.token_input.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.language_combo.setEnabled(True)
        self.split_check.setEnabled(True)
        self.chunk_spin.setEnabled(True)
        self.fast_diarization_check.setEnabled(True)
        self.save_btn.setEnabled(True)
        self.copy_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

        # Switch to transcription tab to show results
        self.tabs.setCurrentIndex(0)

    def save_transcription(self):
        if not self.result_text.toPlainText():
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Transcription",
            "",
            "Text Files (*.txt);;All Files (*)"
        )

        if save_path:
            try:
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write(self.result_text.toPlainText())
                self.status_label.setText(f"Transcription saved to {save_path}")
                self.update_log(f"Transcription saved to {save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save file: {str(e)}")

    def copy_to_clipboard(self):
        if not self.result_text.toPlainText():
            return

        clipboard = QApplication.clipboard()
        clipboard.setText(self.result_text.toPlainText())
        self.status_label.setText("Transcription copied to clipboard")
        self.update_log("Transcription copied to clipboard")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TranscriptionApp()
    window.show()
    sys.exit(app.exec_())