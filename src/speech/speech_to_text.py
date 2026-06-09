import os
import wave
import logging
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

class SpeechToTextManager:
    """
    Manages audio recording from microphone and transcription using Faster-Whisper.
    """
    def __init__(self, model_size: str = "tiny"):
        """
        Initializes the SpeechToTextManager and loads the WhisperModel.
        """
        # Try to load model using GPU if CUDA is available, else fall back to CPU
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            logger.info(f"Loading Faster-Whisper model '{model_size}' on device: {device} (compute_type: {compute_type})...")
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        except Exception as e:
            logger.warning(
                f"Failed to load Whisper on GPU ({e}). "
                "Falling back to CPU with int8 computation..."
            )
            try:
                self.model = WhisperModel(model_size, device="cpu", compute_type="int8")
                logger.info("Successfully loaded Faster-Whisper model on CPU.")
            except Exception as cpu_err:
                logger.error(f"Failed to load Whisper model on CPU: {cpu_err}")
                
                # Provide a descriptive and helpful troubleshooting message
                help_msg = (
                    f"Failed to download or load Whisper model '{model_size}' from Hugging Face.\n\n"
                    "Troubleshooting Guide:\n"
                    "1. Switch to a smaller model by adding this to your .env file:\n"
                    "   WHISPER_MODEL=tiny\n\n"
                    "2. If you are behind a firewall or restricted network, use the Hugging Face mirror by adding this to your .env file:\n"
                    "   HF_ENDPOINT=https://hf-mirror.com\n\n"
                    "3. Alternatively, manually download model files and set WHISPER_MODEL in .env to the local directory path.\n\n"
                    f"Original Error: {cpu_err}"
                )
                raise RuntimeError(help_msg)

    def record_audio(self, filename: str = "temp_recording.wav", sample_rate: int = 16000) -> None:
        """
        Records audio from the microphone until the user presses Enter.
        
        Args:
            filename (str): The filename to save the recorded audio.
            sample_rate (int): Sample rate for recording (16kHz is optimal for Whisper).
        """
        recorded_data = []

        def callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio record status warning: {status}")
            recorded_data.append(indata.copy())

        # Start capturing in a background stream
        logger.info(f"Recording microphone input at {sample_rate}Hz...")
        print("\n>>> Recording started. Speak into your microphone.")
        print(">>> Press [Enter] to STOP recording...")
        
        try:
            with sd.InputStream(samplerate=sample_rate, channels=1, callback=callback, dtype='float32'):
                # Block the main thread until the user hits Enter
                input()
        except Exception as e:
            logger.error(f"Failed to capture audio stream: {e}")
            raise RuntimeError(f"Audio recording failed: {e}")

        print(">>> Recording stopped. Processing audio...")

        if not recorded_data:
            raise ValueError("No audio data captured. Please make sure your microphone is connected.")

        # Concatenate and save to WAV
        try:
            audio_np = np.concatenate(recorded_data, axis=0)
            
            # Convert float32 array (-1.0 to 1.0) to int16 (PCM 16-bit)
            audio_int16 = (audio_np * 32767).astype(np.int16)
            
            # Save the WAV file
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit PCM = 2 bytes
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())
                
            logger.info(f"Audio successfully recorded and saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to write audio file: {e}")
            raise RuntimeError(f"Failed to save audio file: {e}")

    def transcribe_audio(self, filename: str = "temp_recording.wav") -> str:
        """
        Transcribes the given audio WAV file using Faster-Whisper.
        Automatically deletes the audio file after transcription.
        
        Args:
            filename (str): The WAV file path to transcribe.
            
        Returns:
            str: The transcribed text.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Audio file to transcribe not found: {filename}")

        logger.info(f"Starting transcription of {filename} using Faster-Whisper...")
        try:
            segments, info = self.model.transcribe(filename, beam_size=5)
            
            transcription_text = ""
            for segment in segments:
                transcription_text += segment.text + " "
                
            transcription_text = transcription_text.strip()
            logger.info(f"Transcription completed. Language: {info.language} (prob: {info.language_probability:.2f})")
            return transcription_text
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"Transcription error: {e}")
        finally:
            # Clean up the audio file
            try:
                if os.path.exists(filename):
                    os.remove(filename)
                    logger.info(f"Cleaned up temporary audio file: {filename}")
            except Exception as clean_err:
                logger.warning(f"Could not delete temporary audio file '{filename}': {clean_err}")
