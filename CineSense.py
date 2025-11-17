# -*- coding: utf-8 -*-
"""BDA_Project_1.ipynb

# Big Data Analytics 2024 Project 1

Name: Mohammed Abuzer Khanzade

Course: MSc Data Science

Birkbeck University of London

# CineSense
# 1. **Introduction**
CineSense is a unique startup that deals with video processing using the most advanced natural language processing and computer vision techniques to extract valuable insight from video content on social media. In this age of businesses being run with a data-driven approach for analyses and future decisions, knowing the sentiment and emotional response of the audience is very vital. CineSense wants to bridge this gap by providing insightful information extracted from users' video content on social media platforms like YouTube.

This project involves developing a Python application using multiprocessing, threading, or asynchronous programming concepts to download and analyze YouTube videos.

The tasks include:

*   Downloading YouTube videos.
*   Extracting audio from videos.
*   Transcribing audio to text.
*   Performing sentiment analysis.
*   Translating text into Spanish.
*   Extracting emotions from the text.

# 2. Setup and Installation
 It is essential to install all the necessary libraries before implementing any functionality. This ensures that all dependencies are met, and the code can run smoothly. The following libraries are required for downloading videos, extracting audio, transcribing audio, analyzing sentiment, and extracting emotions:

*   **pytube**: Used for downloading YouTube videos.
*   **moviepy**: Used for video editing and audio extraction.
*   **speechrecognition**: Used for transcribing audio to text.
*   **textblob**: Used for text processing, sentiment analysis, and translation.
*   **nrclex**: Used for extracting emotions from text.
*  **spacy and nltk**: Libraries for natural language processing tasks.
*   **nltk.download('punkt')**: Downloads the tokenizer required by NLTK.
*   **textblob.download_corpora**: Downloads the necessary corpora for TextBlob.
"""

!pip install pytube moviepy speechrecognition textblob nrclex spacy nltk

import nltk
nltk.download('punkt')
!python -m textblob.download_corpora

"""# 3. Creating Video URLs
This code defines a list of YouTube video URLs and writes them to a text file named **'video_urls.txt'**. This file will be used as the source for downloading videos. The function verifies the content of the file by reading and printing it.
"""

# Creats video_urls.txt file
def main_create_urls():
    urls = [
        "https://youtu.be/9h2bKsJ7j_c?si=JmzURQROnXAJbrr5",
        "https://youtu.be/hCXYxufbDag?si=qh1OCN6r9q7iTL4B",
        "https://youtu.be/qYyxoor5Hk4?si=SoFp54_InaCAJ7hk",
        "https://youtu.be/HisYsqqszq0?si=ClaUeFCAQnhxucc0",
        "https://youtu.be/XR4Vy2a3MqY?si=dUuqAUs5yNPH2e9H",
        "https://youtu.be/Y_9v5yPi2DE?si=cOmotMGkj20W8dwN",
        "https://youtu.be/NbqKRCefJhU?si=NRByXUnCLNiOtEME",
        "https://youtu.be/1aA1WGON49E?si=JGpFu6e9S-1f1Av2",
        "https://youtu.be/XALBGkjkUPQ?si=plNbDeIQMGCQOlbF",
        "https://youtu.be/nyhRNwTfydU?si=mkzLzO0Shw1lw6v3",
        "https://youtu.be/n1fGPpuaDpw?si=hXAChznyw65EBeXf",
        "https://youtu.be/5Intdml2m-0?si=X7Fzg7TvaQItoJoc"
    ]

    with open('video_urls.txt', 'w') as f:
        for url in urls:
            f.write(f"{url}\n")

    # Verifyies the contents of the video_urls.txt file
    with open('video_urls.txt', 'r') as file:
        content = file.read()
        print(content)

main_create_urls()

"""# 4. Reading URLs from File
This code reads the URLs from the **'video_urls.txt'** file and prints them to verify the contents. The **'read_urls'** function reads the URLs from the specified file and returns them as a list of strings. The **'main_read_urls'** function reads the URLs using **'read_urls'** and prints them.

"""

# Utility function to read URLs from file
def read_urls(file_path):
    with open(file_path, 'r') as file:
        urls = file.readlines()
    urls = [url.strip() for url in urls]
    return urls

def main_read_urls():
    urls = read_urls('video_urls.txt')
    print(urls)

main_read_urls()

"""# 5. Downloading videos
This section of the code handles downloading YouTube videos using both serial and parallel processing. It also logs the download activities. It defines a VideoDownloader class to manage video downloads. It includes methods for logging downloads, downloading videos, and handling downloads using both serial and parallel processing. The **'main_download'** function initiates the download process, first using serial and then parallel execution, and measures the time taken for each method.
"""

from pytube import YouTube
from concurrent.futures import ThreadPoolExecutor
from threading import Semaphore, Lock
import time
import os
import logging
from datetime import datetime

# Setup logging to ensure the log file is saved in the correct directory
log_file_path = '/content/download_log.txt'
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(message)s')
log_lock = Lock()

# Test writing to the log file
with open(log_file_path, 'a') as log_file:
    log_file.write('Log file created for debugging purposes.\n')

class VideoDownloader:
    def __init__(self, max_threads=5):
        self.semaphore = Semaphore(max_threads)
        self.output_path = '/content/video_output'
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def log_download(self, url):
        with log_lock:
            timestamp = datetime.now().strftime("%H:%M:%S, %d %B %Y")
            logging.info(f'"Timestamp": {timestamp}, "URL":"{url}", "Download":True')

    def download_video(self, url):
        try:
            yt = YouTube(url)
            stream = yt.streams.get_highest_resolution()
            title = yt.title.replace(' ', '_').replace('/', '_')
            video_folder = os.path.join(self.output_path, title)
            if not os.path.exists(video_folder):
                os.makedirs(video_folder)
            stream.download(output_path=video_folder)
            self.log_download(url)
            print(f"Downloaded: {yt.title}")
        except Exception as e:
            print(f"Failed to download {url}: {e}")
            with open(log_file_path, 'a') as log_file:
                log_file.write(f'Failed to download {url}: {e}\n')

    def download_video_thread(self, url):
        with self.semaphore:
            self.download_video(url)

    def serial_download(self, urls):
        start_time = time.time()
        for url in urls:
            self.download_video(url)
        end_time = time.time()
        print(f"Serial download time: {end_time - start_time} seconds")

    def parallel_download(self, urls):
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(self.download_video_thread, urls)
        end_time = time.time()
        print(f"Parallel download time: {end_time - start_time} seconds")


def main_download():
    urls = read_urls('/content/video_urls.txt')
    downloader = VideoDownloader()
    print("Starting serial download...")
    downloader.serial_download(urls)
    print("Starting parallel download...")
    downloader.parallel_download(urls)

main_download()

# Verify file creation
!ls -l /content

"""# Complexity Analysis

**Time Complexity:**

The time complexity is primarily dependent on network speed and response time of YouTube server. Parallel downloading significantly reduces total download time by leveraging multiple threads.

*   **Serial Download:** O(n), where n is the number of videos.

*   **Parallel Download:** O(n/k), where k is the number of threads, assuming ideal conditions without any significant overhead.

**Space Complexity:**

Both serial and parallel executions have a space complexity of O(1) for the individual download processes, as each video download does not depend on the size of the input but rather the space required to store the videos.

# 6. VideoAnalyzer Class: Comprehensive video Analysis
In this code **'VideoAnalyzer'** class is defined. It is designed to perform various tasks on video files, including extracting audio, transcribing audio, analyzing sentiment, translating text, and extracting emotions. below is the brief overview of each method and its functionality:

**Imports**

*   **'moviepy.editor'**: For handling video files
*   **'moviepy.editor'**: For handling video files
*   **'TextBlob'**: For text processing, sentiment analysis, and translation.
*   **'NRCLex'**: For extracting emotions from text.

**Class Initialization**


*   The **'VideoAnalyzer'** class initializes with an **'output_path'** to save processed files, defaulting to **'video_output'**

**Methods:**


1.   **'extract_audio(video_folder)':**

   *   Extracts audio from the first **'.mp4'** video file in the specified folder.
   *   Saves the audio as a **'.wav'** file.


2.   **'transcribe_audio(audio_path, language='en-US')'**:

   *   Converts audio from a **'.wav'** file to text using Google's Web Speech API.
   *   Handles various exceptions and returns the transcribed text.


3.   **'analyze_sentiment(text)'**:

   *   Analyzes the sentiment of the text using TextBlob.
   *   Returns polarity (positive/negative) and subjectivity scores of the text.


4.   **'translate_text(text, from_lang='en', to_lang='es')'**:

   *   Translates text from English to Spanish using TextBlob.
   *   Handles errors and ensures the input is a string.


5.   **'extract_emotions(text)'**:

   *   Extracts emotional content from the given text using NRCLex.
   *   Returns a dictionary with the frequencies of various emotions.
"""

from moviepy.editor import VideoFileClip
import speech_recognition as sr
from textblob import TextBlob
from nrclex import NRCLex

class VideoAnalyzer:
    def __init__(self, output_path='video_output'):
        self.output_path = output_path

    def extract_audio(self, video_folder):
        try:
            video_file = [f for f in os.listdir(video_folder) if f.endswith('.mp4')][0]
            video_path = os.path.join(video_folder, video_file)
            video = VideoFileClip(video_path)
            audio_path = video_path.replace('.mp4', '.wav')
            video.audio.write_audiofile(audio_path)
            return audio_path
        except Exception as e:
            print(f"Failed to extract audio from {video_path}: {e}")
            return None

    def transcribe_audio(self, audio_path, language='en-US'):
        recognizer = sr.Recognizer()
        try:
            with sr.AudioFile(audio_path) as source:
                audio = recognizer.record(source)
            text = recognizer.recognize_google(audio, language=language)
            return text
        except sr.RequestError as e:
            print(f"Could not request results from Google Web Speech API; {e}")
        except sr.UnknownValueError:
            print(f"Google Web Speech API could not understand audio {audio_path}")
            return ""
        except Exception as e:
            print(f"Error transcribing {audio_path}: {e}")
            return ""

    def analyze_sentiment(self, text):
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity

    def translate_text(self, text, from_lang='en', to_lang='es'):
        print(f"Type of text to be translated: {type(text)}")  # Debugging output
        if isinstance(text, str):
            try:
                blob = TextBlob(text)
                translated = blob.translate(from_lang=from_lang, to=to_lang)
                return str(translated)
            except Exception as e:
                print(f"Translation error: {e}")
                return text
        else:
            print("Error: Input is not a string, cannot translate.")
            return text

    def extract_emotions(self, text):
        emotions = NRCLex(text)
        return emotions.affect_frequencies

"""# 7. Extracting Audio from Video Files
 The **'extract_audio'** method extracts audio from video files and saves them as **.wav** files. The **'main_extract_audio'** function iterates through the downloaded video folders and extracts audio for each video.
"""

def main_extract_audio():
    analyzer = VideoAnalyzer()
    video_folders = [os.path.join('video_output', f) for f in os.listdir('video_output') if os.path.isdir(os.path.join('video_output', f))]
    for video_folder in video_folders:
        audio_path = analyzer.extract_audio(video_folder)
        if audio_path:
            print(f"Extracted audio to {audio_path}")
        else:
            print(f"Skipping audio extraction for {video_folder}")

main_extract_audio()

"""# 8. Transcribing Audio to Text
This code transcribes the audio extracted from the videos to text and saves the transcriptions as text files. The **'transcribe_audio'** method uses the Google Web Speech API to convert audio to text. The **'main_transcribe_audio'** function iterates through the audio files in the video folders and transcribes each audio file.
"""

def main_transcribe_audio():
    analyzer = VideoAnalyzer()
    video_folders = [os.path.join('video_output', f) for f in os.listdir('video_output') if os.path.isdir(os.path.join('video_output', f))]
    for video_folder in video_folders:
        audio_files = [f for f in os.listdir(video_folder) if f.endswith('.wav')]
        if audio_files:
            audio_path = os.path.join(video_folder, audio_files[0])
            transcription = analyzer.transcribe_audio(audio_path)
            transcription_path = audio_path.replace('.wav', '.txt')
            with open(transcription_path, 'w', encoding='utf-8', errors='ignore') as file:
                file.write(transcription)
            print(f"Transcription for {audio_path}: {transcription}")
        else:
            print(f"Audio file for {video_folder} does not exist.")

main_transcribe_audio()

"""# 9. Analyzing Sentiments
This code analyzes the sentiment of the transcribed text, calculating the polarity and subjectivity of each video transcription. The **'analyze_sentiment'** method uses the TextBlob library to determine the sentiment. The **'main_analyze_sentiment'** function iterates through the transcription files in the video folders and analyzes the sentiment of each transcription.
"""

def main_analyze_sentiment():
    analyzer = VideoAnalyzer()
    video_folders = [os.path.join('video_output', f) for f in os.listdir('video_output') if os.path.isdir(os.path.join('video_output', f))]
    for video_folder in video_folders:
        transcription_files = [f for f in os.listdir(video_folder) if f.endswith('.txt') and not '_emotions.txt' in f and not '_translated.txt' in f]
        if transcription_files:
            transcription_path = os.path.join(video_folder, transcription_files[0])
            with open(transcription_path, 'r', encoding='utf-8', errors='ignore') as file:
                transcription = file.read()
            polarity, subjectivity = analyzer.analyze_sentiment(transcription)
            print(f"Video: {os.path.basename(video_folder)} - Polarity: {polarity}, Subjectivity: {subjectivity}")
        else:
            print(f"Transcription file for {video_folder} does not exist.")

main_analyze_sentiment()

"""# 10. Translating Text
It translates the transcribed text into Spanish using the TextBlob library. The **'translate_text'** method translates the text from English to Spanish. The **'main_translate_text'** function iterates through the transcription files in the video folders and translates each transcription.
"""

def main_translate_text():
    analyzer = VideoAnalyzer()
    video_folders = [os.path.join('video_output', f) for f in os.listdir('video_output') if os.path.isdir(os.path.join('video_output', f))]

    for video_folder in video_folders:
        transcription_files = [f for f in os.listdir(video_folder) if f.endswith('.txt') and not f.endswith('_translated.txt') and not f.endswith('_emotions.txt')]

        if transcription_files:
            transcription_path = os.path.join(video_folder, transcription_files[0])

            with open(transcription_path, 'r', encoding='utf-8', errors='ignore') as file:
                transcription = file.read()

            if isinstance(transcription, str):
                translation = analyzer.translate_text(transcription, from_lang='en', to_lang='es')

                translation_path = transcription_path.replace('.txt', '_translated.txt')
                with open(translation_path, 'w', encoding='utf-8', errors='ignore') as file:
                    file.write(translation)

                print(f"Translation for {transcription_path}: {translation}")  # Display the full translation
            else:
                print(f"Error: Transcription is not a string in file {transcription_path}")
        else:
            print(f"Transcription file for {video_folder} does not exist.")

main_translate_text()

"""# 11. Extracting Emotions from Text
This code extracts the emotions from the transcribed text using the NRCLex library. The **'extract_emotions'** method uses NRCLex to analyze the emotions in the text. The **'main_extract_emotions'** function iterates through the transcription files in the video folders and extracts emotions for each transcription.
"""

def main_extract_emotions():
    analyzer = VideoAnalyzer()
    video_folders = [os.path.join('video_output', f) for f in os.listdir('video_output') if os.path.isdir(os.path.join('video_output', f))]
    for video_folder in video_folders:
        transcription_files = [f for f in os.listdir(video_folder) if f.endswith('.txt') and not f.endswith('_translated.txt') and not f.endswith('_emotions.txt')]
        if transcription_files:
            transcription_path = os.path.join(video_folder, transcription_files[0])
            with open(transcription_path, 'r', encoding='utf-8', errors='ignore') as file:
                transcription = file.read()
            emotions = analyzer.extract_emotions(transcription)
            emotions_path = transcription_path.replace('.txt', '_emotions.txt')
            with open(emotions_path, 'w', encoding='utf-8', errors='ignore') as file:
                file.write(str(emotions))
            print(f"Emotions for {transcription_path}: {emotions}")
        else:
            print(f"Transcription file for {video_folder} does not exist.")

main_extract_emotions()

"""# 12. Main Execution Flow
This code defines the main function that manages the entire workflow from downloading videos to extracting emotions. It sequentially calls the functions defined in the previous cells to create video URLs, read URLs, download videos, extract audio, transcribe audio, analyze sentiment, translate text, and extract emotions.
"""

def main():
    # Create video URLs
    main_create_urls()

    # Read URLs
    main_read_urls()

    # Download videos
    main_download()

    # Extract audio
    main_extract_audio()

    # Transcribe audio
    main_transcribe_audio()

    # Analyze sentiment
    main_analyze_sentiment()

    # Translate text
    main_translate_text()

    # Extract emotions
    main_extract_emotions()

main()

"""# 13. Comparing Audio Extraction Methods
This code compares the performance of serial, threading, and multiprocessing methods for extracting audio from videos. The **'serial_extract_audio'** function extracts audio serially. The **'threading_extract_audio'** function uses threading to extract audio in parallel. The **'multiprocessing_extract_audio'** function uses multiprocessing for parallel extraction. The **'main_compare_audio_extraction'** function compares the execution times of these methods.
"""

import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

def serial_extract_audio(videos):
    start_time = time.time()
    analyzer = VideoAnalyzer()
    for video_folder in videos:
        analyzer.extract_audio(video_folder)
    end_time = time.time()
    print(f"Serial extract audio time: {end_time - start_time} seconds")

def threading_extract_audio(videos):
    start_time = time.time()
    analyzer = VideoAnalyzer()
    with ThreadPoolExecutor(max_workers=5) as executor:
        executor.map(analyzer.extract_audio, videos)
    end_time = time.time()
    print(f"Threading extract audio time: {end_time - start_time} seconds")

def multiprocessing_extract_audio(videos):
    start_time = time.time()
    analyzer = VideoAnalyzer()
    with Pool(processes=5) as pool:
        pool.map(analyzer.extract_audio, videos)
    end_time = time.time()
    print(f"Multiprocessing extract audio time: {end_time - start_time} seconds")

def main_compare_audio_extraction():
    video_folders = [os.path.join('video_output', f) for f in os.listdir('video_output') if os.path.isdir(os.path.join('video_output', f))]

    print("Starting serial extract audio...")
    serial_extract_audio(video_folders)

    print("Starting threading extract audio...")
    threading_extract_audio(video_folders)

    print("Starting multiprocessing extract audio...")
    multiprocessing_extract_audio(video_folders)

main_compare_audio_extraction()

"""# 14. Strategy for Each Solution
The strategy for developing the solution involved the following steps.

*   **Library Installation:** Installing necessary libraries for video downloading, audio extraction, transcription, sentiment analysis, translation, and emotion extraction.

*   **Video URL Management:** Creating and reading video URLs from a text file.

*   **Video Downloading:** The download task was implemented using both serial and parallel processing. In the parallel approach, threads and semaphores were used to control the number of simultaneous downloads to avoid YouTube blocks.

*   **Logging:** Logging was implemented to record which video was downloaded by which process or thread. This was achieved using the logging module and a mutex to ensure thread-safe logging.

*   **Audio Processing:** Extracting audio from downloaded videos, transcribing audio to text, and saving transcriptions.

*   **Text Analysis:** Performing sentiment analysis, translating text to Spanish, and extracting emotions from the text. Each task was implemented in a separate function to maintain modularity.

*   **Comparing Methods:** The performance of serial, threading, and multiprocessing methods for extracting audio was compared. Multiprocessing showed significant performance improvement over serial and threading methods.

# 15. Conclusion
The project successfully demonstrated the development of a Python application for downloading and analyzing YouTube videos using various parallel processing techniques. By utilizing threads and multiprocessing, the overall execution time was significantly reduced compared to the serial approach. By comparing serial and parallel processing methods, significant performance improvements were observed, making it feasible to analyze large volumes of social media video content efficiently. This approach enables CineSense to provide valuable insights to businesses, enhancing their understanding of audience sentiments and emotions.

"""
