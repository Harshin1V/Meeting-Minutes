# Meeting Minutes Generator using Whisper and LLaMA

This project is an AI-powered application that generates comprehensive meeting minutes from audio recordings using OpenAI's Whisper for transcription and Meta's LLaMA for text generation. It provides a structured summary including executive summaries, action items, decisions made, and next steps.

## Features
- **Audio Transcription**: Convert audio files (MP3/WAV) into text using Whisper.
- **AI-Generated Meeting Minutes**: Summarizes the transcript using LLaMA 3.1.
- **Interactive UI**: Built with Streamlit for easy user interaction.
- **Downloadable Results**: Download meeting minutes in markdown format.

## Prerequisites
Make sure you have the following installed:
- Python 3.9+
- CUDA (if using GPU)
- Hugging Face account with access to Whisper and LLaMA models

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/meeting-minutes-generator.git
cd meeting-minutes-generator

# Install dependencies
pip install -r requirements.txt
```

## Environment Setup
Create a `.env` file to store your Hugging Face token:

```
HF_TOKEN=your_huggingface_token
```

Alternatively, you can set it in your terminal:

```bash
export HF_TOKEN=your_huggingface_token
```

## How to Run

1. Launch the application using Streamlit:

```bash
streamlit run app.py
```

2. Upload an audio file (MP3 or WAV).
3. The app will perform transcription using Whisper and generate structured meeting minutes using LLaMA.
4. View or download the generated minutes in markdown format.

## Project Structure
```bash
.
├── app.py                   # Main Streamlit app
├── requirements.txt          # Project dependencies
├── Meeting_Minutes.md        # Sample output file
├── .env                      # Environment variables (optional)
└── README.md                 # This file
```

## Troubleshooting
- **Model Download Errors**: Ensure your Hugging Face token has access to the required models.
- **CUDA Not Available**: Verify GPU installation using `torch.cuda.is_available()`.
- **Audio File Not Uploading**: Ensure the file is in MP3 or WAV format.

## License
This project is licensed under the MIT License. Feel free to use and modify it!

## Contact
- **LinkedIn**: [Harshini Vutukuri](https://linkedin.com/in/harshini-vutukuri)
- **Email**: [harshinivutukuri@example.com](mailto:harshinivutukuri@example.com)

Happy Coding! 🚀

