import streamlit as st
import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
from huggingface_hub import login
import os
import re

# Extract Meeting Minutes using regex
def extract_meeting_minutes(text):
    # Match starting from '**Meeting Title:**' till the end
    match = re.search(r"[*]{2}\s*Meeting Title\s*[:][*]{2}[\s\S]*$", text, re.IGNORECASE)
    if match:
        return match.group(0).strip()
    else:
        return "Meeting Title section not found."



# App Title
st.set_page_config(page_title="Meeting Minutes Generator", page_icon="📝")

st.title("📝 Meeting Minutes Generator using Whisper and LLaMA")
st.write("Upload an audio file, and this app will generate detailed meeting minutes.")

# Load Hugging Face Token
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    st.error("Hugging Face token not found. Please set HF_TOKEN in environment variables.")
    st.stop()
login(hf_token)

# Upload Audio File
audio_file = st.file_uploader("Upload your audio file (MP3/WAV) here...", type=["mp3", "wav"])

if audio_file:
    st.audio(audio_file, format='audio/mp3')
    st.write("Audio uploaded successfully. Starting transcription...")

    # Load Whisper Model
    st.spinner("Loading Whisper Medium for transcription...")
    AUDIO_MODEL = "openai/whisper-medium"
    speech_model = AutoModelForSpeechSeq2Seq.from_pretrained(AUDIO_MODEL, torch_dtype=torch.float16).to('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoProcessor.from_pretrained(AUDIO_MODEL)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=speech_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Perform Transcription
    with open("temp_audio.mp3", "wb") as f:
        f.write(audio_file.read())

    result = pipe("temp_audio.mp3", return_timestamps=True)
    transcription = result["text"]
    st.success("Transcription Complete!")
    # st.text_area("Transcription Output", transcription, height=300)

    # Generate Meeting Minutes
    st.spinner("Generating meeting minutes using LLaMA 3.1...")

    LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(LLAMA, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        LLAMA,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_auth_token=hf_token
    )

    system_message =  "You are an expert meeting notes specialist with years of experience capturing the most important elements of business discussions. Your task is to transform raw meeting transcripts into clear, structured, and actionable meeting minutes."

    user_prompt = f"""I need comprehensive meeting minutes from the following transcript. The meeting minutes should follow a professional business format with these sections:

    • Meeting Title: [Extract from context or leave blank for me to fill]
    • Date and Time: [Extract from context or leave blank]
    • Participants: [Identify and list all participants mentioned]
    • Location/Platform: [Extract from context or leave blank]

    Please structure the minutes as follows:

    1. EXECUTIVE SUMMARY (2-3 sentences capturing the main purpose and outcome)

    2. AGENDA ITEMS DISCUSSED
    • Format each agenda item as a clear heading
    • Include detailed discussion points under each item
    • Capture key decisions made
    • Highlight different perspectives mentioned

    3. ACTION ITEMS
    • List each action item in the format: [Action] - [Owner] - [Due Date]
    • Ensure every action has an owner (if mentioned)
    • Include any mentioned deadlines or timelines

    4. KEY DECISIONS
    • List all final decisions made during the meeting
    • Include any voting outcomes if mentioned

    5. OPEN ISSUES
    • Note any unresolved questions or topics deferred to future meetings

    6. NEXT STEPS
    • Include information about follow-up meetings
    • Highlight priority items to be addressed next

    Format the minutes in professional Markdown with clear headings, bullet points, and emphasis on key information. Focus on capturing the substance of the discussion rather than word-for-word transcription.

    {transcription}
    """
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]

    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    streamer = TextStreamer(tokenizer)

    outputs = model.generate(inputs, max_new_tokens=2000, streamer=streamer)
    # Decode and extract
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print('response:',response)
    meeting_minutes_only = extract_meeting_minutes(response)
    
    
    minutes_path = "Meeting_Minutes.md"
    with open(minutes_path, "w") as f:
        f.write(meeting_minutes_only)

    # Display Only Meeting Minutes
    st.subheader("Meeting Minutes:")
    st.markdown(meeting_minutes_only)

    # Provide Download Option
    with open(minutes_path, "rb") as f:
        st.download_button("Download Meeting Minutes", f, file_name="Meeting_Minutes.md", mime="text/markdown")



