import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
from IPython.display import display, Markdown
from huggingface_hub import login
import os

# Constants
AUDIO_MODEL = "openai/whisper-medium"
LLAMA = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# Load Hugging Face Token
hf_token = "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"
if not hf_token:
    raise ValueError("Hugging Face token not found. Set HF_TOKEN in environment variables.")
login(hf_token)

# Load Whisper Medium for Audio Transcription
print("Loading Whisper Medium for transcription...")
speech_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    AUDIO_MODEL, torch_dtype=torch.float16, low_cpu_mem_usage=True, use_safetensors=True
).to('cuda' if torch.cuda.is_available() else 'cpu')
processor = AutoProcessor.from_pretrained(AUDIO_MODEL)

# Perform Speech-to-Text using Hugging Face Pipeline
audio_filename = input("Enter the path to your audio file (e.g., input_audio.mp3): ")
print("Performing transcription...")
pipe = pipeline(
    "automatic-speech-recognition",
    model=speech_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch.float16,
    device='cuda' if torch.cuda.is_available() else 'cpu',
)
result = pipe(audio_filename, return_timestamps=True)
transcription = result["text"]
print("Transcription Complete:")
print(transcription)

# Prepare System Message for Meeting Minutes Generation (Generic)
system_message = "You are an AI assistant that generates meeting minutes from transcripts. Provide a clear summary, key discussion points, important takeaways, and action items with assigned owners in markdown format."
user_prompt = f"""Below is a meeting transcript. 
Please generate detailed meeting minutes in markdown format. 
Ensure the minutes include a summary (mentioning attendees, location, and date if available), 
key discussion points, takeaways, and action items with assigned owners.

{transcription}
"""

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": user_prompt}
]

# Load LLaMA 3.1 for Meeting Minutes Generation
# print("Loading LLaMA 3.1 model...")
# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_quant_type="nf4"
# )

# tokenizer = AutoTokenizer.from_pretrained(LLAMA, token=hf_token)
# model = AutoModelForCausalLM.from_pretrained(
#     LLAMA,
#     device_map="auto",
#     quantization_config=quant_config,
#     token=hf_token
# )


model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# token = "your_hf_token"  # Ensure you have access

# Load tokenizer and model without bitsandbytes
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,  # Use BF16 if supported
    device_map="auto",
    use_auth_token=hf_token
)

print("Model loaded successfully without bitsandbytes!")

tokenizer.pad_token = tokenizer.eos_token

# Tokenize Messages using Chat Template
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

# Streamer for Generating Output
streamer = TextStreamer(tokenizer)

# Generate Meeting Minutes
print("Generating meeting minutes...")
outputs = model.generate(inputs, max_new_tokens=2000, streamer=streamer)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Display Results
display(Markdown(response))
