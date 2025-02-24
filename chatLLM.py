# Import libraries for Streamlit and model loading
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel  # Import PEFT for LoRA
import torch
from huggingface_hub import login

# Function to load the fine-tuned model with LoRA
def load_model_and_tokenizer(model_dir, hf_token, base_model_name='microsoft/phi-2'):
    # Authenticate with Hugging Face
    login(token=hf_token)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set

    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, token=hf_token, torch_dtype=torch.float16)

    # Load the LoRA adapters and combine with the base model
    model = PeftModel.from_pretrained(base_model, model_dir, token=hf_token)

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on: {device}")  # For debugging in terminal

    return model, tokenizer

# Main Streamlit app
def main():
    st.title("Chat with Fine-Tuned Phi-2 Model (LoRA)")

    # Input for Hugging Face token
    hf_token = st.text_input("Enter your Hugging Face token", type="password")
    
    # Input for model directory
    model_dir = st.text_input("Enter the path to the fine-tuned LoRA model directory", "my_fine_tuned_phi2_lora")
    
    # Load model and tokenizer when the user confirms the path
    if st.button("Load Model"):
        if not hf_token:
            st.error("Please provide a Hugging Face token")
        else:
            try:
                model, tokenizer = load_model_and_tokenizer(model_dir, hf_token)
                st.session_state['model'] = model
                st.session_state['tokenizer'] = tokenizer
                st.success("Model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading model: {e}")

    # Chat interface
    if 'model' in st.session_state and 'tokenizer' in st.session_state:
        st.subheader("Chat with the Model")
        user_input = st.text_input("Ask something (e.g., 'Summarize the content' or 'What is this about?'):")

        if user_input:
            # Structure the input as a prompt for summarization/reasoning
            prompt = f"Provide a concise summary or answer based on the content:\nQuestion: {user_input}\nAnswer: "
            generator = pipeline(
                "text-generation",
                model=st.session_state['model'],
                tokenizer=st.session_state['tokenizer'],
                device=0 if torch.cuda.is_available() else -1  # 0 for GPU, -1 for CPU
            )
            response = generator(
                prompt,
                max_length=200,
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )[0]['generated_text']
            
            # Extract the answer part
            answer = response.split("Answer:")[1].strip() if "Answer:" in response else response
            st.write(f"Model response: {answer}")
    else:
        st.warning("Please load a model and provide a token first.")

if __name__ == "__main__":
    main()