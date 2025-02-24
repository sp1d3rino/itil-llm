# Import libraries for Streamlit and model loading
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# Function to load the fine-tuned model and tokenizer
def load_model_and_tokenizer(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure padding token is set
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16)
    model.to("cuda" if torch.cuda.is_available() else "cpu")  # Move to GPU if available
    return model, tokenizer

# Main Streamlit app
def main():
    st.title("Chat with Fine-Tuned Phi-2 Model")

    # Input for model directory
    model_dir = st.text_input("Enter the path to the fine-tuned model directory", "my_fine_tuned_phi2")
    
    # Load model and tokenizer when the user confirms the path
    if st.button("Load Model"):
        try:
            model, tokenizer = load_model_and_tokenizer(model_dir)
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
                max_length=200,  # Longer responses for reasoning
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
        st.warning("Please load a model first.")

if __name__ == "__main__":
    main()