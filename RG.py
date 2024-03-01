# Import necessary libraries
import streamlit as st
from transformers import FlaxAutoModelForSeq2SeqLM, AutoTokenizer

# Define constants
MODEL_NAME_OR_PATH = "/content/path/to/save/model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, use_fast=True)
model = FlaxAutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_OR_PATH)
prefix = "items: "

# Additional constants
generation_kwargs = {
    "max_length": 512,
    "min_length": 64,
    "no_repeat_ngram_size": 3,
    "do_sample": True,
    "top_k": 60,
    "top_p": 0.95
}

special_tokens = tokenizer.all_special_tokens
tokens_map = {
    "<sep>": "--",
    "<section>": "\n"
}

# Function to skip special tokens
def skip_special_tokens(text, special_tokens):
    for token in special_tokens:
        text = text.replace(token, "")
    return text

# Function for target post-processing
def target_postprocessing(texts, special_tokens):
    if not isinstance(texts, list):
        texts = [texts]

    new_texts = []
    for text in texts:
        text = skip_special_tokens(text, special_tokens)

        for k, v in tokens_map.items():
            text = text.replace(k, v)

        new_texts.append(text)

    return new_texts

# Function to process generation
def generation_function(texts):
    _inputs = texts if isinstance(texts, list) else [texts]
    inputs = [prefix + inp for inp in _inputs]
    inputs = tokenizer(
        inputs,
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="jax"
    )

    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **generation_kwargs
    )
    generated = output_ids.sequences
    generated_recipe = target_postprocessing(
        tokenizer.batch_decode(generated, skip_special_tokens=False),
        special_tokens
    )
    return generated_recipe

# Streamlit app
def main():
# Set background color and frame color
    st.markdown(
        """
        <style>
            .app-container {
                background-color: #5DADE2;  /* Coral Pink */
                padding: 20px;
                margin: 0;
            }
            .stApp {
                background-color: #85C1E9;  /* Blue color for the frame */
            }
            .title {
                color: #5DADE2 !important;  /* Orange color for the title */
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("🤖AI Recipe📜🍽️🧉 Generator🤖")

    # Get user input
    ingredients_input = st.text_area("Enter ingredients (comma-separated):")

    if st.button("Generate Recipe"):
        if ingredients_input:
            st.info("Generating recipe... Please wait.")
            items = [ingredients_input]
            generated = generation_function(items)

            # Display generated recipe
            for idx, text in enumerate(generated, start=1):
                st.subheader(f"Recipe #{idx}")
                sections = text.split("\n")
                for section in sections:
                    st.write(section)

                # Add a button to copy the generated recipe to clipboard
                st.button(f"Copy Recipe #{idx} to Clipboard", key=f"copy_button_{idx}", on_click=lambda x=text: st.write(copy_to_clipboard(x)))
                st.write("---")

def copy_to_clipboard(text):
    st.write("Recipe copied to clipboard!")
    st.text(text)
    return text

if __name__ == "__main__":
    main()
