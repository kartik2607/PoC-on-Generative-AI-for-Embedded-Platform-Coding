
# PoC On Generative AI for Embedded Platform Code

AI chat model that can convert natural language to code i.e., understand natural language as input and generate the required embedded platform code as the output.

#  How To Use

Using Hugging Face Models

This project leverages Hugging Face models for natural language processing tasks. To integrate these models into your code, follow these steps:

Install the library:

   ```
   pip install accelerate peft transformers
   pip install trl
   pip install sentencepiece
   pip install -U einops
   pip install datasets
   ```
## Hugging Face Login 

To interact with the Hugging Face API, you'll need to obtain an API key. Follow these steps:

1. Visit the [Hugging Face website](https://huggingface.co/) and sign in or create an account.

2. Once logged in, navigate to your account settings or developer settings.

3. Generate an API key.

4. Copy the generated API key.

To use the Hugging Face API in your project:

Store the API key in a secure location, such as a configuration file or environment variable.

Use the Hugging Face CLI to login and verify your authentication status.

```bash
//Run this command in Terminal
huggingface-cli login
```
## Fine-Tuning the Model
   1. Open Terminal

   2. Open the python script `train.py`:
   
      Load the base model from  Hugging face:
      
      ```bash
      //The Base model will be downloaded from hugging face and this "meta-llama/Llama-2-7b-hf" can be found in the hugging face repository of the model
      model_name = "meta-llama/Llama-2-7b-hf"
      ```
   
      Give the path to save the Fine-Tuned model and name
      
      ```bash
      new_model = "./7b1"
      ```
   
      Load the dataset for Training
      
      ```bash
      //The dataset should be in the same directory as the 'train.py' scrpit, if not give the full path
      //The dataset should in JSONL format and it should follow the dataformat of the model, the format can be found in 'dataformat.py' scrpit.
      dataset = load_dataset("json", data_files="final2i.jsonl")
      ```
   3. Run the python script `train.py`

      ```bash
      python3 train.py
      ```

## Inference with the Fine-Tuned Model

   To perform inference using the fine-tuned model, use the following steps:

   1. Open Terminal
   2. Open the python script `inference_pipeline.py`:
      
      Load the fine-tuned model and tokenizer in your inference script:

      ```bash
      model_name = "./7b1"
      ```
   3. Run the python script `inference_pipeline.py`

      ```bash
      python3 inference_pipeline.py
      ```
    
   4. Enter the prompt, and wait for the model to generate output.

## Inference with the Fine-Tuned Model using Client Server UI

   To perform inference using the fine-tuned model with UI, use the following steps:

   1. Open Terminal in DGX server
   2. Open the python script `inference_server.py`:
      
      Load the fine-tuned model and tokenizer in your inference script:

      ```bash
      model_name = "./7b1"
      ```

   3. Run the python script in the DGX server `inference_server.py`

      ```bash
      python3 inference_server.py
      ```

   4. Open Terminal in the Client Side
   5. Run the python script in the client side `inference_client.py
   
      ```bash
      streamlit run --server.fileWatcherType none inference_client.py
      ```

   6. Enter the prompt, and wait for the model to generate output from the server.
