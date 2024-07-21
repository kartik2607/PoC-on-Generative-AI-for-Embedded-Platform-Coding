import socket 
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline 
 
model_name = "/home/dockuser1/users/Interns/kartik-final/7b1" 
 
model = AutoModelForCausalLM.from_pretrained( 
    model_name, 
    device_map="auto" 
) 
 
tokenizer = AutoTokenizer.from_pretrained(model_name) 
model.use_cache = True 
model.eval() 
 
def format_instruction(sample): 
    return f"""### Instruction: 
You are a coding assistant that will write a Solution to resolve the following 
Task: 
  
### Task: 
{sample} 
  
### Solution: 
""".strip() 
 
def chatbot_response(query): 
 
    final = format_instruction(query) 
    pipe = pipeline( 
        task="text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        torch_dtype=torch.float16, 
        repetition_penalty=1.1 
    ) 
    result = pipe( 
        f"{final}", 
        do_sample=True, 
        top_p=0.5, 
        temperature=0.5, 
        top_k=10, 
        num_return_sequences=1, 
        eos_token_id=tokenizer.eos_token_id, 
        max_length=2048 
    ) 
    return result[0]['generated_text'] 
 
def main(): 
    host = '0.0.0.0'  # Listen on all available network interfaces 
    port = 12345  # Choose a port for your server 
 
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    server.bind((host, port)) 
    server.listen(1)  # Listen for a single client connection 
 
    print(f"Server is listening on {host}:{port}") 
 
    while True: 
        client_socket, client_address = server.accept() 
        print(f"Accepted connection from {client_address}") 
 
        while True: 
            data = client_socket.recv(1024).decode('utf-8') 
            if not data: 
                break  # Connection closed by the client 
 
            response = chatbot_response(data) 
 
            client_socket.send(response.encode('utf-8')) 
 
        client_socket.close() 
 
if __name__ == '__main__': 
    main() 