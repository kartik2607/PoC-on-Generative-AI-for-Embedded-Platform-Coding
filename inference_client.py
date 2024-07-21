# chatbot_ui.py 
import streamlit as st 
import socket 
 
def main(): 
    st.title("Chatbot UI") 
 
    # Set the host and port to match the server's configuration 
    host = '192.168.5.37'  # Use loopback IP for local testing 
    port = 12345  # Use the same port as your server 
 
    # Create a socket client to communicate with the server 
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
    client.connect((host, port)) 
 
    st.subheader("Enter your inference prompt:") 
    message = st.text_input("Prompt") 
    if st.button("Send"): 
        # Send the user's query to the server 
        client.send(message.encode('utf-8')) 
 
        # Receive and display the server's response 
        response = client.recv(1024).decode('utf-8') 
        st.subheader("Server's Response:") 
        st.write(response) 
 
    # Close the client socket when done 
    client.close() 
 
if __name__ == '__main__': 
    main()