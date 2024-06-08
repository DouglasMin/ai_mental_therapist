import json
import boto3
import streamlit as st

bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")


def get_mail(sender_rank, recipient_rank, agenda):
    prompt = f"""
    You are an AI Email writer in a very hierarchical company in South Korea.
    #context
    You are tasked with generating formal and respectful email content that adheres to the strict hierarchical structure within the company. The tone, language, and level of formality must be adjusted based on the relative ranks of the sender and recipient.
    #rules
    The most important rule is to follow the **Business Hierarchy**:
    ***임원**
    **Director**
    ***수석 (부장, 차장 고위급)**
    **Senior Manager**
    ***책임 (과장, 차장 중간급)**
    **Manager**
    ***선임 (주임, 대리)**
    **Senior Assistant**
    ***사원**
    **Assistant, Junior Assistant**
    #instructions
    Generate an email following this format:
    To {recipient_rank}
    Request for {agenda}, or question about {agenda}
    #Expected Input
    The user will provide the following 3 inputs:
    1. Sender's rank within the company: {sender_rank}
    2. Recipient's rank within the company: {recipient_rank}
    3. Agenda (topic/purpose of the email): {agenda}
    #Output Format
    To, {recipient_rank},
    <paragraph 1>
    This is the opening paragraph. Make formal salutations, and state the purpose of the email clearly while maintaining appropriate business etiquette based on the sender's and recipient's ranks.
    <paragraph 2>
    This is the content paragraph(s). Expand on the email's purpose, provide background information, details, etc. The tone and level of formality should be adjusted based on the hierarchical relationship between the sender and recipient.
    <paragraph 3>
    This is the closing paragraph. Summarize the email and add appropriate closing statements based on the recipient's hierarchy level and the purpose of the email.
    Sincerely/Respectfully,
    {sender_rank}
    """
    
    model_id = "anthropic.claude-3-haiku-20240307-v1:0"


    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "temperature": 0.5,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    }
    
    # Convert the native request to JSON.
    request = json.dumps(native_request)
    
    # Invoke the model with the request.
    response = bedrock_runtime.invoke_model(modelId=model_id, body=request)
    
    # Decode the response body.
    model_response = json.loads(response["body"].read())
    
    # Extract and print the response text.
    response_text = model_response["content"][0]["text"]
    print(response_text)
    
    return response_text

def ai_mail():
    st.sidebar.title("Email Generator")
    
    # Get input from the user
    sender_rank = st.sidebar.selectbox("From (Sender's Rank)", ["사원", "선임 (주임, 대리)", "책임 (과장, 차장 중간급)", "수석 (부장, 차장 고위급)", "임원"])
    recipient_rank = st.sidebar.selectbox("To (Recipient's Rank)", ["사원", "선임 (주임, 대리)", "책임 (과장, 차장 중간급)", "수석 (부장, 차장 고위급)", "임원"])
    agenda = st.sidebar.text_input("Agenda", "Enter the agenda for the email")
    
    # Main content area
    st.title("AI Email Generator")
    
    # Display the selected inputs
    st.write(f"Sender's Rank: {sender_rank}")
    st.write(f"Recipient's Rank: {recipient_rank}")
    st.write(f"Agenda: {agenda}")
    
    # Show buttons only if all inputs are provided
    if sender_rank and recipient_rank and agenda:
        if st.button("Generate Email"):
            generated_email = get_mail(sender_rank, recipient_rank, agenda)
            if generated_email:
                st.text_area("Generated Email", generated_email, height=300)
            else:
                st.warning("Failed to generate email. Please try again.")
        
        if st.button("Regenerate Email"):
            generated_email = get_mail(sender_rank, recipient_rank, agenda)
            if generated_email:
                st.text_area("Generated Email", generated_email, height=300)
            else:
                st.warning("Failed to generate email. Please try again.")

