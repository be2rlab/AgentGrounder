from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

def create_prompt(config, use_image=False, image_path=None, input_variables=None):
    """Create Ollama API messages."""
    
    if input_variables is None:
        input_variables = []
    
    if use_image and image_path:
        img_content = [{"type": "image_url", "image_url": {"url": image_path}}]
    else:
        img_content = []
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", config.system_prompt),
            HumanMessagePromptTemplate.from_template(
                [
                    {"type": "text", "text": config.user_prompt},
                ] + img_content, 
                input_variables=input_variables)
        ],
    )

    return prompt