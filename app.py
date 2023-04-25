import os

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 
from langchain.document_loaders import PyPDFLoader



# os.environ['OPENAI_API_KEY']

# APP
st.title('🦜🔗 Langchain + OpenAPI')
prompt = st.text_input('Enter a topic here:')


# Template
title_template = PromptTemplate(
    input_variables = ['topic'],
    template = 'Write me an essay title about {topic}'
)

essay_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'],
    template = 'Write me an essay on this title TITLE: {title}  while leveraging this wikipedia reserch:{wikipedia_research} '
)

# Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
essay_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# LLMS
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
essay_chain = LLMChain(llm=llm, prompt=essay_template, verbose=True, output_key='essay', memory=essay_memory)

# Wiki
wiki = WikipediaAPIWrapper()

# Show result if prompt is not empty
if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = essay_chain.run(title=title, wikipedia_research=wiki_research)
    st.write(title)
    st.write(script)
    
    with st.expander('Title History'):
        st.info(title_memory.buffer)
    
    with st.expander('Essay History'):
        st.info(essay_memory.buffer)
        
    with st.expander('Wikipedia Research'): 
        st.info(wiki_research)