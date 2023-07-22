from secret_key import openapi_key
import os
os.environ['OPENAI_API_KEY'] = openapi_key
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

llm = OpenAI(temperature=0.9)

def generate_joke(age):
    prompt_template_name = PromptTemplate(
        input_variables = ['age'],
        template = "What types of joke would a {age} year old boy like? Please provide the best answer. Only state the joke type in one or two words."
    )
    
    type_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key='joke_type')
    
    prompt_template_items = PromptTemplate(
        input_variables = ['joke_type'],
        template = 'Provide 2 examples of suitable {joke_type} jokes. Separate each example with "---" '
        )
    example_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key='joke_examples')

    chain = SequentialChain(
        chains = [type_chain, example_chain],
        input_variables = ['age'],
        output_variables = ['joke_type', 'joke_examples']
        )

    response = chain({'age' : age})
   
    return response

if __name__== "__main__":
    print(generate_joke("10"))