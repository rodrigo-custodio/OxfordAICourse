import streamlit as st
import langchain_helper

st.title("Joke Generator")

age = st.sidebar.selectbox("Pick an Age", ("7", "8","9", "10","11","12", "13","14", "15","16") )


if age:
    response = langchain_helper.generate_joke(age)
    st.header(response['joke_type'].strip())
    joke_examples = response['joke_examples'].strip().split('---')


    st.write("Is the type of joke a child of that age likes. Here are ** Examples of Jokes** in that style")

    for item in joke_examples:
      st.write(item)


