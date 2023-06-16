import streamlit as st
from source.pipelines.testing_pipeline import custom_data,prediction_pipeline 
st.title("Iris Species Detection using ML")
st.write("""
    
        This web application predicts the species type to which it belongs
    """)

st.image("main.jpeg", use_column_width=True)

st.write("""
    To use this app, please fill out the following form and click on the 'Predict' button. You will then see the result.
    """)

st.write("""
    ### Form
    """)

    # Add your loan application form components here

st.write("""
    ---\n
    Made with ❤️ by Chakka Guna Sekhar Venkata Chennaiah.
    """)

st.subheader('Enter Form')
sepal_length = st.number_input('Enter the Sepal Length')
sepal_width = st.number_input('Enter the Sepal Width')
petal_length = st.number_input('Enter the Petal Length')
petal_width= st.number_input('Enter the Petal Width')

data=custom_data(sepal_length,sepal_width,petal_length,petal_width)
pred_df=data.get_data_as_a_dataframe()

pred_pipe=prediction_pipeline()
final_pred=pred_pipe.predict(pred_df)

if st.button('Predict'):
    if final_pred[0]==0:
        st.write('According to your information, the predicted specie is as follows:-')
        st.image('iris-setosa.png',use_column_width=True)
    elif final_pred[0]==1:
        
        st.write('According to your information, the predicted species is Iris Versicolor')
        st.image('iris-versicolor.jpeg',use_column_width=True)
    elif final_pred[0]==2:
        st.write('According to your information, the predicted species is Iris Virginica')
        st.image('iris-virginica.png',use_column_width=True)
