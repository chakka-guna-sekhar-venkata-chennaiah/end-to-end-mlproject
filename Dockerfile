FROM python:3.7
WORKDIR /iris-classification
COPY . .
EXPOSE 9999
RUN python3 -m pip install -r requirements.txt
ENTRYPOINT ['streamlit','run']
CMD ['app.py'] 
