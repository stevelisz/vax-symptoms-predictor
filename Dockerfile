FROM python:3.9
EXPOSE 8501
WORKDIR ./
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
CMD streamlit run streamlit_service.py