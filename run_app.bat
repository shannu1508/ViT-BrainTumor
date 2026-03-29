@echo off
REM Run the Streamlit Brain Tumor Classifier
echo Starting Brain Tumor Classifier...
streamlit run streamlit_app.py --logger.level=error
pause
