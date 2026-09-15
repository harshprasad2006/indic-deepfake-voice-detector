# SatyaDhwani: AI-Cloned & Deepfake Voice Detector for Indian Languages

## 📂 Project Structure
- pp.py: Full interactive Streamlit presentation application with custom UI styling.
- 	rain_and_evaluate.py: End-to-end dataset loading, speaker-disjoint splitting, CNN & XLS-R training, and metric evaluation.
- equirements.txt: Clean, version-pinned dependencies.

## 🚀 Running the Web App Locally
1. Install dependencies:
   `ash
   pip install -r requirements.txt
   `
2. Start the application:
   `ash
   streamlit run app.py
   `
3. Open http://localhost:8501 in your browser.

## ☁️ Running on Google Colab
1. Upload the folder or notebook to Colab.
2. Select **Runtime > Change runtime type > T4 GPU**.
3. Run python train_and_evaluate.py to train models and generate confusion_matrix.png, oc_curve.png, and aseline_cnn.pth.
4. Run Streamlit directly on Colab using:
   `ash
   !streamlit run app.py & npx localtunnel --port 8501
   `
