### Installation Steps

Follow these steps to set up the environment and get the project running:

```bash
# Step 1: Clone the Repository
git clone https://github.com/jomoll/reasoning-reader-study
cd reasoning-reader-study

# [Optional:] 
conda init
conda create -n <your_name>
conda activate <your_name>

# Step 2: Install environment
pip install -r requirements.txt

# Step 3: Run app
python app.py

# Step 4: Open web-app
# click on link in terminal (something like http://127.0.0.1:7860)

# Step 5: Each time you 'submit' an annotation in the app, this will create an entry in the 'logs/reader_study_results.jsonl' file.
# Each time you restart the app by running 'python app.py', it will automatically load your previous progress.
# When you finish the last sample, the app will show errors, this is expected.
# Please keep the logs file and send it to me as soon as you finished all samples, it will contain all your answers.
# Have fun and thank you!! :)