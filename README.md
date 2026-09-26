# FIV1

### NOTE:
This project was created in Google Colab.

### Description:
A fully automated, cloud-native computer vision and data engineering pipeline designed to handle the complex real-world challenges of paleontological image classification. Built specifically to run seamlessly within the Google Colab and Kaggle ecosystems, the notebook automates the entire machine learning lifecycle, from targeted data harvesting and semantic filtering to multi-level deep learning and intelligent inference. 

The core architecture utilizes a hierarchical taxonomic safety net. By linking runtime model layers with live paleontological metadata from the Paleobiology Database API, the system guarantees a biologically accurate fallback prediction (climbing from Species to Genus or Family) whenever fine-grained species identification falls below confidence thresholds.

### Key Features:
-  ConvNeXt-tiny
-  OpenAI's CLIP powered image filtering
-  Google ecosystem integration (Kaggle, Colab)
-  Graceful taxonomic fallback (using the Paleobiology Database API)

#### Required Prerequisites:
- A Google account (for Google Colab)
- A Google Kaggle account (to store the scraped training images)
  
**The following prerequisites are included inside the notebook. Nothing additional is required.**
 - ConvNeXt (Included with Colab)
 - Seaborn (Included with Colab)
 - SciKit learn (Included with Colab)
 - Matplotlib (Included with Colab)
 - OpenAI CLIP
 - NumPy
 - KaggleHub
 - Safetensors
 - Cowsay (for QoL)

**If you are working locally inside an application such as VS Code, make sure to also install these libraries first.**

## Quick Start Guide (for Google Colab):
1. Navigate to the repository housing the main code file and click the Open in Colab badge at the top.
2. Sign up or log into Kaggle. Click your profile picture → Settings → API Tokens → Click Generate New Token.
3. Save the generated API key (the long string of numbers and letters).
4. In Kaggle's left sidebar, click Data Hub → Datasets → New Dataset.
   - Note down your username and dataset slug from the resulting URL: https://kaggle.com[YourUsernameIsHere]/[YourSlugIsHere]/
5. Open the Secrets tab (the key icon) in Google Colab's left sidebar. Add three new secrets with these exact, case-sensitive names:
   - KAGGLE_USERNAME (Your Kaggle account username)
   - KAGGLE_SLUG (The slug created for your dataset)
   - KAGGLE_KEY (Your Kaggle API key)
     
**IMPORTANT: Make sure you toggle Notebook Access to ON for all three!**

6. Navigate to the top menu, select Runtime → Change runtime type, and select T4 GPU to enable hardware acceleration.
7. Run Cells 1 to 4 until completion, and the model is setup!
   
   **Note: I also recommend running Cell 9 immediately post training (for diagnostic data).**
     
### How the Notebook Works:
 - Cell 1:
      - Prerequisites: the first cell installs the necessary packages within the virtual environment to ensure that the project runs as expected.
 - Cell 2:
      - Kaggle Authentication: authenticates your Kaggle credentials (via API key) in order to allow the next few cells of the notebook to run properly.
        **Run this cell TWICE to ensure proper function.**
 - Cell 3:
      - Image scraper + uploader: Scrapes, filters (via OpenAI CLIP) and uploads images from the Wikimedia Commons API to your Kaggle dataset.
 - Cell 4:
      - This is the actual model. Run it AFTER running the previous three cells to ensure the model trains correctly.

**WARNING: Google Colab environments are temporary. If your runtime disconnects or restarts, you must run the setup and authentication cells again to rebuild your environment. Make sure to download your trained model file (`.safetensors`) to your local computer as soon as training finishes so you don't lose your progress!**

#### Cleanup & Restoration:
 - Cell 5:
      - Cleanup: If when checking your Kaggle dataset, you discover that the scraper collected images of genera that are unsatisfactory, use this to deleted the affected (granted, this             deletes the folder itself, so only use this if an entire folder contains mostly/all unsatisfactory training images).
 - Cell 6:
      - Restoration: If you realize you accidentally deleted the wrong genus folder, you can run the code in this cell to rescrape and restore it to your dataset.
         
         
**Note: the individual folders in your Kaggle dataset are ordered by genera (e.g. Allosaurus).**

**WARNING: Do not run either of these cells without filling in all the marked input boxes (marked by [""]). Doing so may result in unintended effects, such as corruption or data loss, affecting your Kaggle database.**

#### Debugging/Troubleshooting: 
**IMPORTANT: Use the following only if cell 4 cannot upload images to your dataset:**
 - Cell 7:
      - Creates a kaggle.json file on your device. This file contains your Kaggle username along with your API key.
 - Cell 8:
      - Uploader for your kaggle.json file that the previous cell should have created. Makes sure everything has the correct permissions to ensure
        smooth uploading in cell 4 if it has previously failed.

#### Diagnostics:
  - Cell 9:
      - Confusion matrix for the model. This cell allows you to visualize which species the model is confusing most. I recommend running this cell immediately post training.
       

**WARNING: DO NOT publicly upload or share your Kaggle API keys and/or Kaggle.json (if applicable) file online.**
**If you publish a version of this code to your own GitHub repository, please make sure to clear your output boxes before doing so. This prevents accidental sharing of aforementioned API keys.**
