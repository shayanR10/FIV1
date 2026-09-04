# FIV1

### NOTE:
This project was created in Google Colab.

### Description:
An end-to-end computer vision and data pipeline that automatically scrapes, filters, and trains a ResNet-18 model to identify dinosaur fossil species, complete with intelligent taxonomic fallback powered by the Paleobiology Database (PBDB). Built with Python 3.

### Key Features:
-  RESNET 18
-  OpenAI's CLIP powered image filtering
-  Google ecosystem integration (Kaggle, Colab)
-  Graceful taxonomic fallback (using the Paleobiology Database API)

#### Required Prerequisites:
- A Google account (for Google Colab)
- A Google Kaggle account (to store the scraped training images)
**The next three prerequisites are included inside the notebook. Nothing additional is required.**
 - OpenAI CLIP
 - NumPy
 - KaggleHub

## Quick Start Guide (for Google Colab):
1. Navigate to the repository housing the main code file and click the Open in Colab badge at the top.
2. Sign up or log into Kaggle. Click your profile picture → Settings → API Tokens → Click Generate New Token.
3. Open the downloaded kaggle.json file with any text editor to view your credentials.
4. In Kaggle's left sidebar, click Data Hub → Datasets → New Dataset.
   - Note down your username and dataset slug from the resulting URL: https://kaggle.com[YourUsername]/[YourSlug]/
5. Open the Secrets tab (the key icon) in Google Colab's left sidebar. Add three new secrets with these exact, case-sensitive names:
   - KAGGLE_USERNAME (Your Kaggle account username)
   - KAGGLE_SLUG (The slug created for your dataset)
   - KAGGLE_KEY (The long string found inside your downloaded kaggle.json file)
**IMPORTANT: Make sure you toggle Notebook Access to ON for all three!**
6. Navigate to the top menu, select Runtime → Change runtime type, and select T4 GPU to enable hardware acceleration.
7. Run each code cell sequentially from top to bottom, and the model is setup!
     
### How the Notebook Works:
 - Cell 0:
      - This is just a small blurb (has #s to signify comments) and does not have anything to do with the actual function. Disregard.
 - Cell 1:
      - Prerequisites: the first cell installs the necessary packages within the virtual environment to ensure that the project runs as expected.
 - Cell 2:
      - Kaggle Authentication: authenticates your Kaggle credentials (via API key) in order to allow the next few cells of the notebook to run properly.
        Run this cell TWICE to ensure proper function.
 - Cell 3:
      - Image scraper + uploader: Scrapes, filters (via OpenAI CLIP) and uploads images from the Wikimedia Commons API to your Kaggle dataset.
 - Cell 4:
      - This is the actual model. Run it AFTER running the previous three cells to ensure the model trains correctly.

**IMPORTANT: Google Colab environments are temporary. If your runtime disconnects or restarts, you must run the setup and authentication cells again to rebuild your environment. Make sure to download your trained model file (`.pth`) to your local computer as soon as training finishes so you don't lose your progress!**

#### Cleanup:
 - Cell 5:
      - Cleanup: If when checking your Kaggle dataset, you discover that the scraper collected images of genera that are unsatisfactory, use this to deleted the affected (granted, this             deletes the folder itself, so only use this if an entire folder contains mostly/all unsatisfactory training images).
 - Cell 6:
      - Restoration: If you realize you accidentally deleted the wrong genera folder, you can run the code in this cell to rescrape and restore it to your dataset.
         
         
**Note: the individual folders in your Kaggle dataset are ordered by genera (e.g. Allosaurus).**

#### DEBUGGING: Use this only if cell 4 does not upload any images to Kaggle:
 - Cell 7:
      - Creates a kaggle.json file on your device.
 - Cell 8:
      - Uploader for your kaggle.json file that the previous cell should have created. Makes sure everything has the correct permissions to ensure
        smooth uploading in cell 4 if it has previously failed.

**IMPORTANT: DO NOT publicly upload or share your Kaggle API keys and/or Kaggle.json (if applicable) file online.**
