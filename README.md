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

## Quick Start Guide (for Google Colab):
1. Go to the main file (where all of the actual code is) and click the "Open in Colab" button.
2. Create a Google Kaggle account @https://www.kaggle.com/ (this is where the training images will be stored).
3. In your Google Kaggle account:
   - Create and save an API key (you can do this by clicking your profile picture post sign up,
     going to "Your API tokens," and clicking "Generate New Token" under "API Tokens (Recommended)"
     Make sure you immediately save your key; you can only view it once.
   - Create a new dataset in Google Kaggle (find and click "Data Hub" in the side menu, click on "Datasets" and then "New Dataset")
   - Note down your account username and dataset slug; you can find it in your newly created dataset URL: (https://www.kaggle.com/datasets/YourUsernameIsHere/YourSlugIsHere)
   - Enter Google Colab and open the Secrets tab (the key icon); create three secrets named "KAGGLE_USERNAME" , "KAGGLE_SLUG" , and "KAGGLE_KEY"
     (have to be these names exactly otherwise this will not work). Paste in your own Kaggle username, dataset slug, and API key, respectively.
   - In Google Colab, connect to a T4 TPU runtime (this ensures the fastest results)
   - Finally, click each code cell in sequential order to ensure everything runs correctly, and the model is set up!
     
