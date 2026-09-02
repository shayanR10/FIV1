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
1. Navigate to the repository housing the main code file and click the Open in Colab badge at the top.
2. Sign up or log into Kaggle. Click your profile picture → Settings → API Tokens → Click Generate New Token.
3. Open the downloaded kaggle.json file with any text editor to view your credentials.
4. In Kaggle's left sidebar, click Data Hub → Datasets → New Dataset.
   - Note down your username and dataset slug from the resulting URL: https://kaggle.com[YourUsername]/[YourSlug]/
5. Open the Secrets tab (the key icon) in Google Colab's left sidebar. Add three new secrets with these exact, case-sensitive names:
   - KAGGLE_USERNAME (Your Kaggle account username)
   - KAGGLE_SLUG (The slug created for your dataset)
   - KAGGLE_KEY (The long string found inside your downloaded kaggle.json file)
   IMPORTANT: Make sure you toggle Notebook Access to ON for all three!
6. Navigate to the top menu, select Runtime → Change runtime type, and select T4 GPU to enable hardware acceleration.
7. Run each code cell sequentially from top to bottom, and the model is setup!
     
