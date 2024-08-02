# Image Classification using VLAD in MATLAB

This project implements an image classification pipeline using Vector of Locally Aggregated Descriptors (VLAD). 
The pipeline involves extracting Dense SIFT features, forming a dictionary, encoding features using VLAD, and training an SVM classifier with various hyperparameter optimizations. 
The goal is to achieve high accuracy in image classification by leveraging the VLAD encoding scheme.



*Visualization of the VLAD encoding process*

## Project Structure

- **VLAD.m**: Main script to run the project, including data loading, feature extraction, dictionary formation, VLAD encoding, and classification.
- **denseSIFTVasilakis.m**: Extracts Dense SIFT features from the dataset.
- **DictionaryFormationVasilakis.m**: Forms a dictionary using the extracted features.
- **VLADNV.m**: Encodes features using VLAD.
- **splitTheDatastore.m**: Splits the image datastore into training and testing sets.

## How to Run

To run this project:
1. Ensure MATLAB is installed on your system.
2. Clone this repository to your local machine.
3. Place your dataset in a directory of your choice.
4. Open MATLAB and navigate to the cloned project directory.
5. Run the `VLAD.m` script to start the image classification pipeline.

```matlab
run('VLAD.m')
```

## License
This code is for teaching/research purposes only.
