# Music-generation-system-based-on-LSTM-and-genetic-algorithm
# README.md

## Program Usage:

### Step 1: Install Required Python Libraries
First, install the required Python libraries specified in the `requirement.txt` file located in the compressed package.  
This project requires **Python==3.6**. It is highly recommended to run this project within a virtual environment created using tools such as Anaconda.  
Additionally, this program uses **MuseScore 3** and **PyCharm** software.

---

### Step 2: Preprocessing
After setting up the environment, start the `preprocess.py` script located in the root directory.  
This script will read the songs from the `Kernscores` dataset in the root directory and preprocess the data.  
After preprocessing, three outputs will be generated:
1. A folder named `dataset` will be created. Each file in this folder corresponds to a song in the original dataset.
2. A file named `file_dataset` will be generated. This file combines all the preprocessed songs in the `dataset` folder into a single file.
3. A file named `mapping.json` will be generated. This file contains a mapping of all unique symbols in the `file_dataset` file.

---

### Step 3: Model Training
Run the `train.py` script in the root directory.  
This script will train the model using the preprocessed data and output a model file named `model.h5`.

---

### Step 4: Melody Generation
Run the `melodygenerator.py` script.  
This script allows you to control the melody generation process using the following hyperparameters:
- **`seed`**: Melody seed
- **`num_steps`**: Number of steps
- **`max_sequence_length`**: Maximum sequence length
- **`temperature`**: Temperature parameter

After execution, two files will be generated:
1. `mel.mid`: The generated melody as a MIDI file.
2. `mel.txt`: The generated melody sequence saved in text format.

---

### Step 5: Melody Encoding
Open the `Mel-harm-GA` folder and move the previously generated `mel.mid` file into the `original-midi` folder.  
Navigate back to the `Mel-harm-GA` folder and run the `midi-encode.py` script.  
This script converts the `mel.mid` file into a file named `melody_corrected.json`.  
The `melody_corrected.json` file contains the melody sequence of `mel.mid` stored in a "pitch-duration pair" format.

---

### Step 6: Melody Harmonization
Run the `geneticmelodyharmonizer.py` script in the `Mel-harm-GA` folder.  
During execution, choose the desired parent selection method (it is recommended to use the **"Fitness Proportionate Selection"** method).  
Once the program finishes, MuseScore will launch and display the generated sheet music containing both melody and chord accompaniment.  
Additionally, the song will be saved in the following formats:
- `mel-chord.mid`: The song as a MIDI file.
- `chord-progress.txt`: The chord progressions in text format.

---

### Additional Notes:
1. **MuseScore Path Configuration**:  
   In all scripts where the following code appears:
   ```python
   from music21 import environment
   env = environment.Environment()
   environment.Environment()['musicxmlPath'] = r'E:\Program Files\MuseScore 3\bin\MuseScore3.exe'  # Set path for music rendering
   environment.Environment()['musescoreDirectPNGPath'] = r'E:\Program Files\MuseScore 3\bin\MuseScore3.exe'  # Set path for graphic rendering
