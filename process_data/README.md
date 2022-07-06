## Process data

This folder has some tools to process UCF101, HMDB51 and Kinetics400 datasets. 

Whereas UCF101 and HMDB51 are around 6 and 2 GB in size, the Kinetics400 train and 
validation splits are around 360 and 26 GB in size, and split into 
around 240 and 17 tar files, respectively.

### 1. Download

Run, e.g., `python download_data.py --d_root main_path --dataset UCF101` to download and arrange the data.

Downloads the videos from source: 
[UCF101 source](https://www.crcv.ucf.edu/data/UCF101.php), 
[HMDB51 source](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads), 
[Kinetics400 source](https://s3.amazonaws.com/kinetics).

Arranges the datasets as follows: 

* UCF101
```
{main_path}/UCF101/videos/{class name}/{video name}.avi
{main_path}/UCF101/splits/trainlist{01/02/03}.txt
{main_path}/UCF101/splits/testlist{01/02/03}}.txt
{main_path}/UCF101/splits/classInd.txt
```

* HMDB51
```
{main_path}/HMDB51/videos/{class name}/{video name}.avi
{main_path}/HMDB51/splits/{class name}_test_split{1/2/3}.txt
```

* Kinetics400
```
{main_path}/Kinetics400/videos/train_split/{class name}/{video name}.mp4
{main_path}/Kinetics400/videos/val_split/{class name}/{video name}.mp4
```
And keeps the downloaded csv files, stored as:
```
{main_path}/Kinetics400/videos/train_split.csv
{main_path}/Kinetics400/videos/val_split.csv
```

### 2. Extract frames

Run, e.g., `python extract_frames.py --d_root main_path --dataset UCF101` to extract video frames. 

### 3. Collect all paths into csvs

Run, e.g., `python write_csvs.py --f_root main_path --dataset UCF101` to collect paths into a csv file.

The csv file is stored under `process_data`, and the `classInd.txt` file listing each class and its class label is created. 

