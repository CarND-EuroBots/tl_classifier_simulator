# Traffic light detection in Udacity's self driving car simulator
## API used
The following Tensorflow API has been used to train the classifier:
https://github.com/tensorflow/models/tree/master/object_detection
## Image samples used for the training:
Image samples used for the training can be found under the following link in the  sim_data-capture folder:
https://drive.google.com/drive/folders/0Bx-GmE9uBTsCNlVFeFVVWldXN1U?usp=sharing
Image classifications are in the same folder, sim_data_large.yaml file.
## Classifier used for the training
Mobilened+SSD classifier has been used for the model training.
Object detection API configuration can be found in this repo, file ssd_mobilenet_v1_sim.config .
SSD mobile net coco frozen graph has been used as initial graph for transfer learning.
The graph can be found under previously mentioned Google Drive link in ssd_mobilenet_v1_coco_11_06_2017 folder.
## Image samples preparation
To prepare image samples it was necessary to export them to sim_train.record and sim_val.record files. 
Python script used to perform this operation can be found in this repo, file: create_sim_data_tf_record.py .
Resulting sim_train.record and sim_val.record files can be found under previously mentioned Google Drive link in data folder.
## Model training
Model has been trained with the following command (path needs to be modified):
'''
python3 object_detection/train.py     --job-dir=/home/ubuntu/anaconda3/lib/python3.6/site-packages/tensorflow/models/train     --module-name object_detection.train     --region us-central1   --train_dir=/home/ubuntu/anaconda3/lib/python3.6/site-packages/tensorflow/models/train     --pipeline_config_path=/home/ubuntu/anaconda3/lib/python3.6/site-packages/tensorflow/models/data/ssd_mobilenet_v1_sim.config
'''
Training checkpoints can be found under the previously mentioned Google Drive link in the train folder.
## Frozen graph export
Frozen graph has been exported with the following command (path needs to be modified):
'''
python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path /home/ubuntu/anaconda3/lib/python3.6/site-packages/tensorflow/models/data/ssd_mobilenet_v1_sim.config \
    --trained_checkpoint_prefix /home/ubuntu/anaconda3/lib/python3.6/site-packages/tensorflow/models/train/model.ckpt-250000 \
    --output_directory=output_inference_graph
'''
Frozen graph can be found under the previously mentioned Google Drive link in the output_inference_graph folder.
## Model inference
Jupyter Notebook example of the trained model inference can be found in this repo, Jupyter Notebook file tl_detectionl_sim.ipynb .
