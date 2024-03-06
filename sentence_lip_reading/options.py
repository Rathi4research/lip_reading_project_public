# Idx of GPU to use with CUDA
gpu = '0'
# Set a non random seed to be able to reproduce the results
seed = 0

# Training method: unseen or overlapped
training_method = 'unseen'

# Name of the model
model_name = 'dense3d_transformer'

# Path to the folder that contains the lip videos (frames)
# video_path = 'D:\\Codebase\\lip_reading_project_public\\sentence_lip_reading\\videos_for_training\\'
video_path = 'D:\\Codebase\\lip_reading_project_public\\sentence_lip_reading\\videos_for_training\\'

# Paths to the txt file that contains the path of the folders for training and validation
training_list = 'D:\\Codebase\\lip_reading_project_public\\sentence_lip_reading\\video_files.txt'
validation_list = 'D:\\Codebase\\lip_reading_project_public\\sentence_lip_reading\\validation_files.txt'

# training_list = '...absolute_path.../fraction_processed_dataset_slr/video_paths_list_training.txt'
# validation_list = '...absolute_path.../fraction_processed_dataset_slr/video_paths_list_validation.txt'

# Path to the folder that contains the alignment files
# alignment_path = 'D:\\Codebase\\lip_reading_project_public\\sentence_lip_reading\\alignments_training\\'
alignment_path = 'D:\\Codebase\\lip_reading_project_public\\sentence_lip_reading\\alignments_training\\'

# Padding value for the video and the text
padding_video = 75
padding_text = 50

# Batch size (max 16 for RTX2070)
batch_size = 16

# Learning rate
learning_rate = 2e-5
decay = 0
momentum=0.9

# Stop the training
# max_epoch = 200
max_epoch = 25
# Display training process to screen
display = 1

# Test and save new model very ... steps
test_step = 2000

# Path to save the model
# save_model_path = f'...path_to_where_you_want.../model_{training_method}_{model_name}'
save_model_path = 'D:\\Codebase\\lip_reading_project_public\\sentence_lip_reading\\weights\\both_model'

# If true: Optimize the model that is located under the 'weights' path or a new model if no path is given
optimize = True

# Comment these 2 lines if you want to train a new model
model = 'transformer'
# weights = 'D:\Codebase\lip_reading_project_public\sentence_lip_reading\weights\model_unseen_dense3d_transformer_loss_0.7470988631248474_wer_0.2406666666666667_cer_0.1106539869955016.pt'
# weights = 'D:\Codebase\lip_reading_project_public\sentence_lip_reading\weights\lrw_resnet18_dctcn_video.pth'
weights = 'D:\\Codebase\\lip_reading_project_public\\sentence_lip_reading\\weights\\new_models1_loss_1.2722039222717285_wer_0.9773333333333334_cer_0.5575882184214666.pt'