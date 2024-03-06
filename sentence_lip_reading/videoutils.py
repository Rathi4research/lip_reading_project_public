import cv2
import os
import os
import shutil

def video_to_frames(video_path, output_folder):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Read and save each frame as JPEG image
    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        # Construct output file path
        output_path = os.path.join(output_folder, f"frame_{i:04d}.jpg")
        # Save frame as JPEG image
        cv2.imwrite(output_path, frame)

    # Release the video capture object
    cap.release()

def copy_mpg_files(input_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over each file in the input directory
    for file in os.listdir(input_dir):
        if file.endswith(".mpg"):
            file_name = os.path.splitext(file)[0]
            file_output_dir = os.path.join(output_dir, file_name)
            if not os.path.exists(file_output_dir):
                os.makedirs(file_output_dir)
                shutil.copy(os.path.join(input_dir, file), os.path.join(file_output_dir, file))
            # else:
            #     continue

            # shutil.copy(os.path.join(input_dir, file), os.path.join(file_output_dir, file))

            # Call function to perform some logic on the copied file
            video_to_frames(os.path.join(file_output_dir, file),file_output_dir)

def write_file_paths(output_dir):
    # Write full paths of copied .mpg files to a text file
    with open("copied_files2.txt", "w") as ofile:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith(".mpg"):
                    # file_path = os.path.join(root, file)
                    file_path = root
                    ofile.write(file_path + "\n")


# Input and output directories
input_directory_path = "D:\\Codebase\\GRID dataset\\s2"
output_directory_path = "D:\\Codebase\\lip_reading_project_public\\sentence_lip_reading\\videos_for_training\\s2"

copy_mpg_files(input_directory_path, output_directory_path)

write_file_paths(output_directory_path)

# # Example usage
# video_path = "D:\\Codebase\\lip_reading_project_public\\sentence_lip_reading\\videos_for_training\s1\\bbas2p\\bbas2p.mpg"  # Change to the path of your video file
# output_folder = "D:\\Codebase\\lip_reading_project_public\\sentence_lip_reading\\videos_for_training\s1\\bbas2p\\"  # Change to the desired output folder name
# video_to_frames(video_path, output_folder)

