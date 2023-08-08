from os import makedirs
import numpy as np
import torch
import cv2
import time
from PIL import Image
from detect import detect_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

makedirs('outputs', exist_ok=True)
video_path = 'downloaded_videos/Рулим по ночному Лос Анджелесу Отличное качество.mp4'
save_name = 'testsave1'

# Load model checkpoint
checkpoint = 'checkpoint_ssd300.pth.tar'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

if __name__ == '__main__':
    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened() == False):
        print('Error while trying to read video. Please check path again')
    # get the frame width, height and framerate
    framerate = int(round(cap.get(cv2.CAP_PROP_FPS)))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    save_name = save_name
    # define codec and create VideoWriter object
    out = cv2.VideoWriter(f"outputs/{save_name}.mp4",
                          cv2.VideoWriter_fourcc(*'mp4v'), framerate,
                          (frame_width, frame_height))
    frame_count = 0  # to count total frames
    total_fps = 0  # to get the final frames per second
    # read until end of video
    while (cap.isOpened()):
        # capture each frame of the video
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image_in = Image.fromarray(frame).convert('RGB')
            # pil_transformed_image = transform(pil_image_in).unsqueeze(0).to(device)
            #
            start_time = time.time()
            # with torch.no_grad():
            #     predictions = model(pil_transformed_image)[0]
            #     print(predictions)
            #     break

            pil_image_out = detect_image(pil_image_in, min_score=0.2, max_overlap=0.2, top_k=200)
            end_time = time.time()

            # convert to NumPy array format
            result_np = np.array(pil_image_out, dtype=np.uint8)
            # convert from RGB to BGR format for OpenCV visualizations
            result_np = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
            # get the FPS
            fps = 1 / (end_time - start_time)
            # add FPS to total FPS
            total_fps += fps
            # increment the frame count
            frame_count += 1
            # press `q` to exit
            wait_time = max(1, int(fps / 4))
            # write the FPS on current frame
            cv2.putText(
                result_np, f"{fps:.3f} FPS", (5, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),
                2
            )
            cv2.imshow('Result', result_np)
            out.write(result_np)
            if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                break
        else:
            break
    # release VideoCapture()
    cap.release()
    # close all frames and video windows
    cv2.destroyAllWindows()
    # calculate and print the average FPS
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")
