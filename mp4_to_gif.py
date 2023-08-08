import moviepy.editor as mp

def convert_mp4_to_gif(input_mp4, output_gif, start_time, end_time, fps=None, gif_width=1000):
    video_clip = mp.VideoFileClip(input_mp4).subclip(start_time, end_time)
    gif_clip = video_clip.resize(width=gif_width)
    gif_clip.write_gif(output_gif, fps=fps)


# Замените следующие значения на свои:
input_mp4 = 'outputs/testsave1.mp4'  # Входное видео в формате MP4
output_gif = 'outputs/g4.gif'  # Имя выходного GIF файла
start_time = 25  # Время начала нарезки (в секундах)
end_time = 33  # Время окончания нарезки (в секундах)

convert_mp4_to_gif(input_mp4, output_gif, start_time, end_time)
