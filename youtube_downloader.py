import pytube
import os


def download_video(video_url, save_path='downloaded_videos'):
    """Скачивает видео с youtube по заданному url"""

    os.makedirs('downloaded_videos', exist_ok=True)
    try:
        yt = pytube.YouTube(video_url)
        print(f'{yt.title=}')
        print(f'{yt.length=}')
        # yt.streams.get_lowest_resolution().download(output_path=save_path)
        yt.streams.get_by_resolution(resolution='720p').download(output_path=save_path)
        print("Видео успешно скачано.")
    except Exception as e:
        print("Произошла ошибка при скачивании видео:", e)


if __name__ == '__main__':
    url = str(input('Введите url ... ')).strip()
    download_video(video_url=url)
