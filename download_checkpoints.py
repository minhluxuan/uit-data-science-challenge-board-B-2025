import gdown
folder_link = "https://drive.google.com/drive/folders/1HmD1ngm1Idy8WfQaqeM3v_gUljMasuLw?usp=sharing"
gdown.download_folder(folder_link, output="./checkpoints", quiet=False, use_cookies=True)