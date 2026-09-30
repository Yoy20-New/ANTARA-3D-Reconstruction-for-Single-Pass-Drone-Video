import os
import urllib.request

def download_file(url, dest):
    print(f"Downloading {os.path.basename(dest)}...")
    if not os.path.exists(dest):
        urllib.request.urlretrieve(url, dest)
        print(f"Successfully downloaded to {dest}")
    else:
        print(f"File {dest} already exists. Skipping.")

if __name__ == '__main__':
    os.makedirs('checkpoints', exist_ok=True)
    
    # SAM 2 Weight
    sam2_url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
    download_file(sam2_url, os.path.join('checkpoints', 'sam2.1_hiera_tiny.pt'))
    
    # SLAM3R Weights (Cached via HF)
    print("Caching SLAM3R HuggingFace weights...")
    try:
        from slam3r.models import Image2PointsModel, Local2WorldModel
        Image2PointsModel.from_pretrained('siyan824/slam3r_i2p')
        Local2WorldModel.from_pretrained('siyan824/slam3r_l2w')
        print("SLAM3R weights successfully cached.")
    except Exception as e:
        print(f"Could not automatically cache SLAM3R weights. Make sure SLAM3R is installed. Error: {e}")
        
    print("All necessary AI weights have been downloaded!")
