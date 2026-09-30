import os
import shutil

print("Applying custom ANTARA patches to submodules...")
shutil.copy("patches/SLAM3R/recon.py", "SLAM3R/recon.py")
shutil.copy("patches/SLAM3R/slam3r/app/app_offline.py", "SLAM3R/slam3r/app/app_offline.py")
print("Patches applied successfully.")
