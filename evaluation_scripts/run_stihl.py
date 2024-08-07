import os
import subprocess

sequnces = {
    "kwald/drosselweg/flaeche1": [
        "2023-08-18", "2023-09-15", "2024-01-13", "2024-04-11", "2024-05-29_1", "2024-05-29_2", "2024-05-29_3", "2024-05-29_4"],
    "kwald/drosselweg/flaeche2": [
        "2023-08-18", "2023-12-21", "2024-01-13", "2024-04-11", "2024-05-29_1", "2024-05-30_1", "2024-05-30_2"],
    "esslingen/hse_dach": [
        "2023-07-20", "2023-11-07", "2024-01-27", "2024-04-14"],
    "esslingen/hse_hinterhof": [
        "2023-07-31", "2023-11-07", "2024-04-14", "2024-05-08", "2024-05-13_1", "2024-05-13_2", "2024-05-24_2"],
    "esslingen/hse_sporthalle": [
        "2023-09-11", "2023-11-23", "2024-02-19", "2024-04-14", "2024-05-07", "2024-05-08_1", "2024-05-08_2", "2024-05-24_1"],
}

base_data_path = "/data_storage/tum-datasets"
base_output_path = "/home/user/projects/DROID-SLAM/output"
d435i_script_path = "/home/user/projects/DROID-SLAM/evaluation_scripts/test_stihl_d435i.py"
pi_cam_script_path = "/home/user/projects/DROID-SLAM/evaluation_scripts/test_stihl_pi-cam-02.py"
t265_script_path = "/home/user/projects/DROID-SLAM/evaluation_scripts/test_stihl_t265.py"

for location, dates in sequnces.items():
    for date in dates:
        print(f"Running Location: {location} - Date: {date}")

        print(f"\tD435i Mono")
        result = subprocess.run([
            "python", 
            d435i_script_path,
            "--datapath", os.path.join(base_data_path, location, date, 'tum', 'd435i'),
            "--outputpath", os.path.join(base_output_path, location, date),
            "--disable_vis"],
            capture_output=True, text=True)

        print(f"\tD435i RGBD")
        result = subprocess.run([
            "python", 
            d435i_script_path,
            "--datapath", os.path.join(base_data_path, location, date, 'tum', 'd435i'),
            "--outputpath", os.path.join(base_output_path, location, date),
            "--disable_vis",
            "--depth"],
            capture_output=True, text=True)

        print(f"\tPi-Cam-02 Mono")
        result = subprocess.run([
            "python", 
            pi_cam_script_path,
            "--datapath", os.path.join(base_data_path, location, date, 'tum', 'pi_cam_02'),
            "--outputpath", os.path.join(base_output_path, location, date),
            "--disable_vis"],
            capture_output=True, text=True)