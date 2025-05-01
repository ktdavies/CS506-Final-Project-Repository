import os
import sys
import subprocess

REPO_NAME = "CS506-Final-Project-Repository"
REPO_URL = "https://github.com/ktdavies/CS506-Final-Project-Repository"

# Clone the repo if it doesn't exist
if not os.path.exists(REPO_NAME):
    print(f"Cloning {REPO_NAME}...")
    subprocess.run(["git", "clone", REPO_URL])

# Change into the repo directory
os.chdir(REPO_NAME)

# Add current directory to Python path so imports work
sys.path.append(os.getcwd())

# Import valid modules only
import CorrAndHTest
import DBSCANbyAirline
import DelayByCity_Visualization
import MultiLinearRegression
import model_final
import climate_change

def main():
    print("Clustering delays by airline using DBSCAN...")
    DBSCANbyAirline.main()
    print("Clustering delays by airline using DBSCAN is done.")

    print("Visualizing delays by city...")
    DelayByCity_Visualization.main()
    print("Visualizing delays by city is done.")
    
    print("Performing correlation and hypothesis testing...")
    CorrAndHTest.main()
    print("Performing correlation and hypothesis testing is done. ")

    print("Running multi-linear regression on delay data...")
    MultiLinearRegression.main()
    print("Running multi-linear regression on delay data is done.")
    
    print("Running the Climate Change file...")
    climate_change.main()
    print("Running the Climate Change file is done.")

    print("Running Prediction Model...")
    model_final.main()
    print("Running Prediction Model is done.")

    print("All tasks completed successfully.")

if __name__ == "__main__":
    main()
