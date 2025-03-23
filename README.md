# ML-End-to-End-HotelBooking-Project 

# Create the conda environment  
$ conda create -n hotelbooking python=3.11 -y
conda activate hotelbooking 
pip install-r requirements.txt 


## Steps to Commit Code in Git

### 1. Initialize Git Repository (If Not Already Initialized)
```bash
git init

2. Check the Status of Your Changes 
```bash
git status

3. Stage Files for Commit
```bash
git add . 

 4. Commit Changes with a Meaningful Message
 ```bash
git commit -m "Your meaningful commit message here"

 5. Connect to a Remote Repository (If Not Already Connected)
 ```bash
 git remote add origin <your-repository-URL> 

 6. Push Changes to Remote Repository
    To push to the main branch: 
    ```bash
git push -u origin main

7. If using a different branch: 
```bash  # To push to another branch, replace "main 
git push origin <branch-name>

8. Pull Latest Changes Before Pushing (To Avoid Merge Conflicts)
```bash 
git pull origin main --rebase 

9. Create and Switch to a New Branch (If Needed) 
```bash 
git checkout -b <new-branch-name>

10. Merge a Branch into Main
First, switch to the main branch: 
```bash
git checkout main
Then, merge the branch: 
```bash
git merge <branch-name>





