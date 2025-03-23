# ML-End-to-End-HotelBooking-Project 

# Create the conda environment  
```bash
$ conda create -n hotelbooking python=3.11 -y

conda activate hotelbooking 

pip install-r requirements.txt 


## Steps to Commit Code in Git

1. Initialize Git Repository (If Not Already Initialized)

git init

2. Check the Status of Your Changes 

git status

3. Stage Files for Commit

git add . 

4. Commit Changes with a Meaningful Message

git commit -m "Your meaningful commit message here"

5. Connect to a Remote Repository (If Not Already Connected)
 
 git remote add origin <your-repository-URL> 

6. Push Changes to Remote Repository
    To push to the main branch: 
  
git push -u origin main

7. If using a different branch: 

# To push to another branch, replace "main 
git push origin <branch-name>

8. Pull Latest Changes Before Pushing (To Avoid Merge Conflicts)

git pull origin main --rebase 

9. Create and Switch to a New Branch (If Needed) 

git checkout -b <new-branch-name>

10. Merge a Branch into Main
First, switch to the main branch: 

git checkout main
Then, merge the branch: 

git merge <branch-name>





