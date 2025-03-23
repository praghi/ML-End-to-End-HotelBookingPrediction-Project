# ML End-to-End Hotel Booking Project

## 1. Create the Conda Environment  
```bash
conda create -n hotelbooking python=3.11 -y
conda activate hotelbooking
pip install -r requirements.txt


2. Steps to Commit Code in Git
Initialize Git Repository (If Not Already Initialized) 
git init

Check the Status of Your Changes
git status

Stage Files for Commit
To stage all changes:
git add .

Commit Changes with a Meaningful Message
git commit -m "Your meaningful commit message here"

Connect to a Remote Repository (If Not Already Connected)
git remote add origin <your-repository-URL>

Push Changes to Remote Repository
To push to the main branch:
git push -u origin main

To push to a different branch:
git push origin <branch-name>

Pull Latest Changes Before Pushing (To Avoid Merge Conflicts)
git pull origin main --rebase

Create and Switch to a New Branch (If Needed)
git checkout -b <new-branch-name>

Merge a Branch into Main
First, switch to the main branch:
git checkout main

Then, merge the branch:
git merge <branch-name>



