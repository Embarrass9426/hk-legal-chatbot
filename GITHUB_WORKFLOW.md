# 🚀 GitHub Workflow Cheat Sheet

A quick reference for the commands you'll use most often.

---

## 🛠️ 1. The Basic Loop (Daily Work)
Use this when you've made changes and want to save them to GitHub.

```powershell
# 1. Check what changed
git status

# 2. Stage all changes
git add .

# 3. Commit with a message
git commit -m "feat: added rag integration to backend"

# 4. Push to GitHub
git push origin main
```

---

## 🌿 2. Branching (Safe Feature Development)
Always work on a branch to keep the `main` branch stable.

```powershell
# Create and switch to a new branch
git checkout -b feature/scraping-logic

# Switch back to main
git checkout main

# List all branches
git branch

# Merge a branch into main (do this while on main)
git merge feature/scraping-logic
```

---

## 🔄 3. Syncing with GitHub
Use these to get the latest code from the cloud.

```powershell
# Download latest changes from GitHub
git pull origin main

# Fetch changes without merging them yet
git fetch origin
```

---

## ⚠️ 4. Undoing & Fixing
When things go wrong.

```powershell
# Undo the last commit but keep the code changes
git reset --soft HEAD~1

# Discard all local changes (DANGER: wipes your work!)
git reset --hard HEAD

# Discard changes in a specific file
git checkout -- path/to/file.py
```

---

## 📦 5. Stashing (Temporary Save)
Use this if you need to switch branches but aren't ready to commit yet.

```powershell
# Save changes to a "stash"
git stash

# Bring changes back
git stash pop
```

---

## 💡 Pro Tips
- **Commit often**: Small commits are easier to fix than huge ones.
- **Descriptive messages**: Use `feat:`, `fix:`, or `docs:` prefixes.
- **Pull before you push**: Always `git pull` before starting work to avoid conflicts.
