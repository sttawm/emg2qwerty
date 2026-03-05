# Team Lead Guide - Adding Teammates

Quick reference for adding new teammates to the project.

## Adding a New Teammate (2 minutes)

### Step 1: Grant Access

```bash
cd /Users/sttawm/dev/emg2qwerty/bootstrap
./grant_access.sh teammate@gmail.com
```

This grants them:
- Read access to shared data bucket
- Read/write access to shared logs bucket
- TensorBoard viewer access

### Step 2: Share the Repo

Send them:
```
Repository: https://github.com/sttawm/emg2qwerty
Setup Guide: See TEAMMATE_SETUP.md in the repo
```

**That's it!** They run `./bootstrap/setup_teammate.sh` and they're ready to train.

---

## What You've Already Done (One-Time)

✅ Created shared resources
✅ Uploaded dataset
✅ Committed `bootstrap/shared_config.env` to git

Teammates just need:
1. Your permission (grant_access.sh)
2. The repo (already on GitHub)
3. Their own GCP project (they create via setup_teammate.sh)

---

## Quick Teammate Checklist

When someone joins:
- [ ] Run: `./grant_access.sh teammate@gmail.com`
- [ ] Send them: Link to repo + "See TEAMMATE_SETUP.md"
- [ ] Done!

---

## Common Questions

**Q: Do I need to upload the dataset for each teammate?**
No! Dataset is uploaded once and shared.

**Q: Who pays for what?**
- You: Dataset storage (~$2/month)
- Them: Their own GPU compute

**Q: Can teammates see each other's experiments?**
Yes! All logs go to the shared bucket, visible in one TensorBoard.

**Q: Can I remove a teammate's access?**
Yes:
```bash
gsutil iam ch -d user:teammate@gmail.com:objectViewer gs://emg2qwerty-team-data
gsutil iam ch -d user:teammate@gmail.com:objectAdmin gs://emg2qwerty-team-logs
```

---

## Your Resources

**Shared Project:** emg2qwerty-team-shared
- Data: gs://emg2qwerty-team-data
- Logs: gs://emg2qwerty-team-logs

**Monitor costs:**
https://console.cloud.google.com/billing/reports?project=emg2qwerty-team-shared
